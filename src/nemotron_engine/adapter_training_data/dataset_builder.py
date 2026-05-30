"""End-to-end adapter SFT dataset builder."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, fields
import json
from pathlib import Path
import random
from typing import Any, Mapping

from nemotron_engine.competition_sprint import CompetitionProblem, parse_competition_prompt
from nemotron_engine.core.schemas import stable_hash

from .hard_example_miner import HardMiningConfig, mine_hard_examples, oversample_hard_examples
from .sft_formatter import SFTExample, format_direct_answer_example, format_family_tagged_example
from .solver_distillation import build_solver_distilled_examples, distill_solver_predictions


class AdapterTrainingDataError(ValueError):
    """Raised when adapter training data construction fails."""


@dataclass(frozen=True)
class AdapterDatasetBuildReport:
    train_csv_path: str
    output_dir: str
    parsed_count: int
    parse_error_count: int
    train_count: int
    validation_count: int
    file_counts: Mapping[str, int]
    family_counts: Mapping[str, int]
    answer_kind_counts: Mapping[str, int]
    hard_family_counts: Mapping[str, int]
    artifact_hashes: Mapping[str, str]
    seed: int
    report_hash: str = ""

    def __post_init__(self) -> None:
        for field_name in ("parsed_count", "parse_error_count", "train_count", "validation_count", "seed"):
            value = int(getattr(self, field_name))
            if value < 0 and field_name != "seed":
                raise AdapterTrainingDataError(f"{field_name} cannot be negative.")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "train_csv_path", _text(self.train_csv_path, "train_csv_path"))
        object.__setattr__(self, "output_dir", _text(self.output_dir, "output_dir"))
        object.__setattr__(self, "file_counts", _sorted_int_map(self.file_counts))
        object.__setattr__(self, "family_counts", _sorted_int_map(self.family_counts))
        object.__setattr__(self, "answer_kind_counts", _sorted_int_map(self.answer_kind_counts))
        object.__setattr__(self, "hard_family_counts", _sorted_int_map(self.hard_family_counts))
        object.__setattr__(self, "artifact_hashes", {str(k): str(v) for k, v in sorted(self.artifact_hashes.items())})
        _set_or_check_hash(self, "report_hash")


def build_adapter_training_datasets(
    train_csv_path: str | Path = "data/nemotron_competition/train.csv",
    output_dir: str | Path = "artifacts/adapter_training",
    *,
    seed: int = 1337,
    validation_per_family: int = 50,
    hard_family_caps: Mapping[str, int] | None = None,
) -> AdapterDatasetBuildReport:
    """Build deterministic SFT data artifacts from labeled competition train rows."""

    train_path = Path(train_csv_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    problems, parse_errors = _load_labeled_train_problems(train_path)
    train_problems, validation_problems = _split_by_family(problems, seed=seed, validation_per_family=validation_per_family)

    direct = tuple(format_direct_answer_example(problem) for problem in train_problems)
    family_tagged = tuple(format_family_tagged_example(problem) for problem in train_problems)
    records = distill_solver_predictions(train_problems, corpus_source_path=str(train_path))
    solver_distilled = build_solver_distilled_examples(train_problems, records)
    hard_problems = mine_hard_examples(train_problems, records)
    hard_config = HardMiningConfig(
        family_caps=hard_family_caps
        or {
            "bit_manipulation": 800,
            "equation_symbolic": 800,
            "gravity_numeric": 500,
            "unit_conversion": 500,
            "cipher_text": 400,
        },
        seed=seed,
        default_cap=300,
    )
    hard_oversampled = oversample_hard_examples(hard_problems, config=hard_config)
    validation = tuple(format_family_tagged_example(problem, source="validation_family_balanced") for problem in validation_problems)

    artifacts = {
        "train_direct.jsonl": direct,
        "train_family_tagged.jsonl": family_tagged,
        "train_solver_distilled.jsonl": solver_distilled,
        "train_hard_oversampled.jsonl": hard_oversampled,
        "validation_family_balanced.jsonl": validation,
    }
    file_counts: dict[str, int] = {}
    artifact_hashes: dict[str, str] = {}
    for filename, rows in artifacts.items():
        path = out / filename
        _write_jsonl(path, rows)
        file_counts[filename] = len(rows)
        artifact_hashes[filename] = _sha256_text(path.read_text(encoding="utf-8"))

    family_counts = Counter(problem.family for problem in problems)
    answer_kind_counts = Counter(problem.answer_kind or "generic_string" for problem in problems)
    hard_family_counts = Counter(problem.family for problem in hard_problems)
    report = AdapterDatasetBuildReport(
        train_csv_path=str(train_path),
        output_dir=str(out),
        parsed_count=len(problems),
        parse_error_count=len(parse_errors),
        train_count=len(train_problems),
        validation_count=len(validation_problems),
        file_counts=file_counts,
        family_counts=dict(family_counts),
        answer_kind_counts=dict(answer_kind_counts),
        hard_family_counts=dict(hard_family_counts),
        artifact_hashes=artifact_hashes,
        seed=seed,
    )
    manifest = _report_to_dict(report)
    manifest["parse_errors"] = parse_errors
    manifest["hard_config"] = _report_to_dict(hard_config)
    manifest["no_test_labels_used"] = True
    manifest_path = out / "dataset_manifest.json"
    _write_json(manifest_path, manifest)
    return report


def _load_labeled_train_problems(path: Path) -> tuple[tuple[CompetitionProblem, ...], list[dict[str, Any]]]:
    problems: list[CompetitionProblem] = []
    errors: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            problem_id = str(row.get("id") or row.get("problem_id") or f"row-{index}")
            try:
                answer = row.get("answer")
                if answer is None:
                    raise AdapterTrainingDataError("train row missing answer.")
                problems.append(parse_competition_prompt(problem_id, str(row.get("prompt") or ""), answer))
            except Exception as exc:
                errors.append({"problem_id": problem_id, "row_index": index, "reason": str(exc)})
    return tuple(problems), errors


def _split_by_family(problems: tuple[CompetitionProblem, ...], *, seed: int, validation_per_family: int) -> tuple[tuple[CompetitionProblem, ...], tuple[CompetitionProblem, ...]]:
    grouped: dict[str, list[CompetitionProblem]] = defaultdict(list)
    for problem in problems:
        grouped[problem.family].append(problem)
    rng = random.Random(seed)
    train: list[CompetitionProblem] = []
    validation: list[CompetitionProblem] = []
    for family in sorted(grouped):
        rows = list(grouped[family])
        rng.shuffle(rows)
        rows.sort(key=lambda item: stable_hash({"seed": seed, "family": family, "id": item.problem_id}))
        count = min(max(1, int(validation_per_family)), max(1, len(rows) // 5), len(rows))
        validation.extend(rows[:count])
        train.extend(rows[count:])
    return tuple(sorted(train, key=lambda item: item.problem_id)), tuple(sorted(validation, key=lambda item: (item.family, item.problem_id)))


def _write_jsonl(path: Path, rows: tuple[SFTExample, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_json_obj(), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False, indent=2)
        handle.write("\n")


def _report_to_dict(record: object) -> dict[str, Any]:
    return {item.name: getattr(record, item.name) for item in fields(record)}


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _set_or_check_hash(instance: object, hash_field: str) -> None:
    expected = stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})
    current = getattr(instance, hash_field)
    if not current:
        object.__setattr__(instance, hash_field, expected)
    elif current != expected:
        raise AdapterTrainingDataError(f"{hash_field} does not match payload.")


def _sorted_int_map(values: Mapping[str, int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(values.items())}


def _text(value: object, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise AdapterTrainingDataError(f"{field_name} must be non-empty.")
    return text


__all__ = ["AdapterDatasetBuildReport", "AdapterTrainingDataError", "build_adapter_training_datasets"]
