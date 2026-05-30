from __future__ import annotations

"""Production CLI entrypoint for router/archetype-training support.

This script prepares manifest-backed router/archetype datasets from current
offline artifacts and can optionally hand them off to an explicit external
training backend. It does not pretend to train a model when no backend is
available.
"""

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.common.constants import OFFLINE_ARTIFACT_VERSION, OFFLINE_SPLIT_RATIOS, OFFLINE_SPLIT_SALT
from src.common.schemas import ParsedProblem
from src.routing.archetype_predictor import get_supported_archetypes

try:  # optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

LOGGER = logging.getLogger("train_router")
SCRIPT_SCHEMA_VERSION = "train_router_script.v1"
ARTIFACT_KIND = "router_training_run"
DEFAULT_OUTPUT_DIR = "artifacts/router/training"


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        populate_by_name=True,
    )


class SplitName(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    CALIBRATION = "calibration"


class SplitStrategy(str, Enum):
    SOURCE = "source"
    MANIFEST = "manifest"
    HASH = "hash"


class RunStatus(str, Enum):
    COMPLETED = "completed"
    READY = "ready"
    INVALID_DATASET = "invalid_dataset"
    MISSING_INPUT = "missing_input"
    UNAVAILABLE_RUNTIME = "unavailable_runtime"


class RouterTrainingScriptConfig(StrictModel):
    run_name: str = "router_training"
    dataset_paths: list[str] = Field(default_factory=list)
    split_manifest_path: str | None = None
    output_dir: str = DEFAULT_OUTPUT_DIR
    artifact_version: str = OFFLINE_ARTIFACT_VERSION
    split_strategy: SplitStrategy = SplitStrategy.SOURCE
    split_ratios: tuple[float, float, float] = OFFLINE_SPLIT_RATIOS
    split_salt: str = OFFLINE_SPLIT_SALT
    min_examples: int = Field(default=4, ge=1)
    min_valid_examples: int = Field(default=1, ge=0)
    min_calibration_examples: int = Field(default=1, ge=0)
    preserve_problem_metadata: bool = True
    emit_jsonl: bool = True
    write_summary_json: bool = True
    allow_unlabeled_examples: bool = False
    overwrite: bool = False
    seed: int = 1729
    trainer_backend: str = "none"
    trainer_command: list[str] = Field(default_factory=list)
    trainer_work_dir: str | None = None
    trainer_env: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_backend(self) -> "RouterTrainingScriptConfig":
        if self.trainer_backend not in {"none", "subprocess"}:
            raise ValueError("trainer_backend must be one of: none, subprocess")
        if self.trainer_backend == "subprocess" and not self.trainer_command:
            raise ValueError("trainer_command is required when trainer_backend='subprocess'")
        if self.split_strategy == SplitStrategy.MANIFEST and not self.split_manifest_path:
            raise ValueError("split_manifest_path is required when split_strategy='manifest'")
        return self


class ArchetypeTrainingExample(StrictModel):
    problem_id: str
    raw_text: str
    domain: str
    target: str = ""
    answer_type: str = ""
    archetype_labels: list[str] = Field(default_factory=list)
    label_scores: dict[str, float] = Field(default_factory=dict)
    source_split: SplitName | None = None
    source_artifact_id: str | None = None
    source_record_ids: list[str] = Field(default_factory=list)
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class SplitSummary(StrictModel):
    split_name: str
    count: int
    label_frequencies: dict[str, int] = Field(default_factory=dict)
    problem_ids: list[str] = Field(default_factory=list)


class DatasetValidationReport(StrictModel):
    valid: bool
    num_examples: int = 0
    num_unique_problem_ids: int = 0
    duplicate_problem_ids: list[str] = Field(default_factory=list)
    split_conflict_problem_ids: list[str] = Field(default_factory=list)
    unsupported_labels: dict[str, list[str]] = Field(default_factory=dict)
    empty_label_problem_ids: list[str] = Field(default_factory=list)
    label_frequencies: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class EmittedArtifactPaths(StrictModel):
    output_root: str
    run_dir: str
    manifest_path: str
    config_snapshot_path: str
    train_path: str | None = None
    valid_path: str | None = None
    calibration_path: str | None = None
    split_summary_path: str | None = None
    dataset_validation_path: str | None = None
    trainer_report_path: str | None = None
    checkpoint_dir: str | None = None
    log_dir: str | None = None
    trained_model_path: str | None = None


class RouterTrainingRunManifest(StrictModel):
    artifact_kind: str = ARTIFACT_KIND
    schema_version: str = SCRIPT_SCHEMA_VERSION
    artifact_version: str = OFFLINE_ARTIFACT_VERSION
    run_name: str
    run_id: str
    status: str
    trainer_backend: str
    dataset_sources: list[dict[str, Any]] = Field(default_factory=list)
    validation: dict[str, Any] = Field(default_factory=dict)
    split_counts: dict[str, int] = Field(default_factory=dict)
    split_label_frequencies: dict[str, dict[str, int]] = Field(default_factory=dict)
    supported_archetypes: list[str] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    trainer_report: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class ResolvedDatasetSource:
    requested_path: str
    resolved_path: str
    source_kind: str
    record_count: int


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _normalize_text(text: str | None) -> str:
    return " ".join((text or "").strip().split())


def _stable_hash(prefix: str, payload: Any) -> str:
    raw = f"{prefix}::{_stable_json(payload)}"
    return f"{prefix}_{sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_stable_json(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_stable_json(row) + "\n")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # pragma: no cover
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass


def _read_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(raw) if raw.strip() else {}
    elif suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is unavailable; cannot read YAML config")
        payload = yaml.safe_load(raw) or {}
    else:
        raise ValueError(f"Unsupported config file type: {path.suffix}")
    if not isinstance(payload, dict):
        raise ValueError("Config file must contain a top-level mapping")
    return payload


def _parse_kv_items(items: Sequence[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid KEY=VALUE item: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid KEY=VALUE item with empty key: {item!r}")
        env[key] = value
    return env


def _merge_cli_overrides(base: dict[str, Any], cli: argparse.Namespace) -> dict[str, Any]:
    merged = dict(base)
    scalars = {
        "run_name": cli.run_name,
        "split_manifest_path": cli.split_manifest,
        "output_dir": cli.output_dir,
        "artifact_version": cli.artifact_version,
        "split_strategy": cli.split_strategy,
        "min_examples": cli.min_examples,
        "min_valid_examples": cli.min_valid_examples,
        "min_calibration_examples": cli.min_calibration_examples,
        "seed": cli.seed,
        "trainer_backend": cli.trainer_backend,
        "trainer_work_dir": cli.trainer_work_dir,
    }
    for key, value in scalars.items():
        if value is not None:
            merged[key] = value
    if cli.dataset:
        merged["dataset_paths"] = list(cli.dataset)
    if cli.trainer_command:
        merged["trainer_command"] = list(cli.trainer_command)
    if cli.trainer_env:
        merged["trainer_env"] = _parse_kv_items(cli.trainer_env)
    for flag in (
        "preserve_problem_metadata",
        "emit_jsonl",
        "write_summary_json",
        "allow_unlabeled_examples",
        "overwrite",
    ):
        value = getattr(cli, flag)
        if value is not None:
            merged[flag] = value
    return merged


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and optionally hand off router/archetype training artifacts")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON/YAML config file")
    parser.add_argument("--dataset", action="append", default=None, help="Dataset or manifest path; may be repeated")
    parser.add_argument("--split-manifest", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--artifact-version", type=str, default=None)
    parser.add_argument("--split-strategy", choices=[item.value for item in SplitStrategy], default=None)
    parser.add_argument("--min-examples", type=int, default=None)
    parser.add_argument("--min-valid-examples", type=int, default=None)
    parser.add_argument("--min-calibration-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--preserve-problem-metadata", dest="preserve_problem_metadata", action="store_true")
    parser.add_argument("--no-preserve-problem-metadata", dest="preserve_problem_metadata", action="store_false")
    parser.set_defaults(preserve_problem_metadata=None)

    parser.add_argument("--emit-jsonl", dest="emit_jsonl", action="store_true")
    parser.add_argument("--no-emit-jsonl", dest="emit_jsonl", action="store_false")
    parser.set_defaults(emit_jsonl=None)

    parser.add_argument("--write-summary-json", dest="write_summary_json", action="store_true")
    parser.add_argument("--no-write-summary-json", dest="write_summary_json", action="store_false")
    parser.set_defaults(write_summary_json=None)

    parser.add_argument("--allow-unlabeled-examples", dest="allow_unlabeled_examples", action="store_true")
    parser.add_argument("--disallow-unlabeled-examples", dest="allow_unlabeled_examples", action="store_false")
    parser.set_defaults(allow_unlabeled_examples=None)

    parser.add_argument("--overwrite", dest="overwrite", action="store_true")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.set_defaults(overwrite=None)

    parser.add_argument("--trainer-backend", choices=["none", "subprocess"], default=None)
    parser.add_argument(
        "--trainer-command",
        nargs=argparse.REMAINDER,
        help=(
            "Subprocess backend command. Supports placeholders: "
            "{run_dir} {train_manifest} {valid_manifest} {calibration_manifest} {manifest} "
            "{checkpoint_dir} {logs_dir} {seed} {run_name} {artifact_version}"
        ),
    )
    parser.add_argument("--trainer-work-dir", type=str, default=None)
    parser.add_argument("--trainer-env", action="append", default=None, help="Extra KEY=VALUE env for trainer")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _load_json_lines(path: Path) -> list[Any]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_dataset_path(path: Path) -> list[Any]:
    if path.is_dir():
        for candidate in (path / "records.jsonl", path / "manifest.json"):
            if candidate.exists():
                return _load_dataset_path(candidate)
        raise ValueError(f"Directory does not contain records.jsonl or manifest.json: {path}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _load_json_lines(path)
    if suffix != ".json":
        raise ValueError(f"Unsupported dataset file type: {path.suffix}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if payload.get("artifact_kind") == "offline_training_corpus":
            sibling = path.with_name("records.jsonl")
            if sibling.exists():
                return _load_json_lines(sibling)
        if payload.get("artifact_kind") == "archetype_training_manifest" and isinstance(payload.get("split_bundle"), Mapping):
            split_bundle = payload["split_bundle"]
            rows: list[Any] = []
            for key in ("train", "valid", "calibration"):
                rows.extend(list(split_bundle.get(key, [])))
            return rows
        for key in ("records", "examples", "problems"):
            value = payload.get(key)
            if isinstance(value, list):
                return list(value)
        raise ValueError("JSON dataset must contain records/examples/problems or a known artifact manifest")
    if isinstance(payload, list):
        return list(payload)
    raise ValueError("Unsupported JSON dataset payload")


def _resolve_dataset_sources(paths: Sequence[str]) -> tuple[list[Any], list[ResolvedDatasetSource]]:
    if not paths:
        raise FileNotFoundError("At least one --dataset path is required")
    loaded: list[Any] = []
    resolved: list[ResolvedDatasetSource] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {raw}")
        rows = _load_dataset_path(path)
        if not rows:
            raise ValueError(f"Dataset resolved but contains no rows: {path}")
        loaded.extend(rows)
        resolved.append(
            ResolvedDatasetSource(
                requested_path=raw,
                resolved_path=str(path),
                source_kind="directory" if path.is_dir() else (path.suffix.lower().lstrip(".") or "file"),
                record_count=len(rows),
            )
        )
    return loaded, resolved


def _load_split_manifest(path: Path) -> dict[str, SplitName]:
    rows = _load_dataset_path(path)
    assignments: dict[str, SplitName] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"Invalid split manifest row type: {type(row)!r}")
        problem_id = _normalize_text(str(row.get("problem_id") or ""))
        if not problem_id:
            raise ValueError("Split manifest row is missing problem_id")
        assignments[problem_id] = SplitName(str(row.get("split")))
    return assignments


def _hash_split(problem_id: str, config: RouterTrainingScriptConfig) -> SplitName:
    ratios = config.split_ratios
    total = sum(ratios)
    normalized = tuple((float(v) / total) if total > 0 else d for v, d in zip(ratios, OFFLINE_SPLIT_RATIOS))
    token = f"{config.split_salt}::{problem_id or 'unknown_problem'}"
    bucket = int(sha1(token.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < normalized[0]:
        return SplitName.TRAIN
    if bucket < normalized[0] + normalized[1]:
        return SplitName.VALID
    return SplitName.CALIBRATION


def _normalize_label_payload(payload: Any, supported: set[str]) -> tuple[list[str], dict[str, float], list[str]]:
    labels: list[str] = []
    scores: dict[str, float] = {}
    unsupported: list[str] = []
    if isinstance(payload, Mapping):
        for label, score in payload.items():
            label_name = _normalize_text(str(label))
            try:
                numeric = float(score)
            except Exception:
                numeric = 0.0
            if label_name in supported:
                scores[label_name] = max(scores.get(label_name, 0.0), max(0.0, numeric))
            elif label_name:
                unsupported.append(label_name)
        labels = sorted(name for name, score in scores.items() if score > 0.0)
        return labels, {key: scores[key] for key in sorted(scores)}, sorted(set(unsupported))

    values: list[str] = []
    if payload is None:
        values = []
    elif isinstance(payload, str):
        norm = _normalize_text(payload)
        values = [norm] if norm else []
    elif isinstance(payload, (list, tuple, set)):
        values = [_normalize_text(str(item)) for item in payload if _normalize_text(str(item))]
    else:
        values = [_normalize_text(str(payload))]

    for label in values:
        if label in supported:
            labels.append(label)
            scores[label] = 1.0
        else:
            unsupported.append(label)
    return sorted(set(labels)), {key: scores[key] for key in sorted(scores)}, sorted(set(unsupported))


def _coerce_example(row: Any, supported: set[str], preserve_metadata: bool) -> tuple[ArchetypeTrainingExample | None, list[str]]:
    if isinstance(row, ArchetypeTrainingExample):
        return row, []

    if isinstance(row, ParsedProblem):
        labels, scores, unsupported = _normalize_label_payload(row.likely_archetypes, supported)
        meta = {
            "difficulty_seed": row.difficulty_seed,
            "parse_quality": dict(row.parse_quality),
            "symmetries": list(row.symmetries),
            "parity_cues": list(row.parity_cues),
            "integrality_constraints": list(row.integrality_constraints),
            "original_metadata": dict(row.metadata),
        } if preserve_metadata else {}
        return ArchetypeTrainingExample(
            problem_id=row.problem_id,
            raw_text=_normalize_text(row.raw_text),
            domain=row.domain.value if hasattr(row.domain, "value") else str(row.domain),
            target=row.target,
            answer_type=row.answer_type,
            archetype_labels=labels,
            label_scores=scores,
            source_metadata=meta,
        ), unsupported

    if not isinstance(row, Mapping):
        raise TypeError(f"Unsupported dataset row type: {type(row)!r}")

    payload = dict(row)
    if "record_id" in payload and "task_family" in payload and "provenance" in payload:
        if str(payload.get("task_family")) != "router":
            return None, []
        input_fields = payload.get("input_fields") if isinstance(payload.get("input_fields"), Mapping) else {}
        target_fields = payload.get("target_fields") if isinstance(payload.get("target_fields"), Mapping) else {}
        provenance = payload.get("provenance") if isinstance(payload.get("provenance"), Mapping) else {}
        source_meta = {
            "record_targets": target_fields,
            "record_features": payload.get("feature_fields") if isinstance(payload.get("feature_fields"), Mapping) else {},
            "provenance_metadata": provenance.get("metadata") if isinstance(provenance.get("metadata"), Mapping) else {},
        }
        text = _normalize_text(
            str(
                input_fields.get("problem_text")
                or source_meta["provenance_metadata"].get("trace_metadata", {}).get("problem_text", "")
                or ""
            )
        )
        domain = _normalize_text(
            str(
                source_meta["provenance_metadata"].get("trace_metadata", {}).get("domain")
                or next(iter((target_fields.get("problem_type") or {}).keys()), "unknown")
            )
        ) or "unknown"
        labels, scores, unsupported = _normalize_label_payload(target_fields.get("archetypes"), supported)
        split = payload.get("split")
        source_split = SplitName(str(split)) if split in {item.value for item in SplitName} else None
        return ArchetypeTrainingExample(
            problem_id=_normalize_text(str(payload.get("problem_id") or "")),
            raw_text=text,
            domain=domain,
            target=_normalize_text(str(source_meta["provenance_metadata"].get("trace_metadata", {}).get("target") or "")),
            answer_type=_normalize_text(str(source_meta["provenance_metadata"].get("trace_metadata", {}).get("answer_type") or "")),
            archetype_labels=labels,
            label_scores=scores,
            source_split=source_split,
            source_artifact_id=_normalize_text(str(provenance.get("upstream_artifact_id") or "")) or None,
            source_record_ids=sorted({
                _normalize_text(str(payload.get("record_id") or "")),
                _normalize_text(str(provenance.get("source_record_id") or "")),
            } - {""}),
            source_metadata=source_meta if preserve_metadata else {},
        ), unsupported

    if {"problem_id", "raw_text", "domain", "archetype_labels"}.issubset(payload):
        example = ArchetypeTrainingExample.model_validate(payload)
        unsupported = [label for label in example.archetype_labels if label not in supported]
        if unsupported:
            labels = [label for label in example.archetype_labels if label in supported]
            scores = {k: float(v) for k, v in example.label_scores.items() if k in supported}
            example = example.model_copy(update={"archetype_labels": labels, "label_scores": scores})
        return example, sorted(set(unsupported))

    parsed_problem = ParsedProblem.model_validate(payload)
    return _coerce_example(parsed_problem, supported, preserve_metadata)


def _merge_examples(examples: Sequence[ArchetypeTrainingExample]) -> tuple[list[ArchetypeTrainingExample], list[str], list[str]]:
    by_problem: dict[str, list[ArchetypeTrainingExample]] = {}
    for example in examples:
        by_problem.setdefault(example.problem_id, []).append(example)
    merged: list[ArchetypeTrainingExample] = []
    duplicates: list[str] = []
    split_conflicts: list[str] = []
    for problem_id, items in sorted(by_problem.items()):
        if len(items) > 1:
            duplicates.append(problem_id)
        source_splits = sorted({item.source_split.value for item in items if item.source_split is not None})
        if len(source_splits) > 1:
            split_conflicts.append(problem_id)
        label_scores: dict[str, float] = {}
        for item in items:
            for label, score in item.label_scores.items():
                label_scores[label] = max(label_scores.get(label, 0.0), float(score))
        merged.append(
            ArchetypeTrainingExample(
                problem_id=problem_id,
                raw_text=max((item.raw_text for item in items), key=len, default=""),
                domain=next((item.domain for item in items if item.domain and item.domain != "unknown"), "unknown"),
                target=next((item.target for item in items if item.target), ""),
                answer_type=next((item.answer_type for item in items if item.answer_type), ""),
                archetype_labels=sorted(label_scores),
                label_scores={key: label_scores[key] for key in sorted(label_scores)},
                source_split=items[0].source_split if len(source_splits) == 1 else None,
                source_artifact_id=next((item.source_artifact_id for item in items if item.source_artifact_id), None),
                source_record_ids=sorted({rid for item in items for rid in item.source_record_ids}),
                source_metadata={"merged_example_count": len(items)},
            )
        )
    return merged, duplicates, split_conflicts


def _assign_splits(
    examples: Sequence[ArchetypeTrainingExample],
    config: RouterTrainingScriptConfig,
    manifest_assignments: Mapping[str, SplitName] | None,
) -> dict[str, list[ArchetypeTrainingExample]]:
    grouped = {SplitName.TRAIN.value: [], SplitName.VALID.value: [], SplitName.CALIBRATION.value: []}
    for example in examples:
        if config.split_strategy == SplitStrategy.MANIFEST and manifest_assignments:
            split = manifest_assignments.get(example.problem_id, _hash_split(example.problem_id, config))
        elif config.split_strategy == SplitStrategy.SOURCE and example.source_split is not None:
            split = example.source_split
        else:
            split = _hash_split(example.problem_id, config)
        grouped[split.value].append(example.model_copy(update={"source_split": split}))
    for rows in grouped.values():
        rows.sort(key=lambda item: item.problem_id)
    return grouped


def _split_summary(name: str, examples: Sequence[ArchetypeTrainingExample]) -> SplitSummary:
    counts: dict[str, int] = {}
    for example in examples:
        for label in example.archetype_labels:
            counts[label] = counts.get(label, 0) + 1
    return SplitSummary(split_name=name, count=len(examples), label_frequencies=dict(sorted(counts.items())), problem_ids=[e.problem_id for e in examples])


def _validate_dataset(
    *,
    examples: Sequence[ArchetypeTrainingExample],
    grouped: Mapping[str, list[ArchetypeTrainingExample]],
    duplicates: Sequence[str],
    split_conflicts: Sequence[str],
    unsupported_labels: Mapping[str, list[str]],
    config: RouterTrainingScriptConfig,
) -> DatasetValidationReport:
    label_frequencies: dict[str, int] = {}
    empty_labels: list[str] = []
    for example in examples:
        if not example.archetype_labels:
            empty_labels.append(example.problem_id)
        for label in example.archetype_labels:
            label_frequencies[label] = label_frequencies.get(label, 0) + 1
    notes: list[str] = []
    if duplicates:
        notes.append("Problem-level duplicates were merged deterministically.")
    if split_conflicts:
        notes.append("Conflicting source splits were neutralized before final assignment.")
    valid = True
    if len(examples) < config.min_examples:
        valid = False
        notes.append(f"Need at least {config.min_examples} total examples.")
    if len(grouped[SplitName.VALID.value]) < config.min_valid_examples:
        valid = False
        notes.append(f"Need at least {config.min_valid_examples} validation examples.")
    if len(grouped[SplitName.CALIBRATION.value]) < config.min_calibration_examples:
        valid = False
        notes.append(f"Need at least {config.min_calibration_examples} calibration examples.")
    if empty_labels and not config.allow_unlabeled_examples:
        valid = False
        notes.append("Unlabeled examples are present but not allowed.")
    if unsupported_labels:
        valid = False
        notes.append("Unsupported archetype labels were found.")
    return DatasetValidationReport(
        valid=valid,
        num_examples=len(examples),
        num_unique_problem_ids=len({e.problem_id for e in examples}),
        duplicate_problem_ids=sorted(set(duplicates)),
        split_conflict_problem_ids=sorted(set(split_conflicts)),
        unsupported_labels={k: sorted(set(v)) for k, v in sorted(unsupported_labels.items()) if v},
        empty_label_problem_ids=sorted(set(empty_labels)),
        label_frequencies=dict(sorted(label_frequencies.items())),
        notes=notes,
    )


def _prepare_dataset(
    rows: Sequence[Any],
    config: RouterTrainingScriptConfig,
) -> tuple[list[ArchetypeTrainingExample], dict[str, list[ArchetypeTrainingExample]], DatasetValidationReport, list[str]]:
    supported = set(get_supported_archetypes())
    examples: list[ArchetypeTrainingExample] = []
    unsupported: dict[str, list[str]] = {}
    for row in rows:
        converted, invalid_labels = _coerce_example(row, supported, config.preserve_problem_metadata)
        if converted is None:
            continue
        examples.append(converted)
        if invalid_labels:
            unsupported[converted.problem_id] = invalid_labels

    if not examples:
        report = DatasetValidationReport(valid=False, notes=["No router-compatible examples were found."])
        return [], {SplitName.TRAIN.value: [], SplitName.VALID.value: [], SplitName.CALIBRATION.value: []}, report, list(supported)

    merged, duplicates, split_conflicts = _merge_examples(examples)
    split_assignments = None
    if config.split_manifest_path:
        split_assignments = _load_split_manifest(Path(config.split_manifest_path).resolve())
    grouped = _assign_splits(merged, config, split_assignments)
    report = _validate_dataset(
        examples=merged,
        grouped=grouped,
        duplicates=duplicates,
        split_conflicts=split_conflicts,
        unsupported_labels=unsupported,
        config=config,
    )
    return merged, grouped, report, sorted(supported)


def _precheck_output_root(root: Path, overwrite: bool) -> None:
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists and overwrite is disabled: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=False)


def _create_run_id(config: RouterTrainingScriptConfig, examples: Sequence[ArchetypeTrainingExample]) -> str:
    payload = {
        "run_name": config.run_name,
        "artifact_version": config.artifact_version,
        "dataset_paths": list(config.dataset_paths),
        "problem_ids": [ex.problem_id for ex in examples],
        "seed": config.seed,
        "split_strategy": config.split_strategy.value,
    }
    return _stable_hash("router_training", payload)


def _run_subprocess_backend(
    *,
    config: RouterTrainingScriptConfig,
    run_dir: Path,
    train_manifest: Path,
    valid_manifest: Path,
    calibration_manifest: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    if config.trainer_backend != "subprocess":
        return {}
    command = list(config.trainer_command)
    if not command:
        raise RuntimeError("Subprocess trainer backend requested but no trainer_command was provided")
    executable = shutil.which(command[0])
    if executable is None:
        raise RuntimeError(f"Trainer executable not found on PATH: {command[0]}")

    checkpoint_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    replacements = {
        "run_dir": str(run_dir),
        "train_manifest": str(train_manifest),
        "valid_manifest": str(valid_manifest),
        "calibration_manifest": str(calibration_manifest),
        "manifest": str(manifest_path),
        "checkpoint_dir": str(checkpoint_dir),
        "logs_dir": str(logs_dir),
        "seed": str(config.seed),
        "run_name": config.run_name,
        "artifact_version": config.artifact_version,
    }
    expanded = [token.format(**replacements) for token in command]
    env = os.environ.copy()
    env.update(config.trainer_env)
    cwd = config.trainer_work_dir or str(run_dir)
    completed = subprocess.run(expanded, cwd=cwd, env=env, text=True, capture_output=True, check=False)
    stdout_path = logs_dir / "trainer_stdout.log"
    stderr_path = logs_dir / "trainer_stderr.log"
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"External trainer exited with code {completed.returncode}. "
            f"See {stdout_path} and {stderr_path}."
        )
    checkpoint = None
    for candidate in sorted(checkpoint_dir.rglob("*")):
        if candidate.is_file():
            checkpoint = candidate
            break
    report = {
        "backend": "subprocess",
        "command": expanded,
        "cwd": cwd,
        "returncode": completed.returncode,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "checkpoint_path": str(checkpoint) if checkpoint else None,
    }
    _write_json(run_dir / "trainer_report.json", report)
    return report


def execute(config: RouterTrainingScriptConfig) -> int:
    _set_seed(config.seed)
    rows, sources = _resolve_dataset_sources(config.dataset_paths)
    examples, grouped, validation, supported = _prepare_dataset(rows, config)
    if not examples:
        raise ValueError("No router-compatible examples were found in the provided dataset artifacts")

    run_id = _create_run_id(config, examples)
    output_root = Path(config.output_dir).resolve() / f"script_{run_id}"
    _precheck_output_root(output_root, config.overwrite)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot_path = output_root / "config_snapshot.json"
    _write_json(config_snapshot_path, config.model_dump(mode="json"))

    train_summary = _split_summary(SplitName.TRAIN.value, grouped[SplitName.TRAIN.value])
    valid_summary = _split_summary(SplitName.VALID.value, grouped[SplitName.VALID.value])
    calibration_summary = _split_summary(SplitName.CALIBRATION.value, grouped[SplitName.CALIBRATION.value])

    manifest_payload = {
        "artifact_kind": "archetype_training_manifest",
        "artifact_version": config.artifact_version,
        "schema_version": "archetype_training_manifest.v1",
        "artifact_id": run_id,
        "run_name": config.run_name,
        "supported_labels": supported,
        "split_strategy": config.split_strategy.value,
        "split_bundle": {
            "train": [row.model_dump(mode="json") for row in grouped[SplitName.TRAIN.value]],
            "valid": [row.model_dump(mode="json") for row in grouped[SplitName.VALID.value]],
            "calibration": [row.model_dump(mode="json") for row in grouped[SplitName.CALIBRATION.value]],
            "train_summary": train_summary.model_dump(mode="json"),
            "valid_summary": valid_summary.model_dump(mode="json"),
            "calibration_summary": calibration_summary.model_dump(mode="json"),
        },
        "validation": validation.model_dump(mode="json"),
        "config": {
            "run_name": config.run_name,
            "artifact_version": config.artifact_version,
            "split_strategy": config.split_strategy.value,
            "min_examples": config.min_examples,
            "min_valid_examples": config.min_valid_examples,
            "min_calibration_examples": config.min_calibration_examples,
            "preserve_problem_metadata": config.preserve_problem_metadata,
            "allow_unlabeled_examples": config.allow_unlabeled_examples,
            "seed": config.seed,
        },
    }
    manifest_path = run_dir / "manifest.json"
    _write_json(manifest_path, manifest_payload)
    validation_path = run_dir / "dataset_validation.json"
    _write_json(validation_path, validation.model_dump(mode="json"))
    split_summary_path = run_dir / "split_summary.json"
    _write_json(
        split_summary_path,
        {
            "train": train_summary.model_dump(mode="json"),
            "valid": valid_summary.model_dump(mode="json"),
            "calibration": calibration_summary.model_dump(mode="json"),
        },
    )

    train_path = valid_path = calibration_path = None
    if config.emit_jsonl:
        train_path = run_dir / "train.jsonl"
        valid_path = run_dir / "valid.jsonl"
        calibration_path = run_dir / "calibration.jsonl"
        _write_jsonl(train_path, (row.model_dump(mode="json") for row in grouped[SplitName.TRAIN.value]))
        _write_jsonl(valid_path, (row.model_dump(mode="json") for row in grouped[SplitName.VALID.value]))
        _write_jsonl(calibration_path, (row.model_dump(mode="json") for row in grouped[SplitName.CALIBRATION.value]))

    status = RunStatus.READY if validation.valid else RunStatus.INVALID_DATASET
    trainer_report: dict[str, Any] = {}
    trained_model_path: str | None = None
    if validation.valid and config.trainer_backend == "subprocess":
        if not (train_path and valid_path and calibration_path):
            raise RuntimeError("Subprocess trainer backend requires JSONL manifests; enable --emit-jsonl")
        try:
            trainer_report = _run_subprocess_backend(
                config=config,
                run_dir=run_dir,
                train_manifest=train_path,
                valid_manifest=valid_path,
                calibration_manifest=calibration_path,
                manifest_path=manifest_path,
            )
            trained_model_path = trainer_report.get("checkpoint_path")
            status = RunStatus.COMPLETED
        except Exception as exc:
            trainer_report = {"backend": "subprocess", "error": str(exc)}
            _write_json(run_dir / "trainer_report.json", trainer_report)
            status = RunStatus.UNAVAILABLE_RUNTIME
    elif validation.valid and config.trainer_backend == "none":
        trainer_report = {
            "backend": "none",
            "message": "No training backend was configured. Dataset manifests were emitted for external consumption.",
        }
        _write_json(run_dir / "trainer_report.json", trainer_report)

    artifacts = EmittedArtifactPaths(
        output_root=str(output_root),
        run_dir=str(run_dir),
        manifest_path=str(manifest_path),
        config_snapshot_path=str(config_snapshot_path),
        train_path=str(train_path) if train_path else None,
        valid_path=str(valid_path) if valid_path else None,
        calibration_path=str(calibration_path) if calibration_path else None,
        split_summary_path=str(split_summary_path),
        dataset_validation_path=str(validation_path),
        trainer_report_path=str(run_dir / "trainer_report.json") if (run_dir / "trainer_report.json").exists() else None,
        checkpoint_dir=str(checkpoints_dir),
        log_dir=str(logs_dir),
        trained_model_path=trained_model_path,
    )

    run_manifest = RouterTrainingRunManifest(
        artifact_version=config.artifact_version,
        run_name=config.run_name,
        run_id=run_id,
        status=status.value,
        trainer_backend=config.trainer_backend,
        dataset_sources=[source.__dict__ for source in sources],
        validation=validation.model_dump(mode="json"),
        split_counts={
            SplitName.TRAIN.value: len(grouped[SplitName.TRAIN.value]),
            SplitName.VALID.value: len(grouped[SplitName.VALID.value]),
            SplitName.CALIBRATION.value: len(grouped[SplitName.CALIBRATION.value]),
        },
        split_label_frequencies={
            SplitName.TRAIN.value: train_summary.label_frequencies,
            SplitName.VALID.value: valid_summary.label_frequencies,
            SplitName.CALIBRATION.value: calibration_summary.label_frequencies,
        },
        supported_archetypes=supported,
        artifacts=artifacts.model_dump(mode="json"),
        trainer_report=trainer_report,
        notes=list(validation.notes),
    )
    _write_json(output_root / "router_training_run_manifest.json", run_manifest.model_dump(mode="json"))

    LOGGER.info(
        "Router training status=%s total=%d train=%d valid=%d calibration=%d run_dir=%s",
        status.value,
        validation.num_examples,
        len(grouped[SplitName.TRAIN.value]),
        len(grouped[SplitName.VALID.value]),
        len(grouped[SplitName.CALIBRATION.value]),
        run_dir,
    )
    if status == RunStatus.READY:
        LOGGER.warning("No training backend executed; manifests are ready for an external trainer.")
        return 0
    if status == RunStatus.COMPLETED:
        return 0
    if status == RunStatus.INVALID_DATASET:
        return 2
    if status == RunStatus.UNAVAILABLE_RUNTIME:
        LOGGER.error("Requested training backend was unavailable or failed; no fake checkpoint was emitted.")
        return 4
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _configure_logging(verbose=args.verbose)

    base_config: dict[str, Any] = {}
    if args.config:
        base_config = _read_config_file(Path(args.config))
    merged_config = _merge_cli_overrides(base_config, args)

    try:
        config = RouterTrainingScriptConfig.model_validate(merged_config)
    except ValidationError as exc:
        LOGGER.error("Invalid router training configuration: %s", exc)
        return 2
    except Exception as exc:
        LOGGER.error("Failed to build router training configuration: %s", exc)
        return 2

    try:
        return execute(config)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError, TypeError) as exc:
        LOGGER.error("train_router failed: %s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("train_router interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
