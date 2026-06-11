from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kaggle_anti086.training.day1_dpo_pair_audit import (
    FORMAT_REASONS,
    KNOWN_REJECTION_REASONS,
    audit_dpo_pairs,
    validate_dpo_pair,
)
from kaggle_anti086.training.day1_dpo_pair_manifest import build_dpo_manifest
from kaggle_anti086.training.day1_teacher_families import SyntheticExample, available_teachers
from kaggle_anti086.training.day1_verified_data_schema import read_jsonl


@dataclass(frozen=True)
class DPOPair:
    schema_version: int
    id: str
    family: str
    rule_id: str
    difficulty: str
    prompt_style: str
    prompt: str
    chosen: str
    rejected: str
    correct_answer: str
    rejected_answer: str | None
    reason_rejected: str
    chosen_verification_status: str
    rejected_verification_status: str
    chosen_trainable: bool
    rejected_trainable: bool
    split: str
    source: str
    seed: int
    metadata: dict[str, Any]


def to_json_dict(pair: Any) -> dict[str, Any]:
    return asdict(pair) if is_dataclass(pair) else dict(pair)


def stable_dpo_pair_id(family: str, prompt: str, chosen: str, rejected: str, seed: int) -> str:
    joined = "\x1f".join(("dpo", family, prompt, chosen, rejected, str(seed)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:24]


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(to_json_dict(row), sort_keys=True, ensure_ascii=False) + "\n")


def read_dpo_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def build_dpo_pair_factory(
    out_dir: Path,
    *,
    phase4_dir: Path,
    train_pairs_per_family: int = 500,
    eval_pairs_per_family: int = 50,
    seed: int = 123,
    require_preconditions: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    input_files = {
        "phase4_manifest": str(phase4_dir / "phase4_verified_data_manifest.json"),
        "phase4_train": str(phase4_dir / "phase4_verified_sft_train.jsonl"),
        "phase4_eval": str(phase4_dir / "phase4_verified_sft_eval.jsonl"),
        "phase4_hard_negatives": str(phase4_dir / "phase4_hard_negative_pairs.jsonl"),
    }
    if require_preconditions:
        _check_phase4_preconditions(Path(input_files["phase4_manifest"]))
    train_rows = read_jsonl(Path(input_files["phase4_train"]))
    eval_rows = read_jsonl(Path(input_files["phase4_eval"]))
    _phase4_pairs = read_jsonl(Path(input_files["phase4_hard_negatives"]))
    train_pairs, rejected_train = _build_split_pairs(train_rows, "train", train_pairs_per_family, seed)
    eval_pairs, rejected_eval = _build_split_pairs(eval_rows, "eval", eval_pairs_per_family, seed + 100000)
    rejected_pairs = [*rejected_train, *rejected_eval]
    audit = audit_dpo_pairs(train_pairs, eval_pairs, rejected_pairs)
    output_files = {
        "train_pairs": str(out_dir / "phase5_dpo_train_pairs.jsonl"),
        "eval_pairs": str(out_dir / "phase5_dpo_eval_pairs.jsonl"),
        "rejected_pairs": str(out_dir / "phase5_dpo_rejected_pairs.jsonl"),
        "audit": str(out_dir / "phase5_dpo_pair_audit_report.json"),
        "manifest": str(out_dir / "phase5_dpo_pair_manifest.json"),
        "decision": str(out_dir / "phase5_decision_report.json"),
    }
    manifest, decision = build_dpo_manifest(input_files=input_files, output_files=output_files, audit=audit)
    write_jsonl(Path(output_files["train_pairs"]), train_pairs)
    write_jsonl(Path(output_files["eval_pairs"]), eval_pairs)
    write_jsonl(Path(output_files["rejected_pairs"]), rejected_pairs)
    _write_json(Path(output_files["audit"]), audit)
    _write_json(Path(output_files["manifest"]), manifest)
    _write_json(Path(output_files["decision"]), decision)
    return {
        "status": manifest["status"],
        "manifest": manifest,
        "audit": audit,
        "decision": decision,
        "output_files": output_files,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Day1 Phase5 DPO hard-negative preference pairs.")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/sprint11"))
    parser.add_argument("--phase4-dir", type=Path, default=Path("artifacts/sprint11"))
    parser.add_argument("--train-pairs-per-family", type=int, default=500)
    parser.add_argument("--eval-pairs-per-family", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)
    result = build_dpo_pair_factory(
        args.out_dir,
        phase4_dir=args.phase4_dir,
        train_pairs_per_family=args.train_pairs_per_family,
        eval_pairs_per_family=args.eval_pairs_per_family,
        seed=args.seed,
    )
    manifest = result["manifest"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "train_pairs": manifest["counts"]["train_pairs"],
                "eval_pairs": manifest["counts"]["eval_pairs"],
                "dpo_pairs_ready_for_phase6": manifest["gates"]["dpo_pairs_ready_for_phase6"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] in {"PASS", "WARN"} else 1


def _build_split_pairs(rows: list[dict[str, Any]], split: str, per_family: int, seed: int) -> tuple[list[DPOPair], list[DPOPair]]:
    teachers = available_teachers()
    by_family: dict[str, list[dict[str, Any]]] = {family: [] for family in teachers}
    for row in rows:
        if row.get("family") in by_family:
            by_family[str(row["family"])].append(row)
    accepted: list[DPOPair] = []
    rejected: list[DPOPair] = []
    seen_ids: set[str] = set()
    seen_keys: set[str] = set()
    for family, family_rows in sorted(by_family.items()):
        made = 0
        for index, row in enumerate(family_rows):
            if made >= per_family:
                break
            pair = _pair_from_row(row, split, seed + index, index)
            key = _pair_key(pair)
            ok, _failures = validate_dpo_pair(pair)
            if ok and pair.id not in seen_ids and key not in seen_keys:
                accepted.append(pair)
                seen_ids.add(pair.id)
                seen_keys.add(key)
                made += 1
            else:
                rejected.append(pair)
        if made < per_family:
            for index, row in enumerate(family_rows[made:], start=made):
                if made >= per_family:
                    break
                extra = _family_negative_from_row(row, split, seed + 10000 + index)
                key = _pair_key(extra)
                ok, _failures = validate_dpo_pair(extra)
                if ok and extra.id not in seen_ids and key not in seen_keys:
                    accepted.append(extra)
                    seen_ids.add(extra.id)
                    seen_keys.add(key)
                    made += 1
                else:
                    rejected.append(extra)
        if made < per_family:
            raise RuntimeError(f"{split}:{family} generated {made}/{per_family} valid DPO pairs")
    return accepted, rejected


def _pair_from_row(row: dict[str, Any], split: str, seed: int, index: int) -> DPOPair:
    reason, rejected, rejected_answer = _format_negative(row, index)
    if index < 5:
        return _make_pair(row, rejected, rejected_answer, reason, split, seed, "format_negative")
    return _family_negative_from_row(row, split, seed)


def _family_negative_from_row(row: dict[str, Any], split: str, seed: int) -> DPOPair:
    teacher = available_teachers()[str(row["family"])]
    example = SyntheticExample(
        family=str(row["family"]),
        prompt=str(row["prompt"]),
        answer=str(row["answer"]),
        trace=str(row["trace"]),
        target_text=str(row["target_text"]),
        rule_id=str(row["rule_id"]),
        difficulty=str(row["difficulty"]),
        prompt_style=str(row["prompt_style"]),
        verification_status=str(row["verification_status"]),
        ambiguity_count=int(row["ambiguity_count"]),
        metadata=dict(row.get("metadata") or {}),
    )
    negative = teacher.generate_hard_negative(example, __import__("random").Random(seed))
    reason = _map_reason(example.family, negative.reason_rejected)
    return _make_pair(row, negative.rejected, negative.rejected_answer, reason, split, seed, "teacher_hard_negative")


def _make_pair(
    row: dict[str, Any],
    rejected: str,
    rejected_answer: str | None,
    reason: str,
    split: str,
    seed: int,
    source: str,
) -> DPOPair:
    pair_id = stable_dpo_pair_id(str(row["family"]), str(row["prompt"]), str(row["target_text"]), rejected, seed)
    return DPOPair(
        schema_version=1,
        id=pair_id,
        family=str(row["family"]),
        rule_id=str(row["rule_id"]),
        difficulty=str(row["difficulty"]),
        prompt_style=str(row["prompt_style"]),
        prompt=str(row["prompt"]),
        chosen=str(row["target_text"]),
        rejected=rejected,
        correct_answer=str(row["answer"]),
        rejected_answer=rejected_answer,
        reason_rejected=reason if reason in KNOWN_REJECTION_REASONS else "unknown_rejection_reason",
        chosen_verification_status="PASS",
        rejected_verification_status="FAIL",
        chosen_trainable=True,
        rejected_trainable=False,
        split=split,
        source=source,
        seed=seed,
        metadata={"source_record_id": row.get("record_id"), **dict(row.get("metadata") or {})},
    )


def _format_negative(row: dict[str, Any], index: int) -> tuple[str, str, str | None]:
    answer = str(row["answer"])
    variants = index % 5
    if variants == 0:
        return "failed_format_no_box", f"Answer: {answer}", answer
    if variants == 1:
        return "failed_format_multiple_boxes", f"\\boxed{{{answer}}}\\boxed{{{answer}}}", answer
    if variants == 2:
        return "failed_format_text_after_box", f"\\boxed{{{answer}}} trailing", answer
    if variants == 3:
        wrong = _wrong_answer(answer)
        return "failed_format_boxed_wrong_answer", f"\\boxed{{{wrong}}}", wrong
    return "failed_format_abstain_placeholder", "\\boxed{ABSTAIN}", "ABSTAIN"


def _wrong_answer(answer: str) -> str:
    if re.fullmatch(r"-?\d+", answer):
        return str(int(answer) + 1)
    return f"{answer}x"


def _map_reason(family: str, reason: str) -> str:
    family_map = {
        "custom_numeral": {"wrong_custom_base": "failed_custom_numeral_wrong_base", "wrong_base": "failed_custom_numeral_wrong_base", "wrong_digit_order": "failed_custom_numeral_wrong_digit_order", "off_by_one": "failed_custom_numeral_off_by_one_mapping"},
        "symbol_mapping": {"swapped_mapping": "failed_symbol_mapping_swapped_mapping"},
        "bit_manipulation": {"wrong_bit_operation": "failed_bit_wrong_operation"},
        "char_cipher": {"wrong_cipher": "failed_char_cipher_wrong_shift"},
        "word_cipher": {"wrong_word_rule": "failed_word_cipher_wrong_word_order"},
        "permutation_sorting": {"wrong_sort_direction": "failed_permutation_wrong_direction", "wrong_sorting_rule": "failed_permutation_wrong_order"},
        "gravity_numeric": {"wrong_physics_formula": "failed_gravity_wrong_constant"},
        "unit_conversion": {"wrong_factor": "failed_unit_wrong_factor"},
        "numeric_formula_safe": {"wrong_formula_coefficient": "failed_numeric_wrong_coefficient", "wrong_formula": "failed_numeric_wrong_coefficient"},
        "equation_operator": {"wrong_operator": "failed_equation_wrong_operator"},
        "sequence_pattern": {"wrong_sequence_next_term": "failed_sequence_wrong_difference"},
    }
    return family_map.get(family, {}).get(reason, "unknown_rejection_reason")


def _check_phase4_preconditions(manifest_path: Path) -> None:
    if not manifest_path.exists():
        raise ValueError(f"missing Phase4 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gates = manifest.get("gates", {})
    if gates.get("phase4_ready_for_phase5") is not True:
        raise ValueError("phase4_ready_for_phase5 is not true")
    if gates.get("phase4_ready_for_lora_training") is not False:
        raise ValueError("phase4_ready_for_lora_training must be false")
    if manifest.get("safe_to_train_lora") is not False:
        raise ValueError("safe_to_train_lora must remain false")
    if manifest.get("no_0_93_evidence") is not True or manifest.get("no_0_95_evidence") is not True:
        raise ValueError("0.93/0.95 evidence flags must remain blocked")


def _pair_key(pair: DPOPair) -> str:
    return "\x1f".join((_norm(pair.prompt), _norm(pair.chosen), _norm(pair.rejected)))


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
