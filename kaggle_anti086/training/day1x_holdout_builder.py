from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kaggle_anti086.training.day1_teacher_families import available_teachers
from kaggle_anti086.training.day1_teacher_trainability_gate import detect_abstain_placeholder, verify_trainable_row
from kaggle_anti086.training.day1_teacher_verifier import count_boxed_answers, extract_boxed_answer
from kaggle_anti086.training.day1_verified_data_schema import write_jsonl
from kaggle_anti086.training.day1x_composed_teachers import REQUIRED_COMPOSED_FAMILIES, generate_composed_rows, make_boxed_target, validate_composed_row
from kaggle_anti086.training.day1x_prompt_augmentation import augment_prompt


SOURCE = "day1x_holdout_synthetic"
P2_FAMILIES = ("equation_operator", "sequence_pattern", "gravity_numeric", "numeric_formula_safe", "composed_sequence_operator", "composed_unit_formula")
HOLDOUT_FILES = {
    "rule": "day1x_holdout_rule.jsonl",
    "prompt_style": "day1x_holdout_prompt_style.jsonl",
    "composed": "day1x_holdout_composed.jsonl",
    "format_traps": "day1x_holdout_format_traps.jsonl",
    "hard_p2": "day1x_holdout_hard_p2.jsonl",
}


def build_rule_holdout(count: int, seed: int) -> list[dict[str, Any]]:
    return _generate_original_holdout(count, seed, "holdout_rule", "unseen_rule_parameters", style_cycle=("table", "story", "examples_query"))


def build_prompt_style_holdout(count: int, seed: int) -> list[dict[str, Any]]:
    base = generate_composed_rows(max(1, (count + len(REQUIRED_COMPOSED_FAMILIES) - 1) // len(REQUIRED_COMPOSED_FAMILIES)), seed + 1000)
    styles = ("public_style_mixed_notation", "public_style_noisy_context", "public_style_multiline", "public_style_symbol_heavy")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(base):
        if len(rows) >= count:
            break
        aug = augment_prompt(row, styles[index % len(styles)], seed + index)
        rows.append(_retag_holdout(aug, "holdout_prompt_style", "unseen_prompt_style", seed, index))
    return rows


def build_composed_holdout(count: int, seed: int) -> list[dict[str, Any]]:
    per_family = max(1, (count + len(REQUIRED_COMPOSED_FAMILIES) - 1) // len(REQUIRED_COMPOSED_FAMILIES))
    rows = generate_composed_rows(per_family, seed + 2000)
    return [_retag_holdout(row, "holdout_composed", "composed_generalization", seed, index) for index, row in enumerate(rows[:count])]


def build_format_trap_holdout(count: int, seed: int) -> list[dict[str, Any]]:
    builders = (_negative_trap, _leading_zero_trap, _unit_box_trap, _comma_distractor_trap, _answer_only_trap)
    rows = []
    for index in range(count):
        row = builders[index % len(builders)](seed, index)
        rows.append(row)
    return rows


def build_hard_p2_holdout(count: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    original = [family for family in P2_FAMILIES if not family.startswith("composed_")]
    rows.extend(_generate_original_holdout(max(1, count // 2), seed + 3000, "holdout_hard_p2", "hard_reasoning_p2", families=original, difficulty="hard"))
    composed_needed = count - len(rows)
    if composed_needed > 0:
        composed = generate_composed_rows(max(1, composed_needed), seed + 4000, families=["composed_sequence_operator", "composed_unit_formula"])
        rows.extend(_retag_holdout(row, "holdout_hard_p2", "hard_reasoning_p2", seed, 5000 + index) for index, row in enumerate(composed[:composed_needed]))
    return rows[:count]


def build_all_holdouts(count_per_holdout: int, seed: int) -> dict[str, list[dict[str, Any]]]:
    return {
        "rule": build_rule_holdout(count_per_holdout, seed),
        "prompt_style": build_prompt_style_holdout(count_per_holdout, seed),
        "composed": build_composed_holdout(count_per_holdout, seed),
        "format_traps": build_format_trap_holdout(count_per_holdout, seed),
        "hard_p2": build_hard_p2_holdout(count_per_holdout, seed),
    }


def validate_holdout_rows(rows: list[dict[str, Any]], split_name: str) -> dict[str, Any]:
    errors: Counter[str] = Counter()
    ids: Counter[str] = Counter(str(row.get("id", "")) for row in rows)
    prompts: Counter[str] = Counter(_prompt_key(str(row.get("prompt", ""))) for row in rows)
    for row in rows:
        if row.get("split") != split_name:
            errors["bad_split"] += 1
        check = verify_trainable_row(row)
        if not check.trainable:
            errors[f"trainability:{check.rejection_reason}"] += 1
        if str(row.get("family", "")).startswith("composed_") and not validate_composed_row(row)[0]:
            errors["composed_validation_failed"] += 1
        if detect_abstain_placeholder(row):
            errors["abstain_placeholder"] += 1
        if count_boxed_answers(str(row.get("target_text", ""))) != 1:
            errors["box_count_not_one"] += 1
        if row.get("ambiguity_count") != 0:
            errors["ambiguity_nonzero"] += 1
        if row.get("verification_status") != "PASS":
            errors["verification_not_pass"] += 1
    duplicate_ids = sum(count - 1 for count in ids.values() if count > 1)
    duplicate_prompts = sum(count - 1 for count in prompts.values() if count > 1)
    if duplicate_ids:
        errors["duplicate_id"] += duplicate_ids
    if duplicate_prompts:
        errors["duplicate_prompt"] += duplicate_prompts
    return {
        "schema_version": 1,
        "split": split_name,
        "rows": len(rows),
        "status": "PASS" if not errors else "FAIL",
        "duplicate_id_count": duplicate_ids,
        "duplicate_prompt_count": duplicate_prompts,
        "errors": dict(sorted(errors.items())),
    }


def write_holdout_files(out_dir: str | Path, count_per_holdout: int, seed: int) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    holdouts = build_all_holdouts(count_per_holdout, seed)
    reports: dict[str, Any] = {}
    global_prompts: set[str] = set()
    global_ids: set[str] = set()
    global_errors: Counter[str] = Counter()
    files: dict[str, str] = {}
    for name, rows in holdouts.items():
        split = f"holdout_{name}"
        report = validate_holdout_rows(rows, split)
        for row in rows:
            prompt = _prompt_key(str(row.get("prompt", "")))
            row_id = str(row.get("id", ""))
            if prompt in global_prompts:
                global_errors["cross_holdout_duplicate_prompt"] += 1
            if row_id in global_ids:
                global_errors["cross_holdout_duplicate_id"] += 1
            global_prompts.add(prompt)
            global_ids.add(row_id)
        path = out / HOLDOUT_FILES[name]
        write_jsonl(path, rows)
        files[name] = str(path)
        reports[name] = report
    status = "PASS" if all(report["status"] == "PASS" for report in reports.values()) and not global_errors else "FAIL"
    return {"status": status, "files": files, "reports": reports, "global_errors": dict(sorted(global_errors.items()))}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Day1X holdout files.")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/sprint11"))
    parser.add_argument("--count-per-holdout", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)
    report = write_holdout_files(args.out_dir, args.count_per_holdout, args.seed)
    print(json.dumps({"status": report["status"], "files": report["files"]}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


def _generate_original_holdout(
    count: int,
    seed: int,
    split: str,
    reason: str,
    *,
    families: Sequence[str] | None = None,
    difficulty: str = "hard",
    style_cycle: Sequence[str] = ("direct", "table", "story", "examples_query"),
) -> list[dict[str, Any]]:
    teachers = available_teachers()
    selected = list(families or teachers.keys())
    rows: list[dict[str, Any]] = []
    attempts = 0
    while len(rows) < count and attempts < count * 100:
        family = selected[attempts % len(selected)]
        teacher = teachers[family]
        rng = random.Random(seed + attempts * 7919 + _offset(family))
        try:
            example = teacher.generate_synthetic(rng, difficulty, style_cycle[attempts % len(style_cycle)])
            solved = teacher.verify_example(example)
            if solved.verification_status != "PASS":
                attempts += 1
                continue
            row = _row_from_example(asdict(example), split, reason, seed, attempts)
            rows.append(row)
        except Exception:
            pass
        attempts += 1
    if len(rows) != count:
        raise RuntimeError(f"{split} generated {len(rows)}/{count}")
    return rows


def _row_from_example(example: dict[str, Any], split: str, reason: str, seed: int, index: int) -> dict[str, Any]:
    prompt = f"{example['prompt']} Holdout key: {split}-{seed}-{index}."
    row = {
        "schema_version": 1,
        "id": _stable_id(split, str(seed), str(index), example["family"], prompt),
        "record_id": _stable_id("record", split, str(seed), str(index), example["family"], prompt),
        "split": split,
        "family": example["family"],
        "rule_id": example.get("rule_id", "day1_holdout"),
        "prompt_style": example["prompt_style"],
        "difficulty": example["difficulty"],
        "prompt": prompt,
        "trace": example["trace"],
        "answer": example["answer"],
        "target_text": example["target_text"],
        "verification_status": "PASS",
        "ambiguity_count": 0,
        "trainable": True,
        "holdout_reason": reason,
        "source": SOURCE,
        "generator": f"{split}_{example['family']}_v1",
        "metadata": {"output_type": "integer", "holdout_reason": reason},
    }
    row["messages"] = [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["target_text"]}]
    return row


def _retag_holdout(row: dict[str, Any], split: str, reason: str, seed: int, index: int) -> dict[str, Any]:
    copied = dict(row)
    copied["prompt"] = f"{row['prompt']} Holdout key: {split}-{seed}-{index}."
    copied["split"] = split
    copied["id"] = _stable_id(split, str(seed), str(index), str(row["family"]), copied["prompt"])
    copied["record_id"] = _stable_id("record", split, str(seed), str(index), str(row["family"]), copied["prompt"])
    copied["source"] = SOURCE
    copied["holdout_reason"] = reason
    copied["metadata"] = {**dict(row.get("metadata") or {}), "holdout_reason": reason}
    copied["messages"] = [{"role": "user", "content": copied["prompt"]}, {"role": "assistant", "content": copied["target_text"]}]
    return copied


def _format_trap_row(prompt: str, trace: str, answer: str, seed: int, index: int, family: str = "numeric_formula_safe") -> dict[str, Any]:
    target = make_boxed_target(trace, answer)
    row = {
        "schema_version": 1,
        "id": _stable_id("format_trap", str(seed), str(index), prompt),
        "record_id": _stable_id("record", "format_trap", str(seed), str(index), prompt),
        "split": "holdout_format_traps",
        "family": family,
        "rule_id": "format_trap_v1",
        "prompt_style": "public_style_noisy_context",
        "difficulty": "trap",
        "prompt": f"{prompt} Holdout key: format-trap-{seed}-{index}.",
        "trace": target,
        "answer": str(answer),
        "target_text": target,
        "verification_status": "PASS",
        "ambiguity_count": 0,
        "trainable": True,
        "holdout_reason": "format_trap",
        "source": SOURCE,
        "generator": "format_trap_v1",
        "metadata": {"output_type": "integer", "holdout_reason": "format_trap"},
    }
    row["messages"] = [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": target}]
    return row


def _negative_trap(seed: int, index: int) -> dict[str, Any]:
    value = -((seed + index) % 97 + 1)
    return _format_trap_row(f"Compute 3 - {abs(value) + 3}; final answer may be negative.", f"3 - {abs(value) + 3} = {value}.", str(value), seed, index)


def _leading_zero_trap(seed: int, index: int) -> dict[str, Any]:
    value = (seed + index) % 90 + 10
    return _format_trap_row(f"Numbers may have leading zeros: add 00{value} and 0005. Normalize the answer.", f"{value} + 5 = {value + 5}.", str(value + 5), seed, index)


def _unit_box_trap(seed: int, index: int) -> dict[str, Any]:
    meters = (seed + index) % 9 + 1
    return _format_trap_row(f"Convert {meters} m to cm. Box only the number, not the unit.", f"{meters} m = {meters * 100} cm.", str(meters * 100), seed, index, "unit_conversion")


def _comma_distractor_trap(seed: int, index: int) -> dict[str, Any]:
    values = [index % 7 + 2, index % 5 + 8, index % 3 + 15]
    answer = max(values) - min(values)
    return _format_trap_row(f"Table values are {values}; return the range as one scalar.", f"max - min = {max(values)} - {min(values)} = {answer}.", str(answer), seed, index)


def _answer_only_trap(seed: int, index: int) -> dict[str, Any]:
    a, b = index % 11 + 4, index % 13 + 7
    return _format_trap_row(f"Answer only after checking: {a}*{b}.", f"{a}*{b} = {a * b}.", str(a * b), seed, index)


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]


def _offset(text: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(text))


def _prompt_key(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.strip())


if __name__ == "__main__":
    raise SystemExit(main())
