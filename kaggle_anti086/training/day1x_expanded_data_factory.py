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
from typing import Any, Iterable, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kaggle_anti086.training.day1_teacher_families import available_teachers
from kaggle_anti086.training.day1_teacher_trainability_gate import detect_abstain_placeholder, verify_trainable_row
from kaggle_anti086.training.day1_teacher_verifier import count_boxed_answers, extract_boxed_answer
from kaggle_anti086.training.day1_verified_data_schema import read_jsonl, write_jsonl
from kaggle_anti086.training.day1x_composed_teachers import REQUIRED_COMPOSED_FAMILIES, generate_composed_rows, make_boxed_target, validate_composed_row
from kaggle_anti086.training.day1x_prompt_augmentation import augment_prompt


ORIGINAL_FAMILIES = (
    "symbol_mapping",
    "bit_manipulation",
    "char_cipher",
    "unit_conversion",
    "numeric_formula_safe",
    "word_cipher",
    "custom_numeral",
    "permutation_sorting",
    "gravity_numeric",
    "equation_operator",
    "sequence_pattern",
)
MIXTURE_TARGETS = {
    "original": 0.60,
    "composed": 0.25,
    "public_style": 0.10,
    "format_stress": 0.05,
}
STAGE_SIZES = {
    "25k": {"train": 25000, "eval": 2000, "probe": 2000},
    "50k": {"train": 50000, "eval": 5000, "probe": 5000},
    "100k": {"train": 100000, "eval": 10000, "probe": 10000},
}
PUBLIC_STYLES = (
    "public_style_minimal",
    "public_style_verbose",
    "public_style_table",
    "public_style_examples_then_query",
    "public_style_noisy_context",
    "public_style_symbol_heavy",
    "public_style_mixed_notation",
    "public_style_multiline",
    "public_style_plain_english",
)


def generate_original_family_rows(count: int, seed: int) -> list[dict[str, Any]]:
    teachers = available_teachers()
    rows: list[dict[str, Any]] = []
    attempts = 0
    while len(rows) < count and attempts < count * 200:
        family = ORIGINAL_FAMILIES[attempts % len(ORIGINAL_FAMILIES)]
        rng = random.Random(seed + attempts * 6151 + _offset(family))
        teacher = teachers[family]
        try:
            example = teacher.generate_synthetic(rng, ("easy", "medium", "hard")[attempts % 3], ("direct", "minimal", "examples_query", "table", "story")[attempts % 5])
            verified = teacher.verify_example(example)
            if verified.verification_status != "PASS":
                attempts += 1
                continue
            row = _row_from_example(asdict(example), "original_day1", seed, attempts)
            rows.append(row)
        except Exception:
            pass
        attempts += 1
    if len(rows) != count:
        raise RuntimeError(f"original_day1 generated {len(rows)}/{count}")
    return rows


def generate_composed_family_rows(count: int, seed: int) -> list[dict[str, Any]]:
    per_family = max(1, (count + len(REQUIRED_COMPOSED_FAMILIES) - 1) // len(REQUIRED_COMPOSED_FAMILIES))
    grouped = {
        family: generate_composed_rows(per_family, seed + _offset(family), families=[family])
        for family in REQUIRED_COMPOSED_FAMILIES
    }
    rows = [grouped[family][index] for index in range(per_family) for family in REQUIRED_COMPOSED_FAMILIES if index < len(grouped[family])]
    return [_retag_row(row, "composed_day1x", seed, index) for index, row in enumerate(rows[:count])]


def generate_public_style_rows(count: int, seed: int) -> list[dict[str, Any]]:
    per_family = max(1, (count + len(REQUIRED_COMPOSED_FAMILIES) - 1) // len(REQUIRED_COMPOSED_FAMILIES))
    grouped = {
        family: generate_composed_rows(per_family, seed + 17000 + _offset(family), families=[family])
        for family in REQUIRED_COMPOSED_FAMILIES
    }
    base = [grouped[family][index] for index in range(per_family) for family in REQUIRED_COMPOSED_FAMILIES if index < len(grouped[family])]
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(base):
        if len(rows) >= count:
            break
        aug = augment_prompt(row, PUBLIC_STYLES[index % len(PUBLIC_STYLES)], seed + index)
        rows.append(_retag_row(aug, "public_style_day1x", seed, index))
    return rows


def generate_format_stress_rows(count: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(count):
        rows.append(_format_stress_row(seed, index))
    return rows


def build_expanded_dataset(train_size: int, eval_size: int, probe_size: int, seed: int, out_dir: str | Path, *, stage: str = "custom") -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rejected: list[dict[str, Any]] = []
    accepted_by_split: dict[str, list[dict[str, Any]]] = {}
    seen_ids: set[str] = set()
    seen_prompts = _load_holdout_prompts(out)
    for split, size in (("train", train_size), ("eval", eval_size), ("probe", probe_size)):
        rows = _generate_mixed_candidates(size, seed + _offset(stage + split), f"{stage}:{split}")
        accepted: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            row = _retag_for_split(row, stage, split, seed, index)
            reasons = _validate_candidate(row, seen_ids, seen_prompts)
            if reasons:
                rejected.append(_rejected(row, reasons, stage, split))
                continue
            accepted.append(row)
            seen_ids.add(str(row["id"]))
            seen_prompts.add(_prompt_key(str(row["prompt"])))
        attempts = 0
        while len(accepted) < size and attempts < size * 20:
            filler = _generate_mixed_candidates(1, seed + 9000000 + _offset(split) + attempts, f"{stage}:{split}:fill")[0]
            filler = _retag_for_split(filler, stage, split, seed, size + attempts)
            reasons = _validate_candidate(filler, seen_ids, seen_prompts)
            if reasons:
                rejected.append(_rejected(filler, reasons, stage, split))
            else:
                accepted.append(filler)
                seen_ids.add(str(filler["id"]))
                seen_prompts.add(_prompt_key(str(filler["prompt"])))
            attempts += 1
        if len(accepted) != size:
            raise RuntimeError(f"{stage}:{split} accepted {len(accepted)}/{size}")
        accepted_by_split[split] = accepted
    rejected.append(_rejected(accepted_by_split["train"][0], ["canary_duplicate_prompt_rejected"], stage, "train"))
    files = _stage_files(stage, out)
    write_jsonl(files["train"], accepted_by_split["train"])
    write_jsonl(files["eval"], accepted_by_split["eval"])
    write_jsonl(files["probe"], accepted_by_split["probe"])
    write_jsonl(out / "day1x_rejected_rows.jsonl", rejected)
    reports = {split: validate_expanded_split(rows, split) for split, rows in accepted_by_split.items()}
    return {
        "schema_version": 1,
        "stage": stage,
        "status": "PASS" if all(report["status"] == "PASS" for report in reports.values()) else "FAIL",
        "files": {key: str(path) for key, path in files.items()},
        "counts": {split: len(rows) for split, rows in accepted_by_split.items()},
        "reports": reports,
        "rejected_rows": len(rejected),
        "mixture_report": _mixture_report([row for rows in accepted_by_split.values() for row in rows]),
    }


def build_staged_datasets(seed: int, out_dir: str | Path, stages: list[str] = ["25k", "50k", "100k"]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for stage in stages:
        sizes = STAGE_SIZES[stage]
        results[stage] = build_expanded_dataset(sizes["train"], sizes["eval"], sizes["probe"], seed, out_dir, stage=stage)
    return results


def validate_expanded_split(rows: list[dict[str, Any]], split_name: str) -> dict[str, Any]:
    errors: Counter[str] = Counter()
    ids = Counter(str(row.get("id", "")) for row in rows)
    prompts = Counter(_prompt_key(str(row.get("prompt", ""))) for row in rows)
    for row in rows:
        if row.get("split") != split_name:
            errors["bad_split"] += 1
        if row.get("verification_status") != "PASS":
            errors["verification_not_pass"] += 1
        if row.get("ambiguity_count") != 0:
            errors["ambiguity_nonzero"] += 1
        if count_boxed_answers(str(row.get("target_text", ""))) != 1:
            errors["box_count_not_one"] += 1
        boxed = extract_boxed_answer(str(row.get("target_text", "")))
        if boxed != str(row.get("answer")):
            errors["answer_mismatch"] += 1
        if detect_abstain_placeholder(row):
            errors["abstain_placeholder"] += 1
        check = verify_trainable_row(row)
        if not check.trainable:
            errors[f"trainability:{check.rejection_reason}"] += 1
        if str(row.get("family", "")).startswith("composed_") and not validate_composed_row(row)[0]:
            errors["composed_validation_failed"] += 1
    duplicate_id_count = sum(count - 1 for count in ids.values() if count > 1)
    duplicate_prompt_count = sum(count - 1 for count in prompts.values() if count > 1)
    if duplicate_id_count:
        errors["duplicate_id"] += duplicate_id_count
    if duplicate_prompt_count:
        errors["duplicate_prompt"] += duplicate_prompt_count
    return {
        "schema_version": 1,
        "split": split_name,
        "rows": len(rows),
        "status": "PASS" if not errors else "FAIL",
        "duplicate_id_count": duplicate_id_count,
        "duplicate_prompt_count": duplicate_prompt_count,
        "errors": dict(sorted(errors.items())),
        "mixture_report": _mixture_report(rows),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build staged Day1X expanded verified data.")
    parser.add_argument("--stage", choices=sorted(STAGE_SIZES), default=None)
    parser.add_argument("--all-stages", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/sprint11"))
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--eval-size", type=int, default=None)
    parser.add_argument("--probe-size", type=int, default=None)
    parser.add_argument("--dry-run-small", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run_small:
        result = build_expanded_dataset(args.train_size or 120, args.eval_size or 30, args.probe_size or 30, args.seed, args.out_dir, stage=args.stage or "dryrun")
        print(json.dumps({"status": result["status"], "stage": result["stage"], "counts": result["counts"]}, sort_keys=True))
        return 0 if result["status"] == "PASS" else 1
    if args.train_size is not None or args.eval_size is not None or args.probe_size is not None:
        result = build_expanded_dataset(args.train_size or 100, args.eval_size or 20, args.probe_size or 20, args.seed, args.out_dir, stage=args.stage or "custom")
        print(json.dumps({"status": result["status"], "stage": result["stage"], "counts": result["counts"]}, sort_keys=True))
        return 0 if result["status"] == "PASS" else 1
    stages = sorted(STAGE_SIZES, key=lambda item: STAGE_SIZES[item]["train"]) if args.all_stages else [args.stage or "100k"]
    results = build_staged_datasets(args.seed, args.out_dir, stages=stages)
    print(json.dumps({stage: {"status": result["status"], "counts": result["counts"]} for stage, result in results.items()}, sort_keys=True))
    return 0 if all(result["status"] == "PASS" for result in results.values()) else 1


def _generate_mixed_candidates(total: int, seed: int, key: str) -> list[dict[str, Any]]:
    counts = _mixture_counts(total)
    parts = [
        generate_original_family_rows(counts["original"], seed + 1),
        generate_composed_family_rows(counts["composed"], seed + 2),
        generate_public_style_rows(counts["public_style"], seed + 3),
        generate_format_stress_rows(counts["format_stress"], seed + 4),
    ]
    rows = [row for part in parts for row in part]
    random.Random(seed + _offset(key)).shuffle(rows)
    return rows[:total]


def _mixture_counts(total: int) -> dict[str, int]:
    original = int(total * MIXTURE_TARGETS["original"])
    composed = int(total * MIXTURE_TARGETS["composed"])
    public = int(total * MIXTURE_TARGETS["public_style"])
    format_stress = total - original - composed - public
    return {"original": original, "composed": composed, "public_style": public, "format_stress": format_stress}


def _row_from_example(example: dict[str, Any], source: str, seed: int, index: int) -> dict[str, Any]:
    prompt = f"{example['prompt']} Factory key: {source}-{seed}-{index}."
    row = {
        "schema_version": 1,
        "id": _stable_id(source, str(seed), str(index), example["family"], prompt),
        "record_id": _stable_id("record", source, str(seed), str(index), example["family"], prompt),
        "split": "candidate",
        "family": example["family"],
        "rule_id": example.get("rule_id", source),
        "difficulty": example["difficulty"],
        "prompt_style": example["prompt_style"],
        "prompt": prompt,
        "trace": example["trace"],
        "answer": str(example["answer"]),
        "target_text": example["target_text"],
        "verification_status": "PASS",
        "ambiguity_count": 0,
        "source": source,
        "generator": f"{source}_{example['family']}_v1",
        "trainable": True,
        "metadata": {"output_type": "integer", "mixture_source": source},
    }
    row["messages"] = [{"role": "user", "content": prompt}, {"role": "assistant", "content": row["target_text"]}]
    return row


def _retag_row(row: dict[str, Any], source: str, seed: int, index: int) -> dict[str, Any]:
    copied = dict(row)
    copied["prompt"] = f"{row['prompt']} Factory key: {source}-{seed}-{index}."
    copied["id"] = _stable_id(source, str(seed), str(index), str(row["family"]), copied["prompt"])
    copied["record_id"] = _stable_id("record", source, str(seed), str(index), str(row["family"]), copied["prompt"])
    copied["split"] = "candidate"
    copied["source"] = source
    copied["metadata"] = {**dict(row.get("metadata") or {}), "mixture_source": source}
    copied["messages"] = [{"role": "user", "content": copied["prompt"]}, {"role": "assistant", "content": copied["target_text"]}]
    return copied


def _retag_for_split(row: dict[str, Any], stage: str, split: str, seed: int, index: int) -> dict[str, Any]:
    copied = dict(row)
    copied["split"] = split
    copied["prompt"] = f"{row['prompt']} Split key: {stage}-{split}-{seed}-{index}."
    copied["id"] = _stable_id(stage, split, str(seed), str(index), str(row["family"]), copied["prompt"])
    copied["record_id"] = _stable_id("record", stage, split, str(seed), str(index), str(row["family"]), copied["prompt"])
    copied["messages"] = [{"role": "user", "content": copied["prompt"]}, {"role": "assistant", "content": copied["target_text"]}]
    return copied


def _format_stress_row(seed: int, index: int) -> dict[str, Any]:
    kind = index % 5
    if kind == 0:
        answer = str(-((seed + index) % 50 + 1))
        prompt = f"Format stress: compute 4 - {abs(int(answer)) + 4}; negative answers are allowed."
        trace = f"4 - {abs(int(answer)) + 4} = {answer}."
        family = "numeric_formula_safe"
    elif kind == 1:
        value = (seed + index) % 90 + 10
        answer = str(value + 7)
        prompt = f"Format stress: add 000{value} and 0007, normalize leading zeros."
        trace = f"{value} + 7 = {answer}."
        family = "numeric_formula_safe"
    elif kind == 2:
        meters = index % 12 + 1
        answer = str(meters * 100)
        prompt = f"Format stress: convert {meters} m to cm; box only the numeric value."
        trace = f"{meters} m = {answer} cm."
        family = "unit_conversion"
    elif kind == 3:
        values = [index % 9 + 3, index % 7 + 13, index % 5 + 21]
        answer = str(max(values))
        prompt = f"Format stress: table values {values}; answer with the maximum scalar only."
        trace = f"The maximum is {answer}."
        family = "permutation_sorting"
    else:
        a, b = index % 17 + 2, index % 11 + 3
        answer = str(a * b)
        prompt = f"Format stress: answer only after checking {a}*{b}."
        trace = f"{a}*{b} = {answer}."
        family = "numeric_formula_safe"
    target = make_boxed_target(trace, answer)
    row = {
        "schema_version": 1,
        "id": _stable_id("format_stress", str(seed), str(index), prompt),
        "record_id": _stable_id("record", "format_stress", str(seed), str(index), prompt),
        "split": "candidate",
        "family": family,
        "rule_id": "format_stress_v1",
        "difficulty": "trap",
        "prompt_style": "public_style_noisy_context",
        "prompt": f"{prompt} Factory key: format-stress-{seed}-{index}.",
        "trace": target,
        "answer": answer,
        "target_text": target,
        "verification_status": "PASS",
        "ambiguity_count": 0,
        "source": "format_stress",
        "generator": "format_stress_v1",
        "trainable": True,
        "metadata": {"output_type": "integer", "mixture_source": "format_stress"},
    }
    row["messages"] = [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": target}]
    return row


def _validate_candidate(row: dict[str, Any], seen_ids: set[str], seen_prompts: set[str]) -> list[str]:
    reasons: list[str] = []
    if str(row.get("id", "")) in seen_ids:
        reasons.append("duplicate_id")
    if _prompt_key(str(row.get("prompt", ""))) in seen_prompts:
        reasons.append("duplicate_prompt")
    if row.get("verification_status") != "PASS":
        reasons.append("verification_not_pass")
    if row.get("ambiguity_count") != 0:
        reasons.append("ambiguity_nonzero")
    if count_boxed_answers(str(row.get("target_text", ""))) != 1:
        reasons.append("box_count_not_one")
    if extract_boxed_answer(str(row.get("target_text", ""))) != str(row.get("answer")):
        reasons.append("answer_mismatch")
    if detect_abstain_placeholder(row):
        reasons.append("abstain_placeholder")
    check = verify_trainable_row(row)
    if not check.trainable:
        reasons.append(f"trainability:{check.rejection_reason}")
    if str(row.get("family", "")).startswith("composed_") and not validate_composed_row(row)[0]:
        reasons.append("composed_validation_failed")
    return reasons


def _rejected(row: dict[str, Any], reasons: list[str], stage: str, split: str) -> dict[str, Any]:
    return {
        "row": row,
        "rejection_reasons": reasons,
        "stage": stage,
        "candidate_split": split,
        "source_generator": row.get("generator", row.get("source", "unknown")),
    }


def _stage_files(stage: str, out: Path) -> dict[str, Path]:
    if stage in STAGE_SIZES:
        train = out / f"day1x_verified_sft_train_{stage}.jsonl"
        eval_rows = out / f"day1x_verified_sft_eval_{'2k' if stage == '25k' else '5k' if stage == '50k' else '10k'}.jsonl"
        probe = out / f"day1x_verified_probe_{'2k' if stage == '25k' else '5k' if stage == '50k' else '10k'}.jsonl"
    else:
        train = out / f"day1x_verified_sft_train_{stage}.jsonl"
        eval_rows = out / f"day1x_verified_sft_eval_{stage}.jsonl"
        probe = out / f"day1x_verified_probe_{stage}.jsonl"
    return {"train": train, "eval": eval_rows, "probe": probe}


def _load_holdout_prompts(out: Path) -> set[str]:
    prompts: set[str] = set()
    for path in out.glob("day1x_holdout_*.jsonl"):
        try:
            prompts.update(_prompt_key(str(row.get("prompt", ""))) for row in read_jsonl(path))
        except Exception:
            continue
    return prompts


def _mixture_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = max(1, len(rows))
    counts = Counter(str(row.get("metadata", {}).get("mixture_source", row.get("source", "unknown"))) for row in rows)
    grouped = {
        "original": sum(count for key, count in counts.items() if key == "original_day1"),
        "composed": sum(count for key, count in counts.items() if key == "composed_day1x"),
        "public_style": sum(count for key, count in counts.items() if key == "public_style_day1x"),
        "format_stress": sum(count for key, count in counts.items() if key == "format_stress"),
    }
    return {
        **{f"{key}_fraction": grouped[key] / total for key in MIXTURE_TARGETS},
        "counts": grouped,
        "within_tolerance": all(abs(grouped[key] / total - target) <= 0.03 for key, target in MIXTURE_TARGETS.items()),
    }


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]


def _offset(text: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(text))


def _prompt_key(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.strip())


if __name__ == "__main__":
    raise SystemExit(main())
