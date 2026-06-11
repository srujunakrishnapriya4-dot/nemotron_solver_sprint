from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Callable, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kaggle_anti086.training.day1_teacher_trainability_gate import (
    detect_abstain_placeholder,
    detect_text_after_final_box,
    verify_trainable_row,
)
from kaggle_anti086.training.day1_teacher_verifier import count_boxed_answers, extract_boxed_answer
from kaggle_anti086.training.day1_verified_data_schema import read_jsonl, write_jsonl


CREATED_BY = "DAY1X_CURRICULUM_VARIANT_BUILDER"
DEFAULT_VARIANT_SIZES = {
    "direct": 50000,
    "short_trace": 50000,
    "mixed_curriculum": 100000,
    "format_heavy": 50000,
    "composed_heavy": 50000,
    "public_style_heavy": 50000,
    "hard_p2_heavy": 30000,
}
HARD_P2_FAMILIES: frozenset[str] = frozenset(
    {
        "equation_operator",
        "sequence_pattern",
        "gravity_numeric",
        "numeric_formula_safe",
        "composed_sequence_operator",
        "composed_unit_formula",
        "composed_symbol_equation",
        "composed_custom_numeral_arithmetic",
    }
)
VARIANT_FILES = {
    "direct": "day1x_sft_direct.jsonl",
    "short_trace": "day1x_sft_short_trace.jsonl",
    "mixed_curriculum": "day1x_sft_mixed_curriculum.jsonl",
    "format_heavy": "day1x_sft_format_heavy.jsonl",
    "composed_heavy": "day1x_sft_composed_heavy.jsonl",
    "public_style_heavy": "day1x_sft_public_style_heavy.jsonl",
    "hard_p2_heavy": "day1x_sft_hard_p2_heavy.jsonl",
}


def build_direct_variant(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    selected = _select_rows(rows, size, seed, "direct", set())
    return [_variant_row(row, "direct", "direct", index, f"\\boxed{{{row['answer']}}}") for index, row in enumerate(selected)]


def build_short_trace_variant(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    selected = _select_rows(rows, size, seed, "short_trace", set())
    return [_variant_row(row, "short_trace", "short_trace", index, _short_trace_target(row)) for index, row in enumerate(selected)]


def build_mixed_curriculum_variant(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    used: set[str] = set()
    counts = _ratio_counts(size, {"direct": 0.20, "short_trace": 0.50, "composed": 0.15, "public_style": 0.10, "format_stress": 0.05})
    specs = [
        ("composed", _is_composed, "short_trace"),
        ("public_style", _is_public_style, "short_trace"),
        ("format_stress", _is_format_stress, "direct"),
        ("direct", lambda row: True, "direct"),
        ("short_trace", lambda row: True, "short_trace"),
    ]
    return _build_weighted(rows, "mixed_curriculum", counts, specs, seed, used)


def build_format_heavy_variant(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    used: set[str] = set()
    counts = _ratio_counts(size, {"direct": 0.70, "short_trace": 0.20, "format_stress": 0.10})
    specs = [
        ("format_stress", _is_format_stress, "direct"),
        ("direct", lambda row: True, "direct"),
        ("short_trace", lambda row: True, "short_trace"),
    ]
    return _build_weighted(rows, "format_heavy", counts, specs, seed, used)


def build_composed_heavy_variant(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    used: set[str] = set()
    counts = _ratio_counts(size, {"composed": 0.40, "original": 0.40, "public_style": 0.20})
    specs = [
        ("composed", _is_composed, "short_trace"),
        ("public_style", _is_public_style, "short_trace"),
        ("original", _is_original, "direct"),
    ]
    return _build_weighted(rows, "composed_heavy", counts, specs, seed, used)


def build_public_style_heavy_variant(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    used: set[str] = set()
    counts = _ratio_counts(size, {"public_style": 0.50, "original": 0.30, "composed": 0.20})
    specs = [
        ("public_style", _is_public_style, "short_trace"),
        ("composed", _is_composed, "short_trace"),
        ("original", _is_original, "direct"),
    ]
    return _build_weighted(rows, "public_style_heavy", counts, specs, seed, used)


def build_hard_p2_heavy_variant(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    pool = [row for row in rows if str(row.get("family")) in HARD_P2_FAMILIES]
    selected = _select_rows(pool, min(size, len(pool)), seed, "hard_p2_heavy", set(), allow_reuse=False)
    return [_variant_row(row, "hard_p2_heavy", "short_trace", index, _short_trace_target(row), "hard_p2") for index, row in enumerate(selected)]


def validate_variant_rows(rows: list[dict[str, Any]], variant_name: str, target_size: int | None = None, allow_cap: bool = False) -> dict[str, Any]:
    errors = Counter()
    ids = Counter(str(row.get("id", "")) for row in rows)
    prompts = Counter(_prompt_key(str(row.get("prompt", ""))) for row in rows)
    family_counts = Counter(str(row.get("family", "UNKNOWN")) for row in rows)
    prompt_style_counts = Counter(str(row.get("prompt_style", "UNKNOWN")) for row in rows)
    difficulty_counts = Counter(str(row.get("difficulty", "UNKNOWN")) for row in rows)
    source_counts = Counter(str(row.get("metadata", {}).get("source_mixture", "unknown")) for row in rows)
    mixture = Counter(str(row.get("metadata", {}).get("curriculum_bucket", row.get("metadata", {}).get("target_style", "unknown"))) for row in rows)
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            errors["malformed_messages"] += 1
            assistant = str(row.get("target_text", ""))
        else:
            if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
                errors["malformed_messages"] += 1
            assistant = str(messages[1].get("content", ""))
            if assistant != str(row.get("target_text", "")):
                errors["assistant_target_mismatch"] += 1
        if count_boxed_answers(assistant) != 1 or detect_text_after_final_box(assistant):
            errors["format_error"] += 1
        if extract_boxed_answer(assistant) != str(row.get("answer")):
            errors["answer_mismatch"] += 1
        if detect_abstain_placeholder(row) or detect_abstain_placeholder(assistant):
            errors["abstain_placeholder"] += 1
        check = verify_trainable_row(
            {
                "family": row.get("family"),
                "prompt": row.get("prompt"),
                "answer": row.get("answer"),
                "target_text": row.get("target_text"),
                "trace": row.get("target_text"),
                "verification_status": row.get("verification_status"),
                "ambiguity_count": row.get("ambiguity_count"),
                "trainable": True,
                "metadata": dict(row.get("metadata") or {}),
            }
        )
        if not check.trainable:
            errors[f"trainability:{check.rejection_reason}"] += 1
        if row.get("verification_status") != "PASS":
            errors["verification_not_pass"] += 1
        if row.get("ambiguity_count") != 0:
            errors["ambiguity_nonzero"] += 1
    duplicate_id_count = sum(count - 1 for count in ids.values() if count > 1)
    duplicate_prompt_count = sum(count - 1 for count in prompts.values() if count > 1)
    if duplicate_id_count:
        errors["duplicate_id"] += duplicate_id_count
    if duplicate_prompt_count:
        errors["duplicate_prompt"] += duplicate_prompt_count
    if target_size is not None and len(rows) < target_size and not allow_cap:
        errors["row_count_below_target"] += 1
    safety_zero = not errors
    status = "PASS" if safety_zero else "FAIL"
    if target_size is not None and len(rows) < target_size and allow_cap and all(key in {"row_count_below_target"} for key in errors):
        status = "WARN"
    return {
        "schema_version": 1,
        "created_by": CREATED_BY,
        "variant": variant_name,
        "status": status,
        "row_count": len(rows),
        "target_size": target_size,
        "rows_by_family": dict(sorted(family_counts.items())),
        "rows_by_prompt_style": dict(sorted(prompt_style_counts.items())),
        "rows_by_difficulty": dict(sorted(difficulty_counts.items())),
        "rows_by_source": dict(sorted(source_counts.items())),
        "mixture_report": dict(sorted(mixture.items())),
        "format_error_count": int(errors["format_error"] + errors["answer_mismatch"]),
        "verification_fail_count": sum(count for key, count in errors.items() if key.startswith("trainability:") or key == "verification_not_pass"),
        "ambiguity_accepted": int(errors["ambiguity_nonzero"]),
        "abstain_accepted": int(errors["abstain_placeholder"]),
        "duplicate_prompt_count": duplicate_prompt_count,
        "duplicate_id_count": duplicate_id_count,
        "errors": dict(sorted(errors.items())),
        "recommended_for_day2_training": status in {"PASS", "WARN"} and errors["format_error"] == 0 and errors["answer_mismatch"] == 0,
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
    }


def write_variant_files(out_dir: str | Path, seed: int = 123, size_per_variant: dict[str, int] | int | None = None) -> dict[str, Any]:
    out = Path(out_dir)
    rows = read_jsonl(out / "day1x_verified_sft_train_100k.jsonl")
    sizes = dict(DEFAULT_VARIANT_SIZES)
    if isinstance(size_per_variant, int):
        sizes = {key: size_per_variant for key in sizes}
    elif isinstance(size_per_variant, dict):
        sizes.update(size_per_variant)
    builders: dict[str, Callable[[list[dict[str, Any]], int, int], list[dict[str, Any]]]] = {
        "direct": build_direct_variant,
        "short_trace": build_short_trace_variant,
        "mixed_curriculum": build_mixed_curriculum_variant,
        "format_heavy": build_format_heavy_variant,
        "composed_heavy": build_composed_heavy_variant,
        "public_style_heavy": build_public_style_heavy_variant,
        "hard_p2_heavy": build_hard_p2_heavy_variant,
    }
    manifests: dict[str, dict[str, Any]] = {}
    files: dict[str, str] = {}
    for offset, variant in enumerate(VARIANT_FILES):
        variant_rows = builders[variant](rows, sizes[variant], seed + offset * 1009)
        manifest = validate_variant_rows(variant_rows, variant, sizes[variant], allow_cap=variant == "hard_p2_heavy")
        path = out / VARIANT_FILES[variant]
        manifest_path = out / f"day1x_variant_manifest_{variant}.json"
        write_jsonl(path, variant_rows)
        _write_json(manifest_path, manifest)
        manifests[variant] = manifest
        files[variant] = str(path)
    return {"status": "PASS" if all(manifest["status"] in {"PASS", "WARN"} for manifest in manifests.values()) else "FAIL", "files": files, "manifests": manifests}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Day1X SFT curriculum variant views.")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/sprint11"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--size-per-variant", type=int, default=None)
    args = parser.parse_args(argv)
    result = write_variant_files(args.out_dir, seed=args.seed, size_per_variant=args.size_per_variant)
    print(json.dumps({"status": result["status"], "variants": {key: value["row_count"] for key, value in result["manifests"].items()}}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


def _build_weighted(
    rows: list[dict[str, Any]],
    variant: str,
    counts: dict[str, int],
    specs: list[tuple[str, Callable[[dict[str, Any]], bool], str]],
    seed: int,
    used: set[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for spec_index, (bucket, predicate, target_style) in enumerate(specs):
        selected = _select_rows([row for row in rows if predicate(row)], counts[bucket], seed + spec_index * 997, f"{variant}:{bucket}", used)
        for row in selected:
            reused = _source_id(row) in used
            used.add(_source_id(row))
            target = f"\\boxed{{{row['answer']}}}" if target_style == "direct" else _short_trace_target(row)
            output.append(_variant_row(row, variant, target_style, len(output), target, bucket, reused_source_row=reused))
    return output


def _variant_row(
    row: dict[str, Any],
    variant: str,
    target_style: str,
    index: int,
    target_text: str,
    bucket: str | None = None,
    *,
    reused_source_row: bool = False,
) -> dict[str, Any]:
    source_id = _source_id(row)
    prompt = f"{row['prompt']} Variant key: {variant}-{target_style}-{index}."
    row_id = _stable_id("day1x-variant", variant, target_style, source_id, str(index), prompt)
    metadata = {
        "source": "day1x_100k",
        "source_row_id": source_id,
        "source_mixture": _mixture_source(row),
        "is_composed": _is_composed(row),
        "is_public_style": _is_public_style(row),
        "is_format_stress": _is_format_stress(row),
        "target_style": target_style,
        "curriculum_bucket": bucket or target_style,
        "reused_source_row": reused_source_row,
    }
    return {
        "schema_version": 1,
        "id": row_id,
        "source_row_id": source_id,
        "variant": variant,
        "family": row.get("family"),
        "prompt_style": row.get("prompt_style"),
        "difficulty": row.get("difficulty"),
        "prompt": prompt,
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": target_text}],
        "answer": str(row.get("answer")),
        "target_text": target_text,
        "verification_status": "PASS",
        "ambiguity_count": 0,
        "trainable": True,
        "metadata": metadata,
    }


def _short_trace_target(row: dict[str, Any]) -> str:
    answer = str(row["answer"])
    source = str(row.get("target_text") or row.get("trace") or f"\\boxed{{{answer}}}")
    lines = [line.strip() for line in source.splitlines() if line.strip()]
    box = f"\\boxed{{{answer}}}"
    non_box = [line for line in lines if "\\boxed{" not in line]
    if not non_box:
        return box
    kept = non_box[:3]
    return "\n".join([*kept, box])


def _select_rows(rows: list[dict[str, Any]], size: int, seed: int, salt: str, used: set[str], allow_reuse: bool = True) -> list[dict[str, Any]]:
    if not rows or size <= 0:
        return []
    pool = list(rows)
    random.Random(seed + _offset(salt)).shuffle(pool)
    selected: list[dict[str, Any]] = []
    for row in pool:
        if len(selected) >= size:
            break
        if _source_id(row) in used and not allow_reuse:
            continue
        selected.append(row)
    cursor = 0
    while len(selected) < size and allow_reuse:
        selected.append(pool[cursor % len(pool)])
        cursor += 1
    return selected[:size]


def _ratio_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    keys = list(ratios)
    counts = {key: int(total * ratios[key]) for key in keys[:-1]}
    counts[keys[-1]] = total - sum(counts.values())
    return counts


def _is_composed(row: dict[str, Any]) -> bool:
    return str(row.get("family", "")).startswith("composed_") or _mixture_source(row) == "composed_day1x"


def _is_public_style(row: dict[str, Any]) -> bool:
    return str(row.get("prompt_style", "")).startswith("public_style") or _mixture_source(row) == "public_style_day1x"


def _is_format_stress(row: dict[str, Any]) -> bool:
    return _mixture_source(row) == "format_stress" or str(row.get("difficulty")) == "trap"


def _is_original(row: dict[str, Any]) -> bool:
    return _mixture_source(row) == "original_day1" and not _is_composed(row) and not _is_public_style(row) and not _is_format_stress(row)


def _mixture_source(row: dict[str, Any]) -> str:
    return str(row.get("metadata", {}).get("mixture_source", row.get("source", "unknown")))


def _source_id(row: dict[str, Any]) -> str:
    return str(row.get("record_id") or row.get("id"))


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]


def _offset(text: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(text))


def _prompt_key(prompt: str) -> str:
    return re.sub(r"\s+", " ", str(prompt).strip())


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
