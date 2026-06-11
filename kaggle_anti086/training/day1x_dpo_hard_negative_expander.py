from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Iterable, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kaggle_anti086.training.day1_teacher_trainability_gate import (
    detect_abstain_placeholder,
    detect_answer_mismatch,
    detect_text_after_final_box,
    verify_trainable_row,
)
from kaggle_anti086.training.day1_teacher_verifier import count_boxed_answers, extract_boxed_answer
from kaggle_anti086.training.day1_verified_data_schema import read_jsonl, write_jsonl


CREATED_BY = "DAY1X_DPO_HARD_NEGATIVE_EXPANSION"
NEGATIVE_TYPES: tuple[str, ...] = (
    "near_miss_answer",
    "off_by_one",
    "sign_flip",
    "wrong_arithmetic",
    "wrong_base_or_mapping",
    "wrong_unit_factor",
    "wrong_bit_width",
    "wrong_sequence_rule",
    "wrong_operator",
    "family_confusion",
    "right_reason_wrong_arithmetic",
    "wrong_reason_right_format",
    "wrong_intermediate_correct_final",
    "correct_intermediate_wrong_final",
    "ambiguous_rule_hallucination",
    "wrong_substep_1",
    "wrong_substep_2",
    "wrong_composition_order",
    "correct_substeps_wrong_final",
    "wrong_family_assumption",
    "format_no_box",
    "format_multiple_boxes",
    "format_text_after_box",
    "format_abstain",
    "right_answer_wrong_format",
)
FORMAT_NEGATIVE_TYPES: frozenset[str] = frozenset(
    {
        "format_no_box",
        "format_multiple_boxes",
        "format_text_after_box",
        "format_abstain",
        "right_answer_wrong_format",
    }
)
FAMILY_PREFERRED_NEGATIVES: dict[str, tuple[str, ...]] = {
    "custom_numeral": ("wrong_base_or_mapping", "near_miss_answer"),
    "symbol_mapping": ("wrong_base_or_mapping", "family_confusion"),
    "bit_manipulation": ("wrong_bit_width", "wrong_operator"),
    "unit_conversion": ("wrong_unit_factor", "off_by_one"),
    "numeric_formula_safe": ("wrong_operator", "wrong_arithmetic"),
    "sequence_pattern": ("wrong_sequence_rule", "off_by_one"),
    "equation_operator": ("wrong_operator", "wrong_arithmetic"),
}
MAJOR_FAMILY_GROUPS: dict[str, tuple[str, ...]] = {
    "original": (
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
    ),
    "composed": (
        "composed_custom_numeral_arithmetic",
        "composed_symbol_equation",
        "composed_cipher_mapping",
        "composed_unit_formula",
        "composed_bit_conversion",
        "composed_sequence_operator",
        "composed_permutation_mapping",
        "composed_word_cipher",
        "composed_gravity_unit",
    ),
}


def make_chosen(row: dict[str, Any], style: str = "original") -> str:
    if style == "direct":
        return f"\\boxed{{{row['answer']}}}"
    if style != "original":
        raise ValueError(f"unsupported_chosen_style:{style}")
    return str(row.get("target_text") or row.get("trace") or "")


def make_rejected(row: dict[str, Any], negative_type: str, seed: int) -> tuple[str, dict[str, Any]]:
    if negative_type not in NEGATIVE_TYPES:
        raise ValueError(f"unknown_negative_type:{negative_type}")
    answer = str(row["answer"])
    wrong = _wrong_answer(answer, negative_type, seed)
    reason = _reason_for_negative(negative_type)
    if negative_type == "format_no_box":
        rejected = f"The final answer is {answer}, but no boxed final answer is provided."
    elif negative_type == "format_multiple_boxes":
        other = _wrong_answer(answer, "format_multiple_boxes", seed + 17)
        rejected = f"First candidate \\boxed{{{answer}}}; conflicting candidate \\boxed{{{other}}}"
    elif negative_type == "format_text_after_box":
        rejected = f"The computation gives a wrong final value.\n\\boxed{{{wrong}}} trailing text"
    elif negative_type == "format_abstain":
        rejected = "I cannot solve the item reliably.\n\\boxed{ABSTAIN}"
    elif negative_type == "right_answer_wrong_format":
        rejected = f"The value is {answer}.\n\\boxed{{{answer}}} extra text after the box"
    else:
        stem = _negative_stem(row, negative_type, wrong)
        rejected = f"{stem}\n\\boxed{{{wrong}}}"
    return rejected, {
        "negative_type": negative_type,
        "reason_rejected": reason,
        "expected_failure": True,
        "wrong_answer": wrong,
    }


def validate_dpo_pair(pair: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for key in ("id", "source_row_id", "prompt", "chosen", "rejected", "answer", "negative_type"):
        if not str(pair.get(key, "")).strip():
            failures.append(f"missing_{key}")
    if pair.get("split") not in {"dpo_train", "dpo_eval"}:
        failures.append("invalid_split")
    negative_type = str(pair.get("negative_type", ""))
    if negative_type not in NEGATIVE_TYPES:
        failures.append("unknown_negative_type")
    if _norm(pair.get("chosen")) == _norm(pair.get("rejected")):
        failures.append("chosen_equals_rejected")
    answer = str(pair.get("answer", ""))
    chosen = str(pair.get("chosen", ""))
    rejected = str(pair.get("rejected", ""))
    if count_boxed_answers(chosen) != 1 or detect_text_after_final_box(chosen):
        failures.append("chosen_format_error")
    if extract_boxed_answer(chosen) != answer:
        failures.append("chosen_answer_mismatch")
    if detect_abstain_placeholder(chosen):
        failures.append("chosen_abstain")
    chosen_check = verify_trainable_row(_trainability_row(pair, "chosen"))
    if not chosen_check.trainable:
        failures.append(f"chosen_not_trainable:{chosen_check.rejection_reason}")
    rejected_check = verify_trainable_row(_trainability_row(pair, "rejected"))
    rejected_boxed = extract_boxed_answer(rejected)
    if rejected_check.trainable:
        failures.append("rejected_accidentally_trainable")
    if not str(pair.get("reason_rejected", "")).strip():
        failures.append("rejected_missing_reason")
    if pair.get("rejected_expected_failure") is not True:
        failures.append("rejected_expected_failure_not_true")
    if pair.get("verification_status") != "PASS":
        failures.append("verification_status_not_pass")
    if pair.get("chosen_trainability_status") != "PASS":
        failures.append("chosen_status_not_pass")
    if pair.get("rejected_trainability_status") != "FAIL":
        failures.append("rejected_status_not_fail")
    if rejected_boxed == answer:
        invalid_format = negative_type in FORMAT_NEGATIVE_TYPES and (
            count_boxed_answers(rejected) != 1 or detect_text_after_final_box(rejected) or detect_abstain_placeholder(rejected)
        )
        if not invalid_format:
            failures.append("rejected_accidentally_correct")
    elif negative_type not in FORMAT_NEGATIVE_TYPES and not detect_answer_mismatch(rejected, answer, pair.get("family"), dict(pair.get("metadata") or {})):
        failures.append("rejected_accidentally_correct")
    return not failures, failures


def generate_dpo_pairs(rows: list[dict[str, Any]], count: int, seed: int, split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if split not in {"dpo_train", "dpo_eval"}:
        raise ValueError(f"invalid_split:{split}")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    order = list(rows)
    random.Random(seed).shuffle(order)
    attempts = 0
    index = 0
    max_attempts = max(count * 10, len(order) * 2)
    while len(accepted) < count and attempts < max_attempts:
        row = order[index % len(order)]
        index += 1
        attempts += 1
        negative_type = _select_negative_type(row, len(accepted), seed + attempts)
        pair = _pair_from_row(row, negative_type, seed + attempts, split, len(accepted))
        prompt_key = _norm(pair["prompt"])
        ok, failures = validate_dpo_pair(pair)
        if ok and pair["id"] not in seen_ids and prompt_key not in seen_prompts:
            accepted.append(pair)
            seen_ids.add(str(pair["id"]))
            seen_prompts.add(prompt_key)
        else:
            rejected.append({"candidate": pair, "rejection_reasons": failures or ["duplicate_id_or_prompt"]})
    if len(accepted) < count:
        rejected.append({"candidate": None, "rejection_reasons": [f"insufficient_valid_pairs:{len(accepted)}/{count}"]})
    return accepted, rejected


def write_dpo_pairs(out_dir: str | Path, train_count: int = 20000, eval_count: int = 2000, seed: int = 123) -> dict[str, Any]:
    out = Path(out_dir)
    train_rows = read_jsonl(out / "day1x_verified_sft_train_100k.jsonl")
    eval_rows = read_jsonl(out / "day1x_verified_sft_eval_10k.jsonl")
    train_pairs, rejected_train = generate_dpo_pairs(train_rows, train_count, seed, "dpo_train")
    eval_pairs, rejected_eval = generate_dpo_pairs(eval_rows, eval_count, seed + 100000, "dpo_eval")
    rejected_candidates = [*rejected_train, *rejected_eval]
    audit = build_dpo_audit_report(train_pairs, eval_pairs, rejected_candidates)
    files = {
        "train_pairs": out / "day1x_dpo_train_pairs_20k.jsonl",
        "eval_pairs": out / "day1x_dpo_eval_pairs_2k.jsonl",
        "rejected_pairs": out / "day1x_dpo_rejected_pairs.jsonl",
        "audit": out / "day1x_dpo_audit_report.json",
    }
    write_jsonl(files["train_pairs"], train_pairs)
    write_jsonl(files["eval_pairs"], eval_pairs)
    write_jsonl(files["rejected_pairs"], rejected_candidates)
    _write_json(files["audit"], audit)
    return {"status": audit["status"], "audit": audit, "output_files": {key: str(value) for key, value in files.items()}}


def build_dpo_audit_report(
    train_pairs: list[dict[str, Any]],
    eval_pairs: list[dict[str, Any]],
    rejected_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    accepted = [*train_pairs, *eval_pairs]
    validation_failures: list[str] = []
    counters = Counter()
    for pair in accepted:
        ok, failures = validate_dpo_pair(pair)
        if not ok:
            counters["validation_fail_count"] += 1
            validation_failures.append(f"{pair.get('id', 'unknown')}:{','.join(failures)}")
        for failure in failures:
            if failure.startswith("chosen_not_trainable") or failure in {"chosen_format_error", "chosen_answer_mismatch"}:
                counters["chosen_format_error_count"] += 1
            if failure == "chosen_answer_mismatch":
                counters["chosen_verification_fail_count"] += 1
            if failure == "chosen_abstain":
                counters["chosen_abstain_count"] += 1
            if failure == "rejected_missing_reason":
                counters["rejected_missing_reason_count"] += 1
            if failure == "chosen_equals_rejected":
                counters["chosen_equals_rejected_count"] += 1
            if failure == "rejected_accidentally_trainable":
                counters["rejected_accidentally_trainable_count"] += 1
            if failure == "rejected_accidentally_correct":
                counters["rejected_accidentally_correct_count"] += 1
            if failure == "unknown_negative_type":
                counters["unknown_negative_type_count"] += 1
    ids = [str(pair.get("id")) for pair in accepted]
    duplicate_pair_count = len(ids) - len(set(ids))
    train_prompts = {_norm(pair.get("prompt")) for pair in train_pairs}
    eval_prompts = {_norm(pair.get("prompt")) for pair in eval_pairs}
    source_train = {str(pair.get("source_row_id")) for pair in train_pairs}
    source_eval = {str(pair.get("source_row_id")) for pair in eval_pairs}
    split_overlap = len(train_prompts & eval_prompts) + len(source_train & source_eval)
    negative_type_counts = Counter(str(pair.get("negative_type", "UNKNOWN")) for pair in accepted)
    family_counts = Counter(str(pair.get("family", "UNKNOWN")) for pair in accepted)
    format_negative_count = sum(negative_type_counts.get(item, 0) for item in FORMAT_NEGATIVE_TYPES)
    composed_pair_count = sum(count for family, count in family_counts.items() if family.startswith("composed_"))
    original_pair_count = len(accepted) - composed_pair_count
    represented_groups = {
        group
        for group, families in MAJOR_FAMILY_GROUPS.items()
        if any(family_counts.get(family, 0) for family in families)
    }
    safety = {
        "chosen_format_error_count": int(counters["chosen_format_error_count"]),
        "chosen_verification_fail_count": int(counters["chosen_verification_fail_count"]),
        "chosen_ambiguity_count": 0,
        "chosen_abstain_count": int(counters["chosen_abstain_count"]),
        "rejected_missing_reason_count": int(counters["rejected_missing_reason_count"]),
    }
    blocked: list[str] = []
    if len(train_pairs) < 20000:
        blocked.append("train_pairs_below_20000")
    if len(eval_pairs) < 2000:
        blocked.append("eval_pairs_below_2000")
    if duplicate_pair_count:
        blocked.append("duplicate_pairs")
    if split_overlap:
        blocked.append("train_eval_overlap")
    if counters["chosen_equals_rejected_count"]:
        blocked.append("chosen_equals_rejected")
    if counters["rejected_accidentally_trainable_count"]:
        blocked.append("rejected_accidentally_trainable")
    if counters["rejected_accidentally_correct_count"]:
        blocked.append("rejected_accidentally_correct")
    if counters["unknown_negative_type_count"]:
        blocked.append("unknown_negative_type")
    if any(safety.values()):
        blocked.append("chosen_or_reason_safety_counter_nonzero")
    if len(negative_type_counts) < 15:
        blocked.append("negative_type_diversity_below_15")
    if represented_groups != set(MAJOR_FAMILY_GROUPS):
        blocked.append("missing_major_family_group")
    status = "PASS"
    if blocked:
        safety_zero = not any(safety.values()) and not counters["rejected_accidentally_trainable_count"] and not counters["rejected_accidentally_correct_count"]
        if len(train_pairs) >= 10000 and len(eval_pairs) >= 1000 and safety_zero and set(blocked) <= {"negative_type_diversity_below_15", "train_pairs_below_20000", "eval_pairs_below_2000"}:
            status = "WARN"
        else:
            status = "FAIL"
    return {
        "schema_version": 1,
        "created_by": CREATED_BY,
        "status": status,
        "train_pairs": len(train_pairs),
        "eval_pairs": len(eval_pairs),
        "rejected_candidate_count": len(rejected_candidates),
        "duplicate_pair_count": duplicate_pair_count,
        "train_eval_prompt_overlap_count": split_overlap,
        "chosen_equals_rejected_count": int(counters["chosen_equals_rejected_count"]),
        "rejected_accidentally_trainable_count": int(counters["rejected_accidentally_trainable_count"]),
        "rejected_accidentally_correct_count": int(counters["rejected_accidentally_correct_count"]),
        "unknown_negative_type_count": int(counters["unknown_negative_type_count"]),
        "format_negative_count": format_negative_count,
        "negative_type_counts": dict(sorted(negative_type_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "composed_pair_count": composed_pair_count,
        "original_pair_count": original_pair_count,
        "safety": safety,
        "validation_fail_count": int(counters["validation_fail_count"]),
        "validation_failures": validation_failures[:100],
        "ready_for_curriculum_variants": status in {"PASS", "WARN"} and not any(safety.values()),
        "training_authorized": True,
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
        "blocked_reasons": sorted(set(blocked)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Day1X DPO hard-negative train/eval pairs.")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/sprint11"))
    parser.add_argument("--train-count", type=int, default=20000)
    parser.add_argument("--eval-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)
    result = write_dpo_pairs(args.out_dir, train_count=args.train_count, eval_count=args.eval_count, seed=args.seed)
    audit = result["audit"]
    print(json.dumps({"status": audit["status"], "train_pairs": audit["train_pairs"], "eval_pairs": audit["eval_pairs"]}, sort_keys=True))
    return 0 if audit["status"] in {"PASS", "WARN"} else 1


def _pair_from_row(row: dict[str, Any], negative_type: str, seed: int, split: str, index: int) -> dict[str, Any]:
    chosen = make_chosen(row)
    rejected, meta = make_rejected(row, negative_type, seed)
    source_row_id = str(row.get("record_id") or row.get("id"))
    pair_id = _stable_id("day1x-dpo", split, source_row_id, negative_type, str(seed), str(index))
    source_metadata = dict(row.get("metadata") or {})
    is_composed = str(row.get("family", "")).startswith("composed_")
    return {
        "schema_version": 1,
        "id": pair_id,
        "source_row_id": source_row_id,
        "split": split,
        "family": row.get("family"),
        "prompt_style": row.get("prompt_style"),
        "difficulty": row.get("difficulty"),
        "prompt": row.get("prompt"),
        "chosen": chosen,
        "rejected": rejected,
        "answer": str(row.get("answer")),
        "chosen_boxed_answer": extract_boxed_answer(chosen),
        "rejected_boxed_answer": extract_boxed_answer(rejected),
        "negative_type": negative_type,
        "reason_rejected": meta["reason_rejected"],
        "rejected_expected_failure": True,
        "verification_status": "PASS",
        "chosen_trainability_status": "PASS",
        "rejected_trainability_status": "FAIL",
        "metadata": {
            "source_family": row.get("family"),
            "source_generator": row.get("generator", row.get("source", "unknown")),
            "source_prompt_style": row.get("prompt_style"),
            "source_mixture": source_metadata.get("mixture_source", row.get("source", "unknown")),
            "is_composed": is_composed,
            "subfamilies": list(row.get("subfamilies", [])),
            "negative_seed": seed,
            "wrong_answer": meta.get("wrong_answer"),
        },
    }


def _select_negative_type(row: dict[str, Any], index: int, seed: int) -> str:
    family = str(row.get("family", ""))
    if index % 5 == 4:
        return tuple(sorted(FORMAT_NEGATIVE_TYPES))[(index // 5) % len(FORMAT_NEGATIVE_TYPES)]
    if family.startswith("composed_"):
        choices = ("wrong_substep_1", "wrong_substep_2", "wrong_composition_order", "correct_substeps_wrong_final", "wrong_family_assumption")
    else:
        choices = FAMILY_PREFERRED_NEGATIVES.get(family, ())
    if choices and index % 3 != 0:
        return choices[(index + seed) % len(choices)]
    non_format = [item for item in NEGATIVE_TYPES if item not in FORMAT_NEGATIVE_TYPES]
    return non_format[index % len(non_format)]


def _wrong_answer(answer: str, negative_type: str, seed: int) -> str:
    text = str(answer).strip()
    if re.fullmatch(r"[+-]?\d+", text):
        value = int(text)
        if negative_type in {"sign_flip", "family_confusion"}:
            wrong = -value if value != 0 else seed % 17 + 1
        elif negative_type in {"off_by_one", "near_miss_answer"}:
            wrong = value + (1 if seed % 2 == 0 else -1)
        elif negative_type == "wrong_unit_factor":
            wrong = value * 10 if value != 0 else 10
        elif negative_type == "wrong_bit_width":
            wrong = value + (2 ** ((seed % 4) + 1))
        elif negative_type == "wrong_sequence_rule":
            wrong = value + ((seed % 5) + 2)
        elif negative_type == "wrong_operator":
            wrong = value - ((seed % 7) + 2)
        else:
            wrong = value + ((seed % 11) + 3)
        if wrong == value:
            wrong += 1
        return str(wrong)
    if "," in text:
        parts = [part.strip() for part in text.split(",")]
        if len(parts) > 1:
            parts[0], parts[-1] = parts[-1], parts[0]
            wrong_text = ",".join(parts)
            if wrong_text != text:
                return wrong_text
    if len(text) > 1:
        wrong_text = text[::-1]
        if wrong_text != text:
            return wrong_text
    suffix = chr(ord("a") + seed % 26)
    wrong_text = f"{text}{suffix}"
    return wrong_text if wrong_text != text else f"{text}x"


def _negative_stem(row: dict[str, Any], negative_type: str, wrong: str) -> str:
    family = str(row.get("family", "unknown"))
    stems = {
        "near_miss_answer": "A nearby value is selected instead of the exact verified answer.",
        "off_by_one": "The final step shifts the value by one.",
        "sign_flip": "The sign convention is flipped at the end.",
        "wrong_arithmetic": "The setup is copied, but the arithmetic is evaluated incorrectly.",
        "wrong_base_or_mapping": "The mapping/base is applied with a swapped symbol.",
        "wrong_unit_factor": "The unit conversion uses the wrong factor.",
        "wrong_bit_width": "The bit operation uses the wrong width.",
        "wrong_sequence_rule": "The sequence continuation follows the wrong rule.",
        "wrong_operator": "The stated operator is replaced by a different operation.",
        "family_confusion": f"The solution treats this {family} item as a different family.",
        "right_reason_wrong_arithmetic": "The reason names the right rule but computes the final arithmetic incorrectly.",
        "wrong_reason_right_format": "The response keeps the required final format but justifies the answer with the wrong rule.",
        "wrong_intermediate_correct_final": "A correct-looking intermediate is followed by the wrong final value.",
        "correct_intermediate_wrong_final": "The intermediate step is correct, then the final answer is transcribed incorrectly.",
        "ambiguous_rule_hallucination": "An unstated alternate rule is hallucinated.",
        "wrong_substep_1": "The first composed substep is solved incorrectly.",
        "wrong_substep_2": "The second composed substep is solved incorrectly.",
        "wrong_composition_order": "The composed substeps are applied in the wrong order.",
        "correct_substeps_wrong_final": "The substeps are listed, but the final composition is wrong.",
        "wrong_family_assumption": "A composed family assumption is swapped for an incompatible one.",
    }
    return f"{stems.get(negative_type, 'The rejected answer is explicitly wrong')} This gives {wrong}."


def _reason_for_negative(negative_type: str) -> str:
    if negative_type in FORMAT_NEGATIVE_TYPES:
        return "format_invalid"
    if negative_type == "ambiguous_rule_hallucination":
        return "ambiguous_rule"
    if negative_type in {"family_confusion", "wrong_family_assumption"}:
        return "family_confusion"
    if negative_type.startswith("wrong_substep") or negative_type in {"wrong_composition_order", "correct_substeps_wrong_final"}:
        return "substep_failure"
    return "answer_mismatch"


def _trainability_row(pair: dict[str, Any], target_key: str) -> dict[str, Any]:
    return {
        "family": pair.get("family"),
        "prompt": pair.get("prompt"),
        "answer": pair.get("answer"),
        "trace": pair.get(target_key),
        "target_text": pair.get(target_key),
        "verification_status": "PASS",
        "ambiguity_count": 0,
        "trainable": True,
        "metadata": dict(pair.get("metadata") or {}),
    }


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
