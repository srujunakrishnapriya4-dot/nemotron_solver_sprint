from __future__ import annotations

from collections import Counter
import re
from typing import Any, Iterable, Mapping

from kaggle_anti086.training.day1_teacher_trainability_gate import (
    KNOWN_TRAINABLE_FAMILIES,
    detect_abstain_placeholder,
    detect_answer_mismatch,
    detect_unsafe_metadata,
    verify_trainable_row,
)


FORMAT_REASONS: frozenset[str] = frozenset(
    {
        "failed_format_no_box",
        "failed_format_multiple_boxes",
        "failed_format_text_after_box",
        "failed_format_boxed_wrong_answer",
        "failed_format_abstain_placeholder",
    }
)

KNOWN_REJECTION_REASONS: frozenset[str] = FORMAT_REASONS | frozenset(
    {
        "failed_custom_numeral_wrong_base",
        "failed_custom_numeral_wrong_digit_order",
        "failed_custom_numeral_off_by_one_mapping",
        "failed_custom_numeral_roundtrip",
        "failed_symbol_mapping_swapped_mapping",
        "failed_symbol_mapping_unseen_symbol_guess",
        "failed_symbol_mapping_partial_mapping",
        "failed_bit_wrong_operation",
        "failed_bit_wrong_width",
        "failed_bit_decimal_binary_confusion",
        "failed_bit_invalid_binary",
        "failed_char_cipher_wrong_shift",
        "failed_char_cipher_atbash_vs_caesar",
        "failed_char_cipher_reverse_confusion",
        "failed_char_cipher_case_mismatch",
        "failed_word_cipher_wrong_word_order",
        "failed_word_cipher_wrong_extraction",
        "failed_word_cipher_semantic_hallucination",
        "failed_word_cipher_partial_mapping",
        "failed_permutation_wrong_order",
        "failed_permutation_wrong_direction",
        "failed_permutation_tie_policy",
        "failed_permutation_wrong_permutation",
        "failed_gravity_wrong_constant",
        "failed_gravity_inverse_formula",
        "failed_gravity_rounding_or_unstated_constant",
        "failed_unit_wrong_factor",
        "failed_unit_inverse_conversion",
        "failed_unit_off_by_10",
        "failed_unit_rounding_ambiguity",
        "failed_numeric_wrong_coefficient",
        "failed_numeric_wrong_intercept",
        "failed_numeric_linear_vs_quadratic",
        "failed_numeric_ambiguous_extrapolation",
        "failed_equation_wrong_operator",
        "failed_equation_swapped_direction",
        "failed_equation_non_exact_division",
        "failed_equation_affine_coefficient",
        "failed_sequence_wrong_difference",
        "failed_sequence_wrong_ratio",
        "failed_sequence_wrong_recurrence",
        "failed_sequence_ambiguous_continuation",
        "rejected_equals_chosen",
        "rejected_accidentally_trainable",
        "rejected_answer_matches_correct",
        "unknown_rejection_reason",
        "malformed_pair",
    }
)


def validate_dpo_pair(pair: Any) -> tuple[bool, list[str]]:
    data = _as_dict(pair)
    failures: list[str] = []
    family = data.get("family")
    metadata = dict(data.get("metadata") or {})
    if not data.get("id"):
        failures.append("missing_id")
    if family not in KNOWN_TRAINABLE_FAMILIES:
        failures.append("unknown_family")
    for key in ("prompt", "chosen", "rejected", "correct_answer"):
        if not str(data.get(key, "")).strip():
            failures.append(f"missing_{key}")
    if data.get("split") not in {"train", "eval"}:
        failures.append("invalid_split")
    if _norm(data.get("chosen")) == _norm(data.get("rejected")):
        failures.append("chosen_equals_rejected")
    reason = data.get("reason_rejected")
    if reason not in KNOWN_REJECTION_REASONS or reason in {
        "rejected_equals_chosen",
        "rejected_accidentally_trainable",
        "rejected_answer_matches_correct",
        "unknown_rejection_reason",
        "malformed_pair",
    }:
        failures.append("unknown_rejection_reason")
    if detect_abstain_placeholder(str(data.get("chosen", ""))):
        failures.append("chosen_abstain_placeholder")
    if detect_unsafe_metadata(metadata):
        failures.append("unsafe_metadata")
    if data.get("source") == "model_guess_unverified":
        failures.append("model_guess_unverified")
    chosen_row = _trainability_row(data, target_key="chosen")
    chosen_check = verify_trainable_row(chosen_row)
    if not chosen_check.trainable:
        failures.append("chosen_not_trainable")
    if data.get("chosen_verification_status") != "PASS" or data.get("chosen_trainable") is not True:
        failures.append("chosen_status_fields_invalid")
    rejected_row = _trainability_row(data, target_key="rejected")
    rejected_check = verify_trainable_row(rejected_row)
    if rejected_check.trainable:
        failures.append("rejected_accidentally_trainable")
    if data.get("rejected_verification_status") != "FAIL" or data.get("rejected_trainable") is not False:
        failures.append("rejected_status_fields_invalid")
    if reason not in FORMAT_REASONS and not detect_answer_mismatch(
        str(data.get("rejected", "")), str(data.get("correct_answer", "")), family, metadata
    ):
        failures.append("rejected_answer_matches_correct")
    return not failures, failures


def audit_dpo_pairs(
    train_pairs: Iterable[Any],
    eval_pairs: Iterable[Any],
    rejected_pairs: Iterable[Any],
) -> dict[str, Any]:
    train = [_as_dict(pair) for pair in train_pairs]
    eval_rows = [_as_dict(pair) for pair in eval_pairs]
    rejected = [_as_dict(pair) for pair in rejected_pairs]
    accepted = [*train, *eval_rows]
    failures: list[str] = []
    warnings: list[str] = []
    validation_fail_count = 0
    rejected_trainable = 0
    chosen_equals = 0
    unknown_reason = 0
    unsafe_metadata = 0
    rejected_answer_matches = 0
    for pair in accepted:
        ok, reasons = validate_dpo_pair(pair)
        if not ok:
            validation_fail_count += 1
            failures.append(f"{pair.get('id', 'unknown')}: {','.join(reasons)}")
        if "rejected_accidentally_trainable" in reasons:
            rejected_trainable += 1
        if "chosen_equals_rejected" in reasons:
            chosen_equals += 1
        if "unknown_rejection_reason" in reasons:
            unknown_reason += 1
        if "unsafe_metadata" in reasons:
            unsafe_metadata += 1
        if "rejected_answer_matches_correct" in reasons:
            rejected_answer_matches += 1
    ids = [pair.get("id") for pair in accepted]
    duplicate_pair_count = len(ids) - len(set(ids))
    if duplicate_pair_count:
        failures.append(f"duplicate_pair_count={duplicate_pair_count}")
    train_prompts = {_prompt_key(pair.get("prompt", "")) for pair in train}
    eval_prompts = {_prompt_key(pair.get("prompt", "")) for pair in eval_rows}
    overlap = train_prompts & eval_prompts
    if overlap:
        failures.append(f"train_eval_prompt_overlap_count={len(overlap)}")
    family_train_counts = Counter(pair.get("family", "UNKNOWN") for pair in train)
    family_eval_counts = Counter(pair.get("family", "UNKNOWN") for pair in eval_rows)
    for family in sorted(KNOWN_TRAINABLE_FAMILIES):
        if family_train_counts.get(family, 0) == 0:
            failures.append(f"{family}: zero_train_pairs")
        if family_eval_counts.get(family, 0) == 0:
            failures.append(f"{family}: zero_eval_pairs")
    reason_counts = Counter(pair.get("reason_rejected", "UNKNOWN") for pair in accepted)
    format_negative_count = sum(count for reason, count in reason_counts.items() if reason in FORMAT_REASONS)
    family_negative_count = len(accepted) - format_negative_count
    status = "PASS"
    if (
        validation_fail_count
        or duplicate_pair_count
        or overlap
        or chosen_equals
        or rejected_trainable
        or unknown_reason
        or unsafe_metadata
        or any("zero_" in failure for failure in failures)
    ):
        status = "FAIL"
    elif len(train) < 5500 or len(eval_rows) < 550:
        status = "WARN"
        warnings.append("pair_counts_below_default_threshold")
    return {
        "schema_version": 1,
        "created_by": "DAY1_PHASE5_DPO_PAIR_AUDIT",
        "status": status,
        "train_pair_count": len(train),
        "eval_pair_count": len(eval_rows),
        "rejected_pair_count": len(rejected),
        "family_train_counts": dict(sorted(family_train_counts.items())),
        "family_eval_counts": dict(sorted(family_eval_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "format_negative_count": format_negative_count,
        "family_negative_count": family_negative_count,
        "validation_fail_count": validation_fail_count,
        "duplicate_pair_count": duplicate_pair_count,
        "train_eval_prompt_overlap_count": len(overlap),
        "chosen_equals_rejected_count": chosen_equals,
        "rejected_accidentally_trainable_count": rejected_trainable,
        "rejected_answer_matches_correct_count": rejected_answer_matches,
        "unknown_reason_count": unknown_reason,
        "unsafe_metadata_count": unsafe_metadata,
        "failures": failures[:100],
        "warnings": warnings[:100],
        "dpo_pairs_ready_for_phase6": status in {"PASS", "WARN"} and validation_fail_count == 0,
        "safe_to_train_lora": False,
        "safe_to_package": False,
        "safe_to_submit": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
    }


def _trainability_row(data: Mapping[str, Any], *, target_key: str) -> dict[str, Any]:
    return {
        "family": data.get("family"),
        "prompt": data.get("prompt"),
        "answer": data.get("correct_answer"),
        "trace": data.get(target_key),
        "target_text": data.get(target_key),
        "verification_status": "PASS",
        "ambiguity_count": 0,
        "trainable": True,
        "metadata": dict(data.get("metadata") or {}),
    }


def _as_dict(pair: Any) -> dict[str, Any]:
    if isinstance(pair, Mapping):
        return dict(pair)
    if hasattr(pair, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(pair)
    return dict(pair)


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _prompt_key(prompt: str) -> str:
    return _norm(prompt)
