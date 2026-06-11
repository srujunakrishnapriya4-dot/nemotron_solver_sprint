from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from collections import Counter
import re
from typing import Any, Iterable, Mapping

from kaggle_anti086.training.day1_teacher_verifier import count_boxed_answers, extract_boxed_answer


DAY1X_COMPOSED_TRAINABLE_FAMILIES: frozenset[str] = frozenset(
    {
        "composed_custom_numeral_arithmetic",
        "composed_symbol_equation",
        "composed_cipher_mapping",
        "composed_unit_formula",
        "composed_bit_conversion",
        "composed_sequence_operator",
        "composed_permutation_mapping",
        "composed_word_cipher",
        "composed_gravity_unit",
    }
)
KNOWN_TRAINABLE_FAMILIES: frozenset[str] = frozenset(
    {
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
    }
) | DAY1X_COMPOSED_TRAINABLE_FAMILIES
PROTOCOL_ONLY_FAMILIES: frozenset[str] = frozenset({"model_verified_fallback", "composed_hidden_style"})
KNOWN_FAMILIES: frozenset[str] = KNOWN_TRAINABLE_FAMILIES | PROTOCOL_ONLY_FAMILIES
INTEGER_FAMILIES: frozenset[str] = frozenset(
    {
        "unit_conversion",
        "gravity_numeric",
        "equation_operator",
        "sequence_pattern",
        "numeric_formula_safe",
    }
) | DAY1X_COMPOSED_TRAINABLE_FAMILIES


@dataclass(frozen=True)
class TrainabilityCheck:
    verification_status: str
    trainable: bool
    family: str | None
    answer: str | None
    boxed_answer: str | None
    ambiguity_count: int
    rejection_reason: str | None
    checks: dict[str, bool]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class BatchTrainabilityReport:
    total_rows: int
    pass_rows: int
    fail_rows: int
    duplicate_prompt_count: int
    abstain_placeholder_count: int
    multiple_box_count: int
    text_after_box_count: int
    answer_mismatch_count: int
    ambiguity_nonzero_count: int
    unknown_family_count: int
    unsafe_metadata_count: int
    family_counts: dict[str, int]
    family_pass_counts: dict[str, int]
    family_fail_counts: dict[str, int]
    rejection_reasons: dict[str, int]
    trainable: bool


def detect_abstain_placeholder(row_or_text: Any) -> bool:
    if isinstance(row_or_text, str):
        values = {"text": row_or_text}
    else:
        row = dict(row_or_text) if isinstance(row_or_text, Mapping) else _row_to_dict(row_or_text)
        values = {key: row.get(key) for key in ("answer", "expected_answer", "target_text", "trace") if key in row}
    placeholders = {"ABSTAIN", "N/A", "UNKNOWN", "UNVERIFIED", "TODO", "PLACEHOLDER"}
    for key, value in values.items():
        if value is None:
            if key in {"answer", "expected_answer"}:
                return True
            continue
        text = str(value).strip()
        if text == "" or any(token in text.upper() for token in placeholders):
            return True
    return False


def detect_multiple_boxes(text: str) -> bool:
    return count_boxed_answers(str(text)) > 1


def detect_text_after_final_box(text: str) -> bool:
    text = str(text)
    token = r"\boxed{"
    start = text.rfind(token)
    if start < 0:
        return False
    index = start + len(token)
    depth = 1
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return bool(text[index + 1 :].strip())
        index += 1
    return False


def detect_answer_mismatch(target_text: str, expected_answer: str, family: str | None = None, metadata: dict[str, Any] | None = None) -> bool:
    if count_boxed_answers(str(target_text)) != 1:
        return True
    boxed = extract_boxed_answer(str(target_text))
    if boxed is None:
        return True
    return normalize_answer_for_family(boxed, family, metadata) != normalize_answer_for_family(expected_answer, family, metadata)


def detect_duplicate_prompt(prompt: str, seen: set[str]) -> bool:
    normalized = re.sub(r"\s+", " ", str(prompt).strip())
    if normalized in seen:
        return True
    seen.add(normalized)
    return False


def detect_unsafe_metadata(metadata: dict[str, Any]) -> bool:
    if not isinstance(metadata, dict):
        return True
    if metadata.get("verification_status") not in {None, "PASS"}:
        return True
    if metadata.get("ambiguity_count") not in {None, 0}:
        return True
    if metadata.get("trainable") is False:
        return True
    if metadata.get("unsafe") is True:
        return True
    if metadata.get("source") == "model_guess_unverified":
        return True
    if metadata.get("local_recovery_claim") is True:
        return True
    if metadata.get("router_integration") is True:
        return True
    if metadata.get("family") == "optional_not_implemented":
        return True
    if metadata.get("rejection_reason") is not None:
        return True
    return False


def normalize_answer_for_family(answer: str, family: str | None, metadata: dict[str, Any] | None = None) -> str:
    text = str(answer).strip()
    metadata = metadata or {}
    if family == "permutation_sorting":
        return ",".join(part.strip() for part in text.split(","))
    integer_like = family in INTEGER_FAMILIES or metadata.get("output_type") == "integer"
    if family == "bit_manipulation" and metadata.get("output_base", "decimal") == "decimal":
        integer_like = True
    if family == "custom_numeral" and metadata.get("output_type") == "integer":
        integer_like = True
    if integer_like and re.fullmatch(r"[+-]?\d+", text):
        value = int(text)
        return "0" if value == 0 else str(value)
    return text


def verify_trainable_row(row: Any, seen_prompts: set[str] | None = None) -> TrainabilityCheck:
    try:
        data = _row_to_dict(row)
    except Exception:
        return _failed_check(None, None, None, 1, "malformed_row", {})

    metadata = dict(data.get("metadata") or {})
    family = data.get("family")
    prompt = data.get("prompt")
    answer = data.get("answer") if data.get("answer") is not None else data.get("expected_answer")
    target_text = data.get("target_text")
    trace = data.get("trace")
    verification_status = data.get("verification_status")
    ambiguity_count = int(data.get("ambiguity_count") if data.get("ambiguity_count") is not None else 1)
    row_rejection_reason = data.get("rejection_reason")
    trainable_flag = data.get("trainable", True)
    boxed_answer = extract_boxed_answer(str(target_text)) if target_text is not None else None
    box_count = count_boxed_answers(str(target_text)) if target_text is not None else 0
    metadata["_box_count"] = box_count
    seen = seen_prompts if seen_prompts is not None else set()

    checks = {
        "known_family": family in KNOWN_FAMILIES,
        "verification_pass": verification_status == "PASS",
        "ambiguity_zero": ambiguity_count == 0,
        "no_abstain_placeholder": not detect_abstain_placeholder(data),
        "exactly_one_box": target_text is not None and box_count == 1,
        "no_text_after_final_box": target_text is not None and not detect_text_after_final_box(str(target_text)),
        "answer_match": target_text is not None and answer is not None and not detect_answer_mismatch(str(target_text), str(answer), family, metadata),
        "prompt_present": bool(str(prompt).strip()) if prompt is not None else False,
        "answer_present": answer is not None and str(answer).strip() != "",
        "target_text_present": bool(str(target_text).strip()) if target_text is not None else False,
        "trace_present": bool(str(trace).strip()) if trace is not None else False,
        "duplicate_prompt": detect_duplicate_prompt(str(prompt), seen) if prompt is not None else False,
        "safe_metadata": not detect_unsafe_metadata(metadata),
    }
    if family in PROTOCOL_ONLY_FAMILIES:
        checks["safe_metadata"] = False

    rejection = _first_rejection(checks, row_rejection_reason, trainable_flag)
    trainable = rejection is None
    return TrainabilityCheck(
        verification_status="PASS" if trainable else "FAIL",
        trainable=trainable,
        family=family,
        answer=None if answer is None else str(answer),
        boxed_answer=boxed_answer,
        ambiguity_count=ambiguity_count,
        rejection_reason=rejection,
        checks=checks,
        metadata=metadata,
    )


def verify_trainable_batch(rows: Iterable[Any]) -> tuple[list[TrainabilityCheck], BatchTrainabilityReport]:
    seen: set[str] = set()
    checks = [verify_trainable_row(row, seen) for row in rows]
    family_counts = Counter(check.family or "UNKNOWN" for check in checks)
    family_pass_counts = Counter(check.family or "UNKNOWN" for check in checks if check.trainable)
    family_fail_counts = Counter(check.family or "UNKNOWN" for check in checks if not check.trainable)
    rejection_reasons = Counter(check.rejection_reason or "PASS" for check in checks if not check.trainable)
    report = BatchTrainabilityReport(
        total_rows=len(checks),
        pass_rows=sum(1 for check in checks if check.trainable),
        fail_rows=sum(1 for check in checks if not check.trainable),
        duplicate_prompt_count=sum(1 for check in checks if check.checks.get("duplicate_prompt")),
        abstain_placeholder_count=sum(1 for check in checks if not check.checks.get("no_abstain_placeholder", True)),
        multiple_box_count=sum(1 for check in checks if check.metadata.get("_box_count", 0) > 1),
        text_after_box_count=sum(1 for check in checks if not check.checks.get("no_text_after_final_box", True)),
        answer_mismatch_count=sum(1 for check in checks if not check.checks.get("answer_match", True)),
        ambiguity_nonzero_count=sum(1 for check in checks if not check.checks.get("ambiguity_zero", True)),
        unknown_family_count=sum(1 for check in checks if not check.checks.get("known_family", True)),
        unsafe_metadata_count=sum(1 for check in checks if not check.checks.get("safe_metadata", True)),
        family_counts=dict(sorted(family_counts.items())),
        family_pass_counts=dict(sorted(family_pass_counts.items())),
        family_fail_counts=dict(sorted(family_fail_counts.items())),
        rejection_reasons=dict(sorted(rejection_reasons.items())),
        trainable=all(check.trainable for check in checks),
    )
    return checks, report


def assert_batch_trainable(rows: Iterable[Any]) -> None:
    _checks, report = verify_trainable_batch(rows)
    if not report.trainable:
        raise ValueError(
            f"Batch not trainable: total={report.total_rows} fail={report.fail_rows} reasons={report.rejection_reasons}"
        )


def verify_custom_numeral(row: Any) -> TrainabilityCheck:
    return verify_trainable_row(row)


def verify_symbol_mapping(row: Any) -> TrainabilityCheck:
    return verify_trainable_row(row)


def verify_bit_manipulation(row: Any) -> TrainabilityCheck:
    return verify_trainable_row(row)


def verify_char_cipher(row: Any) -> TrainabilityCheck:
    return verify_trainable_row(row)


def verify_word_cipher(row: Any) -> TrainabilityCheck:
    return verify_trainable_row(row)


def verify_permutation_sorting(row: Any) -> TrainabilityCheck:
    return verify_trainable_row(row)


def verify_gravity_numeric(row: Any) -> TrainabilityCheck:
    return verify_trainable_row(row)


def verify_unit_conversion(row: Any) -> TrainabilityCheck:
    return verify_trainable_row(row)


def verify_numeric_formula_safe(row: Any) -> TrainabilityCheck:
    return verify_trainable_row(row)


def verify_equation_operator(row: Any) -> TrainabilityCheck:
    return verify_trainable_row(row)


def verify_sequence_pattern(row: Any) -> TrainabilityCheck:
    return verify_trainable_row(row)


def _row_to_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    if is_dataclass(row):
        return asdict(row)
    data = {key: getattr(row, key) for key in dir(row) if not key.startswith("_") and not callable(getattr(row, key))}
    return data


def _first_rejection(checks: dict[str, bool], row_rejection_reason: Any, trainable_flag: Any) -> str | None:
    if not checks["known_family"]:
        return "unknown_family"
    if not checks["prompt_present"]:
        return "missing_prompt"
    if not checks["answer_present"]:
        return "missing_answer"
    if not checks["target_text_present"]:
        return "missing_target_text"
    if not checks["trace_present"]:
        return "missing_trace"
    if not checks["no_abstain_placeholder"]:
        return "abstain_placeholder"
    if not checks["exactly_one_box"]:
        return "multiple_boxes" if checks["target_text_present"] else "missing_target_text"
    if not checks["no_text_after_final_box"]:
        return "text_after_final_box"
    if not checks["answer_match"]:
        return "answer_mismatch"
    if not checks["ambiguity_zero"]:
        return "ambiguity_nonzero"
    if not checks["verification_pass"]:
        return "verification_not_pass"
    if checks["duplicate_prompt"]:
        return "duplicate_prompt"
    if not checks["safe_metadata"]:
        return "unsafe_metadata"
    if row_rejection_reason is not None:
        return "row_rejection_reason_present"
    if trainable_flag is False:
        return "unsafe_metadata"
    return None


def _failed_check(
    family: str | None,
    answer: str | None,
    boxed_answer: str | None,
    ambiguity_count: int,
    reason: str,
    metadata: dict[str, Any],
) -> TrainabilityCheck:
    return TrainabilityCheck(
        verification_status="FAIL",
        trainable=False,
        family=family,
        answer=answer,
        boxed_answer=boxed_answer,
        ambiguity_count=ambiguity_count,
        rejection_reason=reason,
        checks={
            "known_family": False,
            "verification_pass": False,
            "ambiguity_zero": False,
            "no_abstain_placeholder": False,
            "exactly_one_box": False,
            "no_text_after_final_box": False,
            "answer_match": False,
            "prompt_present": False,
            "answer_present": False,
            "target_text_present": False,
            "trace_present": False,
            "duplicate_prompt": False,
            "safe_metadata": False,
        },
        metadata=metadata,
    )
