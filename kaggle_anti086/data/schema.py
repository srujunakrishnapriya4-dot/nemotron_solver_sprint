from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


ALLOWED_FAMILIES = {
    "bit_manipulation",
    "numeric_formula",
    "gravity_numeric",
    "unit_conversion",
    "roman_numeral",
    "custom_numeral",
    "word_cipher",
    "char_cipher",
    "symbol_mapping",
    "digit_symbol_mapping",
    "equation_operator",
    "sequence_pattern",
    "permutation_sorting",
    "table_lookup",
    "grid_mapping",
    "date_time_rule",
    "format_only",
    "unknown",
}

ALLOWED_SPLITS = {
    "train",
    "private_like_eval",
    "family_hard_eval",
    "rule_holdout_eval",
    "anti_leak_eval",
    "parent_calibrated_eval",
    "dev",
    "quarantine",
}

ALLOWED_VERIFICATION_STATUSES = {"verified", "unverified", "abstained", "unsupported", "unsafe", "quarantined"}

REQUIRED_FIELDS = {
    "id",
    "family",
    "subfamily",
    "rule_id",
    "prompt",
    "source",
    "solver_name",
    "verification_status",
    "difficulty",
    "split",
    "leakage_group",
}


class RowValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ValidatedRow:
    id: str
    family: str
    subfamily: str
    rule_id: str
    prompt: str
    answer: str | None
    source: str
    solver_name: str
    verification_status: str
    difficulty: float
    split: str
    leakage_group: str
    raw: dict[str, Any]


def validate_row(row: dict, *, context: str | None = None) -> ValidatedRow:
    prefix = f"{context}: " if context else ""
    for field in REQUIRED_FIELDS:
        _require_non_empty(row, field, prefix)
    if str(row.get("split")) != "quarantine":
        _require_non_empty(row, "answer", prefix)

    family = str(row["family"]).strip()
    split = str(row["split"]).strip()
    status = str(row["verification_status"]).strip()
    if family not in ALLOWED_FAMILIES:
        raise RowValidationError(f"{prefix}unknown family: {family}")
    if split not in ALLOWED_SPLITS:
        raise RowValidationError(f"{prefix}unknown split: {split}")
    if status not in ALLOWED_VERIFICATION_STATUSES:
        raise RowValidationError(f"{prefix}unknown verification_status: {status}")
    if split == "train" and status != "verified":
        raise RowValidationError(f"{prefix}train rows must be verified")
    if split == "train" and family == "equation_operator" and status != "verified":
        raise RowValidationError(f"{prefix}unsupported equation/operator row cannot be train")
    if split == "train" and status in {"unsupported", "unsafe", "abstained", "quarantined", "unverified"}:
        raise RowValidationError(f"{prefix}{status} rows cannot be trainable")
    try:
        difficulty = float(row["difficulty"])
    except (TypeError, ValueError) as exc:
        raise RowValidationError(f"{prefix}difficulty must be numeric") from exc
    return ValidatedRow(
        id=str(row["id"]).strip(),
        family=family,
        subfamily=str(row["subfamily"]).strip(),
        rule_id=str(row["rule_id"]).strip(),
        prompt=str(row["prompt"]).strip(),
        answer=None if row.get("answer") is None else str(row.get("answer")).strip(),
        source=str(row["source"]).strip(),
        solver_name=str(row["solver_name"]).strip(),
        verification_status=status,
        difficulty=difficulty,
        split=split,
        leakage_group=str(row["leakage_group"]).strip(),
        raw=dict(row),
    )


def validate_rows(rows: list[dict], *, context: str | None = None) -> dict:
    by_family: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    by_status: Counter[str] = Counter()
    failures: list[dict] = []
    valid = 0
    for idx, row in enumerate(rows):
        try:
            parsed = validate_row(row, context=f"{context or 'rows'}[{idx}]")
        except RowValidationError as exc:
            failures.append({"index": idx, "id": row.get("id"), "message": str(exc)})
            continue
        valid += 1
        by_family[parsed.family] += 1
        by_split[parsed.split] += 1
        by_status[parsed.verification_status] += 1
    return {
        "row_count": len(rows),
        "valid_count": valid,
        "failure_count": len(failures),
        "by_family": dict(sorted(by_family.items())),
        "by_split": dict(sorted(by_split.items())),
        "by_verification_status": dict(sorted(by_status.items())),
        "failures": failures,
    }


def is_trainable_row(row: dict) -> bool:
    try:
        parsed = validate_row(row)
    except RowValidationError:
        return False
    return parsed.split == "train" and parsed.verification_status == "verified"


def row_key(row: dict) -> str:
    return str(row.get("id", "")).strip()


def _require_non_empty(row: dict, field: str, prefix: str) -> None:
    if field not in row:
        raise RowValidationError(f"{prefix}missing {field}")
    value = row.get(field)
    if value is None or str(value).strip() == "":
        raise RowValidationError(f"{prefix}empty {field}")
