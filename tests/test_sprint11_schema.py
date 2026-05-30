from __future__ import annotations

import pytest

from kaggle_anti086.data.schema import RowValidationError, is_trainable_row, validate_row, validate_rows


def valid_row(**overrides) -> dict:
    row = {
        "id": "r1",
        "family": "bit_manipulation",
        "subfamily": "xor_mask",
        "rule_id": "bit_xor_mask_001",
        "prompt": "What is x?",
        "answer": "1010",
        "source": "unit",
        "solver_name": "fixture",
        "verification_status": "verified",
        "difficulty": 0.4,
        "split": "train",
        "leakage_group": "lg1",
    }
    row.update(overrides)
    return row


def test_valid_row_passes() -> None:
    parsed = validate_row(valid_row())
    assert parsed.id == "r1"
    assert is_trainable_row(valid_row())


@pytest.mark.parametrize("field", ["rule_id", "leakage_group", "verification_status"])
def test_missing_required_fields_fail(field: str) -> None:
    row = valid_row()
    row.pop(field)
    with pytest.raises(RowValidationError):
        validate_row(row)


def test_unknown_family_fails() -> None:
    with pytest.raises(RowValidationError):
        validate_row(valid_row(family="made_up"))


@pytest.mark.parametrize("status", ["unverified", "unsupported", "unsafe"])
def test_bad_train_status_fails(status: str) -> None:
    with pytest.raises(RowValidationError):
        validate_row(valid_row(verification_status=status))


def test_unsupported_equation_train_row_fails() -> None:
    with pytest.raises(RowValidationError):
        validate_row(valid_row(family="equation_operator", verification_status="unsupported"))


def test_quarantine_row_can_omit_answer() -> None:
    row = valid_row(split="quarantine", verification_status="quarantined")
    row.pop("answer")
    assert validate_row(row).answer is None


def test_validate_rows_returns_counts_and_failures() -> None:
    report = validate_rows([valid_row(), valid_row(id="bad", family="nope")])
    assert report["valid_count"] == 1
    assert report["by_family"]["bit_manipulation"] == 1
    assert report["failure_count"] == 1
