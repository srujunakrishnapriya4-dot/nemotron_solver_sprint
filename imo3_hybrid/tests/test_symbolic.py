from __future__ import annotations

from src.common.schemas import FailureType
from src.symbolic import validators


def test_symbolic_partial_support_remains_unsupported_and_weak() -> None:
    result = validators.validate_relation_bundle(
        bundle=("2 + 1 = 3", "x + z = y"),
        domain_hint="algebra",
    )

    assert result.status is validators.CompositeValidationStatus.UNSUPPORTED
    assert result.passed is False
    assert result.score <= 0.42
    assert result.failure_type is FailureType.COVERAGE_GAP
    assert result.metadata["partial_support"] is True
    assert result.metadata["support_ratio"] > 0.0
    assert result.metadata["unsupported_ratio"] > 0.0


def test_symbolic_contradiction_produces_failure_and_no_pass_signal() -> None:
    result = validators.validate_single_statement(
        "x = x + 1",
        domain_hint="algebra",
    )

    assert result.status is validators.CompositeValidationStatus.CONTRADICTION
    assert result.passed is False
    assert result.score == 0.0
    assert result.contradiction_found is True
    assert result.failure_type is FailureType.SYMBOLIC_MISMATCH
