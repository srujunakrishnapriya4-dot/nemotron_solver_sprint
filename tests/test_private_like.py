from __future__ import annotations

from dataclasses import replace
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.evaluation.private_like import (
    PrivateLikeError,
    PrivateLikeGateConfig,
    PrivateLikeReport,
    evaluate_private_like_gate,
)
from nemotron_engine.evaluation.transfer_harness import TransferExample, TransferHarnessConfig, evaluate_transfer_slices


def ex(problem_id: str, completion: str = r"\boxed{42}", metadata: dict | None = None) -> TransferExample:
    return TransferExample(problem_id, "fam", "prim", "fmt", "private_like", "42", completion, "integer", metadata or {})


def transfer(examples: list[TransferExample]):
    return evaluate_transfer_slices(examples, TransferHarnessConfig(required_slices=("private_like",), allow_small_slices=True))


def test_private_like_slice_required() -> None:
    empty_transfer = evaluate_transfer_slices([], TransferHarnessConfig(required_slices=(), allow_small_slices=True))

    with pytest.raises(PrivateLikeError):
        evaluate_private_like_gate([], empty_transfer)


def test_private_like_count_mismatch_rejects() -> None:
    examples = [ex("p1")]
    report = transfer(examples)

    with pytest.raises(PrivateLikeError):
        evaluate_private_like_gate([], report)


def test_insufficient_accuracy_regression_and_errors_reject() -> None:
    examples = [ex("p1", "no box"), ex("p2", r"\boxed{0}")]
    report = transfer(examples)
    private = evaluate_private_like_gate(examples, report, PrivateLikeGateConfig(min_examples=2))

    assert private.passed is False
    assert "extraction_error_rate" in private.failure_reasons
    assert "accuracy_below_threshold" in private.failure_reasons
    assert "private_like_regression" in private.failure_reasons


def test_contamination_metadata_rejects() -> None:
    examples = [ex("p1", metadata={"contamination_flags": ("x",)}), ex("p2")]
    report = transfer(examples)
    private = evaluate_private_like_gate(examples, report)

    assert private.passed is False
    assert "contamination" in private.failure_reasons


def test_report_hash_deterministic_and_forgery_rejected() -> None:
    examples = [ex("p1"), ex("p2")]
    first = evaluate_private_like_gate(examples, transfer(examples))
    second = evaluate_private_like_gate(examples, transfer(examples))

    assert first.report_hash == second.report_hash
    with pytest.raises(PrivateLikeError):
        replace(first, report_hash="forged")


def test_private_like_report_rejects_bad_rates_and_passed_true_errors() -> None:
    with pytest.raises(PrivateLikeError):
        PrivateLikeReport(1, 1, 1.0, 1.0, 0.0, math.nan, 0.0, 0, True)
    with pytest.raises(PrivateLikeError):
        PrivateLikeReport(1, 1, 1.0, 1.0, 0.0, 0.0, math.inf, 0, True)
    with pytest.raises(PrivateLikeError):
        PrivateLikeReport(1, 1, 1.0, 1.0, 0.0, 0.0, 0.1, 0, True)
    with pytest.raises(PrivateLikeError):
        PrivateLikeReport(1, 1, 1.0, 1.0, 0.0, 0.1, 0.0, 0, True)
    with pytest.raises(PrivateLikeError):
        PrivateLikeReport(1, 1, 1.0, 1.0, 0.0, 0.0, 0.0, 1, True)


def test_private_like_report_direct_forgery_rejected() -> None:
    with pytest.raises(PrivateLikeError):
        PrivateLikeReport(1, 0, 1.0, 1.0, 0.0, 0.0, 0.0, 0, True)
