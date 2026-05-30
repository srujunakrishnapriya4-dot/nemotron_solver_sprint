from __future__ import annotations

from dataclasses import replace
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.evaluation.transfer_harness import (
    TRANSFER_SLICE_NAMES,
    TransferEvaluationReport,
    TransferExample,
    TransferHarnessConfig,
    TransferHarnessError,
    evaluate_transfer_slices,
)


def ex(problem_id: str = "p1", split: str = "private_like", completion: str = r"\boxed{42}", expected: str = "42") -> TransferExample:
    return TransferExample(
        problem_id=problem_id,
        family_id="fam",
        primitive_family_id="prim",
        format_family_id="fmt",
        split=split,
        expected_answer=expected,
        completion=completion,
        answer_type="integer",
        metadata={},
    )


def config(*, required: tuple[str, ...] = ("private_like",), allow_small: bool = True, baseline: float = 1.0) -> TransferHarnessConfig:
    return TransferHarnessConfig(
        required_slices=required,
        allow_small_slices=allow_small,
        baseline_accuracies={name: baseline for name in required},
        min_examples_per_slice=2,
    )


def test_transfer_example_rejects_unsupported_split_and_answer_type() -> None:
    with pytest.raises(TransferHarnessError):
        ex(split="not_a_slice")
    with pytest.raises(TransferHarnessError):
        TransferExample("p", "f", "p", "fmt", "private_like", "1", r"\boxed{1}", "bad_type", {})


def test_scores_exact_boxed_answers_through_pass1_scorer() -> None:
    report = evaluate_transfer_slices([ex()], config())

    assert report.passed is True
    assert report.overall_accuracy == 1.0


def test_no_box_and_multiple_box_are_extraction_errors() -> None:
    report = evaluate_transfer_slices(
        [
            ex("p1", completion="answer is 42"),
            ex("p2", completion=r"\boxed{1} then \boxed{42}"),
        ],
        config(allow_small=True),
    )

    failures = " ".join(report.slice_results[0].failures)
    assert report.passed is False
    assert "extraction_error" in failures


def test_slice_gap_thresholds_enforced_and_no_hidden_shortcut() -> None:
    report = evaluate_transfer_slices([ex(expected="42", completion=r"\boxed{41}")], config(allow_small=True))

    assert report.passed is False
    assert any("wrong_answer" in failure for failure in report.slice_results[0].failures)
    assert any("transfer_gap" in failure for failure in report.slice_results[0].failures)


def test_small_slice_rejected_unless_allowed() -> None:
    rejected = evaluate_transfer_slices([ex()], config(allow_small=False))
    allowed = evaluate_transfer_slices([ex()], config(allow_small=True))

    assert rejected.passed is False
    assert allowed.passed is True


def test_required_slices_missing_unless_not_required_or_small_allowed() -> None:
    with pytest.raises(TransferHarnessError):
        TransferEvaluationReport(
            config_hash="cfg",
            total_examples=0,
            slice_results=(),
            overall_accuracy=0.0,
            passed=True,
            required_slices=("private_like",),
            allow_small_slices=False,
        )


def test_transfer_report_rejects_forged_hash_and_inconsistent_counts() -> None:
    report = evaluate_transfer_slices([ex()], config())

    with pytest.raises(TransferHarnessError):
        replace(report, report_hash="forged")
    with pytest.raises(TransferHarnessError):
        replace(report, total_examples=99)


def test_transfer_config_rejects_nan_inf_and_invalid_ranges() -> None:
    with pytest.raises(TransferHarnessError):
        TransferHarnessConfig(baseline_accuracies={"private_like": math.nan})
    with pytest.raises(TransferHarnessError):
        TransferHarnessConfig(baseline_accuracies={"private_like": math.inf})
    with pytest.raises(TransferHarnessError):
        TransferHarnessConfig(baseline_accuracies={"private_like": 1.5})
    with pytest.raises(TransferHarnessError):
        TransferHarnessConfig(gap_thresholds={"private_like": -0.1})


def test_transfer_slice_result_rejects_passed_true_zero_total_or_failures() -> None:
    with pytest.raises(TransferHarnessError):
        from nemotron_engine.evaluation.transfer_harness import TransferSliceResult

        TransferSliceResult("private_like", 0, 0, 0.0, 0.0, 0.0, True)
    with pytest.raises(TransferHarnessError):
        from nemotron_engine.evaluation.transfer_harness import TransferSliceResult

        TransferSliceResult("private_like", 1, 1, 1.0, 1.0, 0.0, True, failures=("bad",))


def test_transfer_report_rejects_passed_true_with_failed_slice() -> None:
    report = evaluate_transfer_slices([ex(completion=r"\boxed{0}")], config())

    with pytest.raises(TransferHarnessError):
        TransferEvaluationReport(
            config_hash=report.config_hash,
            total_examples=report.total_examples,
            slice_results=report.slice_results,
            overall_accuracy=report.overall_accuracy,
            passed=True,
            required_slices=("private_like",),
            allow_small_slices=True,
        )


def test_all_required_default_slices_can_pass_with_two_examples_each() -> None:
    examples = [ex(f"{name}-{i}", split=name) for name in TRANSFER_SLICE_NAMES for i in range(2)]

    report = evaluate_transfer_slices(examples, TransferHarnessConfig())

    assert report.passed is True
    assert report.total_examples == 16
    assert report.report_hash == evaluate_transfer_slices(examples, TransferHarnessConfig()).report_hash
