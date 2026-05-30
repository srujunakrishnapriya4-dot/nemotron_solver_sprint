from __future__ import annotations

from dataclasses import replace

import pytest

from nemotron_engine.evaluation.transfer_harness import TransferHarnessConfig
from nemotron_engine.inference_eval.completion_capture import CompletionCaptureReport, CompletionRecord
from nemotron_engine.inference_eval.prompt_batch import PromptBatch, PromptExample
from nemotron_engine.inference_eval.transfer_bridge import (
    CompletionTransferConfig,
    CompletionTransferError,
    CompletionTransferReport,
    build_transfer_examples_from_completions,
    evaluate_completions_with_transfer_harness,
)


def prompt_batch(ids: tuple[str, ...] = ("p1", "p2"), split: str = "private_like") -> PromptBatch:
    examples = tuple(
        PromptExample(pid, f"Prompt {pid}", split, f"fam-{pid}", "prim", "fmt", "integer", split)
        for pid in ids
    )
    return PromptBatch("batch-" + split, examples, "serving")


def capture(batch: PromptBatch, completions: dict[str, str] | None = None) -> CompletionCaptureReport:
    values = completions or {item.problem_id: r"\boxed{42}" for item in batch.examples}
    records = tuple(
        CompletionRecord(
            item.problem_id,
            item.prompt_hash,
            values[item.problem_id],
            batch.serving_config_hash,
            "inv",
            "res",
        )
        for item in batch.examples
        if item.problem_id in values
    )
    return CompletionCaptureReport(records, batch.batch_hash, batch.serving_config_hash, "inv", "res", len(records))


def cfg(non_exact: bool = False) -> CompletionTransferConfig:
    return CompletionTransferConfig(
        TransferHarnessConfig(required_slices=("private_like",), allow_small_slices=True),
        non_submission_exact=non_exact,
    )


def test_expected_answers_only_needed_in_transfer_bridge() -> None:
    batch = prompt_batch()
    report = capture(batch)
    examples = build_transfer_examples_from_completions(batch, report, {"p1": "42", "p2": "42"})

    assert len(examples) == 2
    assert examples[0].expected_answer == "42"
    assert not hasattr(batch.examples[0], "expected_answer")


def test_missing_expected_answer_and_extra_completion_rejected() -> None:
    batch = prompt_batch()
    report = capture(batch)
    with pytest.raises(CompletionTransferError, match="missing expected"):
        build_transfer_examples_from_completions(batch, report, {"p1": "42"})
    extra_record = CompletionRecord("extra", "prompt", r"\boxed{1}", "serving", "inv", "res")
    extra_report = CompletionCaptureReport(report.records + (extra_record,), batch.batch_hash, "serving", "inv", "res", 3)
    with pytest.raises(CompletionTransferError):
        build_transfer_examples_from_completions(batch, extra_report, {"p1": "42", "p2": "42"})


def test_no_box_and_multiple_box_failures_propagate_through_pass8() -> None:
    batch = prompt_batch()
    no_box = evaluate_completions_with_transfer_harness(batch, capture(batch, {"p1": "no box", "p2": r"\boxed{42}"}), {"p1": "42", "p2": "42"}, config=cfg())
    multi = evaluate_completions_with_transfer_harness(
        batch,
        capture(batch, {"p1": r"\boxed{1} and \boxed{42}", "p2": r"\boxed{42}"}),
        {"p1": "42", "p2": "42"},
        config=cfg(),
    )

    assert no_box.passed is False
    assert multi.passed is False
    assert "extraction_error" in " ".join(no_box.transfer_report.slice_results[0].failures)
    assert "extraction_error" in " ".join(multi.transfer_report.slice_results[0].failures)


def test_non_submission_exact_blocks_passed_and_direct_passed_true_rejected() -> None:
    batch = prompt_batch()
    report = evaluate_completions_with_transfer_harness(batch, capture(batch), {"p1": "42", "p2": "42"}, config=cfg(non_exact=True))

    assert report.non_submission_exact is True
    assert report.passed is False
    with pytest.raises(CompletionTransferError):
        replace(report, passed=True, errors=())


def test_forged_report_hash_and_failed_pass8_passed_true_rejected() -> None:
    batch = prompt_batch()
    report = evaluate_completions_with_transfer_harness(batch, capture(batch), {"p1": "42", "p2": "42"}, config=cfg())
    failed = evaluate_completions_with_transfer_harness(batch, capture(batch, {"p1": "bad", "p2": "bad"}), {"p1": "42", "p2": "42"}, config=cfg())

    with pytest.raises(CompletionTransferError, match="report_hash"):
        replace(report, report_hash="forged")
    with pytest.raises(CompletionTransferError):
        CompletionTransferReport(
            failed.prompt_batch_hash,
            failed.completion_capture_hash,
            failed.transfer_report_hash,
            failed.transfer_report,
            failed.transfer_examples,
            True,
            False,
        )
