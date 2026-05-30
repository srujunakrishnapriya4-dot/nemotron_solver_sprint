from __future__ import annotations

from dataclasses import replace

import pytest

from nemotron_engine.evaluation.transfer_harness import TransferHarnessConfig
from nemotron_engine.inference_eval.completion_capture import CompletionCaptureReport, CompletionRecord
from nemotron_engine.inference_eval.private_like_bridge import (
    CompletionPrivateLikeError,
    CompletionPrivateLikeReport,
    evaluate_private_like_from_completions,
)
from nemotron_engine.inference_eval.prompt_batch import PromptBatch, PromptExample
from nemotron_engine.inference_eval.transfer_bridge import CompletionTransferConfig, evaluate_completions_with_transfer_harness


def transfer(split: str = "private_like", completions: tuple[str, str] = (r"\boxed{42}", r"\boxed{42}"), non_exact: bool = False):
    examples = (
        PromptExample("p1", "Prompt 1", split, "fam1", "prim", "fmt", "integer", split),
        PromptExample("p2", "Prompt 2", split, "fam2", "prim", "fmt", "integer", split),
    )
    batch = PromptBatch("batch-" + split, examples, "serving")
    records = tuple(
        CompletionRecord(ex.problem_id, ex.prompt_hash, text, "serving", "inv", "res")
        for ex, text in zip(examples, completions)
    )
    capture = CompletionCaptureReport(records, batch.batch_hash, "serving", "inv", "res", 2)
    return evaluate_completions_with_transfer_harness(
        batch,
        capture,
        {"p1": "42", "p2": "42"},
        config=CompletionTransferConfig(
            TransferHarnessConfig(required_slices=(split,), allow_small_slices=True),
            non_submission_exact=non_exact,
        ),
    )


def test_private_like_slice_required() -> None:
    with pytest.raises(CompletionPrivateLikeError, match="private_like"):
        evaluate_private_like_from_completions(transfer(split="hard_known"))


def test_private_like_gate_called_and_passes_valid_slice() -> None:
    report = evaluate_private_like_from_completions(transfer())

    assert report.passed is True
    assert report.private_like_report.passed is True
    assert report.private_like_report.total == 2


def test_non_submission_exact_blocks_passed_true() -> None:
    report = evaluate_private_like_from_completions(transfer(non_exact=True))

    assert report.passed is False
    with pytest.raises(CompletionPrivateLikeError):
        replace(report, passed=True, errors=())


def test_report_rejects_passed_true_with_failed_pass8_report_and_forged_hash() -> None:
    failed = evaluate_private_like_from_completions(transfer(completions=("no box", r"\boxed{0}")))

    assert failed.private_like_report.passed is False
    with pytest.raises(CompletionPrivateLikeError):
        CompletionPrivateLikeReport(
            failed.completion_transfer_report_hash,
            failed.private_like_report_hash,
            failed.private_like_report,
            True,
        )
    with pytest.raises(CompletionPrivateLikeError, match="report_hash"):
        replace(failed, report_hash="forged")


def test_metadata_score_claims_rejected() -> None:
    report = evaluate_private_like_from_completions(transfer())
    with pytest.raises(CompletionPrivateLikeError):
        replace(report, metadata={"score_guarantee": "95+ guaranteed"})
