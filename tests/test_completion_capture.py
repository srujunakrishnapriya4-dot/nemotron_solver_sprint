from __future__ import annotations

from dataclasses import replace

import pytest

from nemotron_engine.inference_eval.completion_capture import (
    CompletionCaptureError,
    CompletionCaptureReport,
    CompletionRecord,
    capture_completions,
)
from nemotron_engine.inference_eval.inference_contracts import InferenceBackendInvocation, InferenceBackendResult, InferenceCompletion
from nemotron_engine.inference_eval.prompt_batch import PromptBatch, PromptExample


def batch() -> PromptBatch:
    ex = PromptExample("p1", "Prompt", "private_like", "fam", "prim", "fmt", "integer", "private_like")
    return PromptBatch("batch", (ex,), "serving")


def invocation(prompt_batch: PromptBatch) -> InferenceBackendInvocation:
    return InferenceBackendInvocation(
        "inv",
        "external_mock_for_tests",
        prompt_batch.serving_config_hash,
        prompt_batch.batch_hash,
        "model",
        None,
        False,
        len(prompt_batch.examples),
        {},
    )


def result(inv: InferenceBackendInvocation, prompt_batch: PromptBatch, *, text: str = "raw  text\n\\boxed{42}", extra: bool = False) -> InferenceBackendResult:
    completions = [
        InferenceCompletion(prompt_batch.examples[0].problem_id, prompt_batch.examples[0].prompt_hash, text, metadata={"finish": "raw"})
    ]
    if extra:
        completions.append(InferenceCompletion("extra", "prompt-extra", r"\boxed{1}"))
    return InferenceBackendResult(
        "res",
        inv.invocation_hash,
        inv.backend_kind,
        "success",
        tuple(completions),
        (),
        True,
        True,
        None,
        {},
    )


def test_capture_preserves_raw_completion_text_exactly() -> None:
    prompt_batch = batch()
    inv = invocation(prompt_batch)
    raw = "  raw completion\n\nwith spacing \\boxed{42}  "
    report = capture_completions(prompt_batch, inv, result(inv, prompt_batch, text=raw))

    assert report.records[0].completion_text == raw
    assert report.captured_count == 1


def test_missing_extra_and_prompt_hash_mismatch_rejected() -> None:
    prompt_batch = batch()
    inv = invocation(prompt_batch)
    missing = InferenceBackendResult("res", inv.invocation_hash, inv.backend_kind, "failed", (), (), True, True, "no completions", {})
    with pytest.raises(CompletionCaptureError, match="missing completion"):
        capture_completions(prompt_batch, inv, missing)
    inv2 = InferenceBackendInvocation(
        inv.invocation_id,
        inv.backend_kind,
        inv.serving_config_hash,
        inv.prompt_batch_hash,
        inv.model_hash,
        inv.adapter_hash,
        inv.dry_run,
        2,
        inv.metadata,
    )
    with pytest.raises(CompletionCaptureError, match="extra"):
        capture_completions(prompt_batch, inv2, result(inv2, prompt_batch, extra=True))
    bad = InferenceBackendResult(
        "res2",
        inv.invocation_hash,
        inv.backend_kind,
        "success",
        (InferenceCompletion("p1", "bad-prompt-hash", r"\boxed{42}"),),
        (),
        True,
        True,
        None,
        {},
    )
    with pytest.raises(CompletionCaptureError, match="prompt_hash"):
        capture_completions(prompt_batch, inv, bad)


def test_completion_record_rejects_metadata_and_forged_hash() -> None:
    prompt_batch = batch()
    inv = invocation(prompt_batch)
    res = result(inv, prompt_batch)
    record = CompletionRecord("p1", prompt_batch.examples[0].prompt_hash, r"\boxed{42}", "serving", inv.invocation_hash, res.result_hash)

    with pytest.raises(CompletionCaptureError):
        CompletionRecord("p1", prompt_batch.examples[0].prompt_hash, r"\boxed{42}", "serving", inv.invocation_hash, res.result_hash, {"is_correct": True})
    with pytest.raises(CompletionCaptureError, match="completion_hash"):
        replace(record, completion_hash="forged")
    with pytest.raises(CompletionCaptureError, match="completion_text"):
        replace(record, completion_text="")


def test_capture_report_rejects_forged_hash_count_and_source_mismatch() -> None:
    prompt_batch = batch()
    inv = invocation(prompt_batch)
    report = capture_completions(prompt_batch, inv, result(inv, prompt_batch))

    with pytest.raises(CompletionCaptureError, match="report_hash"):
        replace(report, report_hash="forged")
    with pytest.raises(CompletionCaptureError, match="captured_count"):
        replace(report, captured_count=99)
    with pytest.raises(CompletionCaptureError, match="backend_result_hash"):
        CompletionCaptureReport(
            (
                CompletionRecord(
                    report.records[0].problem_id,
                    report.records[0].prompt_hash,
                    report.records[0].completion_text,
                    report.records[0].serving_config_hash,
                    report.records[0].backend_invocation_hash,
                    "other",
                ),
            ),
            report.prompt_batch_hash,
            report.serving_config_hash,
            report.backend_invocation_hash,
            report.backend_result_hash,
            1,
        )
    with pytest.raises(CompletionCaptureError, match="serving_config_hash"):
        CompletionCaptureReport(
            (
                CompletionRecord(
                    report.records[0].problem_id,
                    report.records[0].prompt_hash,
                    report.records[0].completion_text,
                    "other",
                    report.records[0].backend_invocation_hash,
                    report.records[0].backend_result_hash,
                ),
            ),
            report.prompt_batch_hash,
            report.serving_config_hash,
            report.backend_invocation_hash,
            report.backend_result_hash,
            1,
        )
