from __future__ import annotations

from dataclasses import replace

import pytest

from nemotron_engine.evaluation.transfer_harness import TransferHarnessConfig
from nemotron_engine.inference_eval.completion_capture import CompletionCaptureReport, CompletionRecord
from nemotron_engine.inference_eval.evaluation_manifest import (
    InferenceEvaluationManifest,
    InferenceEvaluationManifestError,
    build_inference_evaluation_manifest,
    compute_inference_evaluation_manifest_id,
    validate_inference_evaluation_manifest,
)
from nemotron_engine.inference_eval.inference_contracts import InferenceBackendInvocation, InferenceBackendResult, InferenceCompletion
from nemotron_engine.inference_eval.private_like_bridge import evaluate_private_like_from_completions
from nemotron_engine.inference_eval.prompt_batch import PromptBatch, PromptExample
from nemotron_engine.inference_eval.transfer_bridge import CompletionTransferConfig, evaluate_completions_with_transfer_harness


def evidence():
    examples = (
        PromptExample("p1", "Prompt 1", "private_like", "fam1", "prim", "fmt", "integer", "private_like"),
        PromptExample("p2", "Prompt 2", "private_like", "fam2", "prim", "fmt", "integer", "private_like"),
    )
    batch = PromptBatch("batch", examples, "serving")
    inv = InferenceBackendInvocation("inv", "external_mock_for_tests", "serving", batch.batch_hash, "model", None, False, 2, {})
    result = InferenceBackendResult(
        "res",
        inv.invocation_hash,
        inv.backend_kind,
        "success",
        tuple(InferenceCompletion(ex.problem_id, ex.prompt_hash, r"\boxed{42}") for ex in examples),
        (),
        True,
        True,
        None,
        {},
    )
    capture = CompletionCaptureReport(
        tuple(CompletionRecord(ex.problem_id, ex.prompt_hash, r"\boxed{42}", "serving", inv.invocation_hash, result.result_hash) for ex in examples),
        batch.batch_hash,
        "serving",
        inv.invocation_hash,
        result.result_hash,
        2,
    )
    transfer = evaluate_completions_with_transfer_harness(
        batch,
        capture,
        {"p1": "42", "p2": "42"},
        config=CompletionTransferConfig(TransferHarnessConfig(required_slices=("private_like",), allow_small_slices=True)),
    )
    private = evaluate_private_like_from_completions(transfer)
    return batch, inv, result, capture, transfer, private


def test_valid_evaluated_manifest_passes_and_validates() -> None:
    batch, inv, result, capture, transfer, private = evidence()
    manifest = build_inference_evaluation_manifest(
        prompt_batch=batch,
        invocation=inv,
        result=result,
        capture_report=capture,
        transfer_report=transfer,
        private_like_report=private,
    )

    assert manifest.passed is True
    assert validate_inference_evaluation_manifest(manifest) == manifest


def test_forged_manifest_id_and_hash_rejected() -> None:
    batch, inv, result, capture, transfer, private = evidence()
    manifest = build_inference_evaluation_manifest(
        prompt_batch=batch,
        invocation=inv,
        result=result,
        capture_report=capture,
        transfer_report=transfer,
        private_like_report=private,
    )
    with pytest.raises(InferenceEvaluationManifestError, match="manifest_id"):
        replace(manifest, manifest_id="forged")
    with pytest.raises(InferenceEvaluationManifestError, match="manifest_hash"):
        replace(manifest, manifest_hash="forged")


def test_passed_true_requires_transfer_private_exact_not_dry_and_no_errors() -> None:
    batch, inv, result, capture, transfer, private = evidence()
    manifest = build_inference_evaluation_manifest(
        prompt_batch=batch,
        invocation=inv,
        result=result,
        capture_report=capture,
        transfer_report=transfer,
        private_like_report=private,
    )
    with pytest.raises(InferenceEvaluationManifestError):
        replace(manifest, transfer_report_hash=None)
    with pytest.raises(InferenceEvaluationManifestError):
        replace(manifest, private_like_report_hash=None, metadata={"require_private_like": True})
    with pytest.raises(InferenceEvaluationManifestError):
        replace(manifest, non_submission_exact=True)
    with pytest.raises(InferenceEvaluationManifestError):
        replace(manifest, dry_run=True)
    with pytest.raises(InferenceEvaluationManifestError):
        replace(manifest, errors=("bad",))
    with pytest.raises(InferenceEvaluationManifestError):
        replace(manifest, evaluated=False)


def test_direct_manifest_requires_private_like_by_default_but_allows_explicit_opt_out() -> None:
    batch, inv, result, capture, transfer, _private = evidence()
    payload = {
        "prompt_batch_hash": batch.batch_hash,
        "serving_config_hash": batch.serving_config_hash,
        "inference_invocation_hash": inv.invocation_hash,
        "inference_result_hash": result.result_hash,
        "completion_capture_hash": capture.report_hash,
        "transfer_report_hash": transfer.report_hash,
        "private_like_report_hash": None,
        "non_submission_exact": False,
        "backend_kind": inv.backend_kind,
        "dry_run": False,
        "evaluated": True,
        "passed": True,
        "errors": (),
        "warnings": (),
        "metadata": {},
    }
    with pytest.raises(InferenceEvaluationManifestError, match="private_like_report_hash"):
        InferenceEvaluationManifest(manifest_id=compute_inference_evaluation_manifest_id(payload), **payload)

    opt_out = {**payload, "metadata": {"require_private_like": False}}
    manifest = InferenceEvaluationManifest(manifest_id=compute_inference_evaluation_manifest_id(opt_out), **opt_out)
    assert manifest.passed is True
    assert manifest.private_like_report_hash is None


def test_dry_run_build_cannot_pass() -> None:
    batch, inv, result, capture, transfer, private = evidence()
    dry_inv = InferenceBackendInvocation(
        inv.invocation_id,
        inv.backend_kind,
        inv.serving_config_hash,
        inv.prompt_batch_hash,
        inv.model_hash,
        inv.adapter_hash,
        True,
        inv.expected_completion_count,
        inv.metadata,
    )
    manifest = build_inference_evaluation_manifest(
        prompt_batch=batch,
        invocation=dry_inv,
        result=result,
        capture_report=capture,
        transfer_report=transfer,
        private_like_report=private,
    )

    assert manifest.passed is False
    assert "dry_run" in manifest.errors


def test_unsafe_metadata_rejected() -> None:
    batch, inv, result, capture, transfer, private = evidence()
    with pytest.raises(InferenceEvaluationManifestError):
        build_inference_evaluation_manifest(
            prompt_batch=batch,
            invocation=inv,
            result=result,
            capture_report=capture,
            transfer_report=transfer,
            private_like_report=private,
            metadata={"promotion_success": True},
        )
    with pytest.raises(InferenceEvaluationManifestError):
        build_inference_evaluation_manifest(
            prompt_batch=batch,
            invocation=inv,
            result=result,
            capture_report=capture,
            transfer_report=transfer,
            private_like_report=private,
            metadata={"nested": {"promotion_success": True}},
        )
    with pytest.raises(InferenceEvaluationManifestError):
        build_inference_evaluation_manifest(
            prompt_batch=batch,
            invocation=inv,
            result=result,
            capture_report=capture,
            transfer_report=transfer,
            private_like_report=private,
            metadata={"nested": {"public_score": 0.99}},
        )
    with pytest.raises(InferenceEvaluationManifestError):
        InferenceEvaluationManifest(
            "x",
            "batch",
            "serving",
            "inv",
            "res",
            "capture",
            "transfer",
            "private",
            False,
            "external_mock_for_tests",
            False,
            True,
            False,
            metadata={"score": "95+ guaranteed"},
        )
