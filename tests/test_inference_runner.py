from __future__ import annotations

from dataclasses import replace

import pytest

from nemotron_engine.inference_eval.inference_contracts import InferenceBackendInvocation, InferenceBackendResult, InferenceCompletion
from nemotron_engine.inference_eval.inference_runner import (
    InferenceRunReport,
    InferenceRunnerConfig,
    InferenceRunnerError,
    invoke_inference_backend,
)
from nemotron_engine.inference_eval.prompt_batch import PromptBatch, PromptExample
from nemotron_engine.runtime.serving_config import ServingConfig


def serving(**updates: object) -> ServingConfig:
    data = {"prompt_template_hash": "prompt", "tokenizer_hash": "tokenizer", "model_hash": "model", "batch_size": 4}
    data.update(updates)
    return ServingConfig(**data)


def batch(config: ServingConfig | None = None) -> PromptBatch:
    cfg = config or serving()
    ex = PromptExample("p1", "Prompt without answers", "private_like", "fam", "prim", "fmt", "integer", "private_like")
    from nemotron_engine.core.schemas import stable_hash

    return PromptBatch("batch", (ex,), stable_hash(cfg.to_dict()))


def invocation(prompt_batch: PromptBatch, config: ServingConfig | None = None, *, dry_run: bool = False) -> InferenceBackendInvocation:
    cfg = config or serving()
    from nemotron_engine.core.schemas import stable_hash

    return InferenceBackendInvocation(
        "inv",
        "external_mock_for_tests",
        stable_hash(cfg.to_dict()),
        prompt_batch.batch_hash,
        "model",
        None,
        dry_run,
        len(prompt_batch.examples),
        {},
    )


def success_result(inv: InferenceBackendInvocation, prompt_batch: PromptBatch) -> InferenceBackendResult:
    return InferenceBackendResult(
        "res",
        inv.invocation_hash,
        inv.backend_kind,
        "success",
        (InferenceCompletion("p1", prompt_batch.examples[0].prompt_hash, r"\boxed{42}"),),
        (),
        True,
        True,
        None,
        {},
    )


def test_dry_run_does_not_call_backend_and_returns_no_successful_completions() -> None:
    cfg = serving()
    prompt_batch = batch(cfg)
    inv = invocation(prompt_batch, cfg, dry_run=True)
    called = {"value": False}

    def backend(_inv: InferenceBackendInvocation, _batch: PromptBatch) -> InferenceBackendResult:
        called["value"] = True
        return success_result(_inv, _batch)

    report = invoke_inference_backend(inv, prompt_batch, backend, config=InferenceRunnerConfig(cfg))

    assert called["value"] is False
    assert report.called_backend is False
    assert report.backend_result is not None
    assert report.backend_result.completions == ()
    assert report.status == "dry_run"


def test_non_dry_run_without_backend_raises() -> None:
    cfg = serving()
    prompt_batch = batch(cfg)
    with pytest.raises(InferenceRunnerError, match="backend"):
        invoke_inference_backend(invocation(prompt_batch, cfg), prompt_batch, None, config=InferenceRunnerConfig(cfg))


def test_backend_exception_becomes_failed_report() -> None:
    cfg = serving()
    prompt_batch = batch(cfg)
    inv = invocation(prompt_batch, cfg)

    def backend(_inv: InferenceBackendInvocation, _batch: PromptBatch) -> InferenceBackendResult:
        raise RuntimeError("boom")

    report = invoke_inference_backend(inv, prompt_batch, backend, config=InferenceRunnerConfig(cfg))

    assert report.status == "failed"
    assert report.backend_result is not None
    assert report.backend_result.error == "boom"
    assert report.errors


def test_backend_never_receives_expected_answers() -> None:
    cfg = serving()
    prompt_batch = batch(cfg)
    inv = invocation(prompt_batch, cfg)

    def backend(_inv: InferenceBackendInvocation, received_batch: PromptBatch) -> InferenceBackendResult:
        assert not hasattr(received_batch.examples[0], "expected_answer")
        assert "expected_answer" not in received_batch.examples[0].metadata
        return success_result(_inv, received_batch)

    report = invoke_inference_backend(inv, prompt_batch, backend, config=InferenceRunnerConfig(cfg))

    assert report.completed is True
    assert report.status == "success"


def test_malformed_backend_result_rejected_report() -> None:
    cfg = serving()
    prompt_batch = batch(cfg)
    inv = invocation(prompt_batch, cfg)

    report = invoke_inference_backend(inv, prompt_batch, lambda _inv, _batch: {"bad": "payload"}, config=InferenceRunnerConfig(cfg))

    assert report.status == "rejected"
    assert report.errors


def test_non_submission_exact_config_flagged_and_cannot_pass() -> None:
    cfg = serving(temperature=0.5, strict_submission_mode=False)
    prompt_batch = batch(cfg)
    inv = invocation(prompt_batch, cfg)
    report = invoke_inference_backend(
        inv,
        prompt_batch,
        lambda _inv, _batch: success_result(_inv, _batch),
        config=InferenceRunnerConfig(cfg, allow_non_submission_exact_eval=True),
    )

    assert report.non_submission_exact is True
    assert report.completed is False
    assert report.status == "rejected"
    with pytest.raises(InferenceRunnerError):
        InferenceRunReport(
            report.invocation_hash,
            report.prompt_batch_hash,
            report.serving_config_hash,
            True,
            report.backend_result,
            "success",
            True,
            True,
        )


def test_run_report_rejects_contradictory_lifecycle_states() -> None:
    cfg = serving()
    prompt_batch = batch(cfg)
    inv = invocation(prompt_batch, cfg)
    backend_result = success_result(inv, prompt_batch)

    with pytest.raises(InferenceRunnerError, match="completed"):
        InferenceRunReport(
            inv.invocation_hash,
            prompt_batch.batch_hash,
            prompt_batch.serving_config_hash,
            True,
            backend_result,
            "success",
            False,
            False,
        )
    with pytest.raises(InferenceRunnerError, match="completed"):
        InferenceRunReport(
            inv.invocation_hash,
            prompt_batch.batch_hash,
            prompt_batch.serving_config_hash,
            True,
            backend_result,
            "failed",
            True,
            False,
        )
    with pytest.raises(InferenceRunnerError, match="non-submission-exact"):
        InferenceRunReport(
            inv.invocation_hash,
            prompt_batch.batch_hash,
            prompt_batch.serving_config_hash,
            True,
            backend_result,
            "success",
            True,
            True,
        )
    with pytest.raises(InferenceRunnerError, match="dry-run"):
        InferenceRunReport(
            inv.invocation_hash,
            prompt_batch.batch_hash,
            prompt_batch.serving_config_hash,
            True,
            backend_result,
            "success",
            True,
            False,
            metadata={"dry_run": True},
        )


def test_forged_run_report_hash_rejected_and_unsafe_metadata_rejected() -> None:
    cfg = serving()
    prompt_batch = batch(cfg)
    inv = invocation(prompt_batch, cfg)
    report = invoke_inference_backend(inv, prompt_batch, lambda _inv, _batch: success_result(_inv, _batch), config=InferenceRunnerConfig(cfg))

    with pytest.raises(InferenceRunnerError, match="report_hash"):
        replace(report, report_hash="forged")
    with pytest.raises(InferenceRunnerError):
        InferenceRunnerConfig(cfg, metadata={"leaderboard_success": True})
