from __future__ import annotations

from dataclasses import replace
import json

import pytest

from nemotron_engine.training_backend.adapter_validator import AdapterValidationConfig
from nemotron_engine.training_backend.backend_contracts import BackendArtifactRef, BackendInvocation, BackendResult
from nemotron_engine.training_backend.backend_runner import (
    BackendRunReport,
    BackendRunnerError,
    invoke_training_backend,
)


SHA = "c" * 64


def make_adapter(tmp_path, rank=8):
    path = tmp_path / "adapter"
    path.mkdir()
    (path / "adapter_config.json").write_text(json.dumps({"r": rank}), encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"x")
    return path


def invocation(*, dry_run=False) -> BackendInvocation:
    return BackendInvocation("inv", "external_sft", "sft", "plan", "lora", "dataset", 1, 0, 1, dry_run, "out", {})


def success_mapping(inv: BackendInvocation) -> dict:
    return {
        "result_id": "res",
        "invocation_hash": inv.invocation_hash,
        "backend_kind": inv.backend_kind,
        "stage": inv.stage,
        "status": "success",
        "trained": True,
        "adapter_path": "adapter",
        "checkpoint_path": None,
        "artifacts": [{"path": "adapter/adapter_model.safetensors", "role": "adapter_model", "sha256": SHA, "size_bytes": 1, "required": True}],
        "metrics": [],
        "started": True,
        "finished": True,
        "error": None,
        "metadata": {},
    }


def test_dry_run_does_not_call_backend() -> None:
    called = False
    inv = invocation(dry_run=True)

    def backend(_):
        nonlocal called
        called = True
        return {}

    report = invoke_training_backend(inv, backend)
    assert not called
    assert not report.trained
    assert report.called_backend is False


def test_non_dry_run_backend_none_raises() -> None:
    with pytest.raises(BackendRunnerError, match="explicit backend"):
        invoke_training_backend(invocation(dry_run=False), None)


def test_backend_exception_becomes_failed_report() -> None:
    def backend(_):
        raise RuntimeError("boom")

    report = invoke_training_backend(invocation(), backend)
    assert report.status == "failed"
    assert not report.trained
    assert report.errors


def test_malformed_backend_success_rejected() -> None:
    def backend(inv):
        data = success_mapping(inv)
        data["adapter_path"] = None
        return data

    with pytest.raises((BackendRunnerError, Exception)):
        invoke_training_backend(invocation(), backend)


def test_trained_backend_result_requires_adapter_validation(tmp_path) -> None:
    inv = invocation()

    def backend(_):
        return success_mapping(inv)

    report = invoke_training_backend(inv, backend, adapter_validation_config=AdapterValidationConfig(make_adapter(tmp_path)))
    assert report.trained
    assert report.adapter_validation_report is not None
    assert report.adapter_validation_report.passed


def test_backend_result_with_adapter_rank_over_32_rejected(tmp_path) -> None:
    inv = invocation()

    def backend(_):
        return success_mapping(inv)

    report = invoke_training_backend(inv, backend, adapter_validation_config=AdapterValidationConfig(make_adapter(tmp_path, rank=64)))
    assert not report.trained
    assert report.status == "rejected"
    assert report.errors


def test_run_report_rejects_trained_without_adapter_validation() -> None:
    inv = invocation()
    result = BackendResult(**success_mapping(inv))
    with pytest.raises(BackendRunnerError, match="adapter validation"):
        BackendRunReport(inv.invocation_hash, True, result, None, None, True, "success")


def test_forged_run_report_hash_rejected(tmp_path) -> None:
    inv = invocation()

    def backend(_):
        return success_mapping(inv)

    report = invoke_training_backend(inv, backend, adapter_validation_config=AdapterValidationConfig(make_adapter(tmp_path)))
    with pytest.raises(BackendRunnerError, match="report_hash"):
        replace(report, report_hash="forged")
