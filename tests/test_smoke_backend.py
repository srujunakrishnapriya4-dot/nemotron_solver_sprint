from __future__ import annotations

from dataclasses import replace

import pytest

from nemotron_engine.smoke_training.smoke_backend import (
    SmokeBackendError,
    SmokeBackendResult,
    normalize_smoke_backend_result,
    validate_smoke_backend_result,
)
from nemotron_engine.training_backend import BackendInvocation, BackendResult


SHA = "e" * 64


def invocation(dry_run: bool = False) -> BackendInvocation:
    return BackendInvocation(
        invocation_id="inv",
        backend_kind="external_sft",
        stage="sft",
        training_plan_hash="plan",
        lora_config_hash="lora",
        dataset_manifest_hash="dataset",
        input_row_count=1,
        input_pair_count=0,
        seed=1,
        dry_run=dry_run,
        output_dir=None if dry_run else "out",
    )


def backend_payload(inv: BackendInvocation, *, trained: bool = True, adapter: bool = True):
    return {
        "result_id": "res",
        "invocation_hash": inv.invocation_hash,
        "backend_kind": inv.backend_kind,
        "stage": inv.stage,
        "status": "success" if trained else "dry_run",
        "trained": trained,
        "adapter_path": "adapter" if adapter else None,
        "checkpoint_path": None,
        "artifacts": [
            {"path": "adapter/adapter_model.safetensors", "role": "adapter_model", "sha256": SHA, "size_bytes": 1, "required": True}
        ]
        if adapter
        else [],
        "metrics": [{"name": "loss", "value": 1.0, "step": 0, "split": None, "metadata": {}}],
        "started": trained,
        "finished": trained,
        "error": None,
        "metadata": {},
    }


def test_valid_smoke_backend_result_normalizes() -> None:
    inv = invocation()
    result = normalize_smoke_backend_result(
        {"backend_result": backend_payload(inv), "adapter_dir": "adapter", "training_log": [{"step": 0, "loss": 1.0}]},
        invocation=inv,
        require_log_validation=True,
    )
    assert isinstance(result.backend_result, BackendResult)
    assert validate_smoke_backend_result(result, invocation=inv, require_log_validation=True) is result


def test_trained_without_adapter_dir_rejected() -> None:
    with pytest.raises(SmokeBackendError):
        SmokeBackendResult(backend_payload(invocation()), adapter_dir=None)


def test_trained_backend_result_missing_adapter_evidence_rejected() -> None:
    inv = invocation()
    with pytest.raises(SmokeBackendError):
        SmokeBackendResult(backend_payload(inv, adapter=False), adapter_dir="adapter")


def test_require_log_validation_without_training_log_rejected() -> None:
    inv = invocation()
    result = SmokeBackendResult(backend_payload(inv), adapter_dir="adapter")
    with pytest.raises(SmokeBackendError):
        validate_smoke_backend_result(result, invocation=inv, require_log_validation=True)


def test_forged_result_hash_rejected() -> None:
    result = SmokeBackendResult(backend_payload(invocation()), adapter_dir="adapter", training_log=[{"step": 0, "loss": 1.0}])
    with pytest.raises(SmokeBackendError, match="result_hash"):
        replace(result, result_hash="forged")


@pytest.mark.parametrize("metadata", [{"kaggle_success": True}, {"leaderboard_ready": True}, {"note": "95+ guaranteed"}])
def test_forbidden_backend_metadata_rejected(metadata) -> None:
    with pytest.raises(SmokeBackendError):
        SmokeBackendResult(backend_payload(invocation()), adapter_dir="adapter", metadata=metadata)


def test_malformed_backend_result_payload_rejected() -> None:
    with pytest.raises(SmokeBackendError):
        normalize_smoke_backend_result({"backend_result": {"trained": True}, "adapter_dir": "adapter"})
