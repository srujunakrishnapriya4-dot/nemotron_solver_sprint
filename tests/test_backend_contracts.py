from __future__ import annotations

from dataclasses import replace
import math

import pytest

from nemotron_engine.training_backend.backend_contracts import (
    BackendArtifactRef,
    BackendContractError,
    BackendInvocation,
    BackendMetric,
    BackendResult,
    BackendStatus,
    validate_backend_invocation,
    validate_backend_result,
)


SHA = "a" * 64


def invocation(**updates: object) -> BackendInvocation:
    data = {
        "invocation_id": "inv-1",
        "backend_kind": "external_sft",
        "stage": "sft",
        "training_plan_hash": "plan",
        "lora_config_hash": "lora",
        "dataset_manifest_hash": "dataset",
        "input_row_count": 1,
        "input_pair_count": 0,
        "seed": 1,
        "dry_run": True,
        "output_dir": "out/sft",
        "metadata": {},
    }
    data.update(updates)
    return BackendInvocation(**data)


def trained_result(inv: BackendInvocation, **updates: object) -> BackendResult:
    data = {
        "result_id": "res-1",
        "invocation_hash": inv.invocation_hash,
        "backend_kind": inv.backend_kind,
        "stage": inv.stage,
        "status": "success",
        "trained": True,
        "adapter_path": "adapter",
        "checkpoint_path": None,
        "artifacts": (BackendArtifactRef("adapter/adapter_model.safetensors", "adapter_model", SHA, 10, True),),
        "metrics": (BackendMetric("loss", 1.0, step=1),),
        "started": True,
        "finished": True,
        "error": None,
        "metadata": {},
    }
    data.update(updates)
    return BackendResult(**data)


def test_valid_dry_run_invocation_validates() -> None:
    inv = invocation()
    assert validate_backend_invocation(inv) == inv
    assert inv.invocation_hash == invocation().invocation_hash


def test_invocation_rejects_forged_invocation_hash() -> None:
    inv = invocation()
    with pytest.raises(BackendContractError, match="invocation_hash"):
        replace(inv, invocation_hash="forged")


def test_result_rejects_forged_result_hash() -> None:
    inv = invocation(dry_run=False)
    result = trained_result(inv)
    with pytest.raises(BackendContractError, match="result_hash"):
        replace(result, result_hash="forged")


def test_trained_true_rejected_when_status_failed() -> None:
    inv = invocation(dry_run=False)
    with pytest.raises(BackendContractError, match="status"):
        trained_result(inv, status="failed")


def test_trained_true_rejected_without_adapter_path() -> None:
    inv = invocation(dry_run=False)
    with pytest.raises(BackendContractError, match="adapter_path"):
        trained_result(inv, adapter_path=None)


def test_dry_run_invocation_cannot_produce_trained_true_result() -> None:
    inv = invocation(dry_run=True)
    result = trained_result(inv)
    with pytest.raises(BackendContractError, match="dry_run"):
        validate_backend_result(result, invocation=inv)


def test_metric_nan_inf_rejected() -> None:
    with pytest.raises(BackendContractError, match="finite"):
        BackendMetric("loss", math.nan)
    with pytest.raises(BackendContractError, match="finite"):
        BackendMetric("loss", math.inf)


def test_metadata_kaggle_leaderboard_score_claims_rejected() -> None:
    with pytest.raises(BackendContractError, match="Kaggle"):
        invocation(metadata={"kaggle_success": True})
    with pytest.raises(BackendContractError, match="guarantee"):
        invocation(metadata={"claim": "95+ guaranteed"})
    inv = invocation(dry_run=False)
    with pytest.raises(BackendContractError, match="leaderboard"):
        trained_result(inv, metadata={"leaderboard_success": True})


def test_backend_kind_stage_mismatch_rejected() -> None:
    with pytest.raises(BackendContractError, match="incompatible"):
        invocation(backend_kind="external_sft", stage="dpo")


def test_required_artifact_hashes_needed_when_requested() -> None:
    inv = invocation(dry_run=False)
    with pytest.raises(BackendContractError, match="sha256"):
        trained_result(inv, artifacts=(BackendArtifactRef("adapter/adapter_model.safetensors", "adapter_model", None, None, True),))


def test_backend_invocation_rejects_bad_numeric_and_boolean_types() -> None:
    with pytest.raises(BackendContractError, match="input_row_count"):
        invocation(input_row_count="1")
    with pytest.raises(BackendContractError, match="input_pair_count"):
        invocation(input_pair_count=True)
    with pytest.raises(BackendContractError, match="input_row_count"):
        invocation(input_row_count=-1)
    with pytest.raises(BackendContractError, match="dry_run"):
        invocation(dry_run="false")


def test_trained_true_rejects_missing_lifecycle_error_checkpoint_and_artifacts() -> None:
    inv = invocation(dry_run=False)
    with pytest.raises(BackendContractError, match="started"):
        trained_result(inv, started=False)
    with pytest.raises(BackendContractError, match="started"):
        trained_result(inv, finished=False)
    with pytest.raises(BackendContractError, match="error"):
        trained_result(inv, error="boom")
    with pytest.raises(BackendContractError, match="required adapter artifact"):
        trained_result(inv, artifacts=())
    with pytest.raises(BackendContractError, match="checkpoint_path"):
        trained_result(inv, checkpoint_path="checkpoint")


def test_backend_metric_negative_step_rejected() -> None:
    with pytest.raises(BackendContractError, match="step"):
        BackendMetric("loss", 1.0, step=-1)
