from __future__ import annotations

from dataclasses import replace
import json

import pytest

from nemotron_engine.training_backend.adapter_validator import validate_adapter_output
from nemotron_engine.training_backend.backend_contracts import BackendArtifactRef, BackendInvocation, BackendResult
from nemotron_engine.training_backend.run_manifest import (
    TrainingRunManifest,
    TrainingRunManifestError,
    build_training_run_manifest,
    compute_training_run_manifest_id,
    validate_training_run_manifest,
)
from nemotron_engine.training_backend.training_log_validator import validate_training_log


SHA = "b" * 64


def adapter(tmp_path):
    path = tmp_path / "adapter"
    path.mkdir()
    (path / "adapter_config.json").write_text(json.dumps({"r": 8}), encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"x")
    return path


def invocation() -> BackendInvocation:
    return BackendInvocation("inv", "external_sft", "sft", "plan", "lora", "dataset", 1, 0, 1, False, "out", {})


def result(inv: BackendInvocation) -> BackendResult:
    return BackendResult(
        "res",
        inv.invocation_hash,
        inv.backend_kind,
        inv.stage,
        "success",
        True,
        "adapter",
        None,
        (BackendArtifactRef("adapter/adapter_model.safetensors", "adapter_model", SHA, 1, True),),
        (),
        True,
        True,
        None,
        {},
    )


def test_valid_trained_manifest_passes_with_adapter_validation(tmp_path) -> None:
    inv = invocation()
    adapter_report = validate_adapter_output(adapter(tmp_path))
    log_report = validate_training_log([{"step": 0, "loss": 1.0}])
    manifest = build_training_run_manifest(
        invocation_hash=inv.invocation_hash,
        backend_result=result(inv),
        training_plan_hash=inv.training_plan_hash,
        lora_config_hash=inv.lora_config_hash,
        dataset_manifest_hash=inv.dataset_manifest_hash,
        adapter_validation_report=adapter_report,
        training_log_report=log_report,
    )
    assert manifest.trained
    assert validate_training_run_manifest(manifest, backend_result=result(inv), adapter_validation_report=adapter_report) == manifest


def test_trained_true_rejected_without_adapter_validation_hash() -> None:
    with pytest.raises(TrainingRunManifestError, match="adapter_validation_hash"):
        TrainingRunManifest("m", "inv", "res", None, None, "plan", "lora", "dataset", "sft", True, "adapter", {"a": "b"})


def test_trained_true_rejected_without_adapter_hash() -> None:
    with pytest.raises(TrainingRunManifestError, match="adapter_hash"):
        TrainingRunManifest("m", "inv", "res", "adapter-report", None, "plan", "lora", "dataset", "sft", True, None, {"a": "b"})


def test_trained_false_cannot_claim_promoted_adapter() -> None:
    with pytest.raises(TrainingRunManifestError, match="adapter_hash"):
        TrainingRunManifest("m", "inv", "res", None, None, "plan", "lora", "dataset", "sft", False, "adapter", {})
    with pytest.raises(TrainingRunManifestError, match="promoted"):
        TrainingRunManifest("m", "inv", "res", None, None, "plan", "lora", "dataset", "sft", False, None, {}, metadata={"promoted_adapter": True})


def test_forged_manifest_hash_rejected(tmp_path) -> None:
    inv = invocation()
    adapter_report = validate_adapter_output(adapter(tmp_path))
    manifest = build_training_run_manifest(
        invocation_hash=inv.invocation_hash,
        backend_result=result(inv),
        training_plan_hash=inv.training_plan_hash,
        lora_config_hash=inv.lora_config_hash,
        dataset_manifest_hash=inv.dataset_manifest_hash,
        adapter_validation_report=adapter_report,
    )
    with pytest.raises(TrainingRunManifestError, match="manifest_hash"):
        replace(manifest, manifest_hash="forged")


def test_forged_manifest_id_rejected(tmp_path) -> None:
    inv = invocation()
    adapter_report = validate_adapter_output(adapter(tmp_path))
    manifest = build_training_run_manifest(
        invocation_hash=inv.invocation_hash,
        backend_result=result(inv),
        training_plan_hash=inv.training_plan_hash,
        lora_config_hash=inv.lora_config_hash,
        dataset_manifest_hash=inv.dataset_manifest_hash,
        adapter_validation_report=adapter_report,
    )
    with pytest.raises(TrainingRunManifestError, match="manifest_id"):
        replace(manifest, manifest_id="forged")


def test_build_manifest_rejects_invocation_hash_mismatch(tmp_path) -> None:
    inv = invocation()
    adapter_report = validate_adapter_output(adapter(tmp_path))
    with pytest.raises(TrainingRunManifestError, match="invocation_hash"):
        build_training_run_manifest(
            invocation_hash="other",
            backend_result=result(inv),
            training_plan_hash=inv.training_plan_hash,
            lora_config_hash=inv.lora_config_hash,
            dataset_manifest_hash=inv.dataset_manifest_hash,
            adapter_validation_report=adapter_report,
        )


def test_stage_mismatch_rejected(tmp_path) -> None:
    inv = invocation()
    adapter_report = validate_adapter_output(adapter(tmp_path))
    manifest = build_training_run_manifest(
        invocation_hash=inv.invocation_hash,
        backend_result=result(inv),
        training_plan_hash=inv.training_plan_hash,
        lora_config_hash=inv.lora_config_hash,
        dataset_manifest_hash=inv.dataset_manifest_hash,
        adapter_validation_report=adapter_report,
    )
    bad_result = BackendResult(
        "res2",
        inv.invocation_hash,
        "external_dpo",
        "dpo",
        "success",
        True,
        "adapter",
        None,
        (BackendArtifactRef("adapter/adapter_model.safetensors", "adapter_model", SHA, 1, True),),
        (),
        True,
        True,
        None,
        {},
    )
    with pytest.raises(TrainingRunManifestError, match="stage"):
        validate_training_run_manifest(manifest, backend_result=bad_result, adapter_validation_report=adapter_report)


def test_score_guarantee_metadata_rejected() -> None:
    with pytest.raises(TrainingRunManifestError, match="guarantee"):
        TrainingRunManifest("m", "inv", "res", None, None, "plan", "lora", "dataset", "sft", False, None, {}, metadata={"score": "guaranteed"})


def test_direct_manifest_invalid_stage_rejected() -> None:
    payload = {
        "manifest_id": "placeholder",
        "invocation_hash": "inv",
        "backend_result_hash": "res",
        "adapter_validation_hash": None,
        "training_log_hash": None,
        "training_plan_hash": "plan",
        "lora_config_hash": "lora",
        "dataset_manifest_hash": "dataset",
        "stage": "bad",
        "trained": False,
        "adapter_hash": None,
        "artifact_hashes": {},
    }
    payload["manifest_id"] = compute_training_run_manifest_id(payload)
    with pytest.raises(TrainingRunManifestError, match="stage"):
        TrainingRunManifest(**payload)
