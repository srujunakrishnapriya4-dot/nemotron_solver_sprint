from __future__ import annotations

import dataclasses

import pytest

from nemotron_engine.rehearsal import AdapterEvidenceBundle, AdapterEvidenceError, build_adapter_evidence_bundle
from nemotron_engine.smoke_training import SmokeHandoffReport, SmokeTrainingReport
from nemotron_engine.training_backend import AdapterValidationReport
from test_pass14_end_to_end_rehearsal import (
    SHA_A,
    SHA_B,
    SHA_C,
    SHA_E,
    make_adapter_bundle,
    make_adapter_report,
    make_config,
    make_training_manifest,
)


def test_valid_adapter_evidence_bundle_passes() -> None:
    bundle = make_adapter_bundle()
    assert bundle.evidence_complete is True
    assert bundle.errors == ()


def test_forged_evidence_hash_rejected() -> None:
    data = dataclasses.asdict(make_adapter_bundle())
    data["evidence_hash"] = "forged"
    with pytest.raises(AdapterEvidenceError):
        AdapterEvidenceBundle(**data)


def test_evidence_complete_with_fake_adapter_hash_rejected() -> None:
    data = dataclasses.asdict(make_adapter_bundle())
    data["adapter_hash"] = SHA_E
    data["evidence_hash"] = ""
    with pytest.raises(AdapterEvidenceError):
        AdapterEvidenceBundle(**data)


def test_failed_adapter_validation_rejected() -> None:
    report = AdapterValidationReport(
        adapter_dir="x",
        passed=False,
        adapter_config_valid=False,
        model_file_present=False,
        errors=("failed",),
    )
    with pytest.raises(AdapterEvidenceError):
        build_adapter_evidence_bundle(adapter_validation_report=report)


def test_rank_over_32_rejected_directly() -> None:
    data = dataclasses.asdict(make_adapter_bundle())
    data["rank"] = 33
    data["evidence_hash"] = ""
    with pytest.raises(AdapterEvidenceError):
        AdapterEvidenceBundle(**data)


def test_empty_artifact_hashes_rejected() -> None:
    data = dataclasses.asdict(make_adapter_bundle())
    data["artifact_hashes"] = {}
    data["evidence_hash"] = ""
    with pytest.raises(AdapterEvidenceError):
        AdapterEvidenceBundle(**data)


def test_evidence_complete_empty_artifact_hashes_rejected() -> None:
    data = dataclasses.asdict(make_adapter_bundle())
    data["artifact_hashes"] = {}
    data["evidence_complete"] = True
    data["evidence_hash"] = ""
    with pytest.raises(AdapterEvidenceError):
        AdapterEvidenceBundle(**data)


def test_smoke_required_but_missing_rejected() -> None:
    report = make_adapter_report()
    bundle = build_adapter_evidence_bundle(adapter_validation_report=report, rehearsal_config=make_config(require_smoke_evidence=True))
    assert "smoke_evidence_missing" in bundle.errors
    assert bundle.evidence_complete is False


def test_smoke_training_not_passed_trained_rejected() -> None:
    report = make_adapter_report()
    smoke = SmokeTrainingReport(
        smoke_config_hash=SHA_A,
        smoke_dataset_hash=SHA_B,
        backend_invocation_hash=SHA_C,
        backend_result_hash=None,
        adapter_validation_hash=None,
        training_log_hash=None,
        training_run_manifest_hash=None,
        trained=False,
        passed=True,
        metadata={"dry_run": True, "require_log_validation": False},
    )
    bundle = build_adapter_evidence_bundle(adapter_validation_report=report, smoke_training_report=smoke)
    assert "smoke_training_not_passed_trained" in bundle.errors


def test_smoke_handoff_mismatch_rejected() -> None:
    report = make_adapter_report()
    manifest = make_training_manifest(report)
    smoke = SmokeTrainingReport(
        smoke_config_hash=SHA_A,
        smoke_dataset_hash=SHA_B,
        backend_invocation_hash=SHA_C,
        backend_result_hash=SHA_A,
        adapter_validation_hash=report.report_hash,
        training_log_hash=SHA_C,
        training_run_manifest_hash=manifest.manifest_hash,
        trained=True,
        passed=True,
        metadata={"require_log_validation": False},
    )
    handoff = SmokeHandoffReport(
        training_run_manifest_hash=manifest.manifest_hash,
        adapter_hash="wrong",
        backend_result_hash=SHA_A,
        adapter_validation_hash=report.report_hash,
        training_log_hash=SHA_C,
        metadata={"handoff_only": True, "require_log_validation": False},
    )
    bundle = build_adapter_evidence_bundle(
        adapter_validation_report=report,
        training_run_manifest=manifest,
        smoke_training_report=smoke,
        smoke_handoff_report=handoff,
    )
    assert "smoke_handoff_adapter_hash_mismatch" in bundle.errors


def test_training_manifest_adapter_hash_mismatch_rejected() -> None:
    report = make_adapter_report()
    manifest = make_training_manifest(report, adapter_hash="wrong")
    bundle = build_adapter_evidence_bundle(adapter_validation_report=report, training_run_manifest=manifest)
    assert any("training_run_manifest_invalid" in item or item == "training_manifest_adapter_hash_mismatch" for item in bundle.errors)


def test_evidence_complete_with_errors_rejected() -> None:
    data = dataclasses.asdict(make_adapter_bundle())
    data["errors"] = ("bad",)
    data["evidence_hash"] = ""
    with pytest.raises(AdapterEvidenceError):
        AdapterEvidenceBundle(**data)


def test_fake_metadata_claims_rejected() -> None:
    with pytest.raises(AdapterEvidenceError):
        build_adapter_evidence_bundle(adapter_validation_report=make_adapter_report(), metadata={"kaggle_success": True})
