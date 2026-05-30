from __future__ import annotations

from dataclasses import replace

import pytest

from nemotron_engine.smoke_training.smoke_report import (
    SmokeTrainingReport,
    SmokeTrainingReportError,
    build_smoke_training_report,
    validate_smoke_training_report,
)


def trained_report(**overrides) -> SmokeTrainingReport:
    payload = {
        "smoke_config_hash": "cfg",
        "smoke_dataset_hash": "ds",
        "backend_invocation_hash": "inv",
        "backend_result_hash": "backend",
        "adapter_validation_hash": "adapter-validation",
        "training_log_hash": "log",
        "training_run_manifest_hash": "manifest",
        "trained": True,
        "passed": True,
        "errors": (),
        "warnings": (),
        "metadata": {"require_log_validation": True},
    }
    payload.update(overrides)
    return build_smoke_training_report(**payload)


def test_valid_trained_smoke_report_passes() -> None:
    report = trained_report()
    assert validate_smoke_training_report(report) is report


def test_passed_true_with_errors_rejected() -> None:
    with pytest.raises(SmokeTrainingReportError):
        trained_report(errors=("bad",))


def test_dry_run_pass_allowed_only_with_metadata_and_no_artifact_manifest_evidence() -> None:
    report = SmokeTrainingReport(
        "cfg",
        "ds",
        "inv",
        backend_result_hash=None,
        adapter_validation_hash=None,
        training_log_hash=None,
        training_run_manifest_hash=None,
        trained=False,
        passed=True,
        metadata={"dry_run": True},
    )
    assert report.passed
    with pytest.raises(SmokeTrainingReportError):
        replace(report, adapter_validation_hash="adapter")
    with pytest.raises(SmokeTrainingReportError):
        replace(report, training_run_manifest_hash="manifest")
    with pytest.raises(SmokeTrainingReportError):
        replace(report, backend_result_hash="dry-backend")
    with pytest.raises(SmokeTrainingReportError):
        SmokeTrainingReport("cfg", "ds", "inv", None, None, None, None, False, True)


def test_trained_true_requires_adapter_and_manifest_and_log_when_required() -> None:
    with pytest.raises(SmokeTrainingReportError):
        trained_report(adapter_validation_hash=None)
    with pytest.raises(SmokeTrainingReportError):
        trained_report(training_run_manifest_hash=None)
    with pytest.raises(SmokeTrainingReportError):
        trained_report(training_log_hash=None)
    with pytest.raises(SmokeTrainingReportError):
        trained_report(training_log_hash=None, metadata={})
    assert trained_report(training_log_hash=None, metadata={"require_log_validation": False})


def test_forged_report_hash_and_score_metadata_rejected() -> None:
    report = trained_report()
    with pytest.raises(SmokeTrainingReportError, match="report_hash"):
        replace(report, report_hash="forged")
    with pytest.raises(SmokeTrainingReportError):
        trained_report(metadata={"score": "guaranteed"})
