from __future__ import annotations

import dataclasses

import pytest

from nemotron_engine.packaging import RuntimeValidationReport
from nemotron_engine.rehearsal import RuntimeRehearsalError, RuntimeRehearsalReport, run_runtime_rehearsal
from test_pass14_end_to_end_rehearsal import make_adapter_bundle, make_bridge, make_config, make_package_report


def test_unloaded_dry_run_allowed_only_when_config_allows() -> None:
    bundle = make_adapter_bundle()
    package = make_package_report(make_config(), bundle, make_bridge())
    ok = run_runtime_rehearsal(rehearsal_config=make_config(), adapter_evidence_bundle=bundle, package_rehearsal_report=package)
    assert ok.passed is True
    blocked = run_runtime_rehearsal(rehearsal_config=make_config(allow_runtime_unloaded=False), adapter_evidence_bundle=bundle, package_rehearsal_report=package)
    assert "runtime_validation_missing" in blocked.errors


def test_runtime_fake_loaded_success_rejected_directly() -> None:
    report = make_runtime_base()
    data = dataclasses.asdict(report)
    data["runtime_loaded"] = True
    data["backend_validated"] = False
    data["report_hash"] = ""
    with pytest.raises(RuntimeRehearsalError):
        RuntimeRehearsalReport(**data)


def test_runtime_loaded_flags_without_runtime_hash_rejected_directly() -> None:
    report = make_runtime_base()
    data = dataclasses.asdict(report)
    data["runtime_loaded"] = True
    data["model_loaded"] = True
    data["adapter_loaded"] = True
    data["backend_validated"] = True
    data["unloaded_dry_run"] = False
    data["runtime_validation_hash"] = None
    data["report_hash"] = ""
    with pytest.raises(RuntimeRehearsalError):
        RuntimeRehearsalReport(**data)


def test_runtime_success_metadata_rejected() -> None:
    report = make_runtime_base()
    data = dataclasses.asdict(report)
    data["metadata"] = {"runtime_success": True}
    data["report_hash"] = ""
    with pytest.raises(RuntimeRehearsalError):
        RuntimeRehearsalReport(**data)


def test_runtime_report_failure_blocks() -> None:
    bundle = make_adapter_bundle()
    package = make_package_report(make_config(), bundle, make_bridge())
    runtime = RuntimeValidationReport(
        passed=False,
        dry_run=True,
        runtime_loaded=False,
        model_loaded=False,
        adapter_loaded=False,
        backend_validated=False,
        backend_name=None,
        manifest_hash=None,
        serving_config_hash="serving",
        adapter_checked=False,
        errors=("failed",),
    )
    report = run_runtime_rehearsal(rehearsal_config=make_config(), adapter_evidence_bundle=bundle, package_rehearsal_report=package, runtime_validation_report=runtime)
    assert "runtime_validation_failed" in report.errors


def test_adapter_hash_mismatch_blocks() -> None:
    bundle = make_adapter_bundle()
    package = make_package_report(make_config(), bundle, make_bridge())
    report = run_runtime_rehearsal(rehearsal_config=make_config(), adapter_evidence_bundle=bundle, package_rehearsal_report=package, metadata={"runtime_adapter_hash": "wrong"})
    assert "runtime_adapter_hash_mismatch" in report.errors


def test_package_rehearsal_failure_blocks() -> None:
    bundle = make_adapter_bundle()
    package = dataclasses.replace(make_package_report(make_config(), bundle, make_bridge()), passed=False, errors=("bad",), report_hash="")
    report = run_runtime_rehearsal(rehearsal_config=make_config(), adapter_evidence_bundle=bundle, package_rehearsal_report=package)
    assert "package_rehearsal_failed" in report.errors


def test_warnings_block_when_not_allowed() -> None:
    bundle = make_adapter_bundle()
    package = make_package_report(make_config(allow_warnings=False), bundle, make_bridge())
    report = run_runtime_rehearsal(rehearsal_config=make_config(allow_warnings=False), adapter_evidence_bundle=bundle, package_rehearsal_report=package)
    assert "warnings_not_allowed" in report.errors


def test_forged_report_hash_rejected() -> None:
    report = make_runtime_base()
    data = dataclasses.asdict(report)
    data["report_hash"] = "forged"
    with pytest.raises(RuntimeRehearsalError):
        RuntimeRehearsalReport(**data)


def make_runtime_base():
    bundle = make_adapter_bundle()
    package = make_package_report(make_config(), bundle, make_bridge())
    return run_runtime_rehearsal(rehearsal_config=make_config(), adapter_evidence_bundle=bundle, package_rehearsal_report=package)
