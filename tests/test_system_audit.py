from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.release.artifact_safety import ArtifactSafetyFinding, ArtifactSafetyReport
from nemotron_engine.release.import_safety import ImportSafetyFinding, ImportSafetyReport
from nemotron_engine.release.locked_pass_registry import build_locked_pass_registry
from nemotron_engine.release.system_audit import (
    SystemAuditConfig,
    SystemAuditError,
    SystemAuditReport,
    run_system_audit,
)


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = "494 passed, 1 skipped"
PASS10_LOCK_SUMMARY = "559 passed, 1 skipped"
CURRENT_SUMMARY = "595 passed, 1 skipped"


def registry():
    return build_locked_pass_registry(repository_root=ROOT, suite_result_summary=SUMMARY)


def import_report(passed: bool = True):
    if passed:
        return ImportSafetyReport(passed=True, scanned_files=("src/nemotron_engine/app.py",))
    finding = ImportSafetyFinding(path="bad.py", line_number=1, pattern="import torch", severity="error", message="bad")
    return ImportSafetyReport(passed=False, scanned_files=("bad.py",), findings=(finding,), errors=("bad.py:1:import torch",))


def artifact_report(passed: bool = True):
    if passed:
        return ArtifactSafetyReport(passed=True, scanned_paths=("src/nemotron_engine/app.py",))
    finding = ArtifactSafetyFinding(path="model.safetensors", artifact_type="checkpoint", reason="bad")
    return ArtifactSafetyReport(passed=False, scanned_paths=("model.safetensors",), forbidden_artifacts=(finding,))


def audit(**overrides):
    data = {
        "locked_pass_registry": registry(),
        "import_safety_report": import_report(),
        "artifact_safety_report": artifact_report(),
        "scoped_suite_result_summary": SUMMARY,
    }
    data.update(overrides)
    return run_system_audit(**data)


def test_all_green_reports_pass() -> None:
    assert audit().passed


def test_failed_import_safety_blocks() -> None:
    report = audit(import_safety_report=import_report(False))
    assert not report.passed
    assert "import_safety_failed" in report.failure_reasons


def test_failed_artifact_safety_blocks() -> None:
    report = audit(artifact_safety_report=artifact_report(False))
    assert not report.passed
    assert "artifact_safety_failed" in report.failure_reasons


def test_missing_scoped_suite_summary_blocks() -> None:
    report = audit(scoped_suite_result_summary=None)
    assert not report.passed
    assert "scoped_suite_summary_missing" in report.failure_reasons


def test_default_accepted_summaries_include_current_pass10_result() -> None:
    report = audit(scoped_suite_result_summary=CURRENT_SUMMARY)
    assert report.passed


def test_default_accepted_summaries_include_requested_pass10_lock_result() -> None:
    report = audit(scoped_suite_result_summary=PASS10_LOCK_SUMMARY)
    assert report.passed


def test_explicit_accepted_suite_override_still_works() -> None:
    report = audit(scoped_suite_result_summary="custom passed", config=SystemAuditConfig(accepted_suite_summaries=("custom passed",)))
    assert report.passed


def test_warnings_block_when_disallowed() -> None:
    report = audit(metadata={"warnings": ("needs review",)})
    assert not report.passed
    assert "warnings_not_allowed" in report.failure_reasons


def test_forged_system_audit_report_hash_rejected() -> None:
    report = audit()
    with pytest.raises(SystemAuditError):
        replace(report, report_hash="forged")


def test_passed_true_with_failure_reasons_rejected() -> None:
    with pytest.raises(SystemAuditError):
        SystemAuditReport(
            locked_registry_hash="locked",
            import_safety_hash="import",
            artifact_safety_hash="artifact",
            promotion_decision_hash=None,
            preflight_gate_hash=None,
            scoped_suite_result_summary=SUMMARY,
            passed=True,
            failure_reasons=("bad",),
        )


def test_report_hash_deterministic() -> None:
    assert audit().report_hash == audit().report_hash


def test_system_audit_does_not_invent_test_run_metadata() -> None:
    report = audit()
    assert "tests_run" not in report.metadata
    supplied = audit(metadata={"tests_run": True})
    assert supplied.metadata["tests_run"] is True
