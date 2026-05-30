from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.release.release_candidate import (
    ReleaseCandidateConfig,
    ReleaseCandidateError,
    build_release_candidate_report,
)
from nemotron_engine.release.system_audit import SystemAuditReport


SUMMARY = "494 passed, 1 skipped"
PASS10_LOCK_SUMMARY = "559 passed, 1 skipped"
CURRENT_SUMMARY = "595 passed, 1 skipped"


def system_audit(passed: bool = True) -> SystemAuditReport:
    return SystemAuditReport(
        locked_registry_hash="locked",
        import_safety_hash="import",
        artifact_safety_hash="artifact",
        promotion_decision_hash=None,
        preflight_gate_hash=None,
        scoped_suite_result_summary=SUMMARY,
        passed=passed,
        failure_reasons=() if passed else ("bad",),
        component_status={"locked": passed, "import": passed, "artifact": passed, "scoped_suite": passed},
    )


def candidate(**overrides):
    data = {"system_audit_report": system_audit(), "repository_root": "repo", "scoped_suite_result_summary": SUMMARY}
    data.update(overrides)
    return build_release_candidate_report(ReleaseCandidateConfig(**data))


def test_all_green_system_audit_creates_ready_candidate() -> None:
    report = candidate()
    assert report.ready
    assert report.release_level == "internal_candidate"


def test_failed_system_audit_blocks() -> None:
    report = candidate(system_audit_report=system_audit(False))
    assert not report.ready
    assert "system_audit_failed" in report.blockers


def test_blockers_block_ready() -> None:
    report = candidate(blockers=("manual_review",))
    assert not report.ready
    assert "manual_review" in report.blockers


def test_unsupported_release_level_rejected() -> None:
    with pytest.raises(ReleaseCandidateError):
        ReleaseCandidateConfig(system_audit_report=system_audit(), release_level="leaderboard_ready")


@pytest.mark.parametrize(
    "metadata",
    [
        {"kaggle_success": True},
        {"leaderboard_readiness": True},
        {"note": "95+ guaranteed private score"},
        {"note": "private score guaranteed"},
        {"note": "public score guaranteed"},
        {"trained_adapter_available": True},
        {"note": "trained adapter available"},
        {"note": "adapter trained successfully"},
        {"note": "adapter ready"},
        {"note": "adapter success"},
    ],
)
def test_unsafe_metadata_claims_rejected(metadata: dict[str, object]) -> None:
    with pytest.raises(ReleaseCandidateError):
        ReleaseCandidateConfig(system_audit_report=system_audit(), metadata=metadata)


def test_default_accepted_summaries_include_current_pass10_result() -> None:
    report = candidate(scoped_suite_result_summary=CURRENT_SUMMARY)
    assert report.ready


def test_default_accepted_summaries_include_requested_pass10_lock_result() -> None:
    report = candidate(scoped_suite_result_summary=PASS10_LOCK_SUMMARY)
    assert report.ready


def test_explicit_accepted_suite_override_still_works() -> None:
    report = candidate(scoped_suite_result_summary="custom passed", accepted_suite_summaries=("custom passed",))
    assert report.ready


def test_forged_release_candidate_report_hash_rejected() -> None:
    report = candidate()
    with pytest.raises(ReleaseCandidateError):
        replace(report, report_hash="forged")


def test_candidate_id_deterministic() -> None:
    assert candidate().candidate_id == candidate().candidate_id
