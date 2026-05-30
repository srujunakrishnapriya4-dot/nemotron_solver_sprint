from __future__ import annotations

from dataclasses import replace

import pytest

from nemotron_engine.smoke_training.smoke_handoff import (
    SmokeHandoffError,
    SmokeHandoffReport,
    build_smoke_handoff_to_pass11,
    validate_smoke_handoff_report,
)
from nemotron_engine.smoke_training.smoke_report import SmokeTrainingReport


def smoke_report(*, trained: bool = True, passed: bool = True) -> SmokeTrainingReport:
    return SmokeTrainingReport(
        "cfg",
        "ds",
        "inv",
        backend_result_hash="backend" if trained else None,
        adapter_validation_hash="adapter-validation" if trained else None,
        training_log_hash="log" if trained else None,
        training_run_manifest_hash="manifest" if trained else None,
        trained=trained,
        passed=passed,
        metadata={"require_log_validation": trained, "dry_run": not trained},
    )


def test_successful_trained_report_handoff_passes() -> None:
    handoff = build_smoke_handoff_to_pass11(smoke_report(), adapter_hash="adapter-hash", require_log_validation=True)
    assert validate_smoke_handoff_report(handoff) is handoff
    assert handoff.metadata["handoff_only"] is True
    assert handoff.metadata["require_log_validation"] is True


def test_dry_run_or_non_trained_report_handoff_rejected() -> None:
    with pytest.raises(SmokeHandoffError):
        build_smoke_handoff_to_pass11(smoke_report(trained=False), adapter_hash="adapter-hash")
    with pytest.raises(SmokeHandoffError):
        build_smoke_handoff_to_pass11(smoke_report(trained=True, passed=False), adapter_hash="adapter-hash")


def test_missing_adapter_evidence_and_forged_handoff_rejected() -> None:
    with pytest.raises(SmokeHandoffError):
        build_smoke_handoff_to_pass11(smoke_report(), adapter_hash="")
    handoff = build_smoke_handoff_to_pass11(smoke_report(), adapter_hash="adapter-hash")
    with pytest.raises(SmokeHandoffError, match="report_hash"):
        replace(handoff, report_hash="forged")


def test_handoff_report_requires_log_by_default_unless_explicitly_disabled() -> None:
    with pytest.raises(SmokeHandoffError):
        SmokeHandoffReport("manifest", "adapter", "backend", "adapter-validation", None, {"handoff_only": True})
    with pytest.raises(SmokeHandoffError):
        SmokeHandoffReport(
            "manifest",
            "adapter",
            "backend",
            "adapter-validation",
            None,
            {"handoff_only": True, "require_log_validation": True},
        )
    assert SmokeHandoffReport(
        "manifest",
        "adapter",
        "backend",
        "adapter-validation",
        None,
        {"handoff_only": True, "require_log_validation": False},
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {"promote": False},
        {"promotion": False},
        {"package": False},
        {"submit": False},
        {"submission": "none"},
        {"leaderboard": "none"},
        {"Kaggle": "none"},
        {"note": "95+ impossible"},
        {"score": "guaranteed"},
        {"public_score": "not checked"},
        {"private_score": "not checked"},
        {"winner": False},
        {"status": "competition ready"},
    ],
)
def test_handoff_metadata_with_promotion_package_submit_leaderboard_rejected(metadata) -> None:
    with pytest.raises(SmokeHandoffError):
        build_smoke_handoff_to_pass11(smoke_report(), adapter_hash="adapter-hash", metadata=metadata)


def test_handoff_does_not_claim_promotion_package_or_submission() -> None:
    handoff = build_smoke_handoff_to_pass11(smoke_report(), adapter_hash="adapter-hash")
    text = str(handoff.metadata).lower()
    assert "promote" not in text
    assert "package" not in text
    assert "submit" not in text
