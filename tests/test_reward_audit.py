from __future__ import annotations

from dataclasses import replace

import pytest

from nemotron_engine.training.reward_audit import (
    RewardAuditReport,
    RewardAuditError,
    audit_reward_model_signals,
    grpo_admission_decision,
)


def audit(**updates):
    data = {
        "sample_count": 100,
        "false_positive_rate": 0.0,
        "format_failure_rate": 0.0,
        "leakage_reward_rate": 0.0,
        "length_gaming_rate": 0.0,
        "heldout_regression_rate": 0.0,
    }
    data.update(updates)
    return audit_reward_model_signals(**data)


def test_clean_audit_accepted_and_hash_deterministic() -> None:
    first = audit()
    second = audit()
    assert first.accepted
    assert grpo_admission_decision(first)
    assert first.report_hash == second.report_hash


def test_threshold_failures_rejected() -> None:
    assert not audit(sample_count=99).accepted
    assert not audit(false_positive_rate=0.011).accepted
    assert not audit(format_failure_rate=0.006).accepted
    assert not audit(leakage_reward_rate=0.001).accepted
    assert not audit(length_gaming_rate=0.02).accepted
    assert not audit(heldout_regression_rate=0.02).accepted


def test_reward_report_hash_tampering_rejected() -> None:
    with pytest.raises(RewardAuditError, match="report_hash"):
        replace(audit(), report_hash="forged")


def test_forged_accepted_reports_with_failing_thresholds_rejected() -> None:
    kwargs = {
        "sample_count": 100,
        "false_positive_rate": 0.0,
        "false_negative_rate": 0.0,
        "format_failure_rate": 0.0,
        "leakage_reward_rate": 0.0,
        "length_gaming_rate": 0.0,
        "heldout_regression_rate": 0.0,
        "accepted": True,
    }
    for update in (
        {"sample_count": 99},
        {"false_positive_rate": 0.011},
        {"format_failure_rate": 0.006},
        {"leakage_reward_rate": 0.001},
    ):
        with pytest.raises(RewardAuditError, match="hard thresholds"):
            RewardAuditReport(**{**kwargs, **update})
