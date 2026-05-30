from __future__ import annotations

from dataclasses import replace
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.evaluation.public_firewall import (
    PublicFirewallConfig,
    PublicFirewallError,
    PublicFirewallReport,
    evaluate_public_firewall,
)


def test_public_only_improvement_returns_investigate() -> None:
    report = evaluate_public_firewall(PublicFirewallConfig(0.02, 0.0, 0.0, 0.0))

    assert report.decision == "investigate"
    assert "public_only_improvement" in report.reasons


def test_public_gain_private_regression_rejected() -> None:
    report = evaluate_public_firewall(PublicFirewallConfig(0.02, -0.01, 0.0, 0.0))

    assert report.decision == "reject"


def test_public_drop_private_gain_investigated() -> None:
    report = evaluate_public_firewall(PublicFirewallConfig(-0.02, 0.01, 0.02, 0.0))

    assert report.decision == "investigate"


def test_repeated_public_driven_changes_rejected() -> None:
    report = evaluate_public_firewall(PublicFirewallConfig(0.0, 0.01, 0.01, 0.0, public_driven_change_count=3, max_public_driven_changes=2))

    assert report.decision == "reject"


def test_report_hash_deterministic_and_forgery_rejected() -> None:
    report = evaluate_public_firewall(PublicFirewallConfig(0.0, 0.01, 0.01, 0.0))
    again = evaluate_public_firewall(PublicFirewallConfig(0.0, 0.01, 0.01, 0.0))

    assert report.report_hash == again.report_hash
    with pytest.raises(PublicFirewallError):
        replace(report, report_hash="forged")


def test_public_firewall_config_rejects_nan_inf() -> None:
    with pytest.raises(PublicFirewallError):
        PublicFirewallConfig(math.nan, 0.0, 0.0, 0.0)
    with pytest.raises(PublicFirewallError):
        PublicFirewallConfig(0.0, math.inf, 0.0, 0.0)


def test_public_firewall_report_rejects_allow_semantic_forgery() -> None:
    with pytest.raises(PublicFirewallError):
        PublicFirewallReport("allow", 0.1, 0.0, 0.0, 0.0, 0)
    with pytest.raises(PublicFirewallError):
        PublicFirewallReport("allow", 0.1, -0.1, 0.0, 0.0, 0)
    with pytest.raises(PublicFirewallError):
        PublicFirewallReport("allow", 0.1, 0.1, 0.1, -0.1, 0)
    with pytest.raises(PublicFirewallError):
        PublicFirewallReport("allow", 0.0, 0.1, 0.1, 0.0, 3, max_public_driven_changes=2)
