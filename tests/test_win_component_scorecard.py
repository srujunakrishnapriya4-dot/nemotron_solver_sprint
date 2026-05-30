from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.win_system_audit.component_scorecard import score_component  # noqa: E402


def test_component_scorecard_penalizes_fake_metrics() -> None:
    score = score_component("eval", evidence={"exists": True, "contains_token": True, "checks": {"exists": 50, "contains_token": 50}, "fake_metric_detected": True})
    assert score.status == "not_ready"
    assert "fake_metric_detected" in score.reasons
