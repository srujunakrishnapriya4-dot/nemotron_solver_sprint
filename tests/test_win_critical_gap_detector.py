from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.win_system_audit.critical_gap_detector import readiness_from_gaps  # noqa: E402


def test_critical_gap_detector_marks_fake_eval_not_ready() -> None:
    assert readiness_from_gaps(["eval_metrics_hardcoded"]) == "NOT_READY"
    assert readiness_from_gaps([]) == "MAIN_READY"
