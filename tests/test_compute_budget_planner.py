from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.compute_budget_planner import build_compute_budget_plan  # noqa: E402


def test_compute_planner_does_not_schedule_broad_training_on_day2(tmp_path: Path) -> None:
    plan = build_compute_budget_plan(tmp_path / "plan.json", tmp_path / "plan.md")
    day2 = next(day for day in plan["schedule"] if day["day"] == 2)

    assert "micro training only" in day2["tasks"]
    assert "v1/v2/v3 broad training" in day2["forbidden"]
    assert "if micro fails, stop custom training" in plan["stop_loss"]

