from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.competition_sprint.gravity_unit_solver import solve_gravity_unit_problem  # noqa: E402


def test_gravity_unit_solver_ratio_rounding() -> None:
    result = solve_gravity_unit_problem((("2 meters", "5.0"), ("4 meters", "10.0")), "6 meters")
    assert result.verified
    assert result.prediction == "15.0"
