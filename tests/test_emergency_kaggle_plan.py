from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.emergency_score_recovery.kaggle_notebook_plan import emergency_notebook_steps, emergency_strategy_verdict  # noqa: E402


def test_kaggle_notebook_plan_does_not_instruct_blind_training() -> None:
    plan = "\n".join(emergency_notebook_steps()).lower()

    assert "do not train unless" in plan
    assert "blind training" not in plan
    assert "known-good public parent" in plan


def test_strategy_verdict_is_blunt_about_095() -> None:
    verdict = emergency_strategy_verdict()

    assert verdict["can_guarantee_0_95"] is False
    assert "0.85-0.87" in verdict["realistic_near_term_band"]
    assert "known-good public parent" in verdict["submit_if_no_better_candidate"]
