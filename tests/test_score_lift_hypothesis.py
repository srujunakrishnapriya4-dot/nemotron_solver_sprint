from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.score_lift_hypothesis import build_score_lift_hypotheses  # noqa: E402


def test_score_lift_hypothesis_rejects_095_claim_without_hard_family_gains(tmp_path: Path) -> None:
    payload = build_score_lift_hypotheses(tmp_path / "scores.json")

    hard_truths = " ".join(payload["hard_truths"])
    assert "0.95 requires large hidden-family gains" in hard_truths
    assert "If private-like hard splits do not improve" in hard_truths

