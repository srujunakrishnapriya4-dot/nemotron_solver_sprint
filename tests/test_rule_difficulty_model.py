from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.rule_difficulty_model import reject_reason, score_rule  # noqa: E402


def test_rule_difficulty_scores_private_like_hard_bit() -> None:
    score = score_rule({"family": "bit_manipulation", "rule_id": "choice_majority_mask", "parameters": {"composition_depth": 4}, "verification_trace": {"verified": True}})

    assert score.difficulty_score >= 0.7
    assert score.private_like_score > 0.5
    assert reject_reason(score) is None


def test_underdetermined_unverified_rejected() -> None:
    score = score_rule({"family": "bit_manipulation", "rule_id": "ambiguous", "parameters": {"composition_depth": 1}})

    assert reject_reason(score) in {"ambiguity_too_high", "solver_not_verified"}
