from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.day8_submission_strategy import write_day8_submission_strategy  # noqa: E402


def test_day8_strategy_freezes_best_candidate_and_forbids_blind_training(tmp_path: Path) -> None:
    text = write_day8_submission_strategy(tmp_path / "strategy.md")

    assert "Freeze the best candidate" in text
    assert "No last-minute blind training" in text
    assert "fallback to parent" in text.lower()

