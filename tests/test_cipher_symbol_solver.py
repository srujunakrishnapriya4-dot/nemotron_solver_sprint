from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.competition_sprint.cipher_symbol_solver import solve_cipher_symbol_problem  # noqa: E402


def test_cipher_symbol_solver_rejects_unseen_symbol() -> None:
    solved = solve_cipher_symbol_problem((("ab", "xy"),), "ba")
    assert solved.verified and solved.prediction == "yx"
    rejected = solve_cipher_symbol_problem((("ab", "xy"),), "ac")
    assert not rejected.verified
    assert rejected.reason == "unseen_target_symbol"
