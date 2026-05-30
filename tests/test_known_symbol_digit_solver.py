from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.parsing import canonicalize_prompt  # noqa: E402
from nemotron_engine.solvers.known_symbol_digit import solve  # noqa: E402


def test_exact_symbol_to_digit_and_digit_to_symbol_mapping_solved() -> None:
    s2d = solve(canonicalize_prompt("s2d", "AB -> 12\nBA -> 21\nAA -> ?"))
    d2s = solve(canonicalize_prompt("d2s", "12 -> AB\n21 -> BA\n11 -> ?"))

    assert s2d.best_attempt.proof.target_execution.output_value == "11"
    assert d2s.best_attempt.proof.target_execution.output_value == "AA"


def test_mapping_collision_unseen_target_and_ambiguous_mapping_rejected() -> None:
    collision = solve(canonicalize_prompt("c", "A -> 1\nA -> 2\nB -> ?"))
    unseen = solve(canonicalize_prompt("u", "A -> 1\nB -> ?"))
    non_bijective = solve(canonicalize_prompt("n", "A -> 1\nB -> 1\nA -> ?"))

    assert collision.best_attempt is None
    assert collision.rejected_attempts[0].reason == "mapping_collision"
    assert unseen.best_attempt is None
    assert unseen.rejected_attempts[0].reason == "unseen_target_symbol"
    assert non_bijective.best_attempt is None
    assert non_bijective.rejected_attempts[0].reason == "non_bijective_mapping"
