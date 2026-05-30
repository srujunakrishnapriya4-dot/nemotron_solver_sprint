from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.parsing import canonicalize_prompt  # noqa: E402
from nemotron_engine.solvers import solve_numeric  # noqa: E402
from nemotron_engine.solvers.known_binary import solve as solve_binary  # noqa: E402


def test_tiny_full_solver_core_produces_training_safe_proof() -> None:
    problem = canonicalize_prompt("p", "1 -> 2\n2 -> 3\n4 -> ?")
    result = solve_numeric(problem)

    assert result.best_attempt is not None
    assert result.best_attempt.proof is not None
    assert result.best_attempt.proof.is_training_safe is True


def test_malformed_or_ambiguous_problem_has_no_training_safe_best_attempt() -> None:
    ambiguous = solve_binary(canonicalize_prompt("amb", "00 -> 00\n01 -> ?"))

    assert ambiguous.best_attempt is None
