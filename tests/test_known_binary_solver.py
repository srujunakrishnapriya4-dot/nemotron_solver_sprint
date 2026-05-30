from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.parsing import canonicalize_prompt  # noqa: E402
from nemotron_engine.solvers.known_binary import solve  # noqa: E402


def test_bit_not_solved_and_verified() -> None:
    result = solve(canonicalize_prompt("p", "00 -> 11\n10 -> 01\n01 -> ?"))
    assert result.best_attempt is not None
    assert result.best_attempt.proof is not None
    assert result.best_attempt.proof.is_training_safe is True
    assert result.best_attempt.proof.target_execution.output_value == "10"


def test_identity_solved_and_verified() -> None:
    result = solve(canonicalize_prompt("p", "01 -> 01\n10 -> 10\n11 -> ?"))
    assert result.best_attempt is not None
    assert result.best_attempt.proof.target_execution.output_value == "11"


def test_reverse_solved_and_verified() -> None:
    result = solve(canonicalize_prompt("p", "01 -> 10\n11 -> 11\n10 -> ?"))
    assert result.best_attempt is not None
    assert result.best_attempt.proof.target_execution.output_value == "01"


def test_variable_length_and_non_bitstrings_rejected() -> None:
    variable = solve(canonicalize_prompt("p", "01 -> 10\n101 -> ?"))
    non_bit = solve(canonicalize_prompt("p", "02 -> 20\n10 -> ?"))

    assert variable.best_attempt is None
    assert variable.rejected_attempts[0].reason == "variable_length_bitstrings"
    assert non_bit.best_attempt is None
    assert non_bit.rejected_attempts[0].reason == "non_bitstring_value"


def test_ambiguous_binary_candidates_do_not_produce_best_attempt() -> None:
    result = solve(canonicalize_prompt("p", "00 -> 00\n01 -> ?"))

    assert result.best_attempt is None
    assert result.ambiguity_report.ambiguous is True
