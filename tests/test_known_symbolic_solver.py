from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.parsing import canonicalize_prompt  # noqa: E402
from nemotron_engine.solvers.known_symbolic import solve  # noqa: E402


def test_operator_aliases_solved() -> None:
    add = solve(canonicalize_prompt("add", "1 @ 2 -> 3\n3 @ 4 -> 7\n5 @ 6 -> ?"))
    sub = solve(canonicalize_prompt("sub", "5 @ 2 -> 3\n9 @ 4 -> 5\n8 @ 1 -> ?"))
    abs_sub = solve(canonicalize_prompt("abs", "2 @ 5 -> 3\n9 @ 4 -> 5\n1 @ 8 -> ?"))
    mul = solve(canonicalize_prompt("mul", "2 @ 5 -> 10\n3 @ 4 -> 12\n4 @ 6 -> ?"))

    assert add.best_attempt.proof.target_execution.output_value == "11"
    assert sub.best_attempt.proof.target_execution.output_value == "7"
    assert abs_sub.best_attempt.proof.target_execution.output_value == "7"
    assert mul.best_attempt.proof.target_execution.output_value == "24"


def test_inconsistent_unsupported_and_malformed_rejected() -> None:
    inconsistent = solve(canonicalize_prompt("i", "1 @ 2 -> 3\n3 @ 4 -> 12\n5 @ 6 -> ?"))
    malformed = solve(canonicalize_prompt("m", "1 @ two -> 3\n5 @ 6 -> ?"))
    unsafe = solve(canonicalize_prompt("u", "10001 @ 2 -> 20002\n3 @ 4 -> ?"))

    assert inconsistent.best_attempt is None
    assert inconsistent.rejected_attempts[0].reason == "inconsistent_operator_alias"
    assert malformed.best_attempt is None
    assert malformed.rejected_attempts[0].reason == "malformed_binary_expression"
    assert unsafe.best_attempt is None
    assert unsafe.rejected_attempts[0].reason == "inconsistent_operator_alias"


def test_mixed_visible_operator_tokens_rejected() -> None:
    result = solve(canonicalize_prompt("p", "1 + 2 -> 3\n3 * 4 -> 7\n5 + 6 -> ?"))

    assert result.best_attempt is None
    assert result.rejected_attempts[0].reason == "mixed_visible_operator_tokens"


def test_target_operator_mismatch_rejected() -> None:
    result = solve(canonicalize_prompt("p", "1 @ 2 -> 3\n3 @ 4 -> 7\n5 ? 6 -> ?"))

    assert result.best_attempt is None
    assert result.rejected_attempts[0].reason == "target_operator_mismatch"


def test_unsafe_multiplication_target_rejected() -> None:
    result = solve(canonicalize_prompt("p", "2 @ 5 -> 10\n3 @ 4 -> 12\n10001 @ 2 -> ?"))

    assert result.best_attempt is None
    assert not result.accepted_attempts
