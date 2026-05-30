from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.parsing import canonicalize_prompt  # noqa: E402
from nemotron_engine.solvers.known_numeric import enumerate_candidate_programs, solve  # noqa: E402


def test_add_sub_mul_affine_and_digit_sum_solved() -> None:
    add = solve(canonicalize_prompt("add", "1 -> 3\n2 -> 4\n5 -> ?"))
    sub = solve(canonicalize_prompt("sub", "3 -> 1\n5 -> 3\n9 -> ?"))
    mul = solve(canonicalize_prompt("mul", "2 -> 6\n4 -> 12\n5 -> ?"))
    affine = solve(canonicalize_prompt("aff", "1 -> 5\n2 -> 7\n4 -> ?"))
    digit_sum = solve(canonicalize_prompt("sum", "123 -> 6\n111 -> 3\n222 -> ?"))

    assert add.best_attempt.proof.target_execution.output_value == "7"
    assert sub.best_attempt.proof.target_execution.output_value == "7"
    assert mul.best_attempt.proof.target_execution.output_value == "15"
    assert affine.best_attempt.proof.target_execution.output_value == "11"
    assert digit_sum.best_attempt.proof.target_execution.output_value == "6"


def test_inconsistent_examples_and_decimals_rejected() -> None:
    decimals = solve(canonicalize_prompt("dec", "1.0 -> 2\n3.0 -> ?"))
    unsupported = solve(canonicalize_prompt("bad", "A -> B\nC -> ?"))

    assert decimals.best_attempt is None
    assert decimals.rejected_attempts[0].reason == "decimals_not_supported"
    assert unsupported.best_attempt is None
    assert unsupported.rejected_attempts[0].reason == "no_supported_numeric_rule"


def test_ambiguous_numeric_candidates_do_not_produce_best_attempt() -> None:
    result = solve(canonicalize_prompt("p", "1 -> 2\n3 -> ?"))

    assert result.best_attempt is None
    assert not result.accepted_attempts
    assert result.rejected_attempts[0].reason == "insufficient_examples"


def test_one_example_numeric_rule_has_no_accepted_attempts() -> None:
    result = solve(canonicalize_prompt("p", "1 -> 2\n3 -> ?"))

    assert result.best_attempt is None
    assert result.accepted_attempts == ()


def test_affine_requires_two_distinct_x_values() -> None:
    problem = canonicalize_prompt("p", "1 -> 2\n1 -> 2\n3 -> ?")
    programs = enumerate_candidate_programs(problem)

    assert all("affine" not in program.program_id for program in programs)


def test_conflicting_numeric_candidates_do_not_appear_as_accepted_attempts() -> None:
    result = solve(canonicalize_prompt("p", "12 -> 3\n13 -> 4\n21 -> ?"))

    assert result.best_attempt is None
    assert result.accepted_attempts == ()
    assert any(attempt.reason == "conflicting_numeric_target_outputs" for attempt in result.rejected_attempts)


def test_digit_sum_affine_conflict_blocks_best_attempt() -> None:
    result = solve(canonicalize_prompt("p", "12 -> 3\n13 -> 4\n21 -> ?"))

    assert result.best_attempt is None
    assert any(
        set(attempt.metadata.get("conflicting_target_outputs", ())) == {"3", "12"}
        for attempt in result.rejected_attempts
    )
