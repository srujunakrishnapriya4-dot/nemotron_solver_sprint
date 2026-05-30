from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.sprint_solvers import parse_problem, solve_arithmetic_problem  # noqa: E402


def test_add_const_solved_with_two_examples() -> None:
    result = solve_arithmetic_problem(parse_problem("1 -> 3\n2 -> 4\nTarget: 3 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "5"


def test_affine_solved_with_two_distinct_examples() -> None:
    result = solve_arithmetic_problem(parse_problem("1 -> 5\n2 -> 8\nTarget: 3 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "11"
    assert any(str(candidate.metadata["rule_id"]).startswith("affine_") for candidate in result.verified_candidates)


def test_one_example_affine_add_const_underdetermined_abstains() -> None:
    result = solve_arithmetic_problem(parse_problem("1 -> 2\nTarget: 3 -> ?"))

    assert result.status == "abstain"


def test_digit_sum_solved() -> None:
    result = solve_arithmetic_problem(parse_problem("124 -> 7\nTarget: 987 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "24"


def test_list_sum_solved() -> None:
    result = solve_arithmetic_problem(parse_problem("1 2 3 -> 6\n4 5 -> 9\nTarget: 10 20 30 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "60"


def test_binary_expression_add_sub_mul_solved() -> None:
    add = solve_arithmetic_problem(parse_problem("2 + 3 -> 5\n4 + 1 -> 5\nTarget: 6 + 7 -> ?"))
    sub = solve_arithmetic_problem(parse_problem("5 - 2 -> 3\n8 - 1 -> 7\nTarget: 9 - 4 -> ?"))
    mul = solve_arithmetic_problem(parse_problem("2 * 3 -> 6\n4 * 5 -> 20\nTarget: 6 * 7 -> ?"))

    assert add.status == "solved"
    assert add.prediction == "13"
    assert sub.status == "solved"
    assert sub.prediction == "5"
    assert mul.status == "solved"
    assert mul.prediction == "42"


def test_exact_division_solved() -> None:
    result = solve_arithmetic_problem(parse_problem("6 / 3 -> 2\n10 / 5 -> 2\nTarget: 8 / 2 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "4"


def test_non_exact_division_rejected() -> None:
    result = solve_arithmetic_problem(parse_problem("5 / 2 -> 2\nTarget: 6 / 4 -> ?"))

    assert result.status == "abstain"


def test_slash_expression_does_not_solve_as_plus() -> None:
    result = solve_arithmetic_problem(parse_problem("5 / 2 -> 7\nTarget: 4 / 2 -> ?"))

    assert result.status == "abstain"


def test_divide_by_zero_rejected() -> None:
    result = solve_arithmetic_problem(parse_problem("5 / 0 -> 0\nTarget: 4 / 2 -> ?"))

    assert result.status == "abstain"


def test_modulo_by_zero_rejected() -> None:
    result = solve_arithmetic_problem(parse_problem("5 % 0 -> 0\nTarget: 4 % 2 -> ?"))

    assert result.status == "abstain"


def test_known_plus_does_not_consider_multiplication_if_plus_examples_fail() -> None:
    result = solve_arithmetic_problem(parse_problem("2 + 3 -> 6\nTarget: 4 + 5 -> ?"))

    assert result.status == "abstain"


def test_symbolic_operator_can_infer_addition_if_examples_verify() -> None:
    result = solve_arithmetic_problem(parse_problem("2 @ 3 -> 5\n4 @ 1 -> 5\nTarget: 6 @ 7 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "13"


def test_conflicting_verified_candidates_produce_disagreement() -> None:
    result = solve_arithmetic_problem(parse_problem("0 -> 0\n1 -> 1\nTarget: 2 -> ?"))

    assert result.status == "disagreement"
    assert {candidate.target_prediction for candidate in result.verified_candidates} >= {"2", "4"}


def test_output_outside_allowed_range_rejected() -> None:
    result = solve_arithmetic_problem(parse_problem("1000 -> 1000000\nTarget: 1000 -> ?"))

    assert result.status == "abstain"
    assert result.rejected_candidates
