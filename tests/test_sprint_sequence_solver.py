from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.sprint_solvers import parse_problem, solve_sequence_problem  # noqa: E402


def test_arithmetic_progression_next_solved() -> None:
    result = solve_sequence_problem(parse_problem("1 2 3 -> 4\n2 4 6 -> 8\nTarget: 3 6 9 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "12"


def test_geometric_progression_solved() -> None:
    result = solve_sequence_problem(parse_problem("2 4 8 -> 16\n3 9 27 -> 81\nTarget: 4 16 64 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "256"


def test_non_integral_geometric_rejected() -> None:
    result = solve_sequence_problem(parse_problem("2 3 5 -> 7\nTarget: 2 3 5 -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "non_integral_ratio" for candidate in result.rejected_candidates)


def test_fibonacci_like_solved() -> None:
    result = solve_sequence_problem(parse_problem("1 1 2 3 -> 5\n2 3 5 -> 8\nTarget: 3 5 8 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "13"


def test_finite_difference_quadratic_solved() -> None:
    result = solve_sequence_problem(parse_problem("1 4 9 16 -> 25\n4 9 16 25 -> 36\nTarget: 9 16 25 36 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "49"


def test_degree_2_with_exactly_three_terms_abstains_or_rejects() -> None:
    result = solve_sequence_problem(parse_problem("1 4 9 -> 16\nTarget: 16 25 36 -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "insufficient_terms_degree2" for candidate in result.rejected_candidates)


def test_repeat_cycle_solved() -> None:
    result = solve_sequence_problem(parse_problem("1 2 1 2 -> 1\n3 4 3 4 -> 3\nTarget: 5 6 5 6 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "5"


def test_sequence_sum_solved() -> None:
    result = solve_sequence_problem(parse_problem("1 2 3 -> 6\n4 5 6 -> 15\nTarget: 7 8 9 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "24"


def test_insufficient_terms_abstain() -> None:
    result = solve_sequence_problem(parse_problem("1 2 -> 3\n2 3 -> 5\nTarget: 3 4 -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "insufficient_terms" for candidate in result.rejected_candidates)


def test_insufficient_examples_abstain() -> None:
    result = solve_sequence_problem(parse_problem("1 -> 2\n2 -> 4\nTarget: 3 -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "insufficient_examples" for candidate in result.rejected_candidates)


def test_index_degree_2_with_exactly_three_examples_abstains_or_rejects() -> None:
    result = solve_sequence_problem(parse_problem("1 -> 1\n2 -> 4\n3 -> 9\nTarget: 4 -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "insufficient_examples_degree2" for candidate in result.rejected_candidates)


def test_index_fibonacci_unknown_backward_target_abstains() -> None:
    result = solve_sequence_problem(parse_problem("3 -> 2\n4 -> 3\n5 -> 5\nTarget: 2 -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "unsafe_backward_index_extrapolation" for candidate in result.rejected_candidates)


def test_index_arithmetic_unknown_backward_target_abstains() -> None:
    result = solve_sequence_problem(parse_problem("2 -> 4\n3 -> 6\n4 -> 8\nTarget: 1 -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "unsafe_backward_index_extrapolation" for candidate in result.rejected_candidates)


def test_index_degree_2_unknown_backward_target_abstains() -> None:
    result = solve_sequence_problem(parse_problem("1 -> 1\n2 -> 4\n3 -> 9\n4 -> 16\nTarget: 0 -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "unsafe_backward_index_extrapolation" for candidate in result.rejected_candidates)


def test_index_exact_known_target_allowed() -> None:
    result = solve_sequence_problem(parse_problem("1 -> 2\n2 -> 4\n3 -> 6\nTarget: 2 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "4"


def test_index_forward_extrapolation_still_allowed() -> None:
    result = solve_sequence_problem(parse_problem("1 -> 2\n2 -> 5\n3 -> 8\nTarget: 4 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "11"


def test_non_integer_sequence_elements_produce_rejected_diagnostic() -> None:
    result = solve_sequence_problem(parse_problem("1 2 x -> 4\nTarget: 1 2 3 -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "non_integer_sequence" for candidate in result.rejected_candidates)


def test_sequence_target_output_outside_allowed_range_rejected() -> None:
    result = solve_sequence_problem(parse_problem("50000 75000 100000 -> 125000\nTarget: 75000 100000 125000 -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "invalid_output_format" for candidate in result.rejected_candidates)


def test_disagreement_returns_disagreement() -> None:
    result = solve_sequence_problem(parse_problem("1 2 3 -> 6\nTarget: 1 2 4 -> ?"))

    assert result.status == "disagreement"
    assert {candidate.target_prediction for candidate in result.verified_candidates} >= {"7", "8"}
