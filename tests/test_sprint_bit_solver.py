from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.sprint_solvers import enumerate_bit_candidates, parse_problem, solve_bit_problem  # noqa: E402


def test_bit_not_solved() -> None:
    result = solve_bit_problem(parse_problem("0101 -> 1010\n1111 -> 0000\nTarget: 0011 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "12"
    assert any(candidate.metadata["rule_id"] == "bit_not" for candidate in result.verified_candidates)


def test_reverse_bits_solved() -> None:
    result = solve_bit_problem(parse_problem("001 -> 100\n101 -> 101\nTarget: 011 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "6"


def test_rotate_solved() -> None:
    result = solve_bit_problem(parse_problem("1000 -> 0001\n0100 -> 1000\nTarget: 0010 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "4"
    assert any(candidate.metadata["rule_id"] == "rotate_left_1" for candidate in result.verified_candidates)


def test_xor_const_solved_with_at_least_two_examples() -> None:
    result = solve_bit_problem(parse_problem("0000 -> 1010\n1111 -> 0101\nTarget: 0011 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "9"
    assert any(str(candidate.metadata["rule_id"]).startswith("xor_const_") for candidate in result.verified_candidates)


def test_mixed_widths_abstain_or_error() -> None:
    result = solve_bit_problem(parse_problem("010 -> 101\n1111 -> 0000\nTarget: 001 -> ?"))

    assert result.status in {"abstain", "error"}
    assert "mixed_width_bitstrings" in result.errors


def test_non_bit_input_abstain_or_error() -> None:
    result = solve_bit_problem(parse_problem("012 -> 101\n111 -> 000\nTarget: 001 -> ?"))

    assert result.status in {"abstain", "error"}
    assert "non_bit_character" in result.errors


def test_ambiguous_candidates_disagree_and_solver_returns_disagreement() -> None:
    result = solve_bit_problem(parse_problem("00 -> 00\n11 -> 11\nTarget: 01 -> ?"))

    assert result.status == "disagreement"
    assert {candidate.target_prediction for candidate in result.verified_candidates} >= {"1", "2"}


def test_binary_output_converted_to_integer_answer() -> None:
    result = solve_bit_problem(parse_problem("0 -> 1\nTarget: 1 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "0"
    assert result.verified_candidates[0].metadata["raw_candidate_output"] == "0"


def test_no_candidate_may_solve_without_verifying_all_examples() -> None:
    result = solve_bit_problem(parse_problem("00 -> 00\n01 -> 11\nTarget: 10 -> ?"))

    assert all(candidate.example_predictions == ("00", "11") for candidate in result.verified_candidates)


def test_one_example_xor_and_or_candidates_absent_as_underdetermined() -> None:
    candidates = enumerate_bit_candidates(parse_problem("0000 -> 1010\nTarget: 0011 -> ?"))

    rule_ids = {str(candidate.metadata["rule_id"]) for candidate in candidates}
    assert not any(rule_id.startswith(("xor_const_", "and_const_", "or_const_")) for rule_id in rule_ids)


def test_binary_target_decimal_conversion_out_of_range_rejected() -> None:
    result = solve_bit_problem(parse_problem("00000000000000000 -> 11111111111111111\nTarget: 00000000000000000 -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "target_prediction_outside_submission_range" for candidate in result.rejected_candidates)
