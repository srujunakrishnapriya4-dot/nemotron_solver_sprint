from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.sprint_solvers import (  # noqa: E402
    SolverResult,
    SprintSolverError,
    enumerate_dsl_candidates,
    parse_problem,
    solve_dsl_problem,
)


def test_identity_integer_solved() -> None:
    result = solve_dsl_problem(parse_problem("0 -> 0\n1 -> 1\nTarget: 0 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "0"


def test_reverse_digits_solved() -> None:
    result = solve_dsl_problem(parse_problem("12 -> 21\n34 -> 43\nTarget: 56 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "65"


def test_bit_not_solved() -> None:
    result = solve_dsl_problem(parse_problem("0101 -> 1010\n1111 -> 0000\nTarget: 0011 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "12"


def test_add_const_with_two_examples_solved() -> None:
    result = solve_dsl_problem(parse_problem("1 -> 3\n2 -> 4\nTarget: 3 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "5"


def test_one_example_add_const_abstains_as_underdetermined() -> None:
    result = solve_dsl_problem(parse_problem("1 -> 2\nTarget: 3 -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "underdetermined_rule" for candidate in result.rejected_candidates)


def test_affine_small_solved_with_two_distinct_examples() -> None:
    result = solve_dsl_problem(parse_problem("1 -> 5\n2 -> 8\nTarget: 3 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "11"


def test_digit_sum_solved() -> None:
    result = solve_dsl_problem(parse_problem("124 -> 7\n59 -> 14\nTarget: 987 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "24"


def test_sequence_sum_solved() -> None:
    result = solve_dsl_problem(parse_problem("1 2 3 -> 6\n4 5 6 -> 15\nTarget: 7 8 9 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "24"


def test_binary_op_addition_solved() -> None:
    result = solve_dsl_problem(parse_problem("2 + 3 -> 5\n4 + 1 -> 5\nTarget: 6 + 7 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "13"


def test_symbolic_operator_at_infers_addition_only_when_examples_verify() -> None:
    result = solve_dsl_problem(parse_problem("2 @ 3 -> 5\n4 @ 1 -> 5\nTarget: 6 @ 7 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "13"
    assert any(candidate.metadata["rule_id"] == "symbolic_add" for candidate in result.verified_candidates)


def test_exact_division_solved() -> None:
    result = solve_dsl_problem(parse_problem("6 / 3 -> 2\n10 / 5 -> 2\nTarget: 8 / 2 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "4"


def test_non_exact_division_rejected() -> None:
    result = solve_dsl_problem(parse_problem("5 / 2 -> 2\nTarget: 6 / 4 -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "non_exact_division" for candidate in result.rejected_candidates)


def test_division_by_zero_rejected() -> None:
    result = solve_dsl_problem(parse_problem("5 / 0 -> 0\nTarget: 4 / 2 -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "division_by_zero" for candidate in result.rejected_candidates)


def test_modulo_solved_with_two_examples() -> None:
    result = solve_dsl_problem(parse_problem("5 -> 2\n8 -> 2\nTarget: 2 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "2"


def test_one_example_modulo_abstains() -> None:
    result = solve_dsl_problem(parse_problem("5 -> 2\nTarget: 8 -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "underdetermined_rule" for candidate in result.rejected_candidates)


def test_char_bijection_solved() -> None:
    result = solve_dsl_problem(parse_problem("ab -> 12\nba -> 21\nTarget: aba -> ?"))

    assert result.status == "solved"
    assert result.prediction == "121"


def test_char_bijection_unseen_target_symbol_rejected() -> None:
    result = solve_dsl_problem(parse_problem("a -> 1\nTarget: z -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "unseen_target_symbol" for candidate in result.rejected_candidates)


def test_token_bijection_solved_only_when_explicitly_tokenized() -> None:
    solved = solve_dsl_problem(parse_problem("red blue -> 1 2\nblue red -> 2 1\nTarget: red red -> ?"))
    bare = enumerate_dsl_candidates(parse_problem("ab -> 12\nTarget: ab -> ?"))

    assert solved.status == "solved"
    assert solved.prediction == "11"
    assert all(candidate.metadata["rule_id"] != "token_bijection" for candidate in bare)


def test_collision_rejected() -> None:
    result = solve_dsl_problem(parse_problem("ab -> 11\nTarget: a -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "mapping_collision" for candidate in result.rejected_candidates)


def test_disagreement_returns_disagreement() -> None:
    result = solve_dsl_problem(parse_problem("00 -> 00\n11 -> 11\nTarget: 01 -> ?"))

    assert result.status == "disagreement"
    assert {candidate.target_prediction for candidate in result.verified_candidates} >= {"1", "10"}


def test_invalid_output_format_rejected() -> None:
    result = solve_dsl_problem(parse_problem("abc -> cba\nTarget: def -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "invalid_output_format" for candidate in result.rejected_candidates)


def test_max_candidates_budget_exhaustion_returns_abstain_and_records_budget() -> None:
    result = solve_dsl_problem(parse_problem("1 -> 1\n2 -> 2\nTarget: 3 -> ?"), max_candidates=1)

    assert result.status == "abstain"
    assert "budget_exhausted" in result.errors
    assert any(candidate.metadata.get("rejection_reason") == "budget_exhausted" for candidate in result.rejected_candidates)


def test_max_candidates_zero_records_budget_exhausted() -> None:
    result = solve_dsl_problem(parse_problem("1 -> 1\n2 -> 2\nTarget: 3 -> ?"), max_candidates=0)

    assert result.status == "abstain"
    assert "budget_exhausted" in result.errors
    assert any(candidate.metadata.get("rejection_reason") == "budget_exhausted" for candidate in result.rejected_candidates)


def test_max_candidates_negative_records_budget_exhausted() -> None:
    result = solve_dsl_problem(parse_problem("1 -> 1\n2 -> 2\nTarget: 3 -> ?"), max_candidates=-1)

    assert result.status == "abstain"
    assert "budget_exhausted" in result.errors
    assert any(candidate.metadata.get("rejection_reason") == "budget_exhausted" for candidate in result.rejected_candidates)


def test_one_example_sequence_sum_mod_abstains_as_underdetermined() -> None:
    result = solve_dsl_problem(parse_problem("1 2 -> 0\nTarget: 3 4 -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rule_id") == "sequence_sum_then_mod" for candidate in result.rejected_candidates)
    assert any(candidate.metadata.get("rejection_reason") == "underdetermined_rule" for candidate in result.rejected_candidates)


def test_one_example_sequence_product_mod_abstains_as_underdetermined() -> None:
    result = solve_dsl_problem(parse_problem("2 3 -> 0\nTarget: 2 4 -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rule_id") == "sequence_product_then_mod" for candidate in result.rejected_candidates)
    assert any(candidate.metadata.get("rejection_reason") == "underdetermined_rule" for candidate in result.rejected_candidates)


def test_mod_composition_with_one_distinct_derived_value_abstains() -> None:
    result = solve_dsl_problem(parse_problem("1 4 -> 1\n2 3 -> 1\nTarget: 7 8 -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rule_id") == "sequence_sum_then_mod" for candidate in result.rejected_candidates)
    assert any(candidate.metadata.get("rejection_reason") == "underdetermined_rule" for candidate in result.rejected_candidates)


def test_symbolic_exact_division_solved() -> None:
    result = solve_dsl_problem(parse_problem("6 @ 3 -> 2\n8 @ 4 -> 2\nTarget: 10 @ 5 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "2"
    assert any(candidate.metadata["rule_id"] == "symbolic_exact_div" for candidate in result.verified_candidates)


def test_symbolic_non_exact_division_rejected() -> None:
    result = solve_dsl_problem(parse_problem("5 @ 2 -> 2\n9 @ 3 -> 3\nTarget: 7 @ 2 -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "non_exact_division" for candidate in result.rejected_candidates)


def test_deterministic_candidate_ids_hashes() -> None:
    parsed = parse_problem("12 -> 21\n34 -> 43\nTarget: 56 -> ?")
    first = enumerate_dsl_candidates(parsed)
    second = enumerate_dsl_candidates(parsed)

    assert [(candidate.metadata["rule_id"], candidate.candidate_hash) for candidate in first] == [
        (candidate.metadata["rule_id"], candidate.candidate_hash) for candidate in second
    ]


def test_no_target_gold_leakage_used() -> None:
    parsed = parse_problem({"id": "p", "prompt": "1 -> 2\n2 -> 3\nTarget: 3 -> ?", "expected_answer": "999", "gold": "888"})
    result = solve_dsl_problem(parsed)

    assert result.status == "solved"
    assert result.prediction == "4"


def test_solver_result_hash_forgery_still_rejected() -> None:
    result = solve_dsl_problem(parse_problem("1 -> 1\n2 -> 2\nTarget: 3 -> ?"))

    with pytest.raises(SprintSolverError):
        replace(result, result_hash="forged")
