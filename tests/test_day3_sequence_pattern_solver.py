from __future__ import annotations

import re

from kaggle_anti086.solvers.router import route_row
from kaggle_anti086.solvers.sequence_pattern_solver import SequencePatternSolver


def _solve(prompt: str, *, family: str = "sequence_pattern"):
    return SequencePatternSolver().solve({"family": family, "prompt": prompt})


def _best(prompt: str):
    result = _solve(prompt)
    assert not result.abstained, result.reason
    return result.candidates[0]


def test_arithmetic_progression_verified():
    candidate = _best("sequence: 2, 4, 6, 8, ?")

    assert candidate.answer == "10"
    assert candidate.metadata["verification_status"] == "PASS"


def test_constant_sequence_verified():
    candidate = _best("sequence: 7, 7, 7, 7, ?")

    assert candidate.answer == "7"


def test_geometric_progression_verified():
    candidate = _best("sequence: 3, 6, 12, 24, ?")

    assert candidate.answer == "48"


def test_second_difference_quadratic_sequence_verified():
    candidate = _best("sequence: 1, 4, 9, 16, ?")

    assert candidate.answer == "25"


def test_alternating_interleaved_arithmetic_verified():
    candidate = _best("sequence: 1, 10, 3, 13, 5, 16, ?")

    assert candidate.answer == "7"


def test_periodic_cycle_verified():
    candidate = _best("sequence: 1, 2, 3, 1, 2, 3, ?")

    assert candidate.answer == "1"


def test_fibonacci_like_recurrence_verified():
    candidate = _best("sequence: 1, 1, 2, 3, 5, ?")

    assert candidate.answer == "8"


def test_affine_recurrence_verified():
    candidate = _best("sequence: 2, 5, 11, 23, ?")

    assert candidate.answer == "47"


def test_negative_integers_parsed_correctly():
    candidate = _best("sequence: -9, -6, -3, 0, ?")

    assert candidate.answer == "3"


def test_malformed_prompt_abstains():
    result = _solve("sequence: A, B, C, ?")

    assert result.abstained


def test_too_few_terms_abstains():
    result = _solve("sequence: 2, 4, ?")

    assert result.abstained


def test_contradictory_noisy_sequence_abstains():
    result = _solve("sequence: 2, 4, 6, 9, ?")

    assert result.abstained


def test_multiple_candidate_rules_with_different_next_outputs_abstain():
    result = _solve("sequence: 1, 2, 4, ?")

    assert result.abstained
    assert result.reason == "sequence_pattern_ambiguous_different_outputs"


def test_multiple_candidate_rules_with_same_next_output_allowed_with_lower_confidence():
    candidate = _best("sequence: 1, 2, 3, ?")

    assert candidate.answer == "4"
    assert candidate.metadata["ambiguity_count"] > 0
    assert candidate.confidence < 0.9


def test_huge_unsafe_extrapolation_abstains():
    result = _solve("sequence: 100000000, 10000000000, 1000000000000, ?")

    assert result.abstained


def test_non_sequence_family_is_not_routed_into_sequence_solver():
    route = route_row({"family": "numeric_formula", "prompt": "1 -> 2; 2 -> 4; query 3"})

    assert route.family != "sequence_pattern"


def test_output_is_exact_integer_string_no_explanation():
    candidate = _best("sequence: 4, 8, 12, 16, ?")

    assert candidate.answer == "20"
    assert re.fullmatch(r"-?\d+", candidate.answer)
    assert "because" not in candidate.answer.lower()


def test_verification_status_pass_only_when_all_known_terms_verify():
    candidate = _best("examples: [1, 2, 3] -> 4; [7, 8, 9] -> 10; query [11, 12, 13] -> ?")
    failed = _solve("examples: [1, 2, 3] -> 5; [7, 8, 9] -> 10; query [11, 12, 13] -> ?")

    assert candidate.answer == "14"
    assert candidate.metadata["verification_status"] == "PASS"
    assert failed.abstained


def test_abstain_safety_preserved_for_day2_indexed_placeholders():
    result = _solve("[private_like #440] sequence 0: 2, 4, 8, 16, ?")

    assert result.abstained


def test_numeric_formula_known_ambiguous_case_is_not_hidden():
    route = route_row({"family": "numeric_formula", "prompt": "1 -> 1; 2 -> 4. Query 3?"})

    assert route.family == "numeric_formula"


def test_no_training_package_or_submission_authorization_in_solver_metadata():
    candidate = _best("sequence: 2, 4, 6, 8, ?")
    text = str(candidate.metadata).lower()

    assert "submission_authorized" not in text
    assert "package_authorized" not in text
    assert "v2a_150_authorized" not in text
