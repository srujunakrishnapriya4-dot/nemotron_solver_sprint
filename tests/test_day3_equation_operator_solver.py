from __future__ import annotations

import re

from kaggle_anti086.solvers.equation_operator_solver import EquationOperatorSolver
from kaggle_anti086.solvers.router import route_row


def _solve(prompt: str, *, family: str = "equation_operator"):
    return EquationOperatorSolver().solve({"family": family, "prompt": prompt})


def _best(prompt: str):
    result = _solve(prompt)
    assert not result.abstained, result.reason
    return result.candidates[0]


def test_binary_addition_inferred_and_verified():
    candidate = _best("1 @ 2 -> 3; 4 @ 5 -> 9; 8 @ 1 -> 9. query: 7 @ 8")

    assert candidate.answer == "15"
    assert candidate.metadata["verification_status"] == "PASS"


def test_binary_subtraction_inferred_and_verified():
    candidate = _best("8 @ 3 -> 5; 10 @ 4 -> 6; 7 @ 2 -> 5; query: 9 @ 5")

    assert candidate.answer == "4"


def test_binary_multiplication_inferred_and_verified():
    candidate = _best("2 @ 3 -> 6; 4 @ 5 -> 20; 3 @ 7 -> 21; query: 6 @ 8")

    assert candidate.answer == "48"


def test_exact_integer_division_inferred_and_verified():
    candidate = _best("6 @ 3 -> 2; 10 @ 5 -> 2; 9 @ 3 -> 3; 12 @ 4 -> 3; query: 15 @ 3")

    assert candidate.answer == "5"


def test_modulo_inferred_and_verified():
    candidate = _best("7 @ 3 -> 1; 10 @ 4 -> 2; 8 @ 5 -> 3; 9 @ 4 -> 1; query: 11 @ 5")

    assert candidate.answer == "1"


def test_min_and_max_inferred_when_supported():
    min_candidate = _best("2 @ 7 -> 2; 9 @ 3 -> 3; 4 @ 8 -> 4; query: 6 @ 5")
    max_candidate = _best("2 @ 7 -> 7; 9 @ 3 -> 9; 4 @ 8 -> 8; query: 6 @ 5")

    assert min_candidate.answer == "5"
    assert max_candidate.answer == "6"


def test_affine_small_coefficient_operator_inferred_when_unique():
    candidate = _best("1 @ 2 -> 7; 2 @ 3 -> 11; 3 @ 5 -> 17; 4 @ 7 -> 23; query: 5 @ 11")

    assert candidate.answer == "33"
    assert any(name.startswith("affine_") for name in candidate.metadata["operation_names"])


def test_negative_integers_parse_correctly():
    candidate = _best("-2 @ 3 -> 1; 4 @ -5 -> -1; -1 @ -1 -> -2; query: -3 @ 2")

    assert candidate.answer == "-1"


def test_malformed_prompt_abstains():
    result = _solve("operator equation: A @ B -> C. infer target A # B?")

    assert result.abstained


def test_zero_division_candidate_rejected_without_crash():
    result = _solve("6 @ 0 -> 2; 10 @ 0 -> 5; query: 8 @ 0")

    assert result.abstained


def test_contradictory_examples_abstain():
    result = _solve("2 @ 3 -> 5; 2 @ 3 -> 9; query: 4 @ 5")

    assert result.abstained


def test_multifit_candidates_with_different_query_outputs_abstain():
    result = _solve("1 @ 1 -> 2; query: 2 @ 3")

    assert result.abstained
    assert result.reason == "equation_operator_ambiguous_different_outputs"


def test_multifit_candidates_with_same_query_output_allowed_with_lower_confidence():
    candidate = _best("1 @ 1 -> 1; 2 @ 2 -> 2; query: 3 @ 3")

    assert candidate.answer == "3"
    assert candidate.metadata["ambiguity_count"] > 0
    assert candidate.confidence < 0.9


def test_query_parse_failure_abstains():
    result = _solve("1 @ 2 -> 3; 4 @ 5 -> 9. query: ?")

    assert result.abstained


def test_non_equation_family_is_not_routed_into_equation_solver():
    route = route_row({"family": "symbol_mapping", "prompt": "1 @ 2 -> 3; query: 4 @ 5"})

    assert route.family != "equation_operator"


def test_answer_format_is_exact_integer_string_no_explanation():
    candidate = _best("2 @ 3 -> 5; 4 @ 5 -> 9; 6 @ 7 -> 13; query: 8 @ 9")

    assert candidate.answer == "17"
    assert re.fullmatch(r"-?\d+", candidate.answer)
    assert "because" not in candidate.answer.lower()


def test_verification_status_pass_only_when_all_examples_verify():
    candidate = _best("2 @ 3 -> 5; 4 @ 5 -> 9; 6 @ 7 -> 13; query: 8 @ 9")
    failed = EquationOperatorSolver().solve({"family": "equation_operator", "prompt": "2 @ 3 -> 5; 4 @ 5 -> 10; query: 8 @ 9"})

    assert candidate.metadata["verification_status"] == "PASS"
    assert failed.abstained


def test_abstain_safety_preserved_for_day2_symbolic_rows():
    result = _solve("[anti_leak #246] operator equation 0: A @ B -> C. infer target A # B?")

    assert result.abstained


def test_no_training_package_or_submission_authorization_in_solver_metadata():
    candidate = _best("2 @ 3 -> 5; 4 @ 5 -> 9; 6 @ 7 -> 13; query: 8 @ 9")
    text = str(candidate.metadata).lower()

    assert "submission_authorized" not in text
    assert "package_authorized" not in text
    assert "v2a_150_authorized" not in text
