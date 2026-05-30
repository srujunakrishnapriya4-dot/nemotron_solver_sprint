from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.sprint_solvers import parse_problem, solve_modular_problem  # noqa: E402


def test_parity_solved() -> None:
    result = solve_modular_problem(parse_problem("0 -> 0\n1 -> 1\n2 -> 0\n3 -> 1\n4 -> 0\n5 -> 1\nTarget: 6 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "0"


def test_last_digit_solved() -> None:
    result = solve_modular_problem(
        parse_problem("25 -> 5\n36 -> 6\n47 -> 7\n58 -> 8\n60 -> 0\n71 -> 1\n82 -> 2\n93 -> 3\n104 -> 4\nTarget: 115 -> ?")
    )

    assert result.status == "solved"
    assert result.prediction == "5"


def test_x_mod_m_solved_with_multiple_examples() -> None:
    result = solve_modular_problem(parse_problem("5 -> 2\n8 -> 2\n11 -> 2\nTarget: 14 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "2"


def test_affine_mod_solved() -> None:
    result = solve_modular_problem(parse_problem("0 -> 1\n1 -> 3\n2 -> 0\n3 -> 2\nTarget: 4 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "4"


def test_digital_root_solved() -> None:
    result = solve_modular_problem(parse_problem("99 -> 9\n123 -> 6\n100 -> 1\n0 -> 0\n8 -> 8\n10 -> 1\nTarget: 9999 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "9"


def test_digit_sum_mod_solved() -> None:
    result = solve_modular_problem(parse_problem("19 -> 0\n28 -> 0\n37 -> 0\nTarget: 46 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "0"


def test_one_example_modulo_abstains() -> None:
    result = solve_modular_problem(parse_problem("5 -> 2\nTarget: 7 -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "insufficient_examples" for candidate in result.rejected_candidates)


def test_ambiguous_moduli_produce_disagreement_or_abstain() -> None:
    result = solve_modular_problem(parse_problem("1 -> 1\n2 -> 2\nTarget: 3 -> ?"))

    assert result.status == "disagreement"
    assert "ambiguous_moduli" in result.errors


def test_non_integer_input_abstains() -> None:
    result = solve_modular_problem(parse_problem("a -> 1\n2 -> 0\nTarget: 3 -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "non_integer_input" for candidate in result.rejected_candidates)


def test_invalid_example_output_gives_invalid_output_format_diagnostic() -> None:
    result = solve_modular_problem(parse_problem("1 -> nope\n2 -> 0\nTarget: 3 -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "invalid_output_format" for candidate in result.rejected_candidates)


def test_out_of_range_example_output_gives_invalid_output_format_diagnostic() -> None:
    result = solve_modular_problem(parse_problem("1 -> 100000\n2 -> 100000\nTarget: 3 -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "invalid_output_format" for candidate in result.rejected_candidates)


def test_zero_or_negative_modulus_is_not_enumerated() -> None:
    result = solve_modular_problem(parse_problem("0 -> 0\n1 -> 1\n2 -> 0\n3 -> 1\nTarget: 4 -> ?"))

    rule_ids = {str(candidate.metadata["rule_id"]) for candidate in result.verified_candidates + result.rejected_candidates}
    for rule_id in rule_ids:
        if rule_id.startswith(("x_mod_", "digit_sum_mod_", "digit_product_mod_")):
            assert int(rule_id.rsplit("_", 1)[1]) > 0
        if rule_id.startswith("affine_mod_"):
            assert int(rule_id.rsplit("_", 1)[1]) > 0
