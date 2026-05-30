from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.competition_sprint.equation_symbolic_solver import solve_equation_symbolic_problem  # noqa: E402


def test_equation_symbolic_solver_exact_division_and_operator_semantics() -> None:
    result = solve_equation_symbolic_problem((("6 @ 3", "2"), ("8 @ 4", "2")), "10 @ 5")
    assert result.verified
    assert result.prediction == "2"
    plus = solve_equation_symbolic_problem((("2 + 3", "6"),), "4 + 5")
    assert not plus.verified


def test_equation_symbolic_solver_keep_drop_reorder_string_transform() -> None:
    drop = solve_equation_symbolic_problem((("20+44", "2044"), ("15}39", "1539"), ("35}43", "3543")), "18+67")
    assert drop.verified
    assert drop.prediction == "1867"
    reorder = solve_equation_symbolic_problem((("05+12", "1205"), ("02*63", "6302"), ("69+97", "9769")), "18+93")
    assert reorder.verified
    assert reorder.prediction == "9318"


def test_equation_symbolic_solver_char_and_token_bijection() -> None:
    char = solve_equation_symbolic_problem((("ab", "xy"), ("bc", "yz")), "ac")
    assert char.verified
    assert char.prediction == "xz"
    token = solve_equation_symbolic_problem((("red blue", "cat dog"), ("blue red", "dog cat")), "red red")
    assert token.verified
    assert token.prediction == "cat cat"


def test_equation_symbolic_solver_operator_conditioned_dispatch() -> None:
    result = solve_equation_symbolic_problem(
        (
            ("%|*\"|", "%|\"|"),
            ("\\(*[^", "\\([^"),
            ("(%+[@", "(%[@"),
            ("|[*([", "|[(["),
            ("[^-[(", "-^"),
        ),
        "\\(*[#",
    )
    assert result.verified
    assert result.prediction == "\\([#"


def test_equation_symbolic_solver_numeric_operator_mapping_modulo_and_nonexact_division() -> None:
    mod = solve_equation_symbolic_problem((("10 % 4", "2"), ("14 % 5", "4")), "20 % 6")
    assert mod.verified
    assert mod.prediction == "2"
    non_exact = solve_equation_symbolic_problem((("5 @ 2", "2"), ("7 @ 3", "2")), "9 @ 4")
    assert not non_exact.verified


def test_equation_symbolic_solver_ambiguous_disagreement_and_agreement_metadata() -> None:
    disagreement = solve_equation_symbolic_problem((("ab", "a"), ("ac", "a")), "bc")
    assert not disagreement.verified
    assert disagreement.reason == "ambiguous_transform"
    agreement = solve_equation_symbolic_problem((("20+44", "2044"), ("15}39", "1539"), ("35}43", "3543")), "18+67")
    assert agreement.verified
    assert agreement.metadata["ambiguity_count"] >= 1


def test_equation_symbolic_solver_preserves_leading_zeroes_and_rejects_unseen_operator_projection() -> None:
    leading = solve_equation_symbolic_problem((("79*29", "2979"), ("12*65", "6512"), ("47*03", "0347")), "03*47")
    assert leading.verified
    assert leading.prediction == "4703"
    unsafe = solve_equation_symbolic_problem((("79*29", "2979"), ("12*65", "6512"), ("47*03", "0347")), "07+38")
    assert not unsafe.verified


def test_equation_symbolic_solver_constant_insertion_and_affine_programs() -> None:
    inserted = solve_equation_symbolic_problem((("ab", "Xa"), ("cd", "Xc"), ("ef", "Xe")), "gh")
    assert inserted.verified
    assert inserted.prediction == "Xg"
    affine = solve_equation_symbolic_problem((("10 @ 20", "31"), ("31 @ 42", "74"), ("55 @ 12", "68")), "48 @ 23")
    assert affine.verified
    assert affine.prediction == "72"


def test_equation_symbolic_solver_rejects_underdetermined_one_example_deletion() -> None:
    result = solve_equation_symbolic_problem((("abcde", "abcd"),), "vwxyz")
    assert not result.verified


def test_equation_symbolic_solver_symbol_digit_binding_and_unseen_rejection() -> None:
    result = solve_equation_symbolic_problem((("A @ B", "3"), ("B @ C", "5"), ("A @ C", "4")), "C @ A")
    assert result.verified
    assert result.prediction == "4"
    unseen = solve_equation_symbolic_problem((("A @ B", "3"), ("B @ C", "5"), ("A @ C", "4")), "C @ D")
    assert not unseen.verified
    assert unseen.reason == "unseen_target_symbol"


def test_equation_symbolic_solver_nonstandard_visible_operator_arithmetic() -> None:
    product_minus_one = solve_equation_symbolic_problem((("38 - 68", "2583"), ("57 - 16", "911"), ("51 - 33", "1682")), "20 - 75")
    assert product_minus_one.verified
    assert product_minus_one.prediction == "1499"
    exact_div = solve_equation_symbolic_problem((("8 @ 4", "2"), ("12 @ 3", "4")), "15 @ 5")
    assert exact_div.verified
    assert exact_div.prediction == "3"
    non_exact = solve_equation_symbolic_problem((("8 @ 4", "2"), ("12 @ 3", "4")), "10 @ 4")
    assert not non_exact.verified
