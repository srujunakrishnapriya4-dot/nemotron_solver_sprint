from kaggle_anti086.solvers.numeric_formula_solver import NumericFormulaSolver


def _row(prompt: str, family: str = "numeric_formula") -> dict:
    return {
        "id": "numeric_precision",
        "family": family,
        "subfamily": "formula",
        "rule_id": "numeric_precision",
        "prompt": prompt,
        "answer": "0",
        "source": "test",
        "solver_name": "numeric_formula_solver",
        "verification_status": "verified",
        "difficulty": 1,
        "split": "dev",
        "leakage_group": "numeric_precision",
    }


def test_gravity_fit_preserves_two_decimals() -> None:
    result = NumericFormulaSolver().solve(_row("time 1 -> 4.90; time 2 -> 19.60; time 3 -> 44.10. Query time 4?", "gravity_numeric"))
    assert not result.abstained
    assert result.candidates[0].answer == "78.40"
    assert result.candidates[0].subfamily == "gravity_distance"


def test_numeric_linear_fit() -> None:
    result = NumericFormulaSolver().solve(_row("1 -> 3.0; 2 -> 6.0; 3 -> 9.0. Convert 4"))
    assert not result.abstained
    assert result.candidates[0].answer == "12.0"


def test_numeric_quadratic_fit() -> None:
    result = NumericFormulaSolver().solve(_row("1 -> 2; 2 -> 8; 3 -> 18. Query 4?"))
    assert not result.abstained
    assert result.candidates[0].answer == "32"
    assert result.candidates[0].subfamily == "quadratic_scale"


def test_numeric_ambiguous_disagreement_abstains() -> None:
    result = NumericFormulaSolver().solve(_row("1 -> 1; 2 -> 4. Query 3?"))
    assert result.abstained
    assert result.reason == "ambiguous_formula_disagreement"


def test_gravity_family_rejects_non_gravity_formula() -> None:
    result = NumericFormulaSolver().solve(_row("1 -> 2; 2 -> 4; 3 -> 6. Query 4?", "gravity_numeric"))
    assert result.abstained
    assert result.reason == "gravity_formula_not_supported"
