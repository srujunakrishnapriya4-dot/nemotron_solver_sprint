from kaggle_anti086.solvers.numeric_formula_solver import NumericFormulaSolver


def _row(prompt: str, family: str = "numeric_formula") -> dict:
    return {
        "id": f"{family}_1",
        "family": family,
        "subfamily": "formula",
        "rule_id": "formula_rule",
        "prompt": prompt,
        "answer": "0",
        "source": "test",
        "solver_name": "numeric_formula_solver",
        "verification_status": "verified",
        "difficulty": 2,
        "split": "dev",
        "leakage_group": family,
    }


def test_numeric_formula_gravity_fit() -> None:
    prompt = "Falling distance examples: time 1 -> 4.9; time 2 -> 19.6; time 3 -> 44.1. Query time 4?"
    result = NumericFormulaSolver().solve(_row(prompt, "gravity_numeric"))
    assert not result.abstained
    assert result.candidates[0].subfamily == "gravity_distance"
    assert result.candidates[0].answer == "78.4"


def test_numeric_formula_linear_scale() -> None:
    result = NumericFormulaSolver().solve(_row("2 -> 6; 4 -> 12; 8 -> 24. Query 5?"))
    assert not result.abstained
    assert result.candidates[0].answer == "15"


def test_numeric_formula_linear_offset() -> None:
    result = NumericFormulaSolver().solve(_row("2 -> 7; 4 -> 11; 8 -> 19. Query 5?"))
    assert not result.abstained
    assert result.candidates[0].answer == "13"


def test_numeric_formula_quadratic_scale() -> None:
    result = NumericFormulaSolver().solve(_row("2 -> 12; 3 -> 27; 4 -> 48. Query 5?"))
    assert not result.abstained
    assert result.candidates[0].answer == "75"


def test_numeric_formula_inconsistent_examples_abstain() -> None:
    assert NumericFormulaSolver().solve(_row("2 -> 6; 4 -> 999; 8 -> 24. Query 5?")).abstained


def test_numeric_formula_ambiguous_not_low_risk_verified() -> None:
    result = NumericFormulaSolver().solve(_row("1 -> 2; 2 -> 4. Query 3?"))
    assert result.abstained or result.candidates[0].risk != "low"


def test_numeric_formula_missing_query_abstains() -> None:
    assert NumericFormulaSolver().solve(_row("2 -> 6; 4 -> 12; 8 -> 24.")).abstained
