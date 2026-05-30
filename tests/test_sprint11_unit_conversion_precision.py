from kaggle_anti086.solvers.unit_conversion_solver import UnitConversionSolver


def _row(prompt: str) -> dict:
    return {
        "id": "unit_precision",
        "family": "unit_conversion",
        "subfamily": "unit",
        "rule_id": "unit_precision",
        "prompt": prompt,
        "answer": "0",
        "source": "test",
        "solver_name": "unit_conversion_solver",
        "verification_status": "verified",
        "difficulty": 1,
        "split": "dev",
        "leakage_group": "unit_precision",
    }


def test_unit_solver_preserves_two_decimal_trailing_zero_precision() -> None:
    result = UnitConversionSolver().solve(_row("1 -> 1.10; 2 -> 2.20; 3 -> 3.30. Convert 4"))
    assert not result.abstained
    assert result.candidates[0].answer == "4.40"


def test_unit_solver_preserves_three_decimal_precision() -> None:
    result = UnitConversionSolver().solve(_row("10 -> 16.650; 20 -> 33.300; 30 -> 49.950. Input: 40"))
    assert not result.abstained
    assert result.candidates[0].answer == "66.600"


def test_unit_solver_integer_outputs_remain_integer() -> None:
    result = UnitConversionSolver().solve(_row("10 -> 15; 20 -> 25; 30 -> 35. Now solve: 40"))
    assert not result.abstained
    assert result.candidates[0].answer == "45"


def test_unit_solver_ambiguous_different_outputs_abstains() -> None:
    result = UnitConversionSolver().solve(_row("1 -> 1; 2 -> 4. For 3, output ?"))
    assert result.abstained
    assert result.reason in {"ambiguous_fit_disagreement", "inconsistent_examples"}


def test_unit_solver_inconsistent_examples_abstain() -> None:
    assert UnitConversionSolver().solve(_row("1 -> 2; 2 -> 999; 3 -> 6. Convert 4")).abstained


def test_unit_solver_query_forms() -> None:
    forms = [
        "1 -> 2; 2 -> 4; 3 -> 6. 4 -> ?",
        "1 -> 2; 2 -> 4; 3 -> 6. Convert 4",
        "1 -> 2; 2 -> 4; 3 -> 6. What is 4 in the target unit?",
        "1 -> 2; 2 -> 4; 3 -> 6. Now solve: 4",
        "1 -> 2; 2 -> 4; 3 -> 6. Input: 4",
        "1 -> 2; 2 -> 4; 3 -> 6. For 4, output ?",
    ]
    for prompt in forms:
        result = UnitConversionSolver().solve(_row(prompt))
        assert not result.abstained, prompt
        assert result.candidates[0].answer == "8"
