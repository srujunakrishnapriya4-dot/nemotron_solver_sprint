from kaggle_anti086.solvers.answer_normalizer import answers_match
from kaggle_anti086.solvers.unit_conversion_solver import UnitConversionSolver, infer_decimal_places


def _row(prompt: str) -> dict:
    return {
        "id": "unit_1",
        "family": "unit_conversion",
        "subfamily": "ratio_conversion",
        "rule_id": "unit_ratio",
        "prompt": prompt,
        "answer": "16.67",
        "source": "test",
        "solver_name": "unit_conversion_solver",
        "verification_status": "verified",
        "difficulty": 2,
        "split": "dev",
        "leakage_group": "unit",
    }


def test_unit_conversion_multiplicative_fit() -> None:
    prompt = "Examples: 10.08 -> 6.69; 17.83 -> 11.83; 35.85 converts to 23.79. Input 25.11, output ?"
    result = UnitConversionSolver().solve(_row(prompt))
    assert not result.abstained
    assert answers_match(result.candidates[0].answer, "16.66", answer_type="numeric")
    assert result.candidates[0].verified is True


def test_unit_conversion_linear_offset_fit() -> None:
    prompt = "10 -> 15; 20 -> 25; 30 -> 35. Query 40?"
    result = UnitConversionSolver().solve(_row(prompt))
    assert not result.abstained
    assert result.candidates[0].answer == "45"


def test_unit_conversion_rounding_precision_inferred() -> None:
    assert infer_decimal_places(["1.20", "2.345", "9"]) == 3


def test_unit_conversion_inconsistent_examples_abstain() -> None:
    prompt = "10 -> 15; 20 -> 99; 30 -> 35. Query 40?"
    assert UnitConversionSolver().solve(_row(prompt)).abstained


def test_unit_conversion_missing_query_abstains() -> None:
    prompt = "10 -> 15; 20 -> 25; 30 -> 35."
    assert UnitConversionSolver().solve(_row(prompt)).abstained
