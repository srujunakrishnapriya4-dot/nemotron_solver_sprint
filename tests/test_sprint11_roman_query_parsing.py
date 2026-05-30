from kaggle_anti086.solvers.roman_solver import RomanSolver


def _row(prompt: str) -> dict:
    return {
        "id": "roman_query",
        "family": "roman_numeral",
        "subfamily": "standard_roman",
        "rule_id": "roman_query",
        "prompt": prompt,
        "answer": "XXXVIII",
        "source": "test",
        "solver_name": "roman_solver",
        "verification_status": "verified",
        "difficulty": 1,
        "split": "dev",
        "leakage_group": "roman_query",
    }


def test_new_query_forms_parse() -> None:
    forms = [
        "11 -> XI; 15 -> XV. write 38 in the same system",
        "11 -> XI; 15 -> XV. solve for 38",
        "11 -> XI; 15 -> XV. output for 38",
        "11 -> XI; 15 -> XV. target number: 38",
        "11 -> XI; 15 -> XV. Given the examples above, 38",
        "11 -> XI; 15 -> XV. Give the answer in the same system: 38",
    ]
    for prompt in forms:
        result = RomanSolver().solve(_row(prompt))
        assert not result.abstained, prompt
        assert result.candidates[0].answer == "XXXVIII"


def test_roman_query_does_not_select_example_number() -> None:
    result = RomanSolver().solve(_row("11 -> XI; 15 -> XV. Given the examples above, 38"))
    assert not result.abstained
    assert result.candidates[0].metadata["query"] == 38


def test_roman_out_of_range_abstains() -> None:
    result = RomanSolver().solve(_row("11 -> XI; 15 -> XV. solve for 4000"))
    assert result.abstained
    assert result.reason == "query_out_of_range"


def test_roman_contradicted_example_abstains() -> None:
    result = RomanSolver().solve(_row("11 -> XII; 15 -> XV. solve for 38"))
    assert result.abstained
    assert result.reason == "examples_do_not_match_standard_roman"
