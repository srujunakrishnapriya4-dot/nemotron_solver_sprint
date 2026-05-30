from kaggle_anti086.solvers.roman_solver import RomanSolver, int_to_roman


def _row(prompt: str, family: str = "roman_numeral") -> dict:
    return {
        "id": "roman_1",
        "family": family,
        "subfamily": "standard_roman",
        "rule_id": "roman_standard",
        "prompt": prompt,
        "answer": "XXXVIII",
        "source": "test",
        "solver_name": "roman_solver",
        "verification_status": "verified",
        "difficulty": 1,
        "split": "dev",
        "leakage_group": "roman",
    }


def test_int_to_roman_boundaries_and_subtractive() -> None:
    assert int_to_roman(1) == "I"
    assert int_to_roman(4) == "IV"
    assert int_to_roman(9) == "IX"
    assert int_to_roman(38) == "XXXVIII"
    assert int_to_roman(94) == "XCIV"
    assert int_to_roman(3999) == "MMMCMXCIX"


def test_roman_solver_handles_noisy_wonderland_prompt() -> None:
    prompt = "In Wonderland numerals, 11 -> XI, 15 -> XV, 94 -> XCIV. What is 38 in Wonderland numerals?"
    result = RomanSolver().solve(_row(prompt, family="custom_numeral"))
    assert not result.abstained
    assert result.candidates[0].answer == "XXXVIII"
    assert result.candidates[0].verified is True
    assert result.candidates[0].risk == "low"


def test_roman_solver_invalid_range_abstains() -> None:
    assert RomanSolver().solve(_row("11 -> XI. Convert 0")).abstained
    assert RomanSolver().solve(_row("11 -> XI. Convert 4000")).abstained


def test_roman_solver_missing_query_abstains() -> None:
    assert RomanSolver().solve(_row("11 -> XI, 15 -> XV")).abstained
