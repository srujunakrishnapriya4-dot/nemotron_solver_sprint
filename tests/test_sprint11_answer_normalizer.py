from __future__ import annotations

from kaggle_anti086.solvers.answer_normalizer import answers_match, normalize_answer


def test_roman_prefix_noise_extracts_final_roman() -> None:
    assert normalize_answer("38 -> XXXVIII", expected_type="roman").normalized == "XXXVIII"


def test_think_tag_removed_for_roman() -> None:
    assert normalize_answer("</think>\nXXXVIII", expected_type="roman").normalized == "XXXVIII"


def test_boxed_numeric_extracts_value() -> None:
    assert normalize_answer(r"\boxed{154.62}", expected_type="numeric").normalized == "154.62"


def test_numeric_strips_units_and_trailing_zeros() -> None:
    assert normalize_answer("154.620 meters", expected_type="numeric").normalized == "154.62"
    assert normalize_answer("154.6200", expected_type="numeric").normalized == "154.62"


def test_symbol_preserves_punctuation_without_trailing_period() -> None:
    assert normalize_answer("Answer: @&.", expected_type="symbol").normalized == "@&"
    assert normalize_answer("`@&`", expected_type="symbol").normalized == "@&"


def test_binary_preserves_leading_zeroes_and_ignores_explanation() -> None:
    assert normalize_answer(" 00101010\nExplanation...", expected_type="binary").normalized == "00101010"
    assert normalize_answer("Result: 10010111", expected_type="binary").normalized == "10010111"


def test_text_phrase_lowercases_and_collapses_spaces() -> None:
    assert normalize_answer("The answer is: Cat   Book.", expected_type="text_phrase").normalized == "cat book"


def test_answers_match_numeric_roman_and_binary() -> None:
    assert answers_match("154.6200", "154.62", "numeric")
    assert answers_match("answer is XXXVIII", "XXXVIII", "roman")
    assert not answers_match("00101011", "00101010", "binary")
