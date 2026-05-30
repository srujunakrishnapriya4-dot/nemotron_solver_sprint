from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.competition_sprint import (  # noqa: E402
    CompetitionAnswerPolicyError,
    infer_answer_kind,
    validate_competition_answer,
)


def test_bitstring_preserved() -> None:
    assert validate_competition_answer(" 00110100 ") == "00110100"
    assert infer_answer_kind("00110100") == "bitstring"


def test_lowercase_text_preserved() -> None:
    assert validate_competition_answer(" cat imagines book ") == "cat imagines book"
    assert infer_answer_kind("cat imagines book") == "lowercase_text"


def test_roman_decimal_and_symbol_preserved() -> None:
    assert validate_competition_answer("XXXVIII", expected_kind="roman") == "XXXVIII"
    assert validate_competition_answer("16.65", expected_kind="decimal") == "16.65"
    assert validate_competition_answer("@&", expected_kind="symbol_string") == "@&"


def test_empty_nan_inf_and_leakage_rejected() -> None:
    for value in ("", "  ", math.nan, math.inf, "expected_answer=4", "correct answer: 7"):
        with pytest.raises(CompetitionAnswerPolicyError):
            validate_competition_answer(value)
