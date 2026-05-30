from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.competition_sprint import CompetitionPromptAdapterError, parse_competition_prompt  # noqa: E402


BIT_PROMPT = """In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.

Here are some examples of input -> output:
00000000 -> 11111111
11111111 -> 00000000
01010101 -> 10101010
10101010 -> 01010101
00110011 -> 11001100
11001100 -> 00110011
00001111 -> 11110000
11110000 -> 00001111

Now, determine the output for: 00110100"""

CIPHER_PROMPT = """In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:
aaa bbb -> cat dog
bbb aaa -> dog cat
ccc aaa -> book cat
Now, decrypt the following text: aaa ccc"""

ROMAN_PROMPT = """In Alice's Wonderland, numbers are secretly converted into a different numeral system. Some examples are given below:
11 -> XI
15 -> XV
94 -> XCIV
19 -> XIX
Now, write the number 38 in the Wonderland numeral system."""


def test_bit_prompt_parses_examples_and_target() -> None:
    problem = parse_competition_prompt("p", BIT_PROMPT, "11001011")

    assert problem.family == "bit_manipulation"
    assert len(problem.examples) == 8
    assert problem.target_input == "00110100"
    assert problem.answer == "11001011"
    assert problem.answer_kind == "bitstring"


def test_cipher_and_roman_prompts_parse() -> None:
    cipher = parse_competition_prompt("c", CIPHER_PROMPT, "cat book")
    roman = parse_competition_prompt("r", ROMAN_PROMPT, "XXXVIII")

    assert cipher.family == "cipher_text"
    assert cipher.target_input == "aaa ccc"
    assert roman.family == "roman_numeral"
    assert roman.target_input == "38"


def test_real_like_unit_gravity_and_equation_prompts_parse() -> None:
    unit_prompt = """In Alice's Wonderland, a secret unit conversion is applied to measurements.
2 meters becomes 5.0
4 meters becomes 10.0
Now, convert the following measurement: 6 meters"""
    gravity_prompt = """In Alice's Wonderland, the gravitational constant has been secretly changed.
For t = 2s, distance = 8.0 m
For t = 3s, distance = 18.0 m
Now, determine the falling distance for t = 4s"""
    equation_prompt = """In Alice's Wonderland, a secret set of transformation rules is applied to equations.
6 @ 3 = 2
8 @ 4 = 2
Now, determine the result for: 10 @ 5"""

    assert parse_competition_prompt("u", unit_prompt, "15.0").family == "unit_conversion"
    assert parse_competition_prompt("g", gravity_prompt, "32.0").family == "gravity_numeric"
    assert parse_competition_prompt("e", equation_prompt, "2").family == "equation_symbolic"


def test_answer_not_used_for_parsing() -> None:
    problem = parse_competition_prompt("r", ROMAN_PROMPT, "I")

    assert problem.target_input == "38"
    assert problem.answer == "I"


def test_malformed_prompt_rejected() -> None:
    with pytest.raises(CompetitionPromptAdapterError):
        parse_competition_prompt("bad", "No examples here", None)


def test_problem_hash_forgery_rejected() -> None:
    problem = parse_competition_prompt("r", ROMAN_PROMPT, "XXXVIII")

    with pytest.raises(CompetitionPromptAdapterError):
        replace(problem, problem_hash="forged")
