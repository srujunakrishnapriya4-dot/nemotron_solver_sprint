from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.adapter_training_data.sft_formatter import SFTFormatterError, format_direct_answer_example, format_family_tagged_example  # noqa: E402
from nemotron_engine.competition_sprint import parse_competition_prompt  # noqa: E402


PROMPT = """In Alice's Wonderland, numbers are secretly converted into a different numeral system.
11 -> XI
15 -> XV
Now, write the number 38 in the Wonderland numeral system."""


def test_formats_direct_answer_boxed() -> None:
    problem = parse_competition_prompt("roman", PROMPT, "XXXVIII")

    row = format_direct_answer_example(problem)

    assert row.messages[0]["role"] == "user"
    assert row.messages[0]["content"].endswith("Return only the final answer in \\boxed{}.")
    assert row.messages[1]["content"] == "\\boxed{XXXVIII}"


def test_formats_family_tagged_user_prefix() -> None:
    problem = parse_competition_prompt("roman", PROMPT, "XXXVIII")

    row = format_family_tagged_example(problem)

    assert row.messages[0]["content"].startswith("Family: roman_numeral.")
    assert row.source == "family_tagged"


def test_requires_answer_for_sft_formatting() -> None:
    problem = parse_competition_prompt("roman", PROMPT, None)

    with pytest.raises(SFTFormatterError):
        format_direct_answer_example(problem)


def test_sft_example_hash_forgery_rejected() -> None:
    problem = parse_competition_prompt("roman", PROMPT, "XXXVIII")
    row = format_direct_answer_example(problem)

    with pytest.raises(SFTFormatterError):
        replace(row, example_hash="forged")
