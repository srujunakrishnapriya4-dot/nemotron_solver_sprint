from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.scoring.answer_extractor import (  # noqa: E402
    AnswerExtractionError,
    extract_boxed_answer,
    extract_final_answer,
    find_boxed_spans,
    has_truncation_risk,
)


def test_no_box_is_rejected_and_answer_outside_box_is_ignored() -> None:
    with pytest.raises(AnswerExtractionError, match="boxed"):
        extract_boxed_answer("The answer is 42.")

    assert extract_final_answer("The answer is 42.") is None


def test_multiple_boxes_are_rejected() -> None:
    with pytest.raises(AnswerExtractionError, match="Multiple boxed"):
        extract_boxed_answer(r"First \boxed{17}, corrected \boxed{42}.")


def test_empty_box_is_rejected() -> None:
    with pytest.raises(AnswerExtractionError, match="empty"):
        extract_boxed_answer(r"The answer is \boxed{   }.")


def test_malformed_and_truncated_boxes_are_rejected() -> None:
    with pytest.raises(AnswerExtractionError, match="missing closing brace"):
        extract_boxed_answer(r"The answer is \boxed{42")

    with pytest.raises(AnswerExtractionError, match="expected"):
        extract_boxed_answer(r"The answer is \boxed 42}.")

    assert has_truncation_risk(r"The answer is \boxed{42") is True


def test_nested_braces_are_handled_deterministically() -> None:
    result = extract_boxed_answer(r"Therefore \boxed{a_{1}+b}.")

    assert result.raw == "a_{1}+b"
    assert result.value == "a_{1}+b"
    assert result.source == "boxed"


def test_raw_boxed_value_and_span_are_preserved() -> None:
    text = r"Work... final \boxed{007}."
    result = extract_boxed_answer(text)
    spans = find_boxed_spans(text)

    assert result.raw == "007"
    assert result.normalized == "007"
    assert len(spans) == 1
    assert text[spans[0].start : spans[0].end] == r"\boxed{007}"
