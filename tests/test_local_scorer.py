from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.scoring.format_policy import AnswerType, FormatPolicy  # noqa: E402
from nemotron_engine.scoring.local_scorer import accuracy, score_batch, score_completion  # noqa: E402


def test_correct_boxed_answer_passes() -> None:
    result = score_completion(r"Verified final answer: \boxed{42}", "42")

    assert result.correct is True
    assert result.score == 1.0
    assert result.predicted_answer == "42"
    assert result.gold_answer == "42"
    assert result.error is None


def test_wrong_boxed_answer_fails() -> None:
    result = score_completion(r"Verified final answer: \boxed{41}", "42")

    assert result.correct is False
    assert result.score == 0.0
    assert result.error == "Normalized prediction does not match expected answer."


def test_no_box_completion_fails_with_error() -> None:
    result = score_completion("Verified final answer: 42", "42")

    assert result.correct is False
    assert result.error is not None
    assert "boxed" in result.error


def test_multiple_box_completion_fails_with_error() -> None:
    result = score_completion(r"\boxed{17} then \boxed{42}", "42")

    assert result.correct is False
    assert result.error is not None
    assert "Multiple boxed" in result.error


def test_leading_zero_mismatch_fails_when_required() -> None:
    result = score_completion(
        r"Final \boxed{7}",
        "007",
        policy=FormatPolicy(answer_type=AnswerType.INTEGER, preserve_leading_zeros=True),
    )

    assert result.correct is False
    assert result.predicted_answer == "7"
    assert result.gold_answer == "007"


def test_exact_symbol_and_cipher_mismatch_fails() -> None:
    symbol = score_completion(
        r"\boxed{abc}",
        "ABC",
        policy=FormatPolicy(answer_type=AnswerType.SYMBOL_STRING),
    )
    cipher = score_completion(
        r"\boxed{AaZ}",
        "AAZ",
        policy=FormatPolicy(answer_type=AnswerType.CIPHER_STRING),
    )

    assert symbol.correct is False
    assert cipher.correct is False


def test_score_batch_and_accuracy_are_deterministic() -> None:
    results = score_batch(
        [
            {"completion": r"\boxed{1}", "expected_answer": 1},
            {"completion": r"\boxed{2}", "expected_answer": 3},
            {"completion": "bad", "expected_answer": 4},
        ]
    )

    assert [item.score for item in results] == [1.0, 0.0, 0.0]
    assert accuracy(results) == 1 / 3
    assert accuracy([]) == 0.0
