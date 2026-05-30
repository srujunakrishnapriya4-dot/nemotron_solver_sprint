from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.scoring.format_policy import (  # noqa: E402
    AnswerFormatError,
    AnswerType,
    FormatPolicy,
    normalize_answer,
    render_final_answer,
    validate_response_format,
)


def test_integer_plus_normalizes_and_integer_decimal_is_rejected() -> None:
    assert normalize_answer("+3", policy=FormatPolicy(answer_type=AnswerType.INTEGER)).normalized == "3"

    with pytest.raises(AnswerFormatError, match="Invalid integer"):
        normalize_answer("3.0", policy=FormatPolicy(answer_type=AnswerType.INTEGER))


def test_integer_leading_zero_policy_is_explicit() -> None:
    assert normalize_answer("003", policy=FormatPolicy(answer_type=AnswerType.INTEGER)).normalized == "3"
    assert (
        normalize_answer(
            "003",
            policy=FormatPolicy(answer_type=AnswerType.INTEGER, preserve_leading_zeros=True),
        ).normalized
        == "003"
    )


def test_decimal_validation_is_exact_and_enforces_places() -> None:
    policy = FormatPolicy(answer_type=AnswerType.DECIMAL, decimal_places=2)

    assert normalize_answer("+003.10", policy=policy).normalized == "3.10"

    with pytest.raises(AnswerFormatError, match="exactly 2 places"):
        normalize_answer("3.1", policy=policy)

    with pytest.raises(AnswerFormatError, match="Invalid decimal"):
        normalize_answer("3..10", policy=policy)


def test_fraction_zero_denominator_is_rejected_without_reducing() -> None:
    assert normalize_answer("06/008", policy=FormatPolicy(answer_type=AnswerType.FRACTION)).normalized == "6/8"

    with pytest.raises(AnswerFormatError, match="denominator"):
        normalize_answer("1/0", policy=FormatPolicy(answer_type=AnswerType.FRACTION))


def test_bitstring_preserves_exact_value_and_rejects_non_binary() -> None:
    assert normalize_answer("00101", policy=FormatPolicy(answer_type=AnswerType.BITSTRING)).normalized == "00101"

    with pytest.raises(AnswerFormatError, match="bitstring"):
        normalize_answer("10201", policy=FormatPolicy(answer_type=AnswerType.BITSTRING))


def test_symbol_and_cipher_reject_whitespace_and_preserve_case() -> None:
    symbol_policy = FormatPolicy(answer_type=AnswerType.SYMBOL_STRING)
    cipher_policy = FormatPolicy(answer_type=AnswerType.CIPHER_STRING)

    assert normalize_answer("AbC_09", policy=symbol_policy).normalized == "AbC_09"
    assert normalize_answer("XaZ", policy=cipher_policy).normalized == "XaZ"

    with pytest.raises(AnswerFormatError, match="Whitespace"):
        normalize_answer("AB CD", policy=symbol_policy)

    with pytest.raises(AnswerFormatError, match="Whitespace"):
        normalize_answer("XY Z", policy=cipher_policy)


def test_expression_is_not_algebraically_simplified() -> None:
    expression = "(x+1)^2"

    assert normalize_answer(expression, policy=FormatPolicy(answer_type=AnswerType.EXPRESSION)).normalized == expression
    assert render_final_answer("007", policy=FormatPolicy(preserve_leading_zeros=True)) == r"\boxed{007}"


def test_response_format_uses_boxed_answer_and_policy_normalization() -> None:
    check = validate_response_format(
        r"Final answer is \boxed{007}.",
        policy=FormatPolicy(preserve_leading_zeros=True),
    )

    assert check.valid is True
    assert check.answer is not None
    assert check.answer.normalized == "007"
