"""Multi-format answer policy for the Nemotron competition sprint."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import math
import re
from typing import Any


class CompetitionAnswerPolicyError(ValueError):
    """Raised when a competition answer is unsafe or unsupported."""


_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_DECIMAL_RE = re.compile(r"^[+-]?\d+\.\d+$")
_BITSTRING_RE = re.compile(r"^[01]+$")
_ROMAN_RE = re.compile(r"^[IVXLCDM]+$")
_LOWER_TEXT_RE = re.compile(r"^[a-z]+(?: [a-z]+)*$")
_SYMBOL_RE = re.compile(r"^[^\w\s]+$")
_LEAKAGE_RE = re.compile(
    r"\b(expected[_\s]+answer|gold[_\s]+answer|target[_\s]+answer|correct[_\s]+answer)\s*[:=]|\bcorrect\s+answer\s*:",
    re.IGNORECASE,
)


def normalize_competition_answer(answer: Any) -> str:
    """Strip surrounding whitespace while preserving answer representation."""

    if isinstance(answer, bool) or answer is None:
        raise CompetitionAnswerPolicyError("answer must be a non-empty finite value.")
    if isinstance(answer, float):
        if not math.isfinite(answer):
            raise CompetitionAnswerPolicyError("NaN/inf answers are invalid.")
        text = str(answer)
    elif isinstance(answer, int):
        text = str(answer)
    elif isinstance(answer, Decimal):
        if not answer.is_finite():
            raise CompetitionAnswerPolicyError("NaN/inf answers are invalid.")
        text = format(answer, "f")
    else:
        text = str(answer)
    text = text.strip()
    if not text:
        raise CompetitionAnswerPolicyError("answer cannot be empty.")
    if text.lower() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
        raise CompetitionAnswerPolicyError("NaN/inf answers are invalid.")
    if _LEAKAGE_RE.search(text):
        raise CompetitionAnswerPolicyError("answer contains forbidden leakage marker.")
    if _DECIMAL_RE.fullmatch(text):
        try:
            Decimal(text)
        except InvalidOperation as exc:
            raise CompetitionAnswerPolicyError("decimal answer is invalid.") from exc
    return text


def infer_answer_kind(answer: str) -> str:
    """Infer the narrowest supported answer kind."""

    text = normalize_competition_answer(answer)
    if _BITSTRING_RE.fullmatch(text):
        return "bitstring"
    if _INTEGER_RE.fullmatch(text):
        return "integer"
    if _DECIMAL_RE.fullmatch(text):
        return "decimal"
    if _ROMAN_RE.fullmatch(text):
        return "roman"
    if _LOWER_TEXT_RE.fullmatch(text):
        return "lowercase_text"
    if _SYMBOL_RE.fullmatch(text):
        return "symbol_string"
    return "generic_string"


def validate_competition_answer(answer: Any, *, expected_kind: str | None = None) -> str:
    """Validate and return a normalized competition answer string."""

    text = normalize_competition_answer(answer)
    kind = infer_answer_kind(text)
    if expected_kind is not None and kind != expected_kind:
        raise CompetitionAnswerPolicyError(f"expected {expected_kind}, got {kind}.")
    return text


__all__ = [
    "CompetitionAnswerPolicyError",
    "infer_answer_kind",
    "normalize_competition_answer",
    "validate_competition_answer",
]
