"""Typed answer-format policies with exact, string-safe normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
import re
from typing import Any

from .answer_extractor import ANSWER_MAX, ANSWER_MIN


_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_DECIMAL_RE = re.compile(r"^[+-]?(?:\d+\.\d+|\.\d+)$")
_FRACTION_RE = re.compile(r"^(?P<num>[+-]?\d+)/(?P<den>[+-]?\d+)$")
_FREE_SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ROMAN_RE = re.compile(r"^[IVXLCDMivxlcdm]+$")


class AnswerFormatError(ValueError):
    """Raised when an answer violates its declared format policy."""


class AnswerType(str, Enum):
    INTEGER = "integer"
    DECIMAL = "decimal"
    FRACTION = "fraction"
    SYMBOL_STRING = "symbol_string"
    CIPHER_STRING = "cipher_string"
    BITSTRING = "bitstring"
    ROMAN_LIKE = "roman_like"
    EXPRESSION = "expression"
    FREE_SYMBOL = "free_symbol"


@dataclass(frozen=True)
class FormatPolicy:
    """Rules for canonicalizing one final answer string."""

    answer_type: AnswerType = AnswerType.INTEGER
    answer_min: int = ANSWER_MIN
    answer_max: int = ANSWER_MAX
    preserve_leading_zeros: bool = False
    decimal_places: int | None = None
    allow_whitespace: bool = False
    preserve_case: bool = True


@dataclass(frozen=True)
class CanonicalAnswer:
    raw: str
    normalized: str
    answer_type: AnswerType


@dataclass(frozen=True)
class FormatCheck:
    valid: bool
    answer: CanonicalAnswer | None = None
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


def normalize_answer(answer: Any, *, policy: FormatPolicy | None = None) -> CanonicalAnswer:
    """Normalize ``answer`` according to ``policy`` without unsafe float coercion."""

    resolved = policy or FormatPolicy()
    if isinstance(answer, bool):
        raise AnswerFormatError("Boolean answers are invalid.")
    if not isinstance(answer, (str, int)):
        raise AnswerFormatError(f"Unsupported answer type: {type(answer)!r}.")

    raw = str(answer).strip()
    if raw == "":
        raise AnswerFormatError("Answer cannot be empty.")

    normalizers = {
        AnswerType.INTEGER: _normalize_integer,
        AnswerType.DECIMAL: _normalize_decimal,
        AnswerType.FRACTION: _normalize_fraction,
        AnswerType.SYMBOL_STRING: _normalize_symbol_or_cipher,
        AnswerType.CIPHER_STRING: _normalize_symbol_or_cipher,
        AnswerType.BITSTRING: _normalize_bitstring,
        AnswerType.ROMAN_LIKE: _normalize_roman_like,
        AnswerType.EXPRESSION: _normalize_expression,
        AnswerType.FREE_SYMBOL: _normalize_free_symbol,
    }
    normalized = normalizers[resolved.answer_type](raw, resolved)
    return CanonicalAnswer(raw=raw, normalized=normalized, answer_type=resolved.answer_type)


def validate_answer(answer: Any, *, policy: FormatPolicy | None = None) -> CanonicalAnswer:
    """Validate and return the canonical form, raising on policy violations."""

    return normalize_answer(answer, policy=policy)


def render_final_answer(answer: Any, *, policy: FormatPolicy | None = None) -> str:
    canonical = normalize_answer(answer, policy=policy)
    return rf"\boxed{{{canonical.normalized}}}"


def validate_response_format(text: str, *, policy: FormatPolicy | None = None) -> FormatCheck:
    from .answer_extractor import AnswerExtractionError, extract_boxed_answer

    try:
        extracted = extract_boxed_answer(text)
        answer = normalize_answer(extracted.raw, policy=policy)
    except (AnswerExtractionError, AnswerFormatError) as exc:
        return FormatCheck(valid=False, answer=None, errors=(str(exc),))
    return FormatCheck(valid=True, answer=answer)


def is_valid_response_format(text: str, *, policy: FormatPolicy | None = None) -> bool:
    return validate_response_format(text, policy=policy).valid


def _normalize_integer(raw: str, policy: FormatPolicy) -> str:
    if not _INTEGER_RE.fullmatch(raw):
        raise AnswerFormatError(f"Invalid integer answer: {raw!r}.")
    sign = "-" if raw.startswith("-") else ""
    digits = raw[1:] if raw[0] in "+-" else raw
    if policy.preserve_leading_zeros:
        normalized = sign + digits
    else:
        value = int(raw)
        normalized = str(value)
    value = int(raw)
    if value < policy.answer_min or value > policy.answer_max:
        raise AnswerFormatError(
            f"Integer answer {value} is outside allowed range [{policy.answer_min}, {policy.answer_max}]."
        )
    return normalized


def _normalize_decimal(raw: str, policy: FormatPolicy) -> str:
    if not _DECIMAL_RE.fullmatch(raw):
        raise AnswerFormatError(f"Invalid decimal answer: {raw!r}.")
    try:
        Decimal(raw)
    except InvalidOperation as exc:
        raise AnswerFormatError(f"Invalid decimal answer: {raw!r}.") from exc

    fraction = raw.rsplit(".", 1)[1]
    if policy.decimal_places is not None and len(fraction) != policy.decimal_places:
        raise AnswerFormatError(
            f"Decimal answer must have exactly {policy.decimal_places} places; got {len(fraction)}."
        )

    sign = "-" if raw.startswith("-") else ""
    unsigned = raw[1:] if raw[0] in "+-" else raw
    integer_part, fraction_part = unsigned.split(".", 1)
    integer_part = _strip_leading_zeros(integer_part or "0")
    return f"{sign}{integer_part}.{fraction_part}"


def _normalize_fraction(raw: str, policy: FormatPolicy) -> str:
    match = _FRACTION_RE.fullmatch(raw)
    if not match:
        raise AnswerFormatError(f"Invalid fraction answer: {raw!r}.")
    numerator = _normalize_signed_integer_text(match.group("num"))
    denominator = _normalize_signed_integer_text(match.group("den"))
    if int(denominator) == 0:
        raise AnswerFormatError("Fraction denominator cannot be zero.")
    if denominator.startswith("-"):
        numerator = _normalize_signed_integer_text(str(-int(numerator)))
        denominator = denominator[1:]
    return f"{numerator}/{denominator}"


def _normalize_bitstring(raw: str, policy: FormatPolicy) -> str:
    if not raw or any(char not in {"0", "1"} for char in raw):
        raise AnswerFormatError(f"Invalid bitstring answer: {raw!r}.")
    return raw


def _normalize_symbol_or_cipher(raw: str, policy: FormatPolicy) -> str:
    if not policy.allow_whitespace and any(char.isspace() for char in raw):
        raise AnswerFormatError("Whitespace is not allowed in symbol/cipher answers.")
    return raw if policy.preserve_case else raw.lower()


def _normalize_roman_like(raw: str, policy: FormatPolicy) -> str:
    if not _ROMAN_RE.fullmatch(raw):
        raise AnswerFormatError(f"Invalid roman-like answer: {raw!r}.")
    return raw if policy.preserve_case else raw.upper()


def _normalize_expression(raw: str, policy: FormatPolicy) -> str:
    if not policy.allow_whitespace and raw != raw.strip():
        raise AnswerFormatError("Expression answer cannot have outer whitespace.")
    if raw == "":
        raise AnswerFormatError("Expression answer cannot be empty.")
    return raw


def _normalize_free_symbol(raw: str, policy: FormatPolicy) -> str:
    if not _FREE_SYMBOL_RE.fullmatch(raw):
        raise AnswerFormatError(f"Invalid free-symbol answer: {raw!r}.")
    return raw if policy.preserve_case else raw.lower()


def _normalize_signed_integer_text(raw: str) -> str:
    sign = "-" if raw.startswith("-") else ""
    digits = raw[1:] if raw[0] in "+-" else raw
    normalized = _strip_leading_zeros(digits)
    return "0" if normalized == "0" else sign + normalized


def _strip_leading_zeros(digits: str) -> str:
    stripped = digits.lstrip("0")
    return stripped or "0"


__all__ = [
    "AnswerFormatError",
    "AnswerType",
    "CanonicalAnswer",
    "FormatCheck",
    "FormatPolicy",
    "is_valid_response_format",
    "normalize_answer",
    "render_final_answer",
    "validate_answer",
    "validate_response_format",
]
