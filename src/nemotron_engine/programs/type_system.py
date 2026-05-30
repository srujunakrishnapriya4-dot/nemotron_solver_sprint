"""Conservative type helpers for the Pass 3 solver core."""

from __future__ import annotations

import re

from nemotron_engine.core.schemas import DomainKind, DomainSignature, SchemaValidationError


_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_DECIMAL_RE = re.compile(r"^[+-]?(?:\d+\.\d+|\.\d+)$")
_FRACTION_RE = re.compile(r"^[+-]?\d+/[+-]?\d+$")
_SYMBOL_RE = re.compile(r"^[A-Za-z]+$")
_CIPHER_RE = re.compile(r"^[^\s]+$")


def is_integer_string(value: str) -> bool:
    return isinstance(value, str) and bool(_INTEGER_RE.fullmatch(value.strip()))


def is_decimal_string(value: str) -> bool:
    return isinstance(value, str) and bool(_DECIMAL_RE.fullmatch(value.strip()))


def is_fraction_string(value: str) -> bool:
    return isinstance(value, str) and bool(_FRACTION_RE.fullmatch(value.strip()))


def is_bitstring(value: str) -> bool:
    text = value.strip() if isinstance(value, str) else ""
    return bool(text) and set(text) <= {"0", "1"}


def is_digit_sequence(value: str) -> bool:
    text = value.strip() if isinstance(value, str) else ""
    return bool(text) and text.isdigit()


def is_symbol_sequence(value: str) -> bool:
    text = value.strip() if isinstance(value, str) else ""
    return bool(text) and bool(_SYMBOL_RE.fullmatch(text))


def is_cipher_sequence(value: str) -> bool:
    text = value.strip() if isinstance(value, str) else ""
    return bool(text) and bool(_CIPHER_RE.fullmatch(text)) and not text.isalnum()


def infer_value_domain(value: str) -> DomainSignature:
    text = value.strip() if isinstance(value, str) else ""
    if is_decimal_string(text):
        return DomainSignature(DomainKind.DECIMAL, text, text)
    if is_fraction_string(text):
        return DomainSignature(DomainKind.FRACTION, text, text)
    if is_bitstring(text) and len(text) > 1:
        return DomainSignature(DomainKind.BITSTRING, text, text)
    if is_digit_sequence(text):
        return DomainSignature(DomainKind.DIGIT_SEQUENCE, text, text)
    if is_integer_string(text):
        return DomainSignature(DomainKind.INTEGER, text, str(int(text)))
    if is_symbol_sequence(text):
        return DomainSignature(DomainKind.SYMBOL_SEQUENCE, text, text)
    if any(op in text for op in (" + ", " - ", " * ", " @ ", " ? ")):
        return DomainSignature(DomainKind.EQUATION, text, text)
    if " " in text and all(part for part in text.split()):
        return DomainSignature(DomainKind.TOKEN_SEQUENCE, text, text)
    if is_cipher_sequence(text):
        return DomainSignature(DomainKind.CIPHER_SEQUENCE, text, text)
    return DomainSignature(DomainKind.UNKNOWN, text, text)


def domains_compatible(actual: DomainKind | DomainSignature, expected: DomainKind | DomainSignature) -> bool:
    actual_kind = actual.kind if isinstance(actual, DomainSignature) else DomainKind(actual)
    expected_kind = expected.kind if isinstance(expected, DomainSignature) else DomainKind(expected)
    if actual_kind is expected_kind:
        return True
    if expected_kind is DomainKind.SYMBOL_SEQUENCE and actual_kind is DomainKind.CIPHER_SEQUENCE:
        return False
    return False


def require_domain(value: str, expected: DomainKind) -> DomainSignature:
    inferred = infer_value_domain(value)
    if not domains_compatible(inferred, expected):
        raise SchemaValidationError(f"Expected domain {expected.value}, got {inferred.kind.value}.")
    return inferred


__all__ = [
    "domains_compatible",
    "infer_value_domain",
    "is_bitstring",
    "is_cipher_sequence",
    "is_decimal_string",
    "is_digit_sequence",
    "is_fraction_string",
    "is_integer_string",
    "is_symbol_sequence",
    "require_domain",
]
