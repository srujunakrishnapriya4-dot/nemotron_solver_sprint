from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import re


ROMAN_RE = re.compile(r"\bM{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b", re.IGNORECASE)
BINARY_RE = re.compile(r"\b[01]+\b")
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
PREFIX_RE = re.compile(
    r"^\s*(?:the\s+)?(?:final\s+)?(?:answer|result|output|decrypted\s+text)\s*(?:is)?\s*[:=\-]?\s*",
    re.IGNORECASE,
)
THINK_RE = re.compile(r"</think>\s*", re.IGNORECASE)
NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class NormalizedAnswer:
    raw: str
    normalized: str
    answer_type: str
    confidence: float
    transformations: list[str] = field(default_factory=list)


def normalize_answer(raw: str, expected_type: str | None = None) -> NormalizedAnswer:
    text = "" if raw is None else str(raw)
    transformations: list[str] = []
    answer_type = expected_type or _infer_type(text)
    cleaned = text.strip()

    boxed = BOXED_RE.search(cleaned)
    if boxed:
        cleaned = boxed.group(1).strip()
        transformations.append("boxed")

    cleaned = THINK_RE.sub("", cleaned).strip()
    if cleaned != text.strip():
        transformations.append("strip_think")

    if answer_type == "roman":
        return _normalize_roman(text, cleaned, transformations)
    if answer_type == "binary":
        return _normalize_binary(text, cleaned, transformations)
    if answer_type == "numeric":
        return _normalize_numeric(text, cleaned, transformations)
    if answer_type == "symbol":
        return _normalize_symbol(text, cleaned, transformations)
    if answer_type == "text_phrase":
        return _normalize_text_phrase(text, cleaned, transformations)
    return _normalize_generic(text, cleaned, transformations)


def normalize_gold(gold: str, answer_type: str | None = None) -> str:
    return normalize_answer(gold, expected_type=answer_type).normalized


def answers_match(prediction: str, gold: str, answer_type: str | None = None) -> bool:
    pred = normalize_answer(prediction, expected_type=answer_type)
    gold_norm = normalize_answer(gold, expected_type=answer_type or pred.answer_type)
    if (answer_type == "numeric" or pred.answer_type == "numeric" or gold_norm.answer_type == "numeric") and pred.normalized and gold_norm.normalized:
        try:
            return abs(Decimal(pred.normalized) - Decimal(gold_norm.normalized)) <= Decimal("0.000001")
        except InvalidOperation:
            return False
    return pred.normalized == gold_norm.normalized


def _infer_type(text: str) -> str:
    stripped = _strip_prefixes(text.strip())
    if re.fullmatch(r"[01]{4,}", stripped):
        return "binary"
    if _extract_roman(stripped):
        return "roman"
    if NUMERIC_RE.search(stripped):
        return "numeric"
    if re.search(r"[A-Za-z]", stripped):
        return "text_phrase"
    if stripped:
        return "symbol"
    return "generic"


def _strip_prefixes(text: str) -> str:
    text = THINK_RE.sub("", text).strip()
    text = PREFIX_RE.sub("", text).strip()
    if "->" in text:
        left, right = text.split("->", 1)
        if NUMERIC_RE.fullmatch(left.strip()):
            text = right.strip()
    return text


def _normalize_roman(raw: str, cleaned: str, transformations: list[str]) -> NormalizedAnswer:
    value = _strip_prefixes(cleaned)
    roman = _extract_roman(value)
    if roman:
        value = roman.upper()
        transformations.append("roman_extract")
    else:
        value = _safe_trim(value).upper()
    return NormalizedAnswer(raw=raw, normalized=value, answer_type="roman", confidence=0.95 if roman else 0.5, transformations=transformations)


def _extract_roman(text: str) -> str | None:
    candidates = []
    for match in ROMAN_RE.finditer(text):
        token = match.group(0)
        if not token or not re.fullmatch(r"[IVXLCDM]+", token, flags=re.IGNORECASE):
            continue
        if token.upper() in {"I", "V", "X", "L", "C", "D", "M"} and re.search(r"\b[A-Z]{2,}\b", text) and text.strip().upper() != token.upper():
            continue
        candidates.append(token)
    return candidates[-1] if candidates else None


def _normalize_binary(raw: str, cleaned: str, transformations: list[str]) -> NormalizedAnswer:
    value = _strip_prefixes(cleaned)
    match = BINARY_RE.search(value)
    normalized = match.group(0) if match else _safe_trim(value)
    if match:
        transformations.append("binary_extract")
    return NormalizedAnswer(raw=raw, normalized=normalized, answer_type="binary", confidence=0.95 if match else 0.4, transformations=transformations)


def _normalize_numeric(raw: str, cleaned: str, transformations: list[str]) -> NormalizedAnswer:
    value = _strip_prefixes(cleaned)
    match = NUMERIC_RE.search(value)
    if not match:
        return NormalizedAnswer(raw=raw, normalized=_safe_trim(value), answer_type="numeric", confidence=0.2, transformations=transformations)
    number = match.group(0)
    transformations.append("numeric_extract")
    try:
        normalized = _normalize_decimal(number)
    except InvalidOperation:
        normalized = number
    return NormalizedAnswer(raw=raw, normalized=normalized, answer_type="numeric", confidence=0.95, transformations=transformations)


def _normalize_decimal(value: str) -> str:
    decimal = Decimal(value)
    if decimal == decimal.to_integral():
        return str(decimal.quantize(Decimal(1)))
    return format(decimal.normalize(), "f")


def _normalize_symbol(raw: str, cleaned: str, transformations: list[str]) -> NormalizedAnswer:
    value = _strip_prefixes(cleaned)
    value = value.splitlines()[0].strip()
    value = value.strip("`'\" ")
    value = re.sub(r"[.;:,]+$", "", value)
    transformations.append("symbol_trim")
    return NormalizedAnswer(raw=raw, normalized=value, answer_type="symbol", confidence=0.9 if value else 0.1, transformations=transformations)


def _normalize_text_phrase(raw: str, cleaned: str, transformations: list[str]) -> NormalizedAnswer:
    value = _strip_prefixes(cleaned)
    value = value.splitlines()[0].strip()
    value = re.sub(r"[.;:,]+$", "", value)
    value = " ".join(value.lower().split())
    transformations.append("text_phrase_cleanup")
    return NormalizedAnswer(raw=raw, normalized=value, answer_type="text_phrase", confidence=0.85 if value else 0.1, transformations=transformations)


def _normalize_generic(raw: str, cleaned: str, transformations: list[str]) -> NormalizedAnswer:
    value = _safe_trim(_strip_prefixes(cleaned))
    return NormalizedAnswer(raw=raw, normalized=value, answer_type="generic", confidence=0.6 if value else 0.0, transformations=transformations)


def _safe_trim(value: str) -> str:
    return value.strip().strip("`'\" ").strip()
