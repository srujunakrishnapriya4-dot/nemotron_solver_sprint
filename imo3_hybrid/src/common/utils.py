"""Deterministic foundational utilities for IDs, serialization, and canonicalization.

This module is intentionally narrow:
- text canonicalization used by shared identity and hashing code
- deterministic JSON-safe serialization
- stable hashing / fingerprinting / reproducible ID derivation
- stable ordering helpers for reproducible artifacts and state identity

It must remain pure, low-level, and free of domain-specific runtime behavior.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time, timezone
from decimal import Decimal
from enum import Enum
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import unicodedata


_WHITESPACE_RE = re.compile(r"\s+")
_LATEX_DELIMITER_RE = re.compile(r"\\[\[\]()]")
_LATEX_SPACE_RE = re.compile(r"\\[,;:! ]+")


def normalize_unicode(text: str | None, *, form: str = "NFKC") -> str:
    return unicodedata.normalize(form, str(text or ""))


def normalize_whitespace(text: str | None) -> str:
    return _WHITESPACE_RE.sub(" ", normalize_unicode(text)).strip()


def safe_normalize_text(
    text: Any,
    *,
    lowercase: bool = False,
    keep_newlines: bool = False,
    ascii_only: bool = False,
) -> str:
    value = normalize_unicode("" if text is None else str(text))
    if ascii_only:
        value = value.encode("ascii", errors="ignore").decode("ascii")
    if keep_newlines:
        lines = [normalize_whitespace(line) for line in value.splitlines()]
        normalized = "\n".join(line for line in lines if line)
    else:
        normalized = normalize_whitespace(value)
    return normalized.lower() if lowercase else normalized


def normalize_problem_text(
    text: str | None,
    *,
    lowercase: bool = False,
    strip_latex_delimiters: bool = True,
) -> str:
    normalized = normalize_unicode(text)
    normalized = normalized.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    normalized = normalized.replace("\u00d7", "*").replace("\u00f7", "/")
    normalized = normalized.replace("\u00a0", " ")
    if strip_latex_delimiters:
        normalized = _LATEX_DELIMITER_RE.sub("", normalized)
        normalized = _LATEX_SPACE_RE.sub(" ", normalized)
    normalized = normalize_whitespace(normalized)
    return normalized.lower() if lowercase else normalized


def strip_wrapping_delimiters(
    text: str | None,
    *,
    pairs: Sequence[tuple[str, str]] | None = None,
) -> str:
    value = normalize_whitespace(text)
    if not value:
        return ""
    delimiters = tuple(pairs or (("(", ")"), ("[", "]"), ("{", "}")))
    changed = True
    while changed and value:
        changed = False
        for left, right in delimiters:
            if value.startswith(left) and value.endswith(right) and len(value) >= len(left) + len(right):
                candidate = value[len(left): len(value) - len(right)].strip()
                if candidate:
                    value = candidate
                    changed = True
    return value


def format_utc_timestamp(value: datetime | date | time) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    if isinstance(value, time):
        return value.isoformat(timespec="milliseconds")
    return value.isoformat()


def stable_json_dumps(payload: Any, *, ensure_ascii: bool = False, indent: int | None = None) -> str:
    return json.dumps(
        to_json_compatible(payload),
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        ensure_ascii=ensure_ascii,
        indent=indent,
        allow_nan=False,
    )


def safe_json_loads(text: str | bytes | bytearray | None, *, default: Any = None) -> Any:
    if text is None:
        return default
    try:
        raw = text.decode("utf-8") if isinstance(text, (bytes, bytearray)) else str(text)
        raw = raw.strip()
        if not raw:
            return default
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
        return default


def to_json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, (datetime, date, time)):
        return format_utc_timestamp(value)
    if isinstance(value, Enum):
        return to_json_compatible(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Decimal):
        return _canonical_numeric_text(value)
    if isinstance(value, Fraction):
        return _canonical_numeric_text(value)
    if is_dataclass(value):
        return to_json_compatible(asdict(value))
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return to_json_compatible(value.model_dump(mode="json"))
        except Exception:
            return repr(value)
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return to_json_compatible(value.dict())
        except Exception:
            return repr(value)
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return to_json_compatible(value.tolist())
        except Exception:
            return repr(value)
    if isinstance(value, Mapping):
        ordered_keys = sorted(value.keys(), key=stable_sort_key)
        return {str(key): to_json_compatible(value[key]) for key in ordered_keys}
    if isinstance(value, (list, tuple)):
        return [to_json_compatible(item) for item in value]
    if isinstance(value, (set, frozenset)):
        ordered_items = sorted(value, key=stable_sort_key)
        return [to_json_compatible(item) for item in ordered_items]
    if hasattr(value, "__dict__"):
        public = {
            str(key): val
            for key, val in vars(value).items()
            if not str(key).startswith("_")
        }
        if public:
            return to_json_compatible(public)
    return repr(value)


def stable_sort_key(value: Any) -> tuple[str, str]:
    if value is None:
        return ("0:none", "")
    if isinstance(value, bool):
        return ("1:bool", "1" if value else "0")
    if isinstance(value, (int, float, Decimal, Fraction)):
        return ("2:number", _canonical_numeric_text(value))
    if isinstance(value, str):
        return ("3:text", safe_normalize_text(value, lowercase=True))
    if isinstance(value, Mapping):
        return ("4:mapping", stable_json_dumps(value))
    if isinstance(value, (set, frozenset)):
        ordered_items = sorted(value, key=stable_sort_key)
        return ("5:set", stable_json_dumps(ordered_items))
    if isinstance(value, (list, tuple)):
        return ("6:sequence", stable_json_dumps(value))
    return ("7:repr", safe_normalize_text(repr(value), lowercase=True))


def ordered_mapping(mapping: Mapping[Any, Any]) -> dict[str, Any]:
    return {str(key): mapping[key] for key in sorted(mapping.keys(), key=stable_sort_key)}


def stable_hash(prefix: str, payload: Any, *, digest_size: int = 16) -> str:
    normalized_prefix = safe_normalize_text(prefix, lowercase=True).replace(" ", "_")
    raw = f"{normalized_prefix}::{stable_json_dumps(payload)}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[: max(1, int(digest_size))]
    return f"{normalized_prefix}_{digest}"


def stable_id(namespace: str, *components: Any, payload: Any = None, digest_size: int = 16) -> str:
    material: dict[str, Any] = {}
    if components:
        material["components"] = list(components)
    if payload is not None:
        material["payload"] = payload
    return stable_hash(namespace, material or namespace, digest_size=digest_size)


def derive_stable_seed(namespace: str, payload: Any = None, *, modulo: int = 2**32 - 1) -> int:
    digest = stable_hash(namespace, payload if payload is not None else namespace, digest_size=16)
    modulus = max(1, int(modulo))
    return int(hashlib.sha1(digest.encode("utf-8")).hexdigest()[:16], 16) % modulus


def text_fingerprint(text: str | None, *, prefix: str = "text", digest_size: int = 16) -> str:
    normalized = normalize_problem_text(text)
    return stable_hash(prefix, {"text": normalized}, digest_size=digest_size)


def object_fingerprint(payload: Any, *, prefix: str = "object", digest_size: int = 16) -> str:
    return stable_hash(prefix, payload, digest_size=digest_size)


def _canonical_numeric_text(value: int | float | Decimal | Fraction) -> str:
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return str(value.numerator)
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, Decimal):
        normalized = value.normalize()
        if normalized == normalized.to_integral():
            return str(normalized.quantize(Decimal("1")))
        return format(normalized, "f").rstrip("0").rstrip(".")
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if value.is_integer():
            return str(int(value))
        return format(value, ".15g")
    return str(value)


__all__ = [
    "derive_stable_seed",
    "format_utc_timestamp",
    "normalize_problem_text",
    "normalize_unicode",
    "normalize_whitespace",
    "object_fingerprint",
    "ordered_mapping",
    "safe_json_loads",
    "safe_normalize_text",
    "stable_hash",
    "stable_id",
    "stable_json_dumps",
    "stable_sort_key",
    "strip_wrapping_delimiters",
    "text_fingerprint",
    "to_json_compatible",
]
