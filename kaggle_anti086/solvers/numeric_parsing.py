from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import re


NUMBER = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
PAIR_RE = re.compile(
    rf"({NUMBER})\s*(?:->|=|converts?\s+to|becomes|maps?\s+to|output\s+is)\s*({NUMBER})",
    re.IGNORECASE,
)
QUERY_PATTERNS = (
    re.compile(rf"({NUMBER})\s*(?:->|=|converts?\s+to|output\s+is)\s*\?", re.IGNORECASE),
    re.compile(rf"\b(?:convert|input|query|now\s+solve|solve|target|for)\s*[:#]?\s*({NUMBER})\b(?=[^0-9.\-+]*(?:\?|output|result|unit|$))", re.IGNORECASE),
    re.compile(rf"\bwhat\s+is\s+({NUMBER})\b", re.IGNORECASE),
    re.compile(rf"\bfor\s+({NUMBER})\s*,?\s*output\s*\?", re.IGNORECASE),
)


@dataclass(frozen=True)
class RawNumeric:
    raw: str
    value: Decimal
    decimal_places: int


@dataclass(frozen=True)
class RawNumericPair:
    x: RawNumeric
    y: RawNumeric


def parse_raw_numeric(value: str) -> RawNumeric:
    raw = str(value).strip()
    if not re.fullmatch(NUMBER, raw):
        raise ValueError(f"not a numeric token: {value!r}")
    places = _decimal_places(raw)
    return RawNumeric(raw=raw, value=Decimal(raw), decimal_places=places)


def extract_raw_numeric_pairs(prompt: str) -> list[RawNumericPair]:
    return [RawNumericPair(parse_raw_numeric(a), parse_raw_numeric(b)) for a, b in PAIR_RE.findall(prompt or "")]


def extract_raw_query_value(prompt: str, pair_spans: list[tuple[int, int]] | None = None) -> RawNumeric | None:
    text = prompt or ""
    spans = pair_spans if pair_spans is not None else [match.span() for match in PAIR_RE.finditer(text)]
    for pattern in QUERY_PATTERNS:
        for match in reversed(list(pattern.finditer(text))):
            if _inside_any(match.span(1), spans):
                continue
            return parse_raw_numeric(match.group(1))
    if spans:
        tail = text[max(end for _, end in spans) :]
        if "?" in tail or re.search(r"\b(?:query|solve|input|output)\b", tail, flags=re.IGNORECASE):
            numbers = list(re.finditer(NUMBER, tail))
            if numbers:
                return parse_raw_numeric(numbers[-1].group(0))
    return None


def infer_output_precision_from_raw_pairs(pairs: list[RawNumericPair]) -> int:
    if not pairs:
        return 0
    return max(pair.y.decimal_places for pair in pairs)


def format_decimal(value: Decimal | float, precision: int) -> str:
    decimal = value if isinstance(value, Decimal) else Decimal(str(value))
    if precision < 0:
        raise ValueError("precision must be non-negative")
    quant = Decimal(1).scaleb(-precision)
    rounded = decimal.quantize(quant, rounding=ROUND_HALF_UP)
    if precision == 0:
        return str(rounded.quantize(Decimal(1)))
    return format(rounded, f".{precision}f")


def pair_spans(prompt: str) -> list[tuple[int, int]]:
    return [match.span() for match in PAIR_RE.finditer(prompt or "")]


def _decimal_places(raw: str) -> int:
    mantissa = raw.lower().split("e", 1)[0]
    if "." not in mantissa:
        return 0
    return len(mantissa.split(".", 1)[1])


def _inside_any(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(parent_start <= start and end <= parent_end for parent_start, parent_end in spans)
