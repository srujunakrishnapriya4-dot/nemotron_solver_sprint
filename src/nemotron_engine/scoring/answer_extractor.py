"""Strict boxed-answer extraction for final competition outputs."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


ANSWER_MIN = 0
ANSWER_MAX = 99999

_INTEGER_RE = re.compile(r"^[+-]?\d+$")


class AnswerExtractionError(ValueError):
    """Raised when a completion does not contain exactly one valid boxed answer."""


@dataclass(frozen=True)
class BoxedSpan:
    """Character span and raw content for one ``\\boxed{...}`` expression."""

    start: int
    end: int
    content_start: int
    content_end: int
    raw: str


@dataclass(frozen=True)
class ExtractedAnswer:
    """The raw boxed final answer preserved exactly after outer trim."""

    raw: str
    source: str = "boxed"
    start: int | None = None
    end: int | None = None

    @property
    def value(self) -> str:
        return self.raw

    @property
    def normalized(self) -> str:
        return self.raw


def find_boxed_spans(text: str) -> tuple[BoxedSpan, ...]:
    """Find all well-formed boxed spans, rejecting malformed boxed syntax."""

    if text is None:
        raise AnswerExtractionError("Cannot extract an answer from None.")
    body = str(text)
    spans: list[BoxedSpan] = []
    marker = r"\boxed"
    index = 0
    while True:
        start = body.find(marker, index)
        if start < 0:
            return tuple(spans)

        cursor = start + len(marker)
        while cursor < len(body) and body[cursor].isspace():
            cursor += 1
        if cursor >= len(body):
            raise AnswerExtractionError("Malformed boxed answer: missing opening brace.")
        if body[cursor] != "{":
            raise AnswerExtractionError("Malformed boxed answer: expected '{' after \\boxed.")

        depth = 0
        escaped = False
        for pos in range(cursor, len(body)):
            char = body[pos]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    raw = body[cursor + 1 : pos].strip()
                    spans.append(
                        BoxedSpan(
                            start=start,
                            end=pos + 1,
                            content_start=cursor + 1,
                            content_end=pos,
                            raw=raw,
                        )
                    )
                    index = pos + 1
                    break
        else:
            raise AnswerExtractionError("Malformed boxed answer: missing closing brace.")


def has_truncation_risk(text: str) -> bool:
    """Detect obvious incomplete final-answer syntax."""

    if text is None:
        return True
    body = str(text)
    marker = r"\boxed"
    start = body.rfind(marker)
    if start < 0:
        stripped = body.rstrip()
        return stripped.endswith("\\") or stripped.endswith(r"\boxed")

    cursor = start + len(marker)
    while cursor < len(body) and body[cursor].isspace():
        cursor += 1
    if cursor >= len(body) or body[cursor] != "{":
        return True

    depth = 0
    escaped = False
    for pos in range(cursor, len(body)):
        char = body[pos]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return False
    return True


def extract_boxed_answer(text: str) -> ExtractedAnswer:
    """Extract the only valid boxed answer from a completion."""

    if text is None or not str(text).strip():
        raise AnswerExtractionError("Cannot extract an answer from empty text.")
    spans = find_boxed_spans(text)
    if not spans:
        if has_truncation_risk(text):
            raise AnswerExtractionError("No complete boxed final answer found; completion may be truncated.")
        raise AnswerExtractionError("Final answer must appear in exactly one \\boxed{} expression.")
    if len(spans) > 1:
        raise AnswerExtractionError("Multiple boxed answers are not allowed.")

    span = spans[0]
    if span.raw == "":
        raise AnswerExtractionError("Boxed answer cannot be empty.")
    return ExtractedAnswer(raw=span.raw, start=span.start, end=span.end)


def extract_answer(text: str, **_: Any) -> ExtractedAnswer:
    """Backward-compatible alias for strict boxed-answer extraction."""

    return extract_boxed_answer(text)


def extract_final_answer(text: str, **_: Any) -> str | None:
    """Return the raw boxed answer, or ``None`` when strict extraction fails."""

    try:
        return extract_boxed_answer(text).raw
    except AnswerExtractionError:
        return None


def has_valid_answer(text: str, **_: Any) -> bool:
    """Whether ``text`` contains exactly one non-empty boxed answer."""

    return extract_final_answer(text) is not None


def normalize_answer_candidate(
    candidate: Any,
    *,
    answer_min: int = ANSWER_MIN,
    answer_max: int = ANSWER_MAX,
) -> int:
    """Compatibility helper for strict integer submission-row validation."""

    if isinstance(candidate, bool):
        raise AnswerExtractionError("Boolean answers are invalid.")
    if isinstance(candidate, int):
        value = candidate
    elif isinstance(candidate, str):
        text = candidate.strip()
        if not _INTEGER_RE.fullmatch(text):
            raise AnswerExtractionError(f"Unsafe non-integer answer: {candidate!r}")
        value = int(text)
    else:
        raise AnswerExtractionError(f"Unsupported answer type: {type(candidate)!r}")

    if value < answer_min or value > answer_max:
        raise AnswerExtractionError(
            f"Answer {value} is outside allowed range [{answer_min}, {answer_max}]."
        )
    return value


__all__ = [
    "ANSWER_MAX",
    "ANSWER_MIN",
    "AnswerExtractionError",
    "BoxedSpan",
    "ExtractedAnswer",
    "extract_answer",
    "extract_boxed_answer",
    "extract_final_answer",
    "find_boxed_spans",
    "has_truncation_risk",
    "has_valid_answer",
    "normalize_answer_candidate",
]
