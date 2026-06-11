from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import re
from typing import Callable

from kaggle_anti086.solvers.base import BaseSolver
from kaggle_anti086.solvers.types import SolverCandidate, SolverResult, validate_candidate


INT_RE = r"[-+]?\d+"
PREFIX_RE = re.compile(r"^\[[^\]]+\]\s*")
BRACKET_EXAMPLE_RE = re.compile(rf"\[([^\]]+)\]\s*(?:->|=>|=)\s*({INT_RE})")
BRACKET_QUERY_RE = re.compile(r"\[([^\]]+)\]\s*(?:->|=>|=)?\s*\?")
INDEXED_SEQUENCE_RE = re.compile(r"\bsequence\s+\d+\s*:", re.IGNORECASE)
SAFE_ABS_LIMIT = 1_000_000_000


@dataclass(frozen=True)
class SequenceTask:
    examples: tuple[tuple[tuple[int, ...], int], ...]
    query: tuple[int, ...]
    prompt_kind: str


@dataclass(frozen=True)
class SequenceRule:
    name: str
    fn: Callable[[tuple[int, ...]], int | None]
    min_terms: int


class SequencePatternSolver(BaseSolver):
    @property
    def name(self) -> str:
        return "sequence_pattern_solver"

    @property
    def supported_families(self) -> tuple[str, ...]:
        return ("sequence_pattern",)

    def solve(self, row: dict) -> SolverResult:
        family = str(row.get("family", "unknown"))
        prompt = str(row.get("prompt", "") or "")
        if family != "sequence_pattern" and not _looks_like_sequence_prompt(prompt):
            return self.abstain("not_sequence_pattern")
        task = parse_sequence_pattern_prompt(prompt)
        if task is None:
            return self.abstain("sequence_pattern_parse_failed")
        passing: list[tuple[SequenceRule, int]] = []
        for rule in _candidate_rules(prompt):
            prediction = _verified_prediction(rule, task)
            if prediction is not None:
                passing.append((rule, prediction))
        if not passing:
            return self.abstain("sequence_pattern_no_verified_candidate")
        answers = sorted({value for _rule, value in passing})
        ambiguity_count = max(0, len(passing) - 1)
        if len(answers) > 1:
            return self.abstain("sequence_pattern_ambiguous_different_outputs")
        answer = answers[0]
        if abs(answer) > SAFE_ABS_LIMIT:
            return self.abstain("sequence_pattern_unsafe_extrapolation")
        risk = "medium" if ambiguity_count else "low"
        confidence = 0.74 if ambiguity_count else 0.92
        candidate = SolverCandidate(
            answer=str(answer),
            source=self.name,
            family="sequence_pattern",
            subfamily=task.prompt_kind,
            confidence=confidence,
            example_consistency=1.0,
            verified=True,
            risk=risk,
            metadata={
                "verification_status": "PASS",
                "ambiguity_count": ambiguity_count,
                "rule_names": sorted(rule.name for rule, _value in passing),
                "route_reason": "verified_sequence_pattern",
                "examples_verified": len(task.examples),
            },
        )
        validate_candidate(candidate)
        return SolverResult(
            solver_name=self.name,
            family="sequence_pattern",
            candidates=[candidate],
            abstained=False,
            reason="",
            metadata={"prompt_kind": task.prompt_kind, "ambiguity_count": ambiguity_count},
        )


def parse_sequence_pattern_prompt(prompt: str) -> SequenceTask | None:
    body = PREFIX_RE.sub("", str(prompt or "").strip())
    if not body:
        return None
    # Day2 contains indexed placeholder probes such as
    # "sequence 0: 2, 4, 8, 16, ?" whose expected behavior is ABSTAIN. Keep
    # that surface outside this standalone solver until a safe final policy is
    # proven.
    if INDEXED_SEQUENCE_RE.search(body):
        return None
    bracket_task = _parse_bracket_task(body)
    if bracket_task is not None:
        return bracket_task
    if body.count("?") > 1:
        return None
    segment = _sequence_segment(body)
    if "?" in segment:
        segment = segment[: segment.index("?")]
    values = _parse_ints(segment)
    if len(values) < 3:
        return None
    return SequenceTask(examples=(), query=tuple(values), prompt_kind="single_sequence")


def _parse_bracket_task(body: str) -> SequenceTask | None:
    examples: list[tuple[tuple[int, ...], int]] = []
    for match in BRACKET_EXAMPLE_RE.finditer(body):
        seq = tuple(_parse_ints(match.group(1)))
        if len(seq) < 3:
            return None
        examples.append((seq, int(match.group(2))))
    if not examples:
        return None
    query_matches = [match for match in BRACKET_QUERY_RE.finditer(body) if not any(ex.start() <= match.start() < ex.end() for ex in BRACKET_EXAMPLE_RE.finditer(body))]
    if not query_matches:
        return None
    query = tuple(_parse_ints(query_matches[-1].group(1)))
    if len(query) < 3:
        return None
    return SequenceTask(examples=tuple(examples), query=query, prompt_kind="example_to_query_sequence")


def _sequence_segment(body: str) -> str:
    markers = list(re.finditer(r"\b(?:sequence|series|next|query)\b\s*:?", body, flags=re.IGNORECASE))
    if markers:
        segment = body[markers[-1].end() :]
        if segment.lstrip().startswith(":"):
            segment = segment[segment.index(":") + 1 :]
        return segment
    return body


def _parse_ints(text: str) -> list[int]:
    return [int(match.group(0)) for match in re.finditer(INT_RE, text)]


def _looks_like_sequence_prompt(prompt: str) -> bool:
    return bool(re.search(r"\b(?:sequence|series|next)\b", prompt, re.IGNORECASE)) and "?" in prompt


def _candidate_rules(prompt: str) -> list[SequenceRule]:
    rules = [
        SequenceRule("constant", _constant_next, 3),
        SequenceRule("arithmetic", _arithmetic_next, 3),
        SequenceRule("geometric", _geometric_next, 3),
        SequenceRule("constant_second_difference", _quadratic_next, 3),
        SequenceRule("alternating_interleaved_arithmetic", _alternating_arithmetic_next, 5),
        SequenceRule("periodic_cycle", _periodic_next, 5),
        SequenceRule("fibonacci_like", _fibonacci_next, 4),
        SequenceRule("affine_recurrence", _affine_recurrence_next, 4),
        SequenceRule("two_term_linear_recurrence", _two_term_linear_next, 5),
    ]
    lowered = prompt.lower()
    if "digit" in lowered:
        rules.extend(
            [
                SequenceRule("digit_sum", _digit_sum_next, 3),
                SequenceRule("reverse_digits", _reverse_digits_next, 3),
            ]
        )
    return rules


def _verified_prediction(rule: SequenceRule, task: SequenceTask) -> int | None:
    if len(task.query) < rule.min_terms:
        return None
    for seq, expected in task.examples:
        if len(seq) < rule.min_terms:
            return None
        predicted = rule.fn(seq)
        if predicted is None or predicted != expected or abs(predicted) > SAFE_ABS_LIMIT:
            return None
    predicted = rule.fn(task.query)
    if predicted is None or abs(predicted) > SAFE_ABS_LIMIT:
        return None
    return predicted


def _constant_next(seq: tuple[int, ...]) -> int | None:
    if len(set(seq)) == 1:
        return seq[-1]
    return None


def _arithmetic_next(seq: tuple[int, ...]) -> int | None:
    diffs = [b - a for a, b in zip(seq, seq[1:])]
    if len(set(diffs)) == 1:
        return seq[-1] + diffs[-1]
    return None


def _geometric_next(seq: tuple[int, ...]) -> int | None:
    if any(value == 0 for value in seq[:-1]):
        return None
    ratios = [Fraction(b, a) for a, b in zip(seq, seq[1:])]
    if len(set(ratios)) != 1:
        return None
    value = Fraction(seq[-1]) * ratios[-1]
    if value.denominator != 1:
        return None
    return int(value)


def _quadratic_next(seq: tuple[int, ...]) -> int | None:
    diffs = [b - a for a, b in zip(seq, seq[1:])]
    if len(diffs) < 2:
        return None
    second = [b - a for a, b in zip(diffs, diffs[1:])]
    if len(set(second)) == 1:
        return seq[-1] + diffs[-1] + second[-1]
    return None


def _alternating_arithmetic_next(seq: tuple[int, ...]) -> int | None:
    evens = seq[0::2]
    odds = seq[1::2]
    if len(evens) < 2 or len(odds) < 2:
        return None
    even_diffs = [b - a for a, b in zip(evens, evens[1:])]
    odd_diffs = [b - a for a, b in zip(odds, odds[1:])]
    if len(set(even_diffs)) != 1 or len(set(odd_diffs)) != 1:
        return None
    if len(seq) % 2 == 0:
        return evens[-1] + even_diffs[-1]
    return odds[-1] + odd_diffs[-1]


def _periodic_next(seq: tuple[int, ...]) -> int | None:
    n = len(seq)
    for period in range(1, n // 2 + 1):
        if all(seq[index] == seq[index % period] for index in range(n)):
            return seq[n % period]
    return None


def _fibonacci_next(seq: tuple[int, ...]) -> int | None:
    for index in range(2, len(seq)):
        if seq[index] != seq[index - 1] + seq[index - 2]:
            return None
    return seq[-1] + seq[-2]


def _affine_recurrence_next(seq: tuple[int, ...]) -> int | None:
    matches: list[int] = []
    for p in range(-10, 11):
        for q in range(-20, 21):
            if all(seq[index] == p * seq[index - 1] + q for index in range(1, len(seq))):
                matches.append(p * seq[-1] + q)
    return _unique(matches)


def _two_term_linear_next(seq: tuple[int, ...]) -> int | None:
    matches: list[int] = []
    for a in range(-5, 6):
        for b in range(-5, 6):
            if all(seq[index] == a * seq[index - 1] + b * seq[index - 2] for index in range(2, len(seq))):
                matches.append(a * seq[-1] + b * seq[-2])
    return _unique(matches)


def _digit_sum_next(seq: tuple[int, ...]) -> int | None:
    expected = [_digit_sum(value) for value in seq[:-1]]
    if expected == list(seq[1:]):
        return _digit_sum(seq[-1])
    return None


def _reverse_digits_next(seq: tuple[int, ...]) -> int | None:
    expected = [_reverse_digits(value) for value in seq[:-1]]
    if expected == list(seq[1:]):
        return _reverse_digits(seq[-1])
    return None


def _digit_sum(value: int) -> int:
    return sum(int(char) for char in str(abs(value)))


def _reverse_digits(value: int) -> int:
    sign = -1 if value < 0 else 1
    return sign * int(str(abs(value))[::-1])


def _unique(values: list[int]) -> int | None:
    distinct = sorted(set(values))
    return distinct[0] if len(distinct) == 1 else None
