from __future__ import annotations

import re

from kaggle_anti086.solvers.answer_normalizer import normalize_answer
from kaggle_anti086.solvers.base import BaseSolver
from kaggle_anti086.solvers.types import SolverCandidate, SolverResult, validate_candidate


ROMAN_VALUES = (
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
)

EXAMPLE_RE = re.compile(r"\b(\d{1,4})\s*(?:->|=|:)\s*([IVXLCDM]+)\b", re.IGNORECASE)
QUERY_PATTERNS = (
    re.compile(r"\b(?:convert|input|query)\s*[:#]?\s*(\d{1,5})\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+is\s+(\d{1,5})\s+(?:in|as)\b", re.IGNORECASE),
    re.compile(r"\b(?:write|solve\s+for|output\s+for|target\s+number)\s*[:#]?\s*(\d{1,5})\b", re.IGNORECASE),
    re.compile(r"\bgiven\s+the\s+examples\s+above,?\s*(\d{1,5})\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,5})\s*(?:->|=)\s*\?", re.IGNORECASE),
    re.compile(r"\bnumber\s*[:#]?\s*(\d{1,5})\b", re.IGNORECASE),
)
STANDALONE_NUMBER_RE = re.compile(r"\b(\d{1,5})\b")


def int_to_roman(n: int) -> str:
    if n < 1 or n > 3999:
        raise ValueError("Roman numerals only supported for 1..3999")
    remaining = n
    parts: list[str] = []
    for value, numeral in ROMAN_VALUES:
        while remaining >= value:
            parts.append(numeral)
            remaining -= value
    return "".join(parts)


def extract_roman_examples(prompt: str) -> list[tuple[int, str]]:
    examples: list[tuple[int, str]] = []
    for match in EXAMPLE_RE.finditer(prompt or ""):
        number = int(match.group(1))
        numeral = match.group(2).upper()
        examples.append((number, numeral))
    return examples


def extract_roman_query(prompt: str) -> int | None:
    text = prompt or ""
    example_spans = [match.span() for match in EXAMPLE_RE.finditer(text)]
    candidates: list[tuple[int, int]] = []
    for pattern in QUERY_PATTERNS:
        matches = list(pattern.finditer(text))
        for match in matches:
            if _inside_any(match.span(1), example_spans):
                continue
            candidates.append((match.start(1), int(match.group(1))))
    if not candidates and example_spans and _prompt_asks_for_answer(text):
        last_example_end = max(end for _, end in example_spans)
        tail = text[last_example_end:]
        standalone = [match for match in STANDALONE_NUMBER_RE.finditer(tail) if not _inside_any((last_example_end + match.start(1), last_example_end + match.end(1)), example_spans)]
        if standalone:
            match = standalone[-1]
            candidates.append((last_example_end + match.start(1), int(match.group(1))))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[-1][1]


class RomanSolver(BaseSolver):
    @property
    def name(self) -> str:
        return "roman_solver"

    @property
    def supported_families(self) -> tuple[str, ...]:
        return ("roman_numeral", "custom_numeral")

    def solve(self, row: dict) -> SolverResult:
        family = row.get("family", "roman_numeral")
        if family not in self.supported_families:
            return SolverResult(self.name, str(family), [], True, "unsupported_family", {})
        prompt = str(row.get("prompt", ""))
        examples = extract_roman_examples(prompt)
        mismatches = []
        for number, numeral in examples:
            if number < 1 or number > 3999:
                mismatches.append({"number": number, "expected": numeral, "reason": "range"})
                continue
            if int_to_roman(number) != numeral.upper():
                mismatches.append({"number": number, "expected": numeral, "actual": int_to_roman(number)})
        if mismatches:
            return SolverResult(self.name, str(family), [], True, "examples_do_not_match_standard_roman", {"mismatches": mismatches})
        query = extract_roman_query(prompt)
        if query is None:
            return SolverResult(self.name, str(family), [], True, "missing_query", {"example_count": len(examples)})
        if query < 1 or query > 3999:
            return SolverResult(self.name, str(family), [], True, "query_out_of_range", {"query": query})
        answer = int_to_roman(query)
        normalized = normalize_answer(answer, expected_type="roman").normalized
        candidate = SolverCandidate(
            answer=normalized,
            source=self.name,
            family=str(family),
            subfamily="standard_roman",
            confidence=0.98,
            example_consistency=1.0,
            verified=True,
            risk="low",
            metadata={"query": query, "example_count": len(examples)},
        )
        validate_candidate(candidate)
        return SolverResult(self.name, str(family), [candidate], False, "", {"example_count": len(examples)})


def _inside_any(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(parent_start <= start and end <= parent_end for parent_start, parent_end in spans)


def _prompt_asks_for_answer(text: str) -> bool:
    return bool(re.search(r"\b(?:answer|convert|write|solve|output|same\s+system|examples\s+above)\b", text, re.IGNORECASE))
