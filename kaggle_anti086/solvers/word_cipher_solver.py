from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
import re

from kaggle_anti086.solvers.answer_normalizer import normalize_answer
from kaggle_anti086.solvers.base import BaseSolver
from kaggle_anti086.solvers.types import SolverCandidate, SolverResult, validate_candidate


PAIR_PATTERNS = (
    re.compile(r'"([^"]+)"\s*(?:->|=)\s*"([^"]+)"'),
    re.compile(r"`([^`]+)`\s*(?:->|=)\s*`([^`]+)`"),
    re.compile(r"encrypted\s*:\s*([^;\n]+?)\s+plaintext\s*:\s*([a-zA-Z][^;\n?]*)", re.IGNORECASE),
    re.compile(r"cipher\s*:\s*([^;\n]+?)\s+plain(?:text)?\s*:\s*([a-zA-Z][^;\n?]*)", re.IGNORECASE),
    re.compile(r"([a-z][a-z\s]{2,}?)\s*(?:->|=)\s*([a-z][a-z\s]{2,})(?=(?:\n|;|$))", re.IGNORECASE),
)
QUERY_PATTERNS = (
    re.compile(r"encrypted\s*:\s*([^;\n?]+?)\s+plaintext\s*:\s*\?", re.IGNORECASE),
    re.compile(r"decode\s*[:#]?\s*['\"]?([^'\"\n?]+?)['\"]?\s*(?:\?|$)", re.IGNORECASE),
    re.compile(r"([a-z][a-z\s]{2,}?)\s*(?:->|=)\s*\?", re.IGNORECASE),
)


@dataclass(frozen=True)
class MappingResult:
    mapping: MappingProxyType
    conflict: bool
    conflicts: tuple[dict, ...]
    used_pairs: int
    ignored_pairs: int


def extract_phrase_pairs(prompt: str) -> list[tuple[list[str], list[str]]]:
    text = prompt or ""
    pairs: list[tuple[list[str], list[str]]] = []
    seen_spans: list[tuple[int, int]] = []
    for pattern in PAIR_PATTERNS:
        for match in pattern.finditer(text):
            if _overlaps(match.span(), seen_spans):
                continue
            left = _words(match.group(1))
            right = _words(match.group(2))
            if left and right:
                pairs.append((left, right))
                seen_spans.append(match.span())
    return pairs


def extract_query_phrase(prompt: str, examples: list[tuple[list[str], list[str]]]) -> list[str] | None:
    text = prompt or ""
    pair_spans: list[tuple[int, int]] = []
    for pattern in PAIR_PATTERNS:
        pair_spans.extend(match.span() for match in pattern.finditer(text))
    for pattern in QUERY_PATTERNS:
        for match in reversed(list(pattern.finditer(text))):
            if _overlaps(match.span(1), pair_spans):
                continue
            words = _words(match.group(1))
            if words:
                return words
    return None


def build_word_mapping(pairs: list[tuple[list[str], list[str]]]) -> MappingResult:
    mapping: dict[str, str] = {}
    conflicts: list[dict] = []
    used = 0
    ignored = 0
    for encrypted, plain in pairs:
        if len(encrypted) != len(plain):
            ignored += 1
            continue
        used += 1
        for key, value in zip(encrypted, plain):
            old = mapping.get(key)
            if old is not None and old != value:
                conflicts.append({"encrypted": key, "first": old, "second": value})
            mapping[key] = value
    return MappingResult(MappingProxyType(dict(mapping)), bool(conflicts), tuple(conflicts), used, ignored)


class WordCipherSolver(BaseSolver):
    def __init__(self, *, allow_partial: bool = False, min_coverage: float = 1.0) -> None:
        self.allow_partial = allow_partial
        self.min_coverage = min_coverage

    @property
    def name(self) -> str:
        return "word_cipher_solver"

    @property
    def supported_families(self) -> tuple[str, ...]:
        return ("word_cipher",)

    def solve(self, row: dict) -> SolverResult:
        family = str(row.get("family", "word_cipher"))
        if family != "word_cipher":
            return SolverResult(self.name, family, [], True, "unsupported_family", {})
        prompt = str(row.get("prompt", ""))
        pairs = extract_phrase_pairs(prompt)
        query = extract_query_phrase(prompt, pairs)
        result = build_word_mapping(pairs)
        base_metadata = {
            "mapping_size": len(result.mapping),
            "used_pairs": result.used_pairs,
            "ignored_pairs": result.ignored_pairs,
            "query_length": 0 if not query else len(query),
            "unknown_count": 0,
            "coverage_ratio": 0.0 if not query else 1.0,
            "conflicts": list(result.conflicts),
        }
        if result.used_pairs == 0:
            return SolverResult(self.name, family, [], True, "no_aligned_examples", base_metadata)
        if result.conflict:
            return SolverResult(self.name, family, [], True, "mapping_conflict", base_metadata)
        if not query:
            return SolverResult(self.name, family, [], True, "missing_query", base_metadata)
        unknown = [word for word in query if word not in result.mapping]
        coverage = (len(query) - len(unknown)) / len(query) if query else 0.0
        metadata = base_metadata | {"unknown_count": len(unknown), "coverage_ratio": coverage, "unknown": unknown}
        if unknown:
            if not self.allow_partial or coverage < self.min_coverage:
                return SolverResult(self.name, family, [], True, "unknown_query_words", metadata)
        answer_words = [result.mapping[word] for word in query if word in result.mapping]
        if len(answer_words) != len(query):
            return SolverResult(self.name, family, [], True, "unknown_query_words", metadata)
        answer = " ".join(answer_words)
        candidate = SolverCandidate(
            answer=normalize_answer(answer, expected_type="text_phrase").normalized,
            source=self.name,
            family=family,
            subfamily="word_substitution",
            confidence=0.95,
            example_consistency=1.0,
            verified=True,
            risk="low",
            metadata=metadata,
        )
        validate_candidate(candidate)
        return SolverResult(self.name, family, [candidate], False, "", {"used_pairs": result.used_pairs})


def _words(value: str) -> list[str]:
    cleaned = value.strip().strip("`'\" ")
    cleaned = re.sub(r"[.,;:!?]+$", "", cleaned)
    return [word.lower() for word in re.findall(r"[a-zA-Z]+", cleaned)]


def _overlaps(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < other_end and other_start < end for other_start, other_end in spans)
