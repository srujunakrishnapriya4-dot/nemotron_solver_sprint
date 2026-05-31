from __future__ import annotations

from dataclasses import dataclass
import re

from kaggle_anti086.solvers.base import BaseSolver
from kaggle_anti086.solvers.types import SolverCandidate, SolverResult, validate_candidate


PAIR_RE = re.compile(r"(?:input\s*:\s*)?(.+?)\s*(?:->|=>|maps?\s+to|=|:)\s*(.+?)(?=(?:\s*;\s*|\n|$))", re.IGNORECASE)
QUERY_RE = re.compile(r"(?:input|query|target)\s*:\s*(.+?)(?=(?:\s*;\s*|\n|$))", re.IGNORECASE)


@dataclass(frozen=True)
class SymbolMapping:
    mapping: dict[str, str]
    reverse_input: bool
    ignored_pairs: int
    conflicts: tuple[dict, ...]


class SymbolMappingSolver(BaseSolver):
    def __init__(self, *, min_coverage: float = 0.9) -> None:
        self.min_coverage = min_coverage

    @property
    def name(self) -> str:
        return "symbol_mapping_solver"

    @property
    def supported_families(self) -> tuple[str, ...]:
        return ("symbol_mapping", "digit_symbol_mapping")

    def solve(self, row: dict) -> SolverResult:
        family = str(row.get("family", "symbol_mapping"))
        if family not in self.supported_families:
            return SolverResult(self.name, family, [], True, "unsupported_family", {})
        prompt = str(row.get("prompt", ""))
        pairs = extract_symbol_pairs(prompt)
        query = extract_symbol_query(prompt, pairs)
        if len(pairs) < 2:
            return SolverResult(self.name, family, [], True, "insufficient_examples", {"example_count": len(pairs)})
        if query is None:
            return SolverResult(self.name, family, [], True, "missing_query", {"example_count": len(pairs)})
        if any(len(left) != len(right) for left, right in pairs):
            return SolverResult(
                self.name,
                family,
                [],
                True,
                "unsupported_alignment_compression",
                {
                    "example_count": len(pairs),
                    "alignment_mode": "unsupported_variable_length",
                    "reason_for_abstain": "length_mismatch_examples",
                },
            )
        mappings = [mapping for mapping in (_build_mapping(pairs, False), _build_mapping(pairs, True)) if mapping is not None and not mapping.conflicts]
        if not mappings:
            conflict = _build_mapping(pairs, False)
            conflicts = [] if conflict is None else list(conflict.conflicts)
            return SolverResult(
                self.name,
                family,
                [],
                True,
                "mapping_conflict",
                {
                    "conflicts": conflicts,
                    "conflict_source": None if not conflicts else conflicts[0].get("source"),
                    "reason_for_abstain": "conflicting_symbol_mapping",
                },
            )
        predictions = []
        candidates = []
        for mapping in mappings:
            prediction, coverage = _apply_mapping(query, mapping)
            if prediction == "":
                continue
            if coverage < self.min_coverage:
                unknown_count = sum(1 for char in query if char not in mapping.mapping)
                return SolverResult(
                    self.name,
                    family,
                    [],
                    True,
                    "unknown_symbol_coverage_too_low",
                    {
                        "coverage_ratio": coverage,
                        "coverage": coverage,
                        "unknown_symbol_count": unknown_count,
                        "query_length": len(query),
                        "mapping_size": len(mapping.mapping),
                        "alignment_mode": "reverse_substitution" if mapping.reverse_input else "same_length_substitution",
                        "reason_for_abstain": "unknown_query_symbols",
                    },
                )
            predictions.append(prediction)
            candidates.append((mapping, prediction, coverage))
        if not candidates:
            return SolverResult(self.name, family, [], True, "empty_prediction", {})
        if len(set(predictions)) > 1:
            return SolverResult(self.name, family, [], True, "ambiguous_mapping_disagreement", {"predictions": predictions})
        mapping, answer, coverage = sorted(candidates, key=lambda item: (item[0].reverse_input, -item[2]))[0]
        risk = "low" if coverage == 1.0 else "medium"
        candidate = SolverCandidate(
            answer=answer,
            source=self.name,
            family=family,
            subfamily="reversal_substitution" if mapping.reverse_input else "char_substitution",
            confidence=0.92 if risk == "low" else 0.72,
            example_consistency=1.0,
            verified=True,
            risk=risk,
            metadata={
                "mapping_size": len(mapping.mapping),
                "ignored_pairs": mapping.ignored_pairs,
                "coverage_ratio": coverage,
                "coverage": coverage,
                "unknown_symbol_count": sum(1 for char in query if char not in mapping.mapping),
                "query_length": len(query),
                "reverse_input": mapping.reverse_input,
                "alignment_mode": "reverse_substitution" if mapping.reverse_input else "same_length_substitution",
                "reason_for_abstain": "",
            },
        )
        validate_candidate(candidate)
        return SolverResult(self.name, family, [candidate], False, "", candidate.metadata)


def extract_symbol_pairs(prompt: str) -> list[tuple[str, str]]:
    pairs = []
    for segment in re.split(r"[;\n]+", prompt or ""):
        if re.search(r"\b(?:input|query|target)\s*:", segment, re.IGNORECASE):
            continue
        match = PAIR_RE.search(segment)
        if not match:
            continue
        left = _clean_token(match.group(1))
        right = _clean_token(match.group(2))
        if not left or not right or "?" in right.lower() or left.lower() in {"input", "output", "query"}:
            continue
        if _looks_like_words(left, right):
            continue
        pairs.append((left, right))
    return pairs


def extract_symbol_query(prompt: str, pairs: list[tuple[str, str]]) -> str | None:
    text = prompt or ""
    match = re.search(r"\b(?:input|query|target)\s*:\s*(.+?)(?:\s+output\s*\?|\s*;\s*|\n|$)", text, re.IGNORECASE)
    if match:
        token = _clean_token(match.group(1))
        if token and "?" not in token:
            return token
    for match in PAIR_RE.finditer(text):
        right = _clean_token(match.group(2))
        left = _clean_token(match.group(1))
        if "?" in right and left:
            return left
    return None


def _build_mapping(pairs: list[tuple[str, str]], reverse_input: bool) -> SymbolMapping | None:
    mapping: dict[str, str] = {}
    conflicts = []
    ignored = 0
    for left, right in pairs:
        src = left[::-1] if reverse_input else left
        if len(src) != len(right):
            ignored += 1
            continue
        for a, b in zip(src, right):
            old = mapping.get(a)
            if old is not None and old != b:
                conflicts.append({"source": a, "first": old, "second": b})
            mapping[a] = b
    if not mapping:
        return None
    return SymbolMapping(mapping, reverse_input, ignored, tuple(conflicts))


def _apply_mapping(query: str, mapping: SymbolMapping) -> tuple[str, float]:
    src = query[::-1] if mapping.reverse_input else query
    output = []
    known = 0
    for char in src:
        if char in mapping.mapping:
            output.append(mapping.mapping[char])
            known += 1
        else:
            output.append(char)
    coverage = known / len(src) if src else 0.0
    return "".join(output), coverage


def _clean_token(value: str) -> str:
    value = value.strip()
    while re.match(r"^\[[^\]]+\]\s*", value):
        value = re.sub(r"^\[[^\]]+\]\s*", "", value).strip()
    value = re.sub(r"^(?:example|input|output|query)\s*[:#]?\s*", "", value, flags=re.IGNORECASE).strip()
    return value.strip("`\"' ")


def _looks_like_words(left: str, right: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z ]+", left) and re.fullmatch(r"[A-Za-z ]+", right))
