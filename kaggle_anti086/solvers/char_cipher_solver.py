from __future__ import annotations

from dataclasses import dataclass
import re

from kaggle_anti086.solvers.answer_normalizer import normalize_answer
from kaggle_anti086.solvers.base import BaseSolver
from kaggle_anti086.solvers.types import SolverCandidate, SolverResult, validate_candidate
from kaggle_anti086.solvers.word_cipher_solver import WordCipherSolver


PAIR_RE = re.compile(r"(?:encrypted\s*:\s*)?([A-Za-z ]+?)\s*(?:->|=>|=|maps?\s+to|plaintext\s*:)\s*([A-Za-z ]+?)(?=(?:\s*;\s*|\n|$))", re.IGNORECASE)
QUERY_RE = re.compile(r"(?:query|decode|encrypted|input|target)\s*:\s*([A-Za-z ]+?)(?=(?:\s*;\s*|\n|\?|$))", re.IGNORECASE)


@dataclass(frozen=True)
class CharRule:
    subfamily: str
    confidence: float
    risk: str
    params: dict

    def apply(self, text: str) -> str:
        if self.subfamily == "caesar_shift":
            return _shift_text(text, self.params["shift"])
        if self.subfamily == "reverse_string":
            return text[::-1]
        if self.subfamily == "reverse_words":
            return " ".join(reversed(text.split()))
        if self.subfamily == "monoalphabetic_substitution":
            return "".join(self.params["mapping"].get(char.lower(), char).lower() if char.isalpha() else char for char in text.lower())
        raise ValueError(f"unknown char rule: {self.subfamily}")


class CharCipherSolver(BaseSolver):
    def __init__(self, *, min_coverage: float = 0.9) -> None:
        self.min_coverage = min_coverage

    @property
    def name(self) -> str:
        return "char_cipher_solver"

    @property
    def supported_families(self) -> tuple[str, ...]:
        return ("char_cipher",)

    def solve(self, row: dict) -> SolverResult:
        family = str(row.get("family", "char_cipher"))
        if family != "char_cipher":
            return SolverResult(self.name, family, [], True, "unsupported_family", {})
        if WordCipherSolver().solve({"family": "word_cipher", "prompt": row.get("prompt", "")}).candidates:
            return SolverResult(self.name, family, [], True, "word_cipher_has_complete_coverage", {})
        prompt = str(row.get("prompt", ""))
        pairs = extract_char_pairs(prompt)
        query = extract_char_query(prompt)
        if len(pairs) < 2:
            return SolverResult(self.name, family, [], True, "insufficient_examples", {"example_count": len(pairs)})
        if not query:
            return SolverResult(self.name, family, [], True, "missing_query", {"example_count": len(pairs)})
        rules = fit_char_rules(pairs, min_coverage=self.min_coverage)
        if not rules:
            return SolverResult(self.name, family, [], True, "no_matching_char_rule", {"example_count": len(pairs)})
        safe_rules = []
        for rule in rules:
            if rule.subfamily == "monoalphabetic_substitution" and _query_coverage(query, rule.params["mapping"]) < self.min_coverage:
                continue
            safe_rules.append(rule)
        if not safe_rules:
            return SolverResult(self.name, family, [], True, "unknown_character_coverage_too_low", {"query": query})
        predictions = [rule.apply(query).strip().lower() for rule in safe_rules]
        if len(set(predictions)) > 1:
            return SolverResult(self.name, family, [], True, "ambiguous_char_rule_disagreement", {"predictions": predictions})
        best = sorted(safe_rules, key=lambda rule: (-rule.confidence, rule.subfamily))[0]
        answer = normalize_answer(predictions[0], expected_type="text_phrase").normalized
        if not answer or "because" in answer:
            return SolverResult(self.name, family, [], True, "unsafe_char_output", {"prediction": answer})
        candidate = SolverCandidate(
            answer=answer,
            source=self.name,
            family=family,
            subfamily=best.subfamily,
            confidence=best.confidence,
            example_consistency=1.0,
            verified=True,
            risk=best.risk,
            metadata={"rule": best.subfamily, **best.params},
        )
        validate_candidate(candidate)
        return SolverResult(self.name, family, [candidate], False, "", candidate.metadata)


def extract_char_pairs(prompt: str) -> list[tuple[str, str]]:
    pairs = []
    for match in PAIR_RE.finditer(prompt or ""):
        left = _clean(match.group(1))
        right = _clean(match.group(2))
        if left and right and "?" not in right:
            pairs.append((left.lower(), right.lower()))
    return pairs


def extract_char_query(prompt: str) -> str | None:
    text = prompt or ""
    for match in QUERY_RE.finditer(text):
        token = _clean(match.group(1))
        if token and "?" not in token:
            return token.lower()
    for match in PAIR_RE.finditer(text):
        left = _clean(match.group(1))
        right = _clean(match.group(2))
        if "?" in right and left:
            return left.lower()
    return None


def fit_char_rules(pairs: list[tuple[str, str]], *, min_coverage: float = 0.9) -> list[CharRule]:
    rules: list[CharRule] = []
    shifts = [_caesar_shift(left, right) for left, right in pairs]
    if shifts and shifts[0] is not None and all(shift == shifts[0] for shift in shifts):
        rules.append(CharRule("caesar_shift", 0.96, "low", {"shift": shifts[0]}))
    if all(left[::-1] == right for left, right in pairs):
        rules.append(CharRule("reverse_string", 0.95, "low", {}))
    multi_word_pairs = [(left, right) for left, right in pairs if len(left.split()) > 1]
    if multi_word_pairs and all(" ".join(reversed(left.split())) == right for left, right in multi_word_pairs):
        rules.append(CharRule("reverse_words", 0.9, "low", {}))
    mapping, conflicts = _build_char_mapping(pairs)
    if mapping and not conflicts and _mapping_coverage(pairs, mapping) >= min_coverage and not _has_space_conflict(pairs):
        rules.append(CharRule("monoalphabetic_substitution", 0.92, "low", {"mapping": mapping, "coverage_ratio": _mapping_coverage(pairs, mapping)}))
    return _dedupe(rules)


def _caesar_shift(left: str, right: str) -> int | None:
    left_letters = [c for c in left.lower() if c.isalpha()]
    right_letters = [c for c in right.lower() if c.isalpha()]
    if len(left_letters) != len(right_letters) or not left_letters:
        return None
    shifts = [((ord(r) - ord(l)) % 26) for l, r in zip(left_letters, right_letters)]
    return shifts[0] if all(shift == shifts[0] for shift in shifts) else None


def _shift_text(text: str, shift: int) -> str:
    chars = []
    for char in text.lower():
        if char.isalpha():
            chars.append(chr((ord(char) - 97 + shift) % 26 + 97))
        else:
            chars.append(char)
    return "".join(chars)


def _build_char_mapping(pairs: list[tuple[str, str]]) -> tuple[dict[str, str], list[dict]]:
    mapping: dict[str, str] = {}
    conflicts = []
    for left, right in pairs:
        left_letters = [c for c in left.lower() if c.isalpha()]
        right_letters = [c for c in right.lower() if c.isalpha()]
        if len(left_letters) != len(right_letters):
            continue
        for a, b in zip(left_letters, right_letters):
            old = mapping.get(a)
            if old is not None and old != b:
                conflicts.append({"source": a, "first": old, "second": b})
            mapping[a] = b
    return mapping, conflicts


def _mapping_coverage(pairs: list[tuple[str, str]], mapping: dict[str, str]) -> float:
    total = 0
    known = 0
    for left, _ in pairs:
        for char in left.lower():
            if char.isalpha():
                total += 1
                if char in mapping:
                    known += 1
    return known / total if total else 0.0


def _query_coverage(query: str, mapping: dict[str, str]) -> float:
    letters = [char for char in query.lower() if char.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for char in letters if char in mapping) / len(letters)


def _has_space_conflict(pairs: list[tuple[str, str]]) -> bool:
    return any((" " in left) != (" " in right) for left, right in pairs)


def _clean(value: str) -> str:
    return value.strip().strip("`\"' ")


def _dedupe(rules: list[CharRule]) -> list[CharRule]:
    seen = set()
    result = []
    for rule in rules:
        key = (rule.subfamily, tuple(sorted((k, str(v)) for k, v in rule.params.items())))
        if key in seen:
            continue
        seen.add(key)
        result.append(rule)
    return result
