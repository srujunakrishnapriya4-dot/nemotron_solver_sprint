from __future__ import annotations

from dataclasses import dataclass, field
import re

from kaggle_anti086.solvers.bit_transform_solver import PAIR_RE as BIT_PAIR_RE
from kaggle_anti086.solvers.roman_solver import EXAMPLE_RE as ROMAN_EXAMPLE_RE


@dataclass(frozen=True)
class RouterResult:
    family: str
    subfamily_hint: str
    confidence: float
    supported_by_solver: bool
    candidate_solvers: list[str]
    risk: str
    signals: list[str] = field(default_factory=list)


def route_row(row: dict) -> RouterResult:
    family = str(row.get("family", "unknown"))
    prompt = str(row.get("prompt", ""))
    if family == "equation_operator" or re.search(r"\b(?:equation|operator|modulo|exact division)\b", prompt, re.IGNORECASE):
        return RouterResult("equation_operator", "unsupported", 0.8, False, [], "high", ["equation_operator_unsupported"])
    if BIT_PAIR_RE.search(prompt) or family == "bit_manipulation":
        return RouterResult("bit_manipulation", "binary_transform", 0.94, True, ["bit_transform_solver"], "low", ["binary_pairs"])
    if ROMAN_EXAMPLE_RE.search(prompt) or family in {"roman_numeral", "custom_numeral"}:
        return RouterResult(family if family in {"roman_numeral", "custom_numeral"} else "roman_numeral", "standard_roman", 0.93, True, ["roman_solver"], "low", ["roman_examples"])
    if family == "gravity_numeric" or re.search(r"\b(?:gravity|falling|distance|time|0\.5\s*\*\s*g)\b", prompt, re.IGNORECASE):
        return RouterResult("gravity_numeric", "gravity_distance", 0.9, True, ["numeric_formula_solver"], "low", ["gravity_wording"])
    if family == "unit_conversion" or re.search(r"\b(?:convert|unit|target unit|maps? to|converts? to)\b", prompt, re.IGNORECASE):
        return RouterResult("unit_conversion", "unit_conversion", 0.88, True, ["unit_conversion_solver"], "low", ["numeric_pairs", "unit_wording"])
    if family == "numeric_formula" or re.search(r"\b\d+(?:\.\d+)?\s*(?:->|=)\s*\d", prompt):
        return RouterResult("numeric_formula", "numeric_formula", 0.82, True, ["numeric_formula_solver", "unit_conversion_solver"], "medium", ["numeric_pairs"])
    if family == "word_cipher" or re.search(r'["`][A-Za-z ]+["`]\s*(?:->|=)\s*["`][A-Za-z ]+["`]', prompt):
        return RouterResult("word_cipher", "word_substitution", 0.9, True, ["word_cipher_solver", "char_cipher_solver"], "low", ["quoted_phrase_pairs"])
    if family in {"symbol_mapping", "digit_symbol_mapping"} or _symbol_heavy(prompt):
        return RouterResult(family if family in {"symbol_mapping", "digit_symbol_mapping"} else "symbol_mapping", "symbol_transform", 0.86, True, ["symbol_mapping_solver"], "medium", ["symbol_heavy"])
    if family == "char_cipher" or _looks_char_cipher(prompt):
        return RouterResult("char_cipher", "char_cipher", 0.82, True, ["char_cipher_solver"], "medium", ["alphabetic_pairs"])
    return RouterResult(family, "unknown", 0.2, False, [], "high", ["unsupported"])


def _symbol_heavy(prompt: str) -> bool:
    symbols = re.findall(r"[^A-Za-z0-9\s]", prompt)
    return len(symbols) >= 6 and bool(re.search(r"(?:->|=>|maps?\s+to|=)", prompt))


def _looks_char_cipher(prompt: str) -> bool:
    return bool(re.search(r"\b[A-Za-z]{2,}\b\s*(?:->|=|=>)\s*\b[A-Za-z]{2,}\b", prompt))
