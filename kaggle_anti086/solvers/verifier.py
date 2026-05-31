from __future__ import annotations

from dataclasses import dataclass
import re

from kaggle_anti086.solvers.answer_normalizer import normalize_answer
from kaggle_anti086.solvers.types import SolverCandidate


ROMAN_VALID_RE = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
ANSWER_TYPE_BY_FAMILY = {
    "roman_numeral": "roman",
    "custom_numeral": "generic",
    "unit_conversion": "numeric",
    "numeric_formula": "numeric",
    "gravity_numeric": "numeric",
    "bit_manipulation": "binary",
    "word_cipher": "text_phrase",
    "char_cipher": "text_phrase",
    "symbol_mapping": "symbol",
    "digit_symbol_mapping": "symbol",
    "format_only": "generic",
}


@dataclass(frozen=True)
class VerificationResult:
    verified: bool
    reason: str
    risk: str
    normalized_answer: str
    failure_code: str


def verify_candidate(candidate: SolverCandidate, row: dict, *, min_confidence: float = 0.5, min_example_consistency: float = 1.0, allow_high_risk: bool = False) -> VerificationResult:
    if not str(candidate.answer).strip():
        return _fail("empty_answer", candidate.risk, "")
    if candidate.confidence < min_confidence:
        return _fail("confidence_below_threshold", candidate.risk, "")
    if candidate.example_consistency < min_example_consistency:
        return _fail("example_consistency_below_threshold", candidate.risk, "")
    if candidate.risk == "high" and not allow_high_risk:
        return _fail("high_risk_rejected", candidate.risk, "")
    if not candidate.verified:
        return _fail("unverified_candidate_rejected", candidate.risk, "")
    family = str(row.get("family", candidate.family))
    answer_type = ANSWER_TYPE_BY_FAMILY.get(family)
    normalized = normalize_answer(candidate.answer, expected_type=answer_type).normalized
    if _has_verbose_explanation(candidate.answer, family):
        return _fail("verbose_answer_rejected", candidate.risk, normalized)
    family_result = _family_check(family, normalized, candidate, row)
    if not family_result.verified:
        return family_result
    return VerificationResult(True, "accepted", candidate.risk, normalized, "")


def _family_check(family: str, normalized: str, candidate: SolverCandidate, row: dict) -> VerificationResult:
    if family == "bit_manipulation":
        if not re.fullmatch(r"[01]+", normalized):
            return _fail("bit_answer_not_binary", candidate.risk, normalized)
        expected_width = candidate.metadata.get("width") or _infer_output_width(str(row.get("prompt", "")))
        if expected_width and len(normalized) != int(expected_width):
            return _fail("bit_answer_wrong_width", candidate.risk, normalized)
    elif family == "roman_numeral":
        if not normalized or not ROMAN_VALID_RE.fullmatch(normalized):
            return _fail("roman_answer_invalid", candidate.risk, normalized)
    elif family == "custom_numeral":
        custom = str(candidate.answer).strip().strip("`\"'")
        metadata = row.get("metadata", {}) if isinstance(row.get("metadata", {}), dict) else {}
        allowed = metadata.get("allowed_symbols") or metadata.get("custom_numeral_allowed_symbols")
        if not custom:
            return _fail("custom_numeral_empty", candidate.risk, normalized)
        if len(custom) > 64:
            return _fail("custom_numeral_too_long", candidate.risk, normalized)
        if any(ord(char) < 32 or ord(char) == 127 for char in custom):
            return _fail("custom_numeral_control_char", candidate.risk, normalized)
        if re.search(r"\s", custom):
            return _fail("custom_numeral_whitespace", candidate.risk, normalized)
        if re.search(r"\b(?:answer|because|explanation|therefore)\b", custom, re.IGNORECASE):
            return _fail("custom_numeral_verbose", candidate.risk, normalized)
        if allowed is not None:
            allowed_set = set(str(allowed))
            if any(char not in allowed_set for char in custom):
                return _fail("custom_numeral_disallowed_symbol", candidate.risk, normalized)
    elif family in {"numeric_formula", "unit_conversion", "gravity_numeric"}:
        if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", normalized):
            return _fail("numeric_answer_invalid", candidate.risk, normalized)
        if len(normalized.split(".", 1)[1]) > 8 if "." in normalized else False:
            return _fail("numeric_precision_insane", candidate.risk, normalized)
    elif family in {"word_cipher", "char_cipher"}:
        if re.search(r"\b(?:answer|because|therefore|explanation)\b", normalized, re.IGNORECASE):
            return _fail("text_answer_verbose", candidate.risk, normalized)
    elif family in {"symbol_mapping", "digit_symbol_mapping"}:
        if re.search(r"\b(?:answer|because|explanation)\b", candidate.answer, re.IGNORECASE):
            return _fail("symbol_answer_verbose", candidate.risk, normalized)
    elif family == "format_only":
        if not normalized:
            return _fail("format_only_empty", candidate.risk, normalized)
        if re.search(r"\b(?:because|explanation|therefore|reasoning)\b", candidate.answer, re.IGNORECASE):
            return _fail("format_only_verbose", candidate.risk, normalized)
    return VerificationResult(True, "accepted", candidate.risk, normalized, "")


def _has_verbose_explanation(answer: str, family: str) -> bool:
    text = str(answer)
    if family == "bit_manipulation":
        return bool(re.search(r"\b(?:answer|because|explanation|therefore)\b", text, re.IGNORECASE))
    return bool(re.search(r"\n.+\b(?:because|explanation|therefore)\b", text, re.IGNORECASE))


def _infer_output_width(prompt: str) -> int | None:
    match = re.search(r"[01]{4,16}\s*(?:->|=>|=|maps?\s+to)\s*([01]{4,16})", prompt)
    return len(match.group(1)) if match else None


def _fail(code: str, risk: str, normalized: str) -> VerificationResult:
    return VerificationResult(False, code, risk, normalized, code)
