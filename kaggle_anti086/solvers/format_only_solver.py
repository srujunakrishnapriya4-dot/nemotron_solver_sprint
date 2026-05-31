from __future__ import annotations

import re

from kaggle_anti086.solvers.answer_normalizer import answers_match, normalize_answer
from kaggle_anti086.solvers.base import BaseSolver
from kaggle_anti086.solvers.types import SolverCandidate, SolverResult, validate_candidate


RAW_OUTPUT_RE = re.compile(
    r"(?:raw\s+(?:model\s+)?output|candidate|formatted\s+answer|output)\s*[:=]\s*(.+?)(?=(?:\n\s*(?:expected|gold|answer)\s*:)|$)",
    re.IGNORECASE | re.DOTALL,
)


class FormatOnlySolver(BaseSolver):
    @property
    def name(self) -> str:
        return "format_only_solver"

    @property
    def supported_families(self) -> tuple[str, ...]:
        return ("format_only",)

    def solve(self, row: dict) -> SolverResult:
        family = str(row.get("family", ""))
        if family != "format_only":
            return SolverResult(self.name, family or "format_only", [], True, "unsupported_family", {})
        prompt = str(row.get("prompt", ""))
        raw = _extract_raw_output(prompt)
        if not raw:
            return SolverResult(self.name, family, [], True, "missing_raw_output", {})
        answer_type = _answer_type(row, raw)
        normalized = normalize_answer(raw, expected_type=answer_type)
        answer = normalized.normalized
        if not answer:
            return SolverResult(self.name, family, [], True, "empty_normalized_answer", {"raw_output": raw})
        if _looks_verbose(answer):
            return SolverResult(self.name, family, [], True, "verbose_output_rejected", {"raw_output": raw, "normalized": answer})
        gold = str(row.get("answer", ""))
        if gold and gold.upper() != "ABSTAIN" and not answers_match(answer, gold, answer_type=answer_type):
            return SolverResult(
                self.name,
                family,
                [],
                True,
                "normalized_answer_mismatch",
                {"raw_output": raw, "normalized": answer, "gold": gold, "answer_type": answer_type},
            )
        candidate = SolverCandidate(
            answer=answer,
            source=self.name,
            family="format_only",
            subfamily="normalization",
            confidence=0.95,
            example_consistency=1.0,
            verified=True,
            risk="low",
            metadata={"raw_output": raw, "answer_type": answer_type, "normalizer_confidence": normalized.confidence, "transformations": normalized.transformations},
        )
        validate_candidate(candidate)
        return SolverResult(self.name, family, [candidate], False, "", candidate.metadata)


def _extract_raw_output(prompt: str) -> str:
    match = RAW_OUTPUT_RE.search(prompt)
    if match:
        return match.group(1).strip()
    if "extract" in prompt.lower():
        return prompt.rsplit(":", 1)[-1].strip()
    return ""


def _answer_type(row: dict, raw: str) -> str | None:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata", {}), dict) else {}
    answer_type = metadata.get("answer_type")
    if answer_type:
        return str(answer_type)
    gold = str(row.get("answer", ""))
    if re.fullmatch(r"[01]{4,}", gold):
        return "binary"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", gold):
        return "numeric"
    if re.fullmatch(r"[IVXLCDM]+", gold):
        return "roman"
    if gold and not re.search(r"[A-Za-z0-9\s]", gold):
        return "symbol"
    return None


def _looks_verbose(text: str) -> bool:
    return bool(re.search(r"\b(?:because|explanation|therefore|step\s*\d+|reasoning)\b", text, re.IGNORECASE))
