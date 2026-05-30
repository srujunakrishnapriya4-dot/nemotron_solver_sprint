from __future__ import annotations

from collections.abc import Iterable

from kaggle_anti086.solvers.answer_normalizer import normalize_answer
from kaggle_anti086.solvers.base import BaseSolver
from kaggle_anti086.solvers.numeric_formula_solver import NumericFormulaSolver
from kaggle_anti086.solvers.roman_solver import RomanSolver
from kaggle_anti086.solvers.types import SolverCandidate, SolverResult, validate_candidate
from kaggle_anti086.solvers.unit_conversion_solver import UnitConversionSolver
from kaggle_anti086.solvers.word_cipher_solver import WordCipherSolver


RISK_ORDER = {"low": 0, "medium": 1, "high": 2}
ANSWER_TYPE_BY_FAMILY = {
    "roman_numeral": "roman",
    "custom_numeral": "roman",
    "unit_conversion": "numeric",
    "numeric_formula": "numeric",
    "gravity_numeric": "numeric",
    "bit_manipulation": "binary",
    "word_cipher": "text_phrase",
    "char_cipher": "text_phrase",
    "symbol_mapping": "symbol",
    "digit_symbol_mapping": "symbol",
}


class SolverEnsemble:
    def __init__(self, solvers: Iterable[BaseSolver] | None = None, *, min_confidence_to_emit: float = 0.50) -> None:
        self.solvers = list(solvers) if solvers is not None else [
            RomanSolver(),
            UnitConversionSolver(),
            NumericFormulaSolver(),
            WordCipherSolver(),
        ]
        self.min_confidence_to_emit = min_confidence_to_emit

    def run_all(self, row: dict) -> SolverResult:
        family = str(row.get("family", "unknown"))
        candidates: list[SolverCandidate] = []
        abstentions: list[dict] = []
        for solver in self.solvers:
            result = solver.solve(row)
            if result.abstained or not result.candidates:
                abstentions.append({"solver": result.solver_name, "reason": result.reason})
                continue
            for candidate in result.candidates:
                normalized = _normalize_candidate(candidate)
                validate_candidate(normalized)
                candidates.append(normalized)
        if not candidates:
            return SolverResult(
                solver_name="solver_ensemble",
                family=family,
                candidates=[],
                abstained=True,
                reason="all_solvers_abstained",
                metadata={"abstentions": abstentions},
            )
        merged = _merge_same_answer_candidates(candidates)
        ranked = sorted(merged, key=lambda candidate: _rank_key(candidate, family))
        disagreement_warning = _verified_low_risk_disagreement(ranked)
        if ranked[0].confidence < self.min_confidence_to_emit:
            return SolverResult(
                solver_name="solver_ensemble",
                family=family,
                candidates=ranked,
                abstained=True,
                reason="best_candidate_below_confidence_threshold",
                metadata={"abstentions": abstentions, "threshold": self.min_confidence_to_emit},
            )
        return SolverResult(
            solver_name="solver_ensemble",
            family=family,
            candidates=ranked,
            abstained=False,
            reason="",
            metadata={"abstentions": abstentions, "solver_count": len(self.solvers), "disagreement_warning": disagreement_warning},
        )

    def best_candidate(self, row: dict) -> SolverCandidate | None:
        result = self.run_all(row)
        if result.abstained or not result.candidates:
            return None
        return result.candidates[0]


def _normalize_candidate(candidate: SolverCandidate) -> SolverCandidate:
    answer_type = ANSWER_TYPE_BY_FAMILY.get(candidate.family)
    answer = normalize_answer(candidate.answer, expected_type=answer_type).normalized
    return SolverCandidate(
        answer=answer,
        source=candidate.source,
        family=candidate.family,
        subfamily=candidate.subfamily,
        confidence=float(candidate.confidence),
        example_consistency=float(candidate.example_consistency),
        verified=bool(candidate.verified),
        risk=candidate.risk,
        metadata=dict(candidate.metadata),
    )


def _rank_key(candidate: SolverCandidate, row_family: str) -> tuple:
    return (
        0 if candidate.verified else 1,
        -float(candidate.example_consistency),
        -float(candidate.confidence),
        RISK_ORDER.get(candidate.risk, 99),
        0 if candidate.family == row_family else 1,
        candidate.source,
        candidate.answer,
    )


def _merge_same_answer_candidates(candidates: list[SolverCandidate]) -> list[SolverCandidate]:
    grouped: dict[tuple[str, str], list[SolverCandidate]] = {}
    for candidate in candidates:
        key = (candidate.family, candidate.answer)
        grouped.setdefault(key, []).append(candidate)
    merged: list[SolverCandidate] = []
    for (_, _), group in grouped.items():
        if len(group) == 1:
            merged.append(group[0])
            continue
        ranked = sorted(group, key=lambda candidate: _rank_key(candidate, candidate.family))
        best = ranked[0]
        risks = sorted({candidate.risk for candidate in group}, key=lambda risk: RISK_ORDER.get(risk, 99))
        metadata = dict(best.metadata)
        metadata.update(
            {
                "agreeing_sources": sorted(candidate.source for candidate in group),
                "max_confidence": max(candidate.confidence for candidate in group),
                "min_risk": risks[0],
                "source_count": len(group),
            }
        )
        merged.append(
            SolverCandidate(
                answer=best.answer,
                source=best.source,
                family=best.family,
                subfamily=best.subfamily,
                confidence=max(candidate.confidence for candidate in group),
                example_consistency=max(candidate.example_consistency for candidate in group),
                verified=any(candidate.verified for candidate in group),
                risk=risks[0],
                metadata=metadata,
            )
        )
    return merged


def _verified_low_risk_disagreement(candidates: list[SolverCandidate]) -> dict:
    verified_low = [candidate for candidate in candidates if candidate.verified and candidate.risk == "low"]
    answers = sorted({candidate.answer for candidate in verified_low})
    if len(answers) <= 1:
        return {"present": False}
    return {
        "present": True,
        "answers": answers,
        "sources": sorted(candidate.source for candidate in verified_low),
    }
