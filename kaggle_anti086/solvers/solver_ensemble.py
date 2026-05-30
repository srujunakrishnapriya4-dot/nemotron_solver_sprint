from __future__ import annotations

from collections.abc import Iterable

from kaggle_anti086.solvers.answer_normalizer import normalize_answer
from kaggle_anti086.solvers.base import BaseSolver
from kaggle_anti086.solvers.bit_transform_solver import BitTransformSolver
from kaggle_anti086.solvers.char_cipher_solver import CharCipherSolver
from kaggle_anti086.solvers.numeric_formula_solver import NumericFormulaSolver
from kaggle_anti086.solvers.roman_solver import RomanSolver
from kaggle_anti086.solvers.router import route_row
from kaggle_anti086.solvers.symbol_mapping_solver import SymbolMappingSolver
from kaggle_anti086.solvers.types import SolverCandidate, SolverResult, validate_candidate
from kaggle_anti086.solvers.unit_conversion_solver import UnitConversionSolver
from kaggle_anti086.solvers.verifier import verify_candidate
from kaggle_anti086.solvers.word_cipher_solver import WordCipherSolver


RISK_ORDER = {"low": 0, "medium": 1, "high": 2}
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
}


class SolverEnsemble:
    def __init__(self, solvers: Iterable[BaseSolver] | None = None, *, min_confidence_to_emit: float = 0.50) -> None:
        self.solvers = list(solvers) if solvers is not None else [
            BitTransformSolver(),
            SymbolMappingSolver(),
            CharCipherSolver(),
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
        verification_failures: list[dict] = []
        route = route_row(row)
        if not route.supported_by_solver and route.confidence >= 0.75:
            return SolverResult(
                solver_name="solver_ensemble",
                family=family,
                candidates=[],
                abstained=True,
                reason=route.unsupported_reason or "unsupported_route",
                metadata={"route": route.__dict__},
            )
        ordered_solvers = self._ordered_solvers(route.candidate_solvers)
        preferred = set(route.candidate_solvers)
        preferred_candidate_found = False
        for solver in ordered_solvers:
            if preferred_candidate_found and route.confidence >= 0.85 and solver.name not in preferred:
                break
            solver_row = row
            if solver.name in preferred and family in {"unknown", ""} and route.family != family:
                solver_row = dict(row) | {"family": route.family}
            result = solver.solve(solver_row)
            if result.abstained or not result.candidates:
                abstentions.append({"solver": result.solver_name, "reason": result.reason})
                continue
            for candidate in result.candidates:
                normalized = _normalize_candidate(candidate)
                validate_candidate(normalized)
                verified = verify_candidate(normalized, row)
                if verified.verified:
                    if normalized.source in preferred:
                        preferred_candidate_found = True
                    candidates.append(
                        SolverCandidate(
                            answer=verified.normalized_answer,
                            source=normalized.source,
                            family=normalized.family,
                            subfamily=normalized.subfamily,
                            confidence=normalized.confidence,
                            example_consistency=normalized.example_consistency,
                            verified=True,
                            risk=verified.risk,
                            metadata=dict(normalized.metadata) | {"verification": verified.reason},
                        )
                    )
                else:
                    verification_failures.append({"solver": normalized.source, "failure_code": verified.failure_code, "reason": verified.reason})
        if not candidates:
            return SolverResult(
                solver_name="solver_ensemble",
                family=family,
                candidates=[],
                abstained=True,
                reason="all_solvers_abstained",
                metadata={"abstentions": abstentions, "verification_failures": verification_failures, "route": route.__dict__},
            )
        merged = _merge_same_answer_candidates(candidates)
        ranked = sorted(merged, key=lambda candidate: _rank_key(candidate, family))
        disagreement_warning = _verified_low_risk_disagreement(ranked)
        if disagreement_warning["present"]:
            routed_sources = set(route.candidate_solvers)
            routed_low = [candidate for candidate in ranked if candidate.source in routed_sources and candidate.verified and candidate.risk == "low"]
            fallback_low = [candidate for candidate in ranked if candidate.source not in routed_sources and candidate.verified and candidate.risk == "low"]
            if route.confidence >= 0.85 and len(routed_low) == 1 and fallback_low and all(candidate.example_consistency < routed_low[0].example_consistency or candidate.confidence < routed_low[0].confidence for candidate in fallback_low):
                disagreement_warning = disagreement_warning | {"overridden_by_route": True}
            else:
                return SolverResult(
                    solver_name="solver_ensemble",
                    family=family,
                    candidates=ranked,
                    abstained=True,
                    reason="verified_candidate_disagreement",
                    metadata={"abstentions": abstentions, "verification_failures": verification_failures, "route": route.__dict__, "disagreement_warning": disagreement_warning},
                )
        if ranked[0].confidence < self.min_confidence_to_emit:
            return SolverResult(
                solver_name="solver_ensemble",
                family=family,
                candidates=ranked,
                abstained=True,
                reason="best_candidate_below_confidence_threshold",
                metadata={"abstentions": abstentions, "verification_failures": verification_failures, "threshold": self.min_confidence_to_emit, "route": route.__dict__},
            )
        return SolverResult(
            solver_name="solver_ensemble",
            family=family,
            candidates=ranked,
            abstained=False,
            reason="",
            metadata={"abstentions": abstentions, "solver_count": len(self.solvers), "verification_failures": verification_failures, "route": route.__dict__, "disagreement_warning": disagreement_warning},
        )

    def best_candidate(self, row: dict) -> SolverCandidate | None:
        result = self.run_all(row)
        if result.abstained or not result.candidates:
            return None
        return result.candidates[0]

    def _ordered_solvers(self, preferred_names: list[str]) -> list[BaseSolver]:
        by_name = {solver.name: solver for solver in self.solvers}
        ordered: list[BaseSolver] = []
        for name in preferred_names:
            solver = by_name.get(name)
            if solver is not None:
                ordered.append(solver)
        ordered.extend(solver for solver in self.solvers if solver.name not in {item.name for item in ordered})
        return ordered


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
