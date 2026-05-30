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
    def __init__(self, solvers: Iterable[BaseSolver] | None = None) -> None:
        self.solvers = list(solvers) if solvers is not None else [
            RomanSolver(),
            UnitConversionSolver(),
            NumericFormulaSolver(),
            WordCipherSolver(),
        ]

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
        ranked = sorted(candidates, key=_rank_key)
        return SolverResult(
            solver_name="solver_ensemble",
            family=family,
            candidates=ranked,
            abstained=False,
            reason="",
            metadata={"abstentions": abstentions, "solver_count": len(self.solvers)},
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


def _rank_key(candidate: SolverCandidate) -> tuple:
    return (
        0 if candidate.verified else 1,
        -float(candidate.example_consistency),
        -float(candidate.confidence),
        RISK_ORDER.get(candidate.risk, 99),
        candidate.source,
        candidate.answer,
    )
