from kaggle_anti086.solvers.base import BaseSolver
from kaggle_anti086.solvers.solver_ensemble import SolverEnsemble
from kaggle_anti086.solvers.types import SolverCandidate, SolverResult


class FixedSolver(BaseSolver):
    def __init__(self, name: str, candidate: SolverCandidate | None) -> None:
        self._name = name
        self._candidate = candidate

    @property
    def name(self) -> str:
        return self._name

    @property
    def supported_families(self) -> tuple[str, ...]:
        return ("roman_numeral",)

    def solve(self, row: dict) -> SolverResult:
        if self._candidate is None:
            return SolverResult(self.name, row["family"], [], True, "skip", {})
        return SolverResult(self.name, row["family"], [self._candidate], False, "", {})


def _candidate(source: str, answer: str, confidence: float = 0.9, verified: bool = True, risk: str = "low") -> SolverCandidate:
    return SolverCandidate(answer, source, "roman_numeral", "standard_roman", confidence, 1.0, verified, risk, {})


def _row() -> dict:
    return {"family": "roman_numeral", "prompt": "Convert 38", "answer": "XXXVIII"}


def test_duplicate_same_answer_candidates_merge() -> None:
    result = SolverEnsemble([FixedSolver("a", _candidate("a", "xxxviii")), FixedSolver("b", _candidate("b", "XXXVIII", 0.8))]).run_all(_row())
    assert not result.abstained
    assert len(result.candidates) == 1
    assert result.candidates[0].metadata["source_count"] == 2
    assert result.candidates[0].metadata["agreeing_sources"] == ["a", "b"]


def test_verified_beats_unverified_and_low_risk_beats_high_risk() -> None:
    unverified = FixedSolver("unverified", _candidate("unverified", "I", 1.0, verified=False, risk="low"))
    high_risk = FixedSolver("high", _candidate("high", "II", 0.9, verified=True, risk="high"))
    low_risk = FixedSolver("low", _candidate("low", "III", 0.9, verified=True, risk="low"))
    result = SolverEnsemble([unverified, high_risk, low_risk]).run_all(_row())
    assert result.candidates[0].source == "low"


def test_low_confidence_best_causes_abstention() -> None:
    result = SolverEnsemble([FixedSolver("weak", _candidate("weak", "XXXVIII", 0.49))]).run_all(_row())
    assert result.abstained
    assert result.reason in {"best_candidate_below_confidence_threshold", "all_solvers_abstained"}


def test_disagreement_warning_for_verified_low_risk_candidates() -> None:
    result = SolverEnsemble([FixedSolver("a", _candidate("a", "I")), FixedSolver("b", _candidate("b", "V"))]).run_all(_row())
    assert result.abstained
    assert result.reason == "verified_candidate_disagreement"
    assert result.metadata["disagreement_warning"]["present"] is True
