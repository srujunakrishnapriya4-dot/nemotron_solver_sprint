from kaggle_anti086.solvers.base import BaseSolver
from kaggle_anti086.solvers.solver_ensemble import SolverEnsemble
from kaggle_anti086.solvers.types import SolverCandidate, SolverResult


class FixedSolver(BaseSolver):
    def __init__(self, name: str, candidate: SolverCandidate | None = None, reason: str = "nope") -> None:
        self._name = name
        self._candidate = candidate
        self._reason = reason

    @property
    def name(self) -> str:
        return self._name

    @property
    def supported_families(self) -> tuple[str, ...]:
        return ("roman_numeral",)

    def solve(self, row: dict) -> SolverResult:
        if self._candidate is None:
            return SolverResult(self.name, row["family"], [], True, self._reason, {})
        return SolverResult(self.name, row["family"], [self._candidate], False, "", {})


def _candidate(source: str, answer: str, confidence: float, verified: bool = True, risk: str = "low") -> SolverCandidate:
    return SolverCandidate(
        answer=answer,
        source=source,
        family="roman_numeral",
        subfamily="standard_roman",
        confidence=confidence,
        example_consistency=1.0,
        verified=verified,
        risk=risk,
        metadata={},
    )


def _row() -> dict:
    return {"family": "roman_numeral", "prompt": "Convert 38", "answer": "XXXVIII"}


def test_verified_high_confidence_candidate_wins() -> None:
    ensemble = SolverEnsemble([FixedSolver("low", _candidate("low", "I", 0.1)), FixedSolver("high", _candidate("high", "V", 0.9))])
    assert ensemble.best_candidate(_row()).source == "high"


def test_verified_candidate_beats_unverified() -> None:
    ensemble = SolverEnsemble(
        [
            FixedSolver("unverified", _candidate("unverified", "I", 1.0, verified=False)),
            FixedSolver("verified", _candidate("verified", "V", 0.2, verified=True)),
        ]
    )
    assert ensemble.best_candidate(_row()).source == "verified"


def test_risk_ranking_breaks_ties() -> None:
    ensemble = SolverEnsemble(
        [
            FixedSolver("medium", _candidate("medium", "I", 0.9, risk="medium")),
            FixedSolver("low", _candidate("low", "V", 0.9, risk="low")),
        ]
    )
    assert ensemble.best_candidate(_row()).source == "low"


def test_abstained_solvers_ignored_and_all_abstained_returns_abstain() -> None:
    row = _row()
    ensemble = SolverEnsemble([FixedSolver("skip"), FixedSolver("hit", _candidate("hit", "xxxviii", 0.8))])
    assert ensemble.best_candidate(row).source == "hit"
    assert ensemble.best_candidate(row).answer == "XXXVIII"
    all_skip = SolverEnsemble([FixedSolver("skip")]).run_all(row)
    assert all_skip.abstained
    assert all_skip.reason == "all_solvers_abstained"
