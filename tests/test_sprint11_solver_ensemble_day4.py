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


def _candidate(source: str, answer: str, confidence: float = 0.9, risk: str = "low", verified: bool = True) -> SolverCandidate:
    return SolverCandidate(answer, source, "roman_numeral", "standard_roman", confidence, 1.0, verified, risk, {})


def test_ensemble_verifies_merges_and_abstains_on_disagreement() -> None:
    row = {"family": "roman_numeral", "prompt": "11 -> XI; solve for 38"}
    merged = SolverEnsemble([FixedSolver("a", _candidate("a", "XXXVIII")), FixedSolver("b", _candidate("b", "38 -> XXXVIII"))]).run_all(row)
    assert not merged.abstained
    assert merged.candidates[0].metadata["source_count"] == 2
    disagreement = SolverEnsemble([FixedSolver("a", _candidate("a", "I")), FixedSolver("b", _candidate("b", "V"))]).run_all(row)
    assert disagreement.abstained
    assert disagreement.reason == "verified_candidate_disagreement"


def test_ensemble_rejects_low_confidence_and_ignores_abstainers() -> None:
    row = {"family": "roman_numeral", "prompt": "11 -> XI; solve for 38"}
    assert SolverEnsemble([FixedSolver("weak", _candidate("weak", "XXXVIII", confidence=0.4))]).run_all(row).abstained
    hit = SolverEnsemble([FixedSolver("skip", None), FixedSolver("hit", _candidate("hit", "XXXVIII"))]).best_candidate(row)
    assert hit.source == "hit"


def test_ensemble_router_order_and_word_beats_char() -> None:
    bit = SolverEnsemble().run_all({"family": "bit_manipulation", "prompt": "0000 -> 1010; 1111 -> 0101; 0011 -> 1001. Input: 0101"})
    assert not bit.abstained
    assert bit.candidates[0].source == "bit_transform_solver"
    word = SolverEnsemble().best_candidate({"family": "word_cipher", "prompt": '"aaa bbb" -> "cat dog"; encrypted: aaa bbb plaintext: ?'})
    assert word is not None
    assert word.source == "word_cipher_solver"
