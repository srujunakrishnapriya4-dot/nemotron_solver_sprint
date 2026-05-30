from kaggle_anti086.solvers.solver_ensemble import SolverEnsemble
from kaggle_anti086.solvers.base import BaseSolver
from kaggle_anti086.solvers.types import SolverCandidate, SolverResult


def test_equation_operator_route_does_not_fall_through_to_symbol_solver() -> None:
    row = {"family": "equation_operator", "prompt": "!@ -> ab; @# -> bc; input: !# output ?"}
    result = SolverEnsemble().run_all(row)
    assert result.abstained
    assert result.reason == "equation_operator_solver_not_implemented"


def test_high_confidence_word_cipher_not_overridden_by_char_cipher() -> None:
    row = {
        "family": "word_cipher",
        "prompt": '"trb hffk" -> "cat book"; encrypted: trb hffk plaintext: ?',
    }
    result = SolverEnsemble().run_all(row)
    assert not result.abstained
    assert result.candidates[0].source == "word_cipher_solver"
    assert result.candidates[0].answer == "cat book"


def test_high_confidence_bit_route_rejects_text_answer() -> None:
    row = {"family": "bit_manipulation", "prompt": "abc -> def; xyz -> abc; Input: 1010"}
    result = SolverEnsemble().run_all(row)
    assert result.abstained


def test_low_confidence_route_can_try_fallback() -> None:
    row = {"family": "unknown", "prompt": "11 -> XI; 15 -> XV. solve for 38"}
    result = SolverEnsemble().run_all(row)
    assert not result.abstained
    assert result.candidates[0].answer == "XXXVIII"


def test_verified_disagreement_still_abstains_on_ambiguous_route() -> None:
    class A(BaseSolver):
        @property
        def name(self):
            return "a_solver"

        @property
        def supported_families(self):
            return ("unknown",)

        def solve(self, row):
            return SolverResult(self.name, "unknown", [SolverCandidate("AAA", self.name, "unknown", "x", 0.9, 1.0, True, "low", {})], False, "", {})

    class B(A):
        @property
        def name(self):
            return "b_solver"

        def solve(self, row):
            return SolverResult(self.name, "unknown", [SolverCandidate("BBB", self.name, "unknown", "x", 0.9, 1.0, True, "low", {})], False, "", {})

    result = SolverEnsemble(solvers=[A(), B()]).run_all({"family": "unknown", "prompt": "ambiguous plain prompt"})
    assert result.abstained
    assert result.reason == "verified_candidate_disagreement"
