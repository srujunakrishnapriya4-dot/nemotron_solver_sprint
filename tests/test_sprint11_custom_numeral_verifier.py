from kaggle_anti086.solvers.types import SolverCandidate
from kaggle_anti086.solvers.verifier import verify_candidate


def _candidate(answer: str, family: str) -> SolverCandidate:
    return SolverCandidate(answer, "test", family, "sub", 0.9, 1.0, True, "low", {})


def test_standard_roman_invalid_rejected() -> None:
    assert not verify_candidate(_candidate("@@&&", "roman_numeral"), {"family": "roman_numeral"}).verified


def test_custom_numeral_symbolic_output_accepted() -> None:
    result = verify_candidate(_candidate("@@&&", "custom_numeral"), {"family": "custom_numeral"})
    assert result.verified


def test_custom_numeral_verbose_and_empty_rejected() -> None:
    assert not verify_candidate(_candidate("@@&& because rule", "custom_numeral"), {"family": "custom_numeral"}).verified
    assert not verify_candidate(_candidate("", "custom_numeral"), {"family": "custom_numeral"}).verified


def test_custom_numeral_allowed_symbols_enforced() -> None:
    row = {"family": "custom_numeral", "metadata": {"allowed_symbols": "@&"}}
    assert verify_candidate(_candidate("@@&&", "custom_numeral"), row).verified
    assert not verify_candidate(_candidate("@@##", "custom_numeral"), row).verified
