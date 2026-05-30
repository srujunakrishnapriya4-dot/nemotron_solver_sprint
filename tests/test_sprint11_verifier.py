from kaggle_anti086.solvers.types import SolverCandidate
from kaggle_anti086.solvers.verifier import verify_candidate


def _candidate(answer: str, family: str = "bit_manipulation", risk: str = "low", verified: bool = True, confidence: float = 0.9) -> SolverCandidate:
    return SolverCandidate(answer, "test", family, "sub", confidence, 1.0, verified, risk, {"width": 4})


def test_verifier_rejects_empty_verbose_wrong_width_and_high_risk_unverified() -> None:
    assert not verify_candidate(_candidate(""), {"family": "bit_manipulation"}).verified
    assert not verify_candidate(_candidate("The answer is 0101"), {"family": "bit_manipulation"}).verified
    assert not verify_candidate(_candidate("01010"), {"family": "bit_manipulation"}).verified
    assert not verify_candidate(_candidate("0101", risk="high", verified=False), {"family": "bit_manipulation"}).verified


def test_verifier_normalizes_numeric_and_roman_and_preserves_symbol() -> None:
    numeric = verify_candidate(_candidate("154.620 meters", "unit_conversion"), {"family": "unit_conversion"})
    assert numeric.verified
    assert numeric.normalized_answer == "154.62"
    roman = verify_candidate(_candidate("38 -> XXXVIII", "roman_numeral"), {"family": "roman_numeral"})
    assert roman.verified
    assert roman.normalized_answer == "XXXVIII"
    symbol = verify_candidate(_candidate("@&", "symbol_mapping"), {"family": "symbol_mapping"})
    assert symbol.verified
    assert symbol.normalized_answer == "@&"


def test_verifier_accepts_low_risk_verified_candidate() -> None:
    result = verify_candidate(_candidate("0101"), {"family": "bit_manipulation", "prompt": "0000 -> 1111"})
    assert result.verified
