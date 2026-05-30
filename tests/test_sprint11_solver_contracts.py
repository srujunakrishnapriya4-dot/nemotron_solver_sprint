from __future__ import annotations

import pytest

from kaggle_anti086.solvers.base import BaseSolver
from kaggle_anti086.solvers.types import SolverCandidate, validate_candidate


def candidate(**overrides) -> SolverCandidate:
    payload = {
        "answer": "42",
        "source": "fixture",
        "family": "numeric_formula",
        "subfamily": "linear_ax",
        "confidence": 0.8,
        "example_consistency": 1.0,
        "verified": True,
        "risk": "low",
        "metadata": {},
    }
    payload.update(overrides)
    return SolverCandidate(**payload)


def test_valid_solver_candidate_passes() -> None:
    validate_candidate(candidate())


@pytest.mark.parametrize("field,value", [("confidence", -0.1), ("confidence", 1.1), ("example_consistency", 1.2)])
def test_score_ranges_enforced(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        validate_candidate(candidate(**{field: value}))


def test_verified_candidate_requires_answer() -> None:
    with pytest.raises(ValueError):
        validate_candidate(candidate(answer=""))


def test_unknown_risk_fails() -> None:
    with pytest.raises(ValueError):
        validate_candidate(candidate(risk="catastrophic"))


class DummySolver(BaseSolver):
    @property
    def name(self) -> str:
        return "dummy"

    @property
    def supported_families(self) -> tuple[str, ...]:
        return ("numeric_formula",)

    def solve(self, row: dict):
        return self.abstain("not implemented")


def test_base_solver_abstain_contract() -> None:
    result = DummySolver().abstain("not implemented")
    assert result.abstained is True
    assert result.solver_name == "dummy"
    assert result.family == "numeric_formula"
