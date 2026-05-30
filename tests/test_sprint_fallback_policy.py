from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.sprint_solvers import SolverCandidate, SolverResult, SprintSolverError, decide_model_fallback  # noqa: E402


def test_fallback_allowed_after_abstain() -> None:
    decision = decide_model_fallback(SolverResult(status="abstain", prediction=None))

    assert decision.allow_model_fallback is True
    assert decision.reason == "symbolic_abstained"


def test_fallback_rejected_after_disagreement() -> None:
    decision = decide_model_fallback(SolverResult(status="disagreement", prediction=None))

    assert decision.allow_model_fallback is False
    assert decision.reason == "symbolic_disagreement"


def test_fallback_rejected_when_verified_symbolic_answer_exists() -> None:
    candidate = SolverCandidate("unit", ("2",), "4")
    result = SolverResult(status="solved", prediction="4", verified_candidates=(candidate,))
    decision = decide_model_fallback(result)

    assert decision.allow_model_fallback is False
    assert decision.reason == "verified_symbolic_answer_exists"


def test_fallback_decision_hash_forgery_rejected() -> None:
    decision = decide_model_fallback("abstain")

    with pytest.raises(SprintSolverError):
        replace(decision, decision_hash="forged")


def test_fallback_does_not_call_model_apis(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked_import(name: str, *args, **kwargs):
        if name in {"openai", "kaggle", "torch", "transformers"}:
            raise AssertionError(f"forbidden import attempted: {name}")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", blocked_import)

    decision = decide_model_fallback("no_solution")

    assert decision.allow_model_fallback is True
