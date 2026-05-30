"""Decision-only model fallback policy for Sprint 1."""

from __future__ import annotations

from dataclasses import dataclass, fields

from nemotron_engine.core.schemas import stable_hash

from .solver_base import SolverResult, SprintSolverError


_FALLBACK_ALLOWED_STATUSES = {"abstain", "no_solution"}


@dataclass(frozen=True)
class FallbackDecision:
    allow_model_fallback: bool
    symbolic_status: str
    reason: str
    verified_symbolic_answer: str | None = None
    decision_hash: str = ""

    def __post_init__(self) -> None:
        if not self.symbolic_status.strip():
            raise SprintSolverError("symbolic_status must be non-empty.")
        if not self.reason.strip():
            raise SprintSolverError("reason must be non-empty.")
        if self.verified_symbolic_answer is not None:
            object.__setattr__(self, "verified_symbolic_answer", str(self.verified_symbolic_answer))
        expected = stable_hash({item.name: getattr(self, item.name) for item in fields(self) if item.name != "decision_hash"})
        if not self.decision_hash:
            object.__setattr__(self, "decision_hash", expected)
        elif self.decision_hash != expected:
            raise SprintSolverError("decision_hash does not match fallback decision payload.")


def decide_model_fallback(
    symbolic_result: SolverResult | str,
    *,
    verified_symbolic_answer: str | None = None,
) -> FallbackDecision:
    """Return fallback metadata only; this function never calls a model API."""

    if isinstance(symbolic_result, SolverResult):
        status = symbolic_result.status
        answer = verified_symbolic_answer if verified_symbolic_answer is not None else symbolic_result.prediction
    else:
        status = str(symbolic_result)
        answer = verified_symbolic_answer

    if answer is not None and str(answer).strip():
        return FallbackDecision(
            allow_model_fallback=False,
            symbolic_status=status,
            reason="verified_symbolic_answer_exists",
            verified_symbolic_answer=str(answer),
        )
    if status == "disagreement":
        return FallbackDecision(
            allow_model_fallback=False,
            symbolic_status=status,
            reason="symbolic_disagreement",
        )
    if status in _FALLBACK_ALLOWED_STATUSES:
        return FallbackDecision(
            allow_model_fallback=True,
            symbolic_status=status,
            reason="symbolic_abstained",
        )
    return FallbackDecision(
        allow_model_fallback=False,
        symbolic_status=status,
        reason="symbolic_status_not_fallback_eligible",
    )


__all__ = ["FallbackDecision", "decide_model_fallback"]
