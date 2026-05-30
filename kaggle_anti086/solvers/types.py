from __future__ import annotations

from dataclasses import dataclass, field

from kaggle_anti086.data.schema import ALLOWED_FAMILIES


@dataclass(frozen=True)
class SolverCandidate:
    answer: str
    source: str
    family: str
    subfamily: str
    confidence: float
    example_consistency: float
    verified: bool
    risk: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SolverResult:
    solver_name: str
    family: str
    candidates: list[SolverCandidate] = field(default_factory=list)
    abstained: bool = False
    reason: str = ""
    metadata: dict = field(default_factory=dict)


def validate_candidate(candidate: SolverCandidate) -> None:
    if not 0.0 <= float(candidate.confidence) <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    if not 0.0 <= float(candidate.example_consistency) <= 1.0:
        raise ValueError("example_consistency must be in [0, 1]")
    if candidate.verified and not str(candidate.answer).strip():
        raise ValueError("verified candidate must have non-empty answer")
    if not str(candidate.source).strip():
        raise ValueError("source must be non-empty")
    if candidate.family not in ALLOWED_FAMILIES:
        raise ValueError(f"unknown family: {candidate.family}")
    if candidate.risk not in {"low", "medium", "high"}:
        raise ValueError(f"unknown risk: {candidate.risk}")
