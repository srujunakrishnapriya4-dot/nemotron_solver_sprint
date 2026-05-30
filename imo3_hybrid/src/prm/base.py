from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProcessStepInput(BaseModel):
    step_index: int = Field(ge=1)
    description: str = ""
    operator_name: str | None = None
    symbolic_valid: bool = True
    proof_obligation_ids: tuple[str, ...] = Field(default_factory=tuple)
    state_fingerprint: str | None = None


class ProcessObligationInput(BaseModel):
    obligation_id: str
    status: str = "open"
    claim: str = ""
    evidence_kind_required: str = "other"
    originating_node_id: str = ""
    target_goal_id: str | None = None


class ProcessScoreInput(BaseModel):
    problem_id: str = ""
    branch_id: str = ""
    steps: tuple[ProcessStepInput, ...] = Field(default_factory=tuple)
    proof_obligations: tuple[ProcessObligationInput, ...] = Field(default_factory=tuple)
    route_uncertainty: float = Field(0.0, ge=0.0, le=1.0)
    retrieval_support: float = Field(0.0, ge=0.0, le=1.0)
    symbolic_score: float = Field(0.0, ge=0.0, le=1.0)
    symbolic_passed: bool = False
    exact_symbolic_match: bool = False
    logical_consistency_hint: float | None = Field(default=None, ge=0.0, le=1.0)
    completeness_hint: float | None = Field(default=None, ge=0.0, le=1.0)
    answer_present: bool = False
    contradiction_count: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StepQualityScore(BaseModel):
    step_index: int = Field(ge=1)
    step_quality: float = Field(0.0, ge=0.0, le=1.0)
    prefix_quality: float = Field(0.0, ge=0.0, le=1.0)
    obligation_coverage: float = Field(0.0, ge=0.0, le=1.0)
    contradiction_risk: float = Field(0.0, ge=0.0, le=1.0)
    repairability_hint: float = Field(0.0, ge=0.0, le=1.0)
    notes: tuple[str, ...] = Field(default_factory=tuple)


class ProcessScoreSummary(BaseModel):
    step_scores: tuple[StepQualityScore, ...] = Field(default_factory=tuple)
    mean_step_quality: float = Field(0.0, ge=0.0, le=1.0)
    prefix_quality: float = Field(0.0, ge=0.0, le=1.0)
    open_obligation_burden: float = Field(0.0, ge=0.0, le=1.0)
    contradiction_risk: float = Field(0.0, ge=0.0, le=1.0)
    branch_search_value: float = Field(0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "ProcessObligationInput",
    "ProcessScoreInput",
    "ProcessScoreSummary",
    "ProcessStepInput",
    "StepQualityScore",
]
