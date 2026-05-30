from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.common.schemas import EdgeType, FailureType, ToolType

try:
    from src.common.schemas import OperatorType
except Exception:  # pragma: no cover
    OperatorType = None  # type: ignore[assignment]


class _StrictSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        populate_by_name=True,
    )


class TransitionKind(str, Enum):
    DERIVE = "derive"
    EXPAND = "expand"
    MERGE = "merge"
    REPAIR = "repair"
    ROLLBACK = "rollback"
    REFUTE = "refute"


class EdgeScoreDelta(_StrictSchema):
    total_delta: float = 0.0
    confidence_delta: float = 0.0
    verifier_delta: float = 0.0
    symbolic_delta: float = 0.0
    retrieval_delta: float = 0.0
    novelty_delta: float = 0.0
    repairability_delta: float = 0.0

    @property
    def is_neutral(self) -> bool:
        return all(
            abs(v) < 1e-12
            for v in (
                self.total_delta,
                self.confidence_delta,
                self.verifier_delta,
                self.symbolic_delta,
                self.retrieval_delta,
                self.novelty_delta,
                self.repairability_delta,
            )
        )


class EdgeEvidenceDelta(_StrictSchema):
    added_constraints: list[str] = Field(default_factory=list)
    removed_constraints: list[str] = Field(default_factory=list)
    added_invariants: list[str] = Field(default_factory=list)
    removed_invariants: list[str] = Field(default_factory=list)
    added_goals: list[str] = Field(default_factory=list)
    resolved_goals: list[str] = Field(default_factory=list)
    tool_evidence_ids: list[str] = Field(default_factory=list)
    verifier_evidence_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def has_material_change(self) -> bool:
        return any(
            (
                self.added_constraints,
                self.removed_constraints,
                self.added_invariants,
                self.removed_invariants,
                self.added_goals,
                self.resolved_goals,
                self.tool_evidence_ids,
                self.verifier_evidence_ids,
                self.notes,
            )
        )


class EdgeProvenance(_StrictSchema):
    source_kind: Literal["operator", "repair", "merge", "rollback", "controller", "symbolic", "verifier", "manual"] = "operator"
    operator_name: str | None = None
    operator_family: str | None = None
    operator_type: Any | None = None
    tool_type: ToolType = ToolType.NONE
    failure_type: FailureType | None = None
    triggered_by_edge_id: str | None = None
    triggered_by_node_id: str | None = None
    trace_step_id: str | None = None
    retrieval_hit_id: str | None = None
    verifier_label_id: str | None = None
    symbolic_check_id: str | None = None
    branch_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize_operator_type(self) -> "EdgeProvenance":
        if self.operator_type is None or OperatorType is None:
            return self
        if isinstance(self.operator_type, OperatorType):
            return self
        try:
            self.operator_type = OperatorType(self.operator_type)
        except Exception:
            self.operator_type = None
        return self


class EdgeRationale(_StrictSchema):
    summary: str
    detailed_reasoning: str | None = None
    compatibility_basis: list[str] = Field(default_factory=list)
    invariant_notes: list[str] = Field(default_factory=list)
    repair_notes: list[str] = Field(default_factory=list)
    merge_basis: list[str] = Field(default_factory=list)
    rollback_basis: list[str] = Field(default_factory=list)
    refutation_basis: list[str] = Field(default_factory=list)
    evidence_notes: list[str] = Field(default_factory=list)


class StateGraphEdge(_StrictSchema):
    edge_id: str
    source_node_id: str
    target_node_id: str

    transition_kind: TransitionKind
    edge_type: EdgeType

    source_state_hash: str | None = None
    target_state_hash: str | None = None

    parent_branch_id: str | None = None
    child_branch_id: str | None = None

    predecessor_edge_id: str | None = None
    supersedes_edge_id: str | None = None

    merge_parent_node_ids: list[str] = Field(default_factory=list)
    repaired_edge_id: str | None = None
    rollback_to_node_id: str | None = None
    refuted_node_id: str | None = None

    provenance: EdgeProvenance
    rationale: EdgeRationale

    score_delta: EdgeScoreDelta = Field(default_factory=EdgeScoreDelta)
    evidence_delta: EdgeEvidenceDelta = Field(default_factory=EdgeEvidenceDelta)

    transition_confidence: float = Field(0.5, ge=0.0, le=1.0)
    compatibility_score: float | None = Field(default=None, ge=0.0, le=1.0)

    is_reversible: bool = False
    is_pruning_relevant: bool = True
    preserves_invariants: bool | None = None

    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_semantics(self) -> "StateGraphEdge":
        if self.source_node_id == self.target_node_id and self.transition_kind in {
            TransitionKind.DERIVE,
            TransitionKind.EXPAND,
            TransitionKind.REPAIR,
            TransitionKind.MERGE,
            TransitionKind.ROLLBACK,
        }:
            raise ValueError("live transition edge must not use identical source/target ids")
        if self.transition_kind is TransitionKind.MERGE and not self.merge_parent_node_ids:
            raise ValueError("merge transition requires merge_parent_node_ids")
        if self.transition_kind is TransitionKind.REPAIR and not (
            self.repaired_edge_id or self.provenance.failure_type or self.rationale.repair_notes
        ):
            raise ValueError("repair transition requires repaired edge or failure semantics")
        if self.transition_kind is TransitionKind.ROLLBACK and not self.rollback_to_node_id:
            raise ValueError("rollback transition requires rollback_to_node_id")
        if self.transition_kind is TransitionKind.REFUTE and not (
            self.refuted_node_id or self.rationale.refutation_basis
        ):
            raise ValueError("refute transition requires refutation target/basis")
        return self

    @property
    def is_repair(self) -> bool:
        return self.transition_kind is TransitionKind.REPAIR

    @property
    def is_merge(self) -> bool:
        return self.transition_kind is TransitionKind.MERGE

    @property
    def is_rollback(self) -> bool:
        return self.transition_kind is TransitionKind.ROLLBACK

    @property
    def is_refutation(self) -> bool:
        return self.transition_kind is TransitionKind.REFUTE

    @property
    def is_expansion(self) -> bool:
        return self.transition_kind is TransitionKind.EXPAND

    @property
    def is_derivation(self) -> bool:
        return self.transition_kind is TransitionKind.DERIVE

    def canonical_key(self) -> tuple[Any, ...]:
        return (
            self.transition_kind.value,
            self.edge_type.value,
            self.source_node_id,
            self.target_node_id,
            self.source_state_hash,
            self.target_state_hash,
            self.provenance.source_kind,
            self.provenance.operator_name,
            self.provenance.operator_family,
            getattr(self.provenance.operator_type, "value", self.provenance.operator_type),
            self.provenance.tool_type.value,
            getattr(self.provenance.failure_type, "value", self.provenance.failure_type),
            tuple(sorted(self.merge_parent_node_ids)),
            self.repaired_edge_id,
            self.rollback_to_node_id,
            self.refuted_node_id,
            self.transition_confidence,
            self.compatibility_score,
            self.is_reversible,
            self.is_pruning_relevant,
            self.preserves_invariants,
            tuple(sorted(self.tags)),
        )

    def stable_hash(self) -> str:
        payload = {
            "canonical_key": self.canonical_key(),
            "score_delta": self.score_delta.model_dump(mode="json"),
            "evidence_delta": self.evidence_delta.model_dump(mode="json"),
            "provenance": self.provenance.model_dump(mode="json"),
            "rationale": self.rationale.model_dump(mode="json"),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()

    def to_graph_store_record(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "transition_kind": self.transition_kind.value,
            "edge_type": self.edge_type.value,
            "stable_hash": self.stable_hash(),
            "source_state_hash": self.source_state_hash,
            "target_state_hash": self.target_state_hash,
            "parent_branch_id": self.parent_branch_id,
            "child_branch_id": self.child_branch_id,
            "predecessor_edge_id": self.predecessor_edge_id,
            "supersedes_edge_id": self.supersedes_edge_id,
            "merge_parent_node_ids": list(self.merge_parent_node_ids),
            "repaired_edge_id": self.repaired_edge_id,
            "rollback_to_node_id": self.rollback_to_node_id,
            "refuted_node_id": self.refuted_node_id,
            "provenance": self.provenance.model_dump(mode="json"),
            "rationale": self.rationale.model_dump(mode="json"),
            "score_delta": self.score_delta.model_dump(mode="json"),
            "evidence_delta": self.evidence_delta.model_dump(mode="json"),
            "transition_confidence": self.transition_confidence,
            "compatibility_score": self.compatibility_score,
            "is_reversible": self.is_reversible,
            "is_pruning_relevant": self.is_pruning_relevant,
            "preserves_invariants": self.preserves_invariants,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }

    def export_state(self) -> dict[str, Any]:
        return self.to_graph_store_record()

    def with_updates(self, **updates: Any) -> "StateGraphEdge":
        data = self.model_dump(mode="python")
        data.update(updates)
        return StateGraphEdge(**data)

    def short_label(self) -> str:
        op = self.provenance.operator_name or self.provenance.operator_family or "transition"
        return f"{self.transition_kind.value}:{op}"

    @classmethod
    def derive(
        cls,
        *,
        edge_id: str,
        source_node_id: str,
        target_node_id: str,
        rationale_summary: str,
        operator_name: str | None = None,
        operator_family: str | None = None,
        operator_type: Any | None = None,
        transition_confidence: float = 0.5,
        compatibility_score: float | None = None,
        score_delta: EdgeScoreDelta | None = None,
        evidence_delta: EdgeEvidenceDelta | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "StateGraphEdge":
        return cls(
            edge_id=edge_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            transition_kind=TransitionKind.DERIVE,
            edge_type=_edge_type_for_kind(TransitionKind.DERIVE),
            provenance=EdgeProvenance(
                source_kind="operator",
                operator_name=operator_name,
                operator_family=operator_family,
                operator_type=operator_type,
            ),
            rationale=EdgeRationale(summary=rationale_summary),
            score_delta=score_delta or EdgeScoreDelta(),
            evidence_delta=evidence_delta or EdgeEvidenceDelta(),
            transition_confidence=transition_confidence,
            compatibility_score=compatibility_score,
            metadata=metadata or {},
        )

    @classmethod
    def expand(
        cls,
        *,
        edge_id: str,
        source_node_id: str,
        target_node_id: str,
        rationale_summary: str,
        parent_branch_id: str | None = None,
        child_branch_id: str | None = None,
        operator_name: str | None = None,
        operator_family: str | None = None,
        transition_confidence: float = 0.5,
        compatibility_score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "StateGraphEdge":
        return cls(
            edge_id=edge_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            transition_kind=TransitionKind.EXPAND,
            edge_type=_edge_type_for_kind(TransitionKind.EXPAND),
            parent_branch_id=parent_branch_id,
            child_branch_id=child_branch_id,
            provenance=EdgeProvenance(
                source_kind="controller",
                operator_name=operator_name,
                operator_family=operator_family,
            ),
            rationale=EdgeRationale(summary=rationale_summary),
            transition_confidence=transition_confidence,
            compatibility_score=compatibility_score,
            metadata=metadata or {},
        )

    @classmethod
    def merge(
        cls,
        *,
        edge_id: str,
        source_node_id: str,
        target_node_id: str,
        merge_parent_node_ids: list[str],
        rationale_summary: str,
        transition_confidence: float = 0.5,
        score_delta: EdgeScoreDelta | None = None,
        evidence_delta: EdgeEvidenceDelta | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "StateGraphEdge":
        return cls(
            edge_id=edge_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            transition_kind=TransitionKind.MERGE,
            edge_type=_edge_type_for_kind(TransitionKind.MERGE),
            merge_parent_node_ids=merge_parent_node_ids,
            provenance=EdgeProvenance(source_kind="merge"),
            rationale=EdgeRationale(summary=rationale_summary, merge_basis=["canonical_state_match"]),
            score_delta=score_delta or EdgeScoreDelta(),
            evidence_delta=evidence_delta or EdgeEvidenceDelta(),
            transition_confidence=transition_confidence,
            metadata=metadata or {},
        )

    @classmethod
    def repair(
        cls,
        *,
        edge_id: str,
        source_node_id: str,
        target_node_id: str,
        rationale_summary: str,
        failure_type: FailureType | None = None,
        repaired_edge_id: str | None = None,
        operator_name: str | None = None,
        operator_family: str | None = None,
        tool_type: ToolType = ToolType.NONE,
        transition_confidence: float = 0.5,
        compatibility_score: float | None = None,
        score_delta: EdgeScoreDelta | None = None,
        evidence_delta: EdgeEvidenceDelta | None = None,
        preserves_invariants: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "StateGraphEdge":
        return cls(
            edge_id=edge_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            transition_kind=TransitionKind.REPAIR,
            edge_type=_edge_type_for_kind(TransitionKind.REPAIR),
            repaired_edge_id=repaired_edge_id,
            provenance=EdgeProvenance(
                source_kind="repair",
                operator_name=operator_name,
                operator_family=operator_family,
                tool_type=tool_type,
                failure_type=failure_type,
            ),
            rationale=EdgeRationale(summary=rationale_summary, repair_notes=["localized_patch_attempt"]),
            score_delta=score_delta or EdgeScoreDelta(),
            evidence_delta=evidence_delta or EdgeEvidenceDelta(),
            transition_confidence=transition_confidence,
            compatibility_score=compatibility_score,
            preserves_invariants=preserves_invariants,
            metadata=metadata or {},
        )

    @classmethod
    def rollback(
        cls,
        *,
        edge_id: str,
        source_node_id: str,
        target_node_id: str,
        rollback_to_node_id: str,
        rationale_summary: str,
        failure_type: FailureType | None = None,
        transition_confidence: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> "StateGraphEdge":
        return cls(
            edge_id=edge_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            transition_kind=TransitionKind.ROLLBACK,
            edge_type=_edge_type_for_kind(TransitionKind.ROLLBACK),
            rollback_to_node_id=rollback_to_node_id,
            provenance=EdgeProvenance(source_kind="rollback", failure_type=failure_type),
            rationale=EdgeRationale(summary=rationale_summary, rollback_basis=["state_reversion"]),
            transition_confidence=transition_confidence,
            is_reversible=True,
            metadata=metadata or {},
        )

    @classmethod
    def refute(
        cls,
        *,
        edge_id: str,
        source_node_id: str,
        target_node_id: str,
        rationale_summary: str,
        refuted_node_id: str | None = None,
        tool_type: ToolType = ToolType.NONE,
        failure_type: FailureType | None = None,
        transition_confidence: float = 0.5,
        score_delta: EdgeScoreDelta | None = None,
        evidence_delta: EdgeEvidenceDelta | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "StateGraphEdge":
        return cls(
            edge_id=edge_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            transition_kind=TransitionKind.REFUTE,
            edge_type=_edge_type_for_kind(TransitionKind.REFUTE),
            refuted_node_id=refuted_node_id,
            provenance=EdgeProvenance(
                source_kind="verifier" if tool_type is ToolType.NONE else "symbolic",
                tool_type=tool_type,
                failure_type=failure_type,
            ),
            rationale=EdgeRationale(summary=rationale_summary, refutation_basis=["symbolic_or_verifier_contradiction"]),
            score_delta=score_delta or EdgeScoreDelta(total_delta=-1.0, confidence_delta=-0.25),
            evidence_delta=evidence_delta or EdgeEvidenceDelta(),
            transition_confidence=transition_confidence,
            is_pruning_relevant=True,
            metadata=metadata or {},
        )


def _edge_type_for_kind(kind: TransitionKind) -> EdgeType:
    if kind is TransitionKind.DERIVE:
        return EdgeType.DERIVE
    if kind is TransitionKind.EXPAND:
        return getattr(EdgeType, "EXPAND", getattr(EdgeType, "SUBGOAL_EXPAND"))
    if kind is TransitionKind.MERGE:
        return EdgeType.MERGE
    if kind is TransitionKind.REPAIR:
        return EdgeType.REPAIR
    if kind is TransitionKind.ROLLBACK:
        return EdgeType.ROLLBACK
    if kind is TransitionKind.REFUTE:
        return EdgeType.REFUTE
    raise ValueError(f"Unsupported transition kind: {kind!r}")


ReasoningStateEdge = StateGraphEdge
GraphTransitionEdge = StateGraphEdge

__all__ = [
    "TransitionKind",
    "EdgeScoreDelta",
    "EdgeEvidenceDelta",
    "EdgeProvenance",
    "EdgeRationale",
    "StateGraphEdge",
    "ReasoningStateEdge",
    "GraphTransitionEdge",
]
