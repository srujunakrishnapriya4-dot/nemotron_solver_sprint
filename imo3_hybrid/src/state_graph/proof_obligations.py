from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from src.common.utils import safe_normalize_text, stable_hash


class ProofEvidenceKind(str, Enum):
    SYMBOLIC_CHECK = "symbolic_check"
    VERIFIER = "verifier"
    GOAL = "goal"
    CONSTRAINT = "constraint"
    RETRIEVAL = "retrieval"
    REPAIR = "repair"
    OTHER = "other"


class ProofObligationStatus(str, Enum):
    OPEN = "open"
    PARTIAL = "partial"
    DISCHARGED = "discharged"
    CONTRADICTED = "contradicted"
    BLOCKED = "blocked"


def _normalize_text(value: Any) -> str:
    return safe_normalize_text(str(value or ""))


@dataclass(frozen=True)
class ContradictionProvenance:
    source: str
    summary: str
    evidence_id: str | None = None
    failure_type: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProofObligation:
    obligation_id: str
    originating_node_id: str
    claim: str
    normalized_claim: str
    evidence_kind_required: ProofEvidenceKind
    status: ProofObligationStatus = ProofObligationStatus.OPEN
    discharged_by: tuple[str, ...] = field(default_factory=tuple)
    contradiction_provenance: ContradictionProvenance | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    target_goal_id: str | None = None
    source_constraint_ids: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        originating_node_id: str,
        claim: str,
        evidence_kind_required: ProofEvidenceKind | str,
        status: ProofObligationStatus | str = ProofObligationStatus.OPEN,
        discharged_by: Sequence[str] | None = None,
        contradiction_provenance: ContradictionProvenance | None = None,
        notes: Sequence[str] | None = None,
        target_goal_id: str | None = None,
        source_constraint_ids: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        obligation_id: str | None = None,
    ) -> "ProofObligation":
        normalized_claim = _normalize_text(claim)
        evidence_kind = (
            evidence_kind_required
            if isinstance(evidence_kind_required, ProofEvidenceKind)
            else ProofEvidenceKind(str(evidence_kind_required))
        )
        obligation_status = (
            status if isinstance(status, ProofObligationStatus) else ProofObligationStatus(str(status))
        )
        payload = {
            "originating_node_id": originating_node_id,
            "claim": normalized_claim,
            "evidence_kind_required": evidence_kind.value,
            "target_goal_id": target_goal_id,
            "source_constraint_ids": sorted(set(source_constraint_ids or ())),
        }
        return cls(
            obligation_id=obligation_id or stable_hash("proof_obligation", payload),
            originating_node_id=_normalize_text(originating_node_id),
            claim=_normalize_text(claim),
            normalized_claim=normalized_claim,
            evidence_kind_required=evidence_kind,
            status=obligation_status,
            discharged_by=tuple(sorted(set(_normalize_text(item) for item in (discharged_by or ()) if _normalize_text(item)))),
            contradiction_provenance=contradiction_provenance,
            notes=tuple(_normalize_text(item) for item in (notes or ()) if _normalize_text(item)),
            target_goal_id=_normalize_text(target_goal_id) or None,
            source_constraint_ids=tuple(
                sorted(
                    set(_normalize_text(item) for item in (source_constraint_ids or ()) if _normalize_text(item))
                )
            ),
            metadata=dict(metadata or {}),
        )

    def canonical_key(self) -> tuple[Any, ...]:
        return (
            self.normalized_claim,
            self.evidence_kind_required.value,
            self.target_goal_id,
            self.source_constraint_ids,
        )

    def is_open(self) -> bool:
        return self.status in {ProofObligationStatus.OPEN, ProofObligationStatus.PARTIAL}

    def discharge(
        self,
        *,
        discharged_by: str,
        notes: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ProofObligation":
        update_notes = self.notes + tuple(
            _normalize_text(item) for item in (notes or ()) if _normalize_text(item)
        )
        update_metadata = dict(self.metadata)
        update_metadata.update(dict(metadata or {}))
        return ProofObligation.create(
            obligation_id=self.obligation_id,
            originating_node_id=self.originating_node_id,
            claim=self.claim,
            evidence_kind_required=self.evidence_kind_required,
            status=ProofObligationStatus.DISCHARGED,
            discharged_by=self.discharged_by + (_normalize_text(discharged_by),),
            contradiction_provenance=None,
            notes=update_notes,
            target_goal_id=self.target_goal_id,
            source_constraint_ids=self.source_constraint_ids,
            metadata=update_metadata,
        )

    def contradict(
        self,
        *,
        source: str,
        summary: str,
        evidence_id: str | None = None,
        failure_type: str | None = None,
        notes: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ProofObligation":
        update_notes = self.notes + tuple(
            _normalize_text(item) for item in (notes or ()) if _normalize_text(item)
        )
        contradiction = ContradictionProvenance(
            source=_normalize_text(source),
            summary=_normalize_text(summary),
            evidence_id=_normalize_text(evidence_id) or None,
            failure_type=_normalize_text(failure_type) or None,
            metadata=dict(metadata or {}),
        )
        update_metadata = dict(self.metadata)
        update_metadata.update(dict(metadata or {}))
        return ProofObligation.create(
            obligation_id=self.obligation_id,
            originating_node_id=self.originating_node_id,
            claim=self.claim,
            evidence_kind_required=self.evidence_kind_required,
            status=ProofObligationStatus.CONTRADICTED,
            discharged_by=self.discharged_by,
            contradiction_provenance=contradiction,
            notes=update_notes,
            target_goal_id=self.target_goal_id,
            source_constraint_ids=self.source_constraint_ids,
            metadata=update_metadata,
        )

    def add_note(self, note: str) -> "ProofObligation":
        return ProofObligation.create(
            obligation_id=self.obligation_id,
            originating_node_id=self.originating_node_id,
            claim=self.claim,
            evidence_kind_required=self.evidence_kind_required,
            status=self.status,
            discharged_by=self.discharged_by,
            contradiction_provenance=self.contradiction_provenance,
            notes=self.notes + ((_normalize_text(note),) if _normalize_text(note) else ()),
            target_goal_id=self.target_goal_id,
            source_constraint_ids=self.source_constraint_ids,
            metadata=self.metadata,
        )


def obligation_status_counts(obligations: Sequence[ProofObligation]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for obligation in obligations:
        counts[obligation.status.value] = counts.get(obligation.status.value, 0) + 1
    return counts


__all__ = [
    "ContradictionProvenance",
    "ProofEvidenceKind",
    "ProofObligation",
    "ProofObligationStatus",
    "obligation_status_counts",
]
