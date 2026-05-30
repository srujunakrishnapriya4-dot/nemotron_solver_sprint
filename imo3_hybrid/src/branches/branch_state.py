from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from src.common.metrics import clamp01 as _clamp01
from src.common.metrics import compose_branch_score
from src.common.schemas import ParsedProblem, RouteDecision
from src.common.utils import safe_normalize_text as _normalize_text
from src.common.utils import stable_hash as _stable_hash
from src.state_graph.node import ReasoningStateNode
from src.state_graph.proof_obligations import ProofObligation, obligation_status_counts


JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]


class BranchPhase(str, Enum):
    INITIALIZED = "initialized"
    RETRIEVED = "retrieved"
    REASONING = "reasoning"
    VERIFIED = "verified"
    CRITIQUED = "critiqued"
    REPAIRED = "repaired"
    ROLLED_BACK = "rolled_back"
    SOLVED = "solved"
    FAILED = "failed"
    PRUNED = "pruned"


class BranchStepKind(str, Enum):
    INITIAL = "initial"
    RETRIEVAL = "retrieval"
    OPERATOR = "operator"
    SYMBOLIC = "symbolic"
    VERIFIER = "verifier"
    CRITIQUE = "critique"
    REPAIR = "repair"
    ROLLBACK = "rollback"
    FINALIZE = "finalize"


class PatchKind(str, Enum):
    REPAIR = "repair"
    ROLLBACK = "rollback"
    CANDIDATE_UPDATE = "candidate_update"
    PHASE_UPDATE = "phase_update"


@dataclass(frozen=True)
class BranchCursor:
    step_count: int = 0
    retrieval_count: int = 0
    symbolic_count: int = 0
    verifier_count: int = 0
    critique_count: int = 0
    repair_count: int = 0
    candidate_count: int = 0
    prior_snapshot_count: int = 0


@dataclass(frozen=True)
class BranchCandidateState:
    candidate_id: str
    cluster_id: str
    raw_answer: str
    canonical_answer: str
    answer_source: str
    confidence: float = 0.0
    supporting_step_ids: tuple[str, ...] = field(default_factory=tuple)
    supporting_evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        raw_answer: str,
        canonical_answer: str | None = None,
        answer_source: str = "unknown",
        confidence: float = 0.0,
        supporting_step_ids: tuple[str, ...] = (),
        supporting_evidence_ids: tuple[str, ...] = (),
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> "BranchCandidateState":
        raw = _normalize_text(raw_answer)
        canonical = _normalize_text(canonical_answer or raw)
        cluster_id = _stable_hash("candidate_cluster", {"canonical_answer": canonical})
        candidate_id = _stable_hash(
            "candidate",
            {
                "cluster_id": cluster_id,
                "raw_answer": raw,
                "source": answer_source,
                "supporting_step_ids": supporting_step_ids,
            },
        )
        return cls(
            candidate_id=candidate_id,
            cluster_id=cluster_id,
            raw_answer=raw,
            canonical_answer=canonical,
            answer_source=_normalize_text(answer_source),
            confidence=_clamp01(confidence),
            supporting_step_ids=tuple(supporting_step_ids),
            supporting_evidence_ids=tuple(supporting_evidence_ids),
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True)
class BranchOperatorPriorSnapshot:
    snapshot_id: str
    phase: BranchPhase
    operator_prior: Mapping[str, float]
    problem_type: Mapping[str, float]
    archetypes: Mapping[str, float]
    retrieval_support: Mapping[str, float] = field(default_factory=dict)
    source: str = "route"

    @classmethod
    def create(
        cls,
        *,
        phase: BranchPhase,
        operator_prior: Mapping[str, float],
        problem_type: Mapping[str, float],
        archetypes: Mapping[str, float],
        retrieval_support: Mapping[str, float] | None = None,
        source: str = "route",
    ) -> "BranchOperatorPriorSnapshot":
        normalized_prior = {k: float(operator_prior[k]) for k in sorted(operator_prior)}
        payload = {
            "phase": phase.value,
            "operator_prior": normalized_prior,
            "problem_type": {k: float(problem_type[k]) for k in sorted(problem_type)},
            "archetypes": {k: float(archetypes[k]) for k in sorted(archetypes)},
            "retrieval_support": {
                k: float((retrieval_support or {})[k]) for k in sorted(retrieval_support or {})
            },
            "source": source,
        }
        return cls(
            snapshot_id=_stable_hash("prior_snapshot", payload),
            phase=phase,
            operator_prior=normalized_prior,
            problem_type={k: float(problem_type[k]) for k in sorted(problem_type)},
            archetypes={k: float(archetypes[k]) for k in sorted(archetypes)},
            retrieval_support={
                k: float((retrieval_support or {})[k]) for k in sorted(retrieval_support or {})
            },
            source=_normalize_text(source),
        )


@dataclass(frozen=True)
class BranchStep:
    step_id: str
    index: int
    kind: BranchStepKind
    phase: BranchPhase
    description: str
    operator_name: str | None = None
    node_id: str | None = None
    state_fingerprint: str | None = None
    proof_obligation_ids: tuple[str, ...] = field(default_factory=tuple)
    summary_text: str | None = None

    @classmethod
    def create(
        cls,
        *,
        index: int,
        kind: BranchStepKind,
        phase: BranchPhase,
        description: str,
        operator_name: str | None = None,
        node: ReasoningStateNode | None = None,
        summary_text: str | None = None,
    ) -> "BranchStep":
        payload = {
            "index": index,
            "kind": kind.value,
            "phase": phase.value,
            "description": _normalize_text(description),
            "operator_name": _normalize_text(operator_name),
            "node_id": getattr(node, "node_id", None),
            "state_fingerprint": getattr(node, "state_fingerprint", None),
        }
        return cls(
            step_id=_stable_hash("branch_step", payload),
            index=index,
            kind=kind,
            phase=phase,
            description=_normalize_text(description),
            operator_name=_normalize_text(operator_name) or None,
            node_id=getattr(node, "node_id", None),
            state_fingerprint=getattr(node, "state_fingerprint", None),
            proof_obligation_ids=tuple(
                item.obligation_id for item in getattr(node, "proof_obligations", ()) if getattr(item, "obligation_id", None)
            ),
            summary_text=_normalize_text(summary_text) or None,
        )


@dataclass(frozen=True)
class RetrievalEvidence:
    evidence_id: str
    trace_id: str
    score: float
    compatibility_score: float = 0.0
    problem: str = ""
    solution: str = ""
    answer: str = ""
    domain: str = ""
    archetypes: tuple[str, ...] = field(default_factory=tuple)
    operators_used: tuple[str, ...] = field(default_factory=tuple)
    operator_support: Mapping[str, float] = field(default_factory=dict)
    repair_operator_hints: tuple[str, ...] = field(default_factory=tuple)
    branch_continuation_bias: Mapping[str, float] = field(default_factory=dict)
    relevant_obligation_claims: tuple[str, ...] = field(default_factory=tuple)
    relevant_evidence_kinds: tuple[str, ...] = field(default_factory=tuple)
    failure_mode_support: tuple[str, ...] = field(default_factory=tuple)
    compatibility_reasons: tuple[str, ...] = field(default_factory=tuple)
    strategy_summary: str = ""
    source: str = "retrieval"


@dataclass(frozen=True)
class SymbolicEvidence:
    evidence_id: str
    passed: bool
    score: float
    check_name: str
    summary: str
    supporting_step_ids: tuple[str, ...] = field(default_factory=tuple)
    discharged_obligation_ids: tuple[str, ...] = field(default_factory=tuple)
    contradicted_obligation_ids: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class VerifierEvidence:
    evidence_id: str
    probability: float
    logical_consistency: float
    symbolic_agreement: float
    completeness: float
    repairability: float
    summary: str
    answer_correctness_likelihood: float = 0.0
    step_quality: float = 0.0
    prefix_quality: float = 0.0
    open_obligation_burden: float = 0.0
    supporting_obligation_ids: tuple[str, ...] = field(default_factory=tuple)
    failure_step_index: int | None = None
    failure_step_id: str | None = None
    failure_node_id: str | None = None
    failure_obligation_id: str | None = None
    decomposition: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class CritiqueEvidence:
    evidence_id: str
    changed_answer: bool
    summary: str
    critique_score: float
    recommended_answer: str | None = None


@dataclass(frozen=True)
class BranchScoreBreakdown:
    verifier_probability: float = 0.0
    tool_consistency: float = 0.0
    answer_agreement: float = 0.0
    branch_novelty: float = 0.0
    exact_symbolic_check: float = 0.0
    retrieval_support: float = 0.0
    logical_consistency: float = 0.0
    completeness: float = 0.0
    repairability: float = 0.0
    step_quality: float = 0.0
    prefix_quality: float = 0.0
    open_obligation_burden: float = 0.0
    penalty: float = 0.0

    def composite_score(self) -> float:
        legacy = compose_branch_score(
            {
                "verifier_probability": self.verifier_probability,
                "tool_consistency": self.tool_consistency,
                "answer_agreement": self.answer_agreement,
                "branch_novelty": self.branch_novelty,
                "exact_symbolic_check": self.exact_symbolic_check,
                "retrieval_support": self.retrieval_support,
                "penalty": self.penalty,
            }
        )
        enriched = (
            0.04 * self.logical_consistency
            + 0.03 * self.completeness
            + 0.03 * self.repairability
            + 0.06 * self.step_quality
            + 0.08 * self.prefix_quality
            - 0.08 * self.open_obligation_burden
        )
        return _clamp01(legacy + enriched)


@dataclass(frozen=True)
class BranchPatchRecord:
    patch_id: str
    kind: PatchKind
    reason: str
    cursor_before: BranchCursor
    cursor_after: BranchCursor
    phase_before: BranchPhase
    phase_after: BranchPhase
    candidate_id_before: str | None = None
    candidate_id_after: str | None = None


@dataclass(frozen=True)
class BranchRepairRecord:
    repair_id: str
    patch_id: str
    reason: str
    from_step_index: int
    to_step_index: int
    success: bool
    failure_type: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class BranchSummary:
    branch_id: str
    root_branch_id: str
    parent_branch_id: str | None
    phase: BranchPhase
    current_step_count: int
    current_candidate_id: str | None
    current_answer: str | None
    current_canonical_answer: str | None
    retrieval_count: int
    symbolic_count: int
    verifier_count: int
    critique_count: int
    repair_count: int
    open_proof_obligation_count: int
    score: float
    latest_node_id: str | None
    lineage: tuple[str, ...]


@dataclass
class BranchState:
    branch_id: str
    problem_id: str
    root_branch_id: str
    parent_branch_id: str | None
    lineage: tuple[str, ...]
    phase: BranchPhase
    route: RouteDecision
    parsed_problem: ParsedProblem | None = None
    root_node_id: str | None = None
    latest_node_id: str | None = None
    latest_state_fingerprint: str | None = None
    proof_state_fingerprint: str | None = None
    proof_obligations: list[ProofObligation] = field(default_factory=list)
    steps: list[BranchStep] = field(default_factory=list)
    retrieval_evidence: list[RetrievalEvidence] = field(default_factory=list)
    symbolic_evidence: list[SymbolicEvidence] = field(default_factory=list)
    verifier_evidence: list[VerifierEvidence] = field(default_factory=list)
    critique_evidence: list[CritiqueEvidence] = field(default_factory=list)
    candidate_history: list[BranchCandidateState] = field(default_factory=list)
    prior_snapshots: list[BranchOperatorPriorSnapshot] = field(default_factory=list)
    patch_history: list[BranchPatchRecord] = field(default_factory=list)
    repair_history: list[BranchRepairRecord] = field(default_factory=list)
    score_breakdown: BranchScoreBreakdown = field(default_factory=BranchScoreBreakdown)
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    active_cursor: BranchCursor = field(default_factory=BranchCursor)

    @classmethod
    def create(
        cls,
        *,
        route: RouteDecision,
        parsed_problem: ParsedProblem | None = None,
        branch_id: str | None = None,
        parent_branch_id: str | None = None,
        root_branch_id: str | None = None,
        lineage: tuple[str, ...] | None = None,
        root_node: ReasoningStateNode | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> "BranchState":
        pid = route.problem_id if parsed_problem is None else parsed_problem.problem_id
        computed_branch_id = branch_id or _stable_hash(
            "branch",
            {"problem_id": pid, "parent_branch_id": parent_branch_id, "root_node_id": getattr(root_node, "node_id", None)},
        )
        root_id = root_branch_id or parent_branch_id or computed_branch_id
        branch_lineage = tuple(lineage or (() if parent_branch_id is None else (root_id, parent_branch_id)))
        state = cls(
            branch_id=computed_branch_id,
            problem_id=pid,
            root_branch_id=root_id,
            parent_branch_id=parent_branch_id,
            lineage=branch_lineage,
            phase=BranchPhase.INITIALIZED,
            route=route,
            parsed_problem=parsed_problem,
            root_node_id=getattr(root_node, "node_id", None),
            latest_node_id=getattr(root_node, "node_id", None),
            latest_state_fingerprint=getattr(root_node, "state_fingerprint", None),
            proof_state_fingerprint=cls._proof_state_fingerprint(
                getattr(root_node, "proof_obligations", ()) or ()
            ),
            proof_obligations=list(getattr(root_node, "proof_obligations", ()) or ()),
            metadata=dict(metadata or {}),
        )
        state._append_prior_snapshot(phase=BranchPhase.INITIALIZED, source="route_init")
        state._append_step(
            kind=BranchStepKind.INITIAL,
            phase=BranchPhase.INITIALIZED,
            description="branch initialized",
            node=root_node,
            summary_text=getattr(root_node, "summary_text", None),
        )
        return state

    def current_candidate(self) -> BranchCandidateState | None:
        if self.active_cursor.candidate_count <= 0:
            return None
        return self.candidate_history[self.active_cursor.candidate_count - 1]

    def active_steps(self) -> tuple[BranchStep, ...]:
        return tuple(self.steps[: self.active_cursor.step_count])

    def active_retrieval_evidence(self) -> tuple[RetrievalEvidence, ...]:
        return tuple(self.retrieval_evidence[: self.active_cursor.retrieval_count])

    def active_symbolic_evidence(self) -> tuple[SymbolicEvidence, ...]:
        return tuple(self.symbolic_evidence[: self.active_cursor.symbolic_count])

    def active_verifier_evidence(self) -> tuple[VerifierEvidence, ...]:
        return tuple(self.verifier_evidence[: self.active_cursor.verifier_count])

    def active_critique_evidence(self) -> tuple[CritiqueEvidence, ...]:
        return tuple(self.critique_evidence[: self.active_cursor.critique_count])

    def active_proof_obligations(self) -> tuple[ProofObligation, ...]:
        return tuple(self.proof_obligations)

    def active_prior_snapshots(self) -> tuple[BranchOperatorPriorSnapshot, ...]:
        return tuple(self.prior_snapshots[: self.active_cursor.prior_snapshot_count])

    def composite_score(self) -> float:
        return self.score_breakdown.composite_score()

    def set_phase(self, phase: BranchPhase, *, reason: str) -> None:
        before = self.active_cursor
        prior_candidate = self.current_candidate()
        phase_before = self.phase
        self.phase = phase
        self._record_patch(
            kind=PatchKind.PHASE_UPDATE,
            reason=reason,
            cursor_before=before,
            cursor_after=self.active_cursor,
            phase_before=phase_before,
            phase_after=phase,
            candidate_id_before=prior_candidate.candidate_id if prior_candidate else None,
            candidate_id_after=prior_candidate.candidate_id if prior_candidate else None,
        )

    def add_step(
        self,
        *,
        kind: BranchStepKind,
        description: str,
        phase: BranchPhase | None = None,
        operator_name: str | None = None,
        node: ReasoningStateNode | None = None,
        summary_text: str | None = None,
    ) -> BranchStep:
        active_phase = phase or self.phase
        step = self._append_step(
            kind=kind,
            phase=active_phase,
            description=description,
            operator_name=operator_name,
            node=node,
            summary_text=summary_text,
        )
        self.phase = active_phase
        return step

    def set_candidate_answer(
        self,
        *,
        raw_answer: str,
        canonical_answer: str | None = None,
        answer_source: str,
        confidence: float = 0.0,
        supporting_step_ids: tuple[str, ...] = (),
        supporting_evidence_ids: tuple[str, ...] = (),
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> BranchCandidateState:
        before = self.active_cursor
        previous = self.current_candidate()
        candidate = BranchCandidateState.create(
            raw_answer=raw_answer,
            canonical_answer=canonical_answer,
            answer_source=answer_source,
            confidence=confidence,
            supporting_step_ids=supporting_step_ids,
            supporting_evidence_ids=supporting_evidence_ids,
            metadata=metadata,
        )
        self.candidate_history.append(candidate)
        self.active_cursor = BranchCursor(
            step_count=self.active_cursor.step_count,
            retrieval_count=self.active_cursor.retrieval_count,
            symbolic_count=self.active_cursor.symbolic_count,
            verifier_count=self.active_cursor.verifier_count,
            critique_count=self.active_cursor.critique_count,
            repair_count=self.active_cursor.repair_count,
            candidate_count=len(self.candidate_history),
            prior_snapshot_count=self.active_cursor.prior_snapshot_count,
        )
        self._record_patch(
            kind=PatchKind.CANDIDATE_UPDATE,
            reason=f"candidate update via {answer_source}",
            cursor_before=before,
            cursor_after=self.active_cursor,
            phase_before=self.phase,
            phase_after=self.phase,
            candidate_id_before=previous.candidate_id if previous else None,
            candidate_id_after=candidate.candidate_id,
        )
        return candidate

    def attach_retrieval_evidence(
        self,
        *,
        trace_id: str,
        score: float,
        compatibility_score: float = 0.0,
        operator_support: Mapping[str, float] | None = None,
        problem: str = "",
        solution: str = "",
        answer: str = "",
        domain: str = "",
        archetypes: tuple[str, ...] = (),
        operators_used: tuple[str, ...] = (),
        repair_operator_hints: tuple[str, ...] = (),
        branch_continuation_bias: Mapping[str, float] | None = None,
        relevant_obligation_claims: tuple[str, ...] = (),
        relevant_evidence_kinds: tuple[str, ...] = (),
        failure_mode_support: tuple[str, ...] = (),
        compatibility_reasons: tuple[str, ...] = (),
        strategy_summary: str = "",
        source: str = "retrieval",
    ) -> RetrievalEvidence:
        evidence = RetrievalEvidence(
            evidence_id=_stable_hash(
                "retrieval_evidence",
                {"trace_id": trace_id, "score": round(float(score), 6), "source": source},
            ),
            trace_id=_normalize_text(trace_id),
            score=_clamp01(score),
            compatibility_score=_clamp01(compatibility_score),
            problem=_normalize_text(problem),
            solution=_normalize_text(solution),
            answer=_normalize_text(answer),
            domain=_normalize_text(domain),
            archetypes=tuple(archetypes),
            operators_used=tuple(operators_used),
            operator_support={k: float(operator_support[k]) for k in sorted(operator_support or {})},
            repair_operator_hints=tuple(repair_operator_hints),
            branch_continuation_bias={k: float((branch_continuation_bias or {})[k]) for k in sorted(branch_continuation_bias or {})},
            relevant_obligation_claims=tuple(_normalize_text(item) for item in relevant_obligation_claims if _normalize_text(item)),
            relevant_evidence_kinds=tuple(_normalize_text(item) for item in relevant_evidence_kinds if _normalize_text(item)),
            failure_mode_support=tuple(_normalize_text(item) for item in failure_mode_support if _normalize_text(item)),
            compatibility_reasons=tuple(_normalize_text(item) for item in compatibility_reasons if _normalize_text(item)),
            strategy_summary=_normalize_text(strategy_summary),
            source=_normalize_text(source),
        )
        self.retrieval_evidence.append(evidence)
        self.active_cursor = BranchCursor(
            step_count=self.active_cursor.step_count,
            retrieval_count=len(self.retrieval_evidence),
            symbolic_count=self.active_cursor.symbolic_count,
            verifier_count=self.active_cursor.verifier_count,
            critique_count=self.active_cursor.critique_count,
            repair_count=self.active_cursor.repair_count,
            candidate_count=self.active_cursor.candidate_count,
            prior_snapshot_count=self.active_cursor.prior_snapshot_count,
        )
        self.phase = BranchPhase.RETRIEVED
        self._append_prior_snapshot(
            phase=BranchPhase.RETRIEVED,
            source="retrieval_adjustment",
            retrieval_support=evidence.operator_support,
        )
        return evidence

    def attach_symbolic_evidence(
        self,
        *,
        passed: bool,
        score: float,
        check_name: str,
        summary: str,
        supporting_step_ids: tuple[str, ...] = (),
        discharged_obligation_ids: tuple[str, ...] = (),
        contradicted_obligation_ids: tuple[str, ...] = (),
        proof_obligations: tuple[ProofObligation, ...] | None = None,
    ) -> SymbolicEvidence:
        evidence = SymbolicEvidence(
            evidence_id=_stable_hash("symbolic_evidence", {"check_name": check_name, "summary": summary}),
            passed=bool(passed),
            score=_clamp01(score),
            check_name=_normalize_text(check_name),
            summary=_normalize_text(summary),
            supporting_step_ids=tuple(supporting_step_ids),
            discharged_obligation_ids=tuple(discharged_obligation_ids),
            contradicted_obligation_ids=tuple(contradicted_obligation_ids),
        )
        self.symbolic_evidence.append(evidence)
        if proof_obligations is not None:
            self._set_proof_obligations(tuple(proof_obligations))
        self.active_cursor = BranchCursor(
            step_count=self.active_cursor.step_count,
            retrieval_count=self.active_cursor.retrieval_count,
            symbolic_count=len(self.symbolic_evidence),
            verifier_count=self.active_cursor.verifier_count,
            critique_count=self.active_cursor.critique_count,
            repair_count=self.active_cursor.repair_count,
            candidate_count=self.active_cursor.candidate_count,
            prior_snapshot_count=self.active_cursor.prior_snapshot_count,
        )
        self.phase = BranchPhase.VERIFIED
        return evidence

    def attach_verifier_evidence(
        self,
        *,
        probability: float,
        logical_consistency: float,
        symbolic_agreement: float = 0.0,
        completeness: float,
        repairability: float,
        summary: str,
        answer_correctness_likelihood: float | None = None,
        step_quality: float = 0.0,
        prefix_quality: float = 0.0,
        open_obligation_burden: float = 0.0,
        supporting_obligation_ids: tuple[str, ...] = (),
        failure_step_index: int | None = None,
        failure_step_id: str | None = None,
        failure_node_id: str | None = None,
        failure_obligation_id: str | None = None,
        decomposition: Mapping[str, JsonValue] | None = None,
    ) -> VerifierEvidence:
        evidence = VerifierEvidence(
            evidence_id=_stable_hash("verifier_evidence", {"probability": probability, "summary": summary}),
            probability=_clamp01(probability),
            logical_consistency=_clamp01(logical_consistency),
            symbolic_agreement=_clamp01(symbolic_agreement),
            completeness=_clamp01(completeness),
            repairability=_clamp01(repairability),
            answer_correctness_likelihood=_clamp01(
                probability if answer_correctness_likelihood is None else answer_correctness_likelihood
            ),
            step_quality=_clamp01(step_quality),
            prefix_quality=_clamp01(prefix_quality),
            open_obligation_burden=_clamp01(open_obligation_burden),
            summary=_normalize_text(summary),
            supporting_obligation_ids=tuple(supporting_obligation_ids),
            failure_step_index=failure_step_index,
            failure_step_id=_normalize_text(failure_step_id) or None,
            failure_node_id=_normalize_text(failure_node_id) or None,
            failure_obligation_id=_normalize_text(failure_obligation_id) or None,
            decomposition=dict(decomposition or {}),
        )
        self.verifier_evidence.append(evidence)
        self.active_cursor = BranchCursor(
            step_count=self.active_cursor.step_count,
            retrieval_count=self.active_cursor.retrieval_count,
            symbolic_count=self.active_cursor.symbolic_count,
            verifier_count=len(self.verifier_evidence),
            critique_count=self.active_cursor.critique_count,
            repair_count=self.active_cursor.repair_count,
            candidate_count=self.active_cursor.candidate_count,
            prior_snapshot_count=self.active_cursor.prior_snapshot_count,
        )
        self.phase = BranchPhase.VERIFIED
        return evidence

    def attach_critique_evidence(
        self,
        *,
        changed_answer: bool,
        summary: str,
        critique_score: float,
        recommended_answer: str | None = None,
    ) -> CritiqueEvidence:
        evidence = CritiqueEvidence(
            evidence_id=_stable_hash("critique_evidence", {"summary": summary, "changed_answer": changed_answer}),
            changed_answer=bool(changed_answer),
            summary=_normalize_text(summary),
            critique_score=_clamp01(critique_score),
            recommended_answer=_normalize_text(recommended_answer) or None,
        )
        self.critique_evidence.append(evidence)
        self.active_cursor = BranchCursor(
            step_count=self.active_cursor.step_count,
            retrieval_count=self.active_cursor.retrieval_count,
            symbolic_count=self.active_cursor.symbolic_count,
            verifier_count=self.active_cursor.verifier_count,
            critique_count=len(self.critique_evidence),
            repair_count=self.active_cursor.repair_count,
            candidate_count=self.active_cursor.candidate_count,
            prior_snapshot_count=self.active_cursor.prior_snapshot_count,
        )
        self.phase = BranchPhase.CRITIQUED
        return evidence

    def update_score_breakdown(
        self,
        *,
        verifier_probability: float | None = None,
        tool_consistency: float | None = None,
        answer_agreement: float | None = None,
        branch_novelty: float | None = None,
        exact_symbolic_check: float | None = None,
        retrieval_support: float | None = None,
        logical_consistency: float | None = None,
        completeness: float | None = None,
        repairability: float | None = None,
        step_quality: float | None = None,
        prefix_quality: float | None = None,
        open_obligation_burden: float | None = None,
        penalty: float | None = None,
    ) -> BranchScoreBreakdown:
        current = self.score_breakdown
        self.score_breakdown = BranchScoreBreakdown(
            verifier_probability=current.verifier_probability if verifier_probability is None else _clamp01(verifier_probability),
            tool_consistency=current.tool_consistency if tool_consistency is None else _clamp01(tool_consistency),
            answer_agreement=current.answer_agreement if answer_agreement is None else _clamp01(answer_agreement),
            branch_novelty=current.branch_novelty if branch_novelty is None else _clamp01(branch_novelty),
            exact_symbolic_check=current.exact_symbolic_check if exact_symbolic_check is None else _clamp01(exact_symbolic_check),
            retrieval_support=current.retrieval_support if retrieval_support is None else _clamp01(retrieval_support),
            logical_consistency=current.logical_consistency if logical_consistency is None else _clamp01(logical_consistency),
            completeness=current.completeness if completeness is None else _clamp01(completeness),
            repairability=current.repairability if repairability is None else _clamp01(repairability),
            step_quality=current.step_quality if step_quality is None else _clamp01(step_quality),
            prefix_quality=current.prefix_quality if prefix_quality is None else _clamp01(prefix_quality),
            open_obligation_burden=(
                current.open_obligation_burden
                if open_obligation_burden is None
                else _clamp01(open_obligation_burden)
            ),
            penalty=current.penalty if penalty is None else _clamp01(penalty),
        )
        return self.score_breakdown

    def record_repair(
        self,
        *,
        reason: str,
        from_step_index: int,
        to_step_index: int,
        success: bool,
        failure_type: str | None = None,
        notes: str = "",
    ) -> BranchRepairRecord:
        cursor_before = self.active_cursor
        target_step_count = max(0, min(self.active_cursor.step_count, to_step_index))
        self._trim_steps_to_prefix(target_step_count)
        self.active_cursor = BranchCursor(
            step_count=target_step_count,
            retrieval_count=self.active_cursor.retrieval_count,
            symbolic_count=self.active_cursor.symbolic_count,
            verifier_count=self.active_cursor.verifier_count,
            critique_count=self.active_cursor.critique_count,
            repair_count=self.active_cursor.repair_count,
            candidate_count=self.active_cursor.candidate_count,
            prior_snapshot_count=self.active_cursor.prior_snapshot_count,
        )
        patch = self._record_patch(
            kind=PatchKind.REPAIR,
            reason=reason,
            cursor_before=cursor_before,
            cursor_after=self.active_cursor,
            phase_before=self.phase,
            phase_after=BranchPhase.REPAIRED,
            candidate_id_before=self.current_candidate().candidate_id if self.current_candidate() else None,
            candidate_id_after=self.current_candidate().candidate_id if self.current_candidate() else None,
        )
        repair = BranchRepairRecord(
            repair_id=_stable_hash("repair_record", {"reason": reason, "from": from_step_index, "to": to_step_index}),
            patch_id=patch.patch_id,
            reason=_normalize_text(reason),
            from_step_index=from_step_index,
            to_step_index=to_step_index,
            success=bool(success),
            failure_type=_normalize_text(failure_type) or None,
            notes=_normalize_text(notes),
        )
        self.repair_history.append(repair)
        self.active_cursor = BranchCursor(
            step_count=self.active_cursor.step_count,
            retrieval_count=self.active_cursor.retrieval_count,
            symbolic_count=self.active_cursor.symbolic_count,
            verifier_count=self.active_cursor.verifier_count,
            critique_count=self.active_cursor.critique_count,
            repair_count=len(self.repair_history),
            candidate_count=self.active_cursor.candidate_count,
            prior_snapshot_count=self.active_cursor.prior_snapshot_count,
        )
        self.phase = BranchPhase.REPAIRED
        return repair

    def rollback_to_step_prefix(self, keep_steps: int, *, reason: str) -> BranchPatchRecord:
        cursor_before = self.active_cursor
        target = max(0, min(self.active_cursor.step_count, keep_steps))
        phase_before = self.phase
        self._trim_steps_to_prefix(target)
        self.active_cursor = BranchCursor(
            step_count=target,
            retrieval_count=self.active_cursor.retrieval_count,
            symbolic_count=self.active_cursor.symbolic_count,
            verifier_count=self.active_cursor.verifier_count,
            critique_count=self.active_cursor.critique_count,
            repair_count=self.active_cursor.repair_count,
            candidate_count=self.active_cursor.candidate_count,
            prior_snapshot_count=self.active_cursor.prior_snapshot_count,
        )
        self.phase = BranchPhase.ROLLED_BACK
        return self._record_patch(
            kind=PatchKind.ROLLBACK,
            reason=reason,
            cursor_before=cursor_before,
            cursor_after=self.active_cursor,
            phase_before=phase_before,
            phase_after=BranchPhase.ROLLED_BACK,
            candidate_id_before=self.current_candidate().candidate_id if self.current_candidate() else None,
            candidate_id_after=self.current_candidate().candidate_id if self.current_candidate() else None,
        )

    def fork(self, *, branch_label: str, root_node: ReasoningStateNode | None = None) -> "BranchState":
        child_id = _stable_hash(
            "branch",
            {"parent": self.branch_id, "label": branch_label, "step_count": self.active_cursor.step_count},
        )
        child = BranchState.create(
            route=self.route,
            parsed_problem=self.parsed_problem,
            branch_id=child_id,
            parent_branch_id=self.branch_id,
            root_branch_id=self.root_branch_id,
            lineage=self.lineage + (self.branch_id,),
            root_node=root_node,
            metadata=self.metadata,
        )
        child.steps = list(self.steps)
        child.retrieval_evidence = list(self.retrieval_evidence)
        child.symbolic_evidence = list(self.symbolic_evidence)
        child.verifier_evidence = list(self.verifier_evidence)
        child.critique_evidence = list(self.critique_evidence)
        child.candidate_history = list(self.candidate_history)
        child.prior_snapshots = list(self.prior_snapshots)
        child.patch_history = list(self.patch_history)
        child.repair_history = list(self.repair_history)
        child.active_cursor = self.active_cursor
        child.score_breakdown = self.score_breakdown
        child.phase = self.phase
        child.latest_node_id = self.latest_node_id
        child.latest_state_fingerprint = self.latest_state_fingerprint
        child.proof_state_fingerprint = self.proof_state_fingerprint
        child.proof_obligations = list(self.proof_obligations)
        return child

    def build_summary(self) -> BranchSummary:
        candidate = self.current_candidate()
        return BranchSummary(
            branch_id=self.branch_id,
            root_branch_id=self.root_branch_id,
            parent_branch_id=self.parent_branch_id,
            phase=self.phase,
            current_step_count=self.active_cursor.step_count,
            current_candidate_id=candidate.candidate_id if candidate else None,
            current_answer=candidate.raw_answer if candidate else None,
            current_canonical_answer=candidate.canonical_answer if candidate else None,
            retrieval_count=self.active_cursor.retrieval_count,
            symbolic_count=self.active_cursor.symbolic_count,
            verifier_count=self.active_cursor.verifier_count,
            critique_count=self.active_cursor.critique_count,
            repair_count=self.active_cursor.repair_count,
            open_proof_obligation_count=sum(1 for item in self.proof_obligations if item.is_open()),
            score=self.composite_score(),
            latest_node_id=self.latest_node_id,
            lineage=self.lineage,
        )

    def to_dict(self) -> dict[str, object]:
        summary = self.build_summary()
        candidate = self.current_candidate()
        return {
            "branch_id": self.branch_id,
            "problem_id": self.problem_id,
            "root_branch_id": self.root_branch_id,
            "parent_branch_id": self.parent_branch_id,
            "lineage": list(self.lineage),
            "phase": self.phase.value,
            "route_problem_type": dict(self.route.problem_type),
            "route_archetypes": dict(self.route.archetypes),
            "latest_node_id": self.latest_node_id,
            "latest_state_fingerprint": self.latest_state_fingerprint,
            "proof_state_fingerprint": self.proof_state_fingerprint,
            "proof_obligation_status_counts": obligation_status_counts(self.proof_obligations),
            "proof_obligations": [
                {
                    "obligation_id": item.obligation_id,
                    "status": item.status.value,
                    "claim": item.claim,
                    "evidence_kind_required": item.evidence_kind_required.value,
                }
                for item in self.proof_obligations
            ],
            "active_cursor": self.active_cursor.__dict__,
            "current_candidate": None if candidate is None else {
                "candidate_id": candidate.candidate_id,
                "cluster_id": candidate.cluster_id,
                "raw_answer": candidate.raw_answer,
                "canonical_answer": candidate.canonical_answer,
                "answer_source": candidate.answer_source,
                "confidence": candidate.confidence,
            },
            "score_breakdown": self.score_breakdown.__dict__,
            "summary": summary.__dict__,
        }

    def _append_step(
        self,
        *,
        kind: BranchStepKind,
        phase: BranchPhase,
        description: str,
        operator_name: str | None = None,
        node: ReasoningStateNode | None = None,
        summary_text: str | None = None,
    ) -> BranchStep:
        step = BranchStep.create(
            index=self.active_cursor.step_count,
            kind=kind,
            phase=phase,
            description=description,
            operator_name=operator_name,
            node=node,
            summary_text=summary_text,
        )
        self.steps.append(step)
        self.active_cursor = BranchCursor(
            step_count=len(self.steps),
            retrieval_count=self.active_cursor.retrieval_count,
            symbolic_count=self.active_cursor.symbolic_count,
            verifier_count=self.active_cursor.verifier_count,
            critique_count=self.active_cursor.critique_count,
            repair_count=self.active_cursor.repair_count,
            candidate_count=self.active_cursor.candidate_count,
            prior_snapshot_count=self.active_cursor.prior_snapshot_count,
        )
        if node is not None:
            self.latest_node_id = node.node_id
            self.latest_state_fingerprint = node.state_fingerprint
            self._set_proof_obligations(getattr(node, "proof_obligations", ()) or ())
        return step

    def _append_prior_snapshot(
        self,
        *,
        phase: BranchPhase,
        source: str,
        retrieval_support: Mapping[str, float] | None = None,
    ) -> BranchOperatorPriorSnapshot:
        snapshot = BranchOperatorPriorSnapshot.create(
            phase=phase,
            operator_prior=self.route.operator_prior,
            problem_type=self.route.problem_type,
            archetypes=self.route.archetypes,
            retrieval_support=retrieval_support,
            source=source,
        )
        self.prior_snapshots.append(snapshot)
        self.active_cursor = BranchCursor(
            step_count=self.active_cursor.step_count,
            retrieval_count=self.active_cursor.retrieval_count,
            symbolic_count=self.active_cursor.symbolic_count,
            verifier_count=self.active_cursor.verifier_count,
            critique_count=self.active_cursor.critique_count,
            repair_count=self.active_cursor.repair_count,
            candidate_count=self.active_cursor.candidate_count,
            prior_snapshot_count=len(self.prior_snapshots),
        )
        return snapshot

    def _record_patch(
        self,
        *,
        kind: PatchKind,
        reason: str,
        cursor_before: BranchCursor,
        cursor_after: BranchCursor,
        phase_before: BranchPhase,
        phase_after: BranchPhase,
        candidate_id_before: str | None,
        candidate_id_after: str | None,
    ) -> BranchPatchRecord:
        patch = BranchPatchRecord(
            patch_id=_stable_hash(
                "branch_patch",
                {
                    "branch_id": self.branch_id,
                    "kind": kind.value,
                    "reason": reason,
                    "cursor_before": cursor_before.__dict__,
                    "cursor_after": cursor_after.__dict__,
                },
            ),
            kind=kind,
            reason=_normalize_text(reason),
            cursor_before=cursor_before,
            cursor_after=cursor_after,
            phase_before=phase_before,
            phase_after=phase_after,
            candidate_id_before=candidate_id_before,
            candidate_id_after=candidate_id_after,
        )
        self.patch_history.append(patch)
        return patch

    def _trim_steps_to_prefix(self, keep_steps: int) -> None:
        self.steps = self.steps[:keep_steps]
        if self.steps:
            last = self.steps[-1]
            self.latest_node_id = last.node_id
            self.latest_state_fingerprint = last.state_fingerprint
        else:
            self.latest_node_id = self.root_node_id
            self.latest_state_fingerprint = None
        self.proof_state_fingerprint = self._proof_state_fingerprint(self.proof_obligations)

    def _set_proof_obligations(self, obligations: tuple[ProofObligation, ...] | list[ProofObligation] | tuple[object, ...] | list[object]) -> None:
        normalized: list[ProofObligation] = []
        for item in obligations:
            if isinstance(item, ProofObligation):
                normalized.append(item)
                continue
            if not isinstance(item, dict):
                continue
            try:
                normalized.append(
                    ProofObligation.create(
                        obligation_id=item.get("obligation_id"),
                        originating_node_id=item.get("originating_node_id", ""),
                        claim=item.get("claim", ""),
                        evidence_kind_required=item.get("evidence_kind_required", "other"),
                        status=item.get("status", "open"),
                        discharged_by=tuple(item.get("discharged_by", []) or ()),
                        notes=tuple(item.get("notes", []) or ()),
                        target_goal_id=item.get("target_goal_id"),
                        source_constraint_ids=tuple(item.get("source_constraint_ids", []) or ()),
                        metadata=dict(item.get("metadata") or {}),
                    )
                )
            except Exception:
                continue
        self.proof_obligations = normalized
        self.proof_state_fingerprint = self._proof_state_fingerprint(self.proof_obligations)

    @staticmethod
    def _proof_state_fingerprint(obligations: tuple[ProofObligation, ...] | list[ProofObligation]) -> str | None:
        if not obligations:
            return None
        payload = [
            (
                item.obligation_id,
                item.status.value,
                item.normalized_claim,
                item.evidence_kind_required.value,
            )
            for item in obligations
        ]
        return _stable_hash("proof_state", {"obligations": payload})


__all__ = [
    "BranchPhase",
    "BranchStepKind",
    "PatchKind",
    "BranchCursor",
    "BranchCandidateState",
    "BranchOperatorPriorSnapshot",
    "BranchStep",
    "RetrievalEvidence",
    "SymbolicEvidence",
    "VerifierEvidence",
    "CritiqueEvidence",
    "BranchScoreBreakdown",
    "BranchPatchRecord",
    "BranchRepairRecord",
    "BranchSummary",
    "BranchState",
]
