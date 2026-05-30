from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from src.common.utils import safe_normalize_text, stable_hash as _stable_hash
from src.state_graph.proof_obligations import ProofObligation, obligation_status_counts

ConstraintKind = Literal[
    "equality",
    "inequality",
    "ordering",
    "boundedness",
    "positivity",
    "divisibility",
    "congruence",
    "domain_membership",
    "integrality",
    "objective",
    "other",
]
GoalKind = Literal[
    "final_answer",
    "prove",
    "subgoal",
    "case_split",
    "verification",
    "other",
]
EvidenceKind = Literal[
    "symbolic_check",
    "verifier",
    "tool",
    "retrieval",
    "repair",
    "human",
    "other",
]


def _normalize_space(text: str) -> str:
    return safe_normalize_text(text)


def _strip_wrapping_parens(text: str) -> str:
    s = _normalize_space(text)
    while len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        depth = 0
        valid = True
        for idx, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and idx != len(s) - 1:
                    valid = False
                    break
            if depth < 0:
                valid = False
                break
        if not valid or depth != 0:
            break
        s = _normalize_space(s[1:-1])
    return s


def _split_toplevel(expr: str, operator: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and expr.startswith(operator, i):
            parts.append(expr[start:i])
            start = i + len(operator)
            i += len(operator)
            continue
        i += 1
    parts.append(expr[start:])
    return [_normalize_space(p) for p in parts if _normalize_space(p)]


def _canonicalize_commutative(expr: str) -> str:
    s = _strip_wrapping_parens(expr)
    if not s:
        return s

    plus_parts = _split_toplevel(s, "+")
    if len(plus_parts) > 1:
        return " + ".join(sorted(_canonicalize_commutative(p) for p in plus_parts))

    mul_parts = _split_toplevel(s, "*")
    if len(mul_parts) > 1:
        return " * ".join(sorted(_canonicalize_commutative(p) for p in mul_parts))

    return _normalize_space(s)


def _canonicalize_relation(lhs: str | None, relation: str | None, rhs: str | None) -> tuple[str | None, str | None, str | None]:
    left = _canonicalize_commutative(lhs or "") if lhs is not None else None
    right = _canonicalize_commutative(rhs or "") if rhs is not None else None
    rel = _normalize_space(relation or "") or None
    if rel in {"=", "!=", "equiv", "congruent"} and left is not None and right is not None:
        ordered = sorted([left, right])
        left, right = ordered[0], ordered[1]
    return left, rel, right


@dataclass(frozen=True)
class ConstraintRecord:
    constraint_id: str
    kind: ConstraintKind
    lhs: str | None
    relation: str | None
    rhs: str | None
    normalized_lhs: str | None
    normalized_rhs: str | None
    normalized_relation: str | None
    scope: str = "global"
    origin: str = "explicit"
    strength: str = "hard"
    confidence: float = 1.0
    domain_assumptions: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        kind: ConstraintKind,
        lhs: str | None,
        relation: str | None,
        rhs: str | None,
        scope: str = "global",
        origin: str = "explicit",
        strength: str = "hard",
        confidence: float = 1.0,
        domain_assumptions: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ConstraintRecord":
        nl, nr_rel, nr = _canonicalize_relation(lhs, relation, rhs)
        payload = {
            "kind": kind,
            "lhs": nl,
            "relation": nr_rel,
            "rhs": nr,
            "scope": scope,
            "origin": origin,
            "strength": strength,
            "domain_assumptions": sorted(set(domain_assumptions or [])),
        }
        return cls(
            constraint_id=_stable_hash("constraint", payload),
            kind=kind,
            lhs=_normalize_space(lhs or "") or None,
            relation=_normalize_space(relation or "") or None,
            rhs=_normalize_space(rhs or "") or None,
            normalized_lhs=nl,
            normalized_rhs=nr,
            normalized_relation=nr_rel,
            scope=scope,
            origin=origin,
            strength=strength,
            confidence=max(0.0, min(1.0, confidence)),
            domain_assumptions=tuple(sorted(set(domain_assumptions or []))),
            metadata=dict(metadata or {}),
        )

    def canonical_key(self) -> tuple[Any, ...]:
        return (
            self.kind,
            self.normalized_lhs,
            self.normalized_relation,
            self.normalized_rhs,
            self.scope,
            self.origin,
            self.strength,
            self.domain_assumptions,
        )

    def summary(self) -> str:
        core = " ".join(
            part for part in [self.normalized_lhs, self.normalized_relation, self.normalized_rhs] if part
        ).strip()
        return core or self.kind


@dataclass(frozen=True)
class InvariantRecord:
    invariant_id: str
    invariant_type: str
    expression: str
    normalized_expression: str
    scope: str = "global"
    strength: str = "candidate"
    proof_source: str = "unknown"
    confidence: float = 0.5
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        invariant_type: str,
        expression: str,
        scope: str = "global",
        strength: str = "candidate",
        proof_source: str = "unknown",
        confidence: float = 0.5,
        metadata: Mapping[str, Any] | None = None,
    ) -> "InvariantRecord":
        normalized = _canonicalize_commutative(expression)
        payload = {
            "type": invariant_type,
            "expr": normalized,
            "scope": scope,
            "strength": strength,
            "proof_source": proof_source,
        }
        return cls(
            invariant_id=_stable_hash("invariant", payload),
            invariant_type=_normalize_space(invariant_type),
            expression=_normalize_space(expression),
            normalized_expression=normalized,
            scope=scope,
            strength=_normalize_space(strength),
            proof_source=_normalize_space(proof_source),
            confidence=max(0.0, min(1.0, confidence)),
            metadata=dict(metadata or {}),
        )

    def canonical_key(self) -> tuple[Any, ...]:
        return (self.invariant_type, self.normalized_expression, self.scope, self.strength)

    def implies(self, other: "InvariantRecord") -> bool:
        if self.normalized_expression == other.normalized_expression:
            return True
        return self.normalized_expression in other.normalized_expression

    def conflicts(self, other: "InvariantRecord") -> bool:
        a = self.normalized_expression
        b = other.normalized_expression
        if a == b:
            return False
        return (a.startswith("not ") and a[4:] == b) or (b.startswith("not ") and b[4:] == a)

    def dominates(self, other: "InvariantRecord") -> bool:
        if self.implies(other) and self.confidence >= other.confidence:
            return True
        if self.normalized_expression == other.normalized_expression and self.strength >= other.strength:
            return True
        return False


@dataclass(frozen=True)
class GoalRecord:
    goal_id: str
    kind: GoalKind
    text: str
    normalized_text: str
    priority: int = 100
    status: Literal["open", "partial", "satisfied", "blocked"] = "open"
    confidence: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        kind: GoalKind,
        text: str,
        priority: int = 100,
        status: Literal["open", "partial", "satisfied", "blocked"] = "open",
        confidence: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> "GoalRecord":
        normalized = _canonicalize_commutative(text)
        payload = {"kind": kind, "text": normalized, "priority": priority}
        return cls(
            goal_id=_stable_hash("goal", payload),
            kind=kind,
            text=_normalize_space(text),
            normalized_text=normalized,
            priority=priority,
            status=status,
            confidence=max(0.0, min(1.0, confidence)),
            metadata=dict(metadata or {}),
        )

    def canonical_key(self) -> tuple[Any, ...]:
        return (self.kind, self.normalized_text, self.priority, self.status)


@dataclass(frozen=True)
class OperatorApplicationRecord:
    step_id: str
    operator_name: str
    rationale: str = ""
    pre_state_fingerprint: str | None = None
    post_state_fingerprint: str | None = None
    success: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        operator_name: str,
        rationale: str = "",
        pre_state_fingerprint: str | None = None,
        post_state_fingerprint: str | None = None,
        success: bool | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "OperatorApplicationRecord":
        payload = {
            "operator_name": operator_name,
            "rationale": rationale,
            "pre": pre_state_fingerprint,
            "post": post_state_fingerprint,
            "success": success,
        }
        return cls(
            step_id=_stable_hash("opstep", payload),
            operator_name=_normalize_space(operator_name),
            rationale=_normalize_space(rationale),
            pre_state_fingerprint=pre_state_fingerprint,
            post_state_fingerprint=post_state_fingerprint,
            success=success,
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    kind: EvidenceKind
    source: str
    summary: str
    score: float | None = None
    supports: tuple[str, ...] = field(default_factory=tuple)
    refutes: tuple[str, ...] = field(default_factory=tuple)
    payload: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        kind: EvidenceKind,
        source: str,
        summary: str,
        score: float | None = None,
        supports: Sequence[str] | None = None,
        refutes: Sequence[str] | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> "EvidenceRecord":
        clean_score = None if score is None else max(0.0, min(1.0, score))
        core = {
            "kind": kind,
            "source": source,
            "summary": summary,
            "score": clean_score,
            "supports": sorted(set(supports or [])),
            "refutes": sorted(set(refutes or [])),
        }
        return cls(
            evidence_id=_stable_hash("evidence", core),
            kind=kind,
            source=_normalize_space(source),
            summary=_normalize_space(summary),
            score=clean_score,
            supports=tuple(sorted(set(supports or []))),
            refutes=tuple(sorted(set(refutes or []))),
            payload=dict(payload or {}),
        )


@dataclass(frozen=True)
class NodeProvenance:
    problem_id: str
    branch_id: str
    parent_node_ids: tuple[str, ...] = field(default_factory=tuple)
    merged_from_node_ids: tuple[str, ...] = field(default_factory=tuple)
    repair_from_node_id: str | None = None
    source_tags: tuple[str, ...] = field(default_factory=tuple)
    depth: int = 0
    lineage: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class MergeCompatibility:
    equivalent: bool
    compatible: bool
    same_problem: bool
    same_branch: bool
    fingerprint_match: bool
    constraints_match: bool
    invariants_compatible: bool
    goals_compatible: bool
    proof_obligations_compatible: bool
    dominance: Literal["self", "other", "neither", "equal"]
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ReasoningStateNode:
    node_id: str
    provenance: NodeProvenance

    constraints: tuple[ConstraintRecord, ...] = field(default_factory=tuple)
    invariants: tuple[InvariantRecord, ...] = field(default_factory=tuple)
    goals: tuple[GoalRecord, ...] = field(default_factory=tuple)
    proof_obligations: tuple[ProofObligation, ...] = field(default_factory=tuple)

    extracted_objects: Mapping[str, Any] = field(default_factory=dict)
    partial_solution: str = ""
    summary_text: str = ""

    operator_history: tuple[OperatorApplicationRecord, ...] = field(default_factory=tuple)
    evidence: tuple[EvidenceRecord, ...] = field(default_factory=tuple)

    confidence: float = 0.5
    verifier_score: float = 0.0
    symbolic_valid: bool = False
    merge_ready: bool = False
    is_terminal: bool = False

    state_fingerprint: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        problem_id: str,
        branch_id: str,
        constraints: Sequence[ConstraintRecord] | None = None,
        invariants: Sequence[InvariantRecord] | None = None,
        goals: Sequence[GoalRecord] | None = None,
        proof_obligations: Sequence[ProofObligation] | None = None,
        extracted_objects: Mapping[str, Any] | None = None,
        partial_solution: str = "",
        summary_text: str = "",
        operator_history: Sequence[OperatorApplicationRecord] | None = None,
        evidence: Sequence[EvidenceRecord] | None = None,
        confidence: float = 0.5,
        verifier_score: float = 0.0,
        symbolic_valid: bool = False,
        merge_ready: bool = False,
        is_terminal: bool = False,
        parent_node_ids: Sequence[str] | None = None,
        merged_from_node_ids: Sequence[str] | None = None,
        repair_from_node_id: str | None = None,
        source_tags: Sequence[str] | None = None,
        depth: int = 0,
        lineage: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        node_id: str | None = None,
    ) -> "ReasoningStateNode":
        normalized_constraints = cls._normalize_constraints(tuple(constraints or ()))
        normalized_invariants = cls._normalize_invariants(tuple(invariants or ()))
        normalized_goals = cls._normalize_goals(tuple(goals or ()))
        normalized_proof_obligations = cls._normalize_proof_obligations(tuple(proof_obligations or ()))

        provenance = NodeProvenance(
            problem_id=problem_id,
            branch_id=branch_id,
            parent_node_ids=tuple(sorted(set(parent_node_ids or ()))),
            merged_from_node_ids=tuple(sorted(set(merged_from_node_ids or ()))),
            repair_from_node_id=repair_from_node_id,
            source_tags=tuple(sorted(set(source_tags or ()))),
            depth=depth,
            lineage=tuple(lineage or ()),
        )

        fingerprint_material = cls._fingerprint_material_static(
            problem_id=problem_id,
            constraints=normalized_constraints,
            invariants=normalized_invariants,
            goals=normalized_goals,
            proof_obligations=normalized_proof_obligations,
            symbolic_valid=symbolic_valid,
            partial_solution=partial_solution,
            is_terminal=is_terminal,
        )
        fingerprint = _stable_hash("state", fingerprint_material)

        computed_node_id = node_id or _stable_hash(
            "node",
            {
                "problem_id": problem_id,
                "branch_id": branch_id,
                "fingerprint": fingerprint,
                "parents": provenance.parent_node_ids,
                "depth": depth,
            },
        )

        node = cls(
            node_id=computed_node_id,
            provenance=provenance,
            constraints=normalized_constraints,
            invariants=normalized_invariants,
            goals=normalized_goals,
            proof_obligations=normalized_proof_obligations,
            extracted_objects=dict(extracted_objects or {}),
            partial_solution=_normalize_space(partial_solution),
            summary_text=_normalize_space(summary_text),
            operator_history=tuple(operator_history or ()),
            evidence=tuple(evidence or ()),
            confidence=max(0.0, min(1.0, confidence)),
            verifier_score=max(0.0, min(1.0, verifier_score)),
            symbolic_valid=bool(symbolic_valid),
            merge_ready=bool(merge_ready),
            is_terminal=bool(is_terminal),
            state_fingerprint=fingerprint,
            metadata=dict(metadata or {}),
        )
        if not node.summary_text:
            object.__setattr__(node, "summary_text", node.build_summary())
        return node

    @staticmethod
    def _normalize_constraints(constraints: Sequence[ConstraintRecord]) -> tuple[ConstraintRecord, ...]:
        deduped = {c.canonical_key(): c for c in constraints}
        return tuple(sorted(deduped.values(), key=lambda x: x.canonical_key()))

    @staticmethod
    def _normalize_invariants(invariants: Sequence[InvariantRecord]) -> tuple[InvariantRecord, ...]:
        ordered = sorted(invariants, key=lambda x: x.canonical_key())
        kept: list[InvariantRecord] = []
        for inv in ordered:
            replaced = False
            for idx, prev in enumerate(kept):
                if inv.normalized_expression == prev.normalized_expression and inv.scope == prev.scope:
                    if inv.dominates(prev):
                        kept[idx] = inv
                    replaced = True
                    break
            if not replaced:
                kept.append(inv)
        return tuple(sorted(kept, key=lambda x: x.canonical_key()))

    @staticmethod
    def _normalize_goals(goals: Sequence[GoalRecord]) -> tuple[GoalRecord, ...]:
        deduped = {g.canonical_key(): g for g in goals}
        return tuple(sorted(deduped.values(), key=lambda x: x.canonical_key()))

    @staticmethod
    def _normalize_proof_obligations(
        proof_obligations: Sequence[ProofObligation],
    ) -> tuple[ProofObligation, ...]:
        deduped = {item.canonical_key(): item for item in proof_obligations}
        return tuple(
            sorted(
                deduped.values(),
                key=lambda item: (
                    item.status.value,
                    item.evidence_kind_required.value,
                    item.normalized_claim,
                    item.obligation_id,
                ),
            )
        )

    @classmethod
    def _fingerprint_material_static(
        cls,
        *,
        problem_id: str,
        constraints: Sequence[ConstraintRecord],
        invariants: Sequence[InvariantRecord],
        goals: Sequence[GoalRecord],
        proof_obligations: Sequence[ProofObligation],
        symbolic_valid: bool,
        partial_solution: str,
        is_terminal: bool,
    ) -> Mapping[str, Any]:
        return {
            "problem_id": problem_id,
            "constraints": [
                {
                    "kind": c.kind,
                    "lhs": c.normalized_lhs,
                    "relation": c.normalized_relation,
                    "rhs": c.normalized_rhs,
                    "scope": c.scope,
                    "strength": c.strength,
                    "domain_assumptions": list(c.domain_assumptions),
                }
                for c in constraints
            ],
            "invariants": [
                {
                    "type": inv.invariant_type,
                    "expr": inv.normalized_expression,
                    "scope": inv.scope,
                    "strength": inv.strength,
                }
                for inv in invariants
            ],
            "goals": [
                {
                    "kind": g.kind,
                    "text": g.normalized_text,
                    "priority": g.priority,
                    "status": g.status,
                }
                for g in goals
            ],
            "proof_obligations": [
                {
                    "claim": item.normalized_claim,
                    "evidence_kind_required": item.evidence_kind_required.value,
                    "status": item.status.value,
                    "target_goal_id": item.target_goal_id,
                    "source_constraint_ids": list(item.source_constraint_ids),
                }
                for item in proof_obligations
            ],
            "symbolic_valid": bool(symbolic_valid),
            "partial_solution": _canonicalize_commutative(partial_solution) if partial_solution else "",
            "is_terminal": bool(is_terminal),
        }

    def fingerprint_material(self) -> Mapping[str, Any]:
        return self._fingerprint_material_static(
            problem_id=self.provenance.problem_id,
            constraints=self.constraints,
            invariants=self.invariants,
            goals=self.goals,
            proof_obligations=self.proof_obligations,
            symbolic_valid=self.symbolic_valid,
            partial_solution=self.partial_solution,
            is_terminal=self.is_terminal,
        )

    def equivalence_key(self) -> tuple[Any, ...]:
        material = self.fingerprint_material()
        return (
            material["problem_id"],
            tuple(
                (c["kind"], c["lhs"], c["relation"], c["rhs"], c["scope"], c["strength"], tuple(c["domain_assumptions"]))
                for c in material["constraints"]
            ),
            tuple((i["type"], i["expr"], i["scope"], i["strength"]) for i in material["invariants"]),
            tuple((g["kind"], g["text"], g["priority"], g["status"]) for g in material["goals"]),
            tuple(
                (
                    item["claim"],
                    item["evidence_kind_required"],
                    item["status"],
                    item["target_goal_id"],
                    tuple(item["source_constraint_ids"]),
                )
                for item in material["proof_obligations"]
            ),
            material["symbolic_valid"],
            material["partial_solution"],
            material["is_terminal"],
        )

    @property
    def remaining_constraints(self) -> tuple[ConstraintRecord, ...]:
        return self.constraints

    @property
    def active_invariants(self) -> tuple[InvariantRecord, ...]:
        return self.invariants

    @property
    def active_equations(self) -> tuple[str, ...]:
        equations: list[str] = []
        for constraint in self.constraints:
            if constraint.normalized_relation in {"=", "!=", "<", ">", "<=", ">=", "in"}:
                equations.append(constraint.summary())
            elif constraint.kind in {"equality", "inequality", "congruence", "divisibility"}:
                equations.append(constraint.summary())
        return tuple(equations)

    @property
    def discovered_invariants(self) -> tuple[str, ...]:
        return tuple(inv.normalized_expression for inv in self.invariants if inv.normalized_expression)

    @property
    def partial_answer(self) -> str | None:
        candidate = _normalize_space(self.partial_solution)
        if candidate:
            return candidate
        metadata_candidate = _normalize_space(str(self.metadata.get("candidate_answer", "")))
        return metadata_candidate or None

    @property
    def tool_evidence(self) -> tuple[EvidenceRecord, ...]:
        return tuple(ev for ev in self.evidence if ev.kind in {"tool", "symbolic_check", "retrieval"})

    @property
    def verifier_evidence(self) -> tuple[EvidenceRecord, ...]:
        return tuple(ev for ev in self.evidence if ev.kind == "verifier")

    def remaining_open_goals(self) -> tuple[GoalRecord, ...]:
        return tuple(g for g in self.goals if g.status != "satisfied")

    def open_proof_obligations(self) -> tuple[ProofObligation, ...]:
        return tuple(item for item in self.proof_obligations if item.is_open())

    def contradicted_proof_obligations(self) -> tuple[ProofObligation, ...]:
        return tuple(item for item in self.proof_obligations if item.status.value == "contradicted")

    def invariants_conflict(self, other: "ReasoningStateNode") -> bool:
        for a in self.invariants:
            for b in other.invariants:
                if a.conflicts(b):
                    return True
        return False

    def invariant_dominance_against(self, other: "ReasoningStateNode") -> Literal["self", "other", "equal", "neither"]:
        self_ge_other = True
        other_ge_self = True

        for inv_other in other.invariants:
            if not any(inv_self.dominates(inv_other) for inv_self in self.invariants):
                self_ge_other = False
                break

        for inv_self in self.invariants:
            if not any(inv_other.dominates(inv_self) for inv_other in other.invariants):
                other_ge_self = False
                break

        if self_ge_other and other_ge_self:
            return "equal"
        if self_ge_other:
            return "self"
        if other_ge_self:
            return "other"
        return "neither"

    def build_summary(self) -> str:
        pieces: list[str] = []
        open_goals = [g.normalized_text for g in self.remaining_open_goals()[:2]]
        if open_goals:
            pieces.append(f"goals={'; '.join(open_goals)}")
        if self.constraints:
            pieces.append(f"constraints={len(self.constraints)}")
        if self.invariants:
            pieces.append(f"invariants={', '.join(inv.normalized_expression for inv in self.invariants[:2])}")
        if self.operator_history:
            pieces.append(f"last_op={self.operator_history[-1].operator_name}")
        open_obligations = self.open_proof_obligations()
        if open_obligations:
            pieces.append(f"open_obligations={len(open_obligations)}")
        if self.partial_solution:
            pieces.append(f"partial={self.partial_solution[:120]}")
        if self.is_terminal:
            pieces.append("terminal=true")
        return " | ".join(pieces) if pieces else "empty_state"

    def to_text_summary(self) -> str:
        return self.summary_text or self.build_summary()

    def with_updates(
        self,
        *,
        constraints: Sequence[ConstraintRecord] | None = None,
        invariants: Sequence[InvariantRecord] | None = None,
        goals: Sequence[GoalRecord] | None = None,
        proof_obligations: Sequence[ProofObligation] | None = None,
        extracted_objects: Mapping[str, Any] | None = None,
        partial_solution: str | None = None,
        summary_text: str | None = None,
        operator_history: Sequence[OperatorApplicationRecord] | None = None,
        evidence: Sequence[EvidenceRecord] | None = None,
        confidence: float | None = None,
        verifier_score: float | None = None,
        symbolic_valid: bool | None = None,
        merge_ready: bool | None = None,
        is_terminal: bool | None = None,
        metadata: Mapping[str, Any] | None = None,
        parent_node_ids: Sequence[str] | None = None,
        merged_from_node_ids: Sequence[str] | None = None,
        repair_from_node_id: str | None = None,
        source_tags: Sequence[str] | None = None,
        depth: int | None = None,
        lineage: Sequence[str] | None = None,
        node_id: str | None = None,
    ) -> "ReasoningStateNode":
        return ReasoningStateNode.create(
            problem_id=self.provenance.problem_id,
            branch_id=self.provenance.branch_id,
            constraints=constraints if constraints is not None else self.constraints,
            invariants=invariants if invariants is not None else self.invariants,
            goals=goals if goals is not None else self.goals,
            proof_obligations=proof_obligations if proof_obligations is not None else self.proof_obligations,
            extracted_objects=extracted_objects if extracted_objects is not None else self.extracted_objects,
            partial_solution=self.partial_solution if partial_solution is None else partial_solution,
            summary_text=self.summary_text if summary_text is None else summary_text,
            operator_history=operator_history if operator_history is not None else self.operator_history,
            evidence=evidence if evidence is not None else self.evidence,
            confidence=self.confidence if confidence is None else confidence,
            verifier_score=self.verifier_score if verifier_score is None else verifier_score,
            symbolic_valid=self.symbolic_valid if symbolic_valid is None else symbolic_valid,
            merge_ready=self.merge_ready if merge_ready is None else merge_ready,
            is_terminal=self.is_terminal if is_terminal is None else is_terminal,
            parent_node_ids=parent_node_ids if parent_node_ids is not None else self.provenance.parent_node_ids,
            merged_from_node_ids=merged_from_node_ids if merged_from_node_ids is not None else self.provenance.merged_from_node_ids,
            repair_from_node_id=self.provenance.repair_from_node_id if repair_from_node_id is None else repair_from_node_id,
            source_tags=source_tags if source_tags is not None else self.provenance.source_tags,
            depth=self.provenance.depth if depth is None else depth,
            lineage=lineage if lineage is not None else self.provenance.lineage,
            metadata=metadata if metadata is not None else self.metadata,
            node_id=node_id or self.node_id,
        )

    def add_constraints(self, records: Sequence[ConstraintRecord]) -> "ReasoningStateNode":
        return self.with_updates(constraints=self.constraints + tuple(records))

    def add_invariants(self, records: Sequence[InvariantRecord]) -> "ReasoningStateNode":
        return self.with_updates(invariants=self._normalize_invariants(self.invariants + tuple(records)))

    def add_goals(self, records: Sequence[GoalRecord]) -> "ReasoningStateNode":
        return self.with_updates(goals=self.goals + tuple(records))

    def set_proof_obligations(self, records: Sequence[ProofObligation]) -> "ReasoningStateNode":
        return self.with_updates(proof_obligations=self._normalize_proof_obligations(tuple(records)))

    def add_proof_obligations(self, records: Sequence[ProofObligation]) -> "ReasoningStateNode":
        return self.with_updates(
            proof_obligations=self._normalize_proof_obligations(self.proof_obligations + tuple(records))
        )

    def add_operator_step(self, record: OperatorApplicationRecord) -> "ReasoningStateNode":
        return self.with_updates(operator_history=self.operator_history + (record,))

    def add_evidence(self, record: EvidenceRecord) -> "ReasoningStateNode":
        return self.with_updates(evidence=self.evidence + (record,))

    def merge_compatibility(self, other: "ReasoningStateNode") -> MergeCompatibility:
        same_problem = self.provenance.problem_id == other.provenance.problem_id
        same_branch = self.provenance.branch_id == other.provenance.branch_id
        fingerprint_match = self.state_fingerprint == other.state_fingerprint
        constraints_match = self.constraints == other.constraints
        goals_compatible = {g.canonical_key() for g in self.goals} == {g.canonical_key() for g in other.goals}
        invariants_compatible = not self.invariants_conflict(other)
        proof_obligations_compatible = {
            item.canonical_key() for item in self.proof_obligations
        } == {item.canonical_key() for item in other.proof_obligations}
        dominance = self.invariant_dominance_against(other)

        reasons: list[str] = []
        if not same_problem:
            reasons.append("different_problem")
        if fingerprint_match:
            reasons.append("same_fingerprint")
        if constraints_match:
            reasons.append("same_constraints")
        if not invariants_compatible:
            reasons.append("invariant_conflict")
        if dominance in {"self", "other", "equal"}:
            reasons.append(f"invariant_dominance:{dominance}")
        if goals_compatible:
            reasons.append("goals_compatible")
        if proof_obligations_compatible:
            reasons.append("proof_obligations_compatible")

        equivalent = same_problem and fingerprint_match
        compatible = same_problem and invariants_compatible and goals_compatible and proof_obligations_compatible
        return MergeCompatibility(
            equivalent=equivalent,
            compatible=compatible,
            same_problem=same_problem,
            same_branch=same_branch,
            fingerprint_match=fingerprint_match,
            constraints_match=constraints_match,
            invariants_compatible=invariants_compatible,
            goals_compatible=goals_compatible,
            proof_obligations_compatible=proof_obligations_compatible,
            dominance=dominance,
            reasons=tuple(reasons),
        )

    def is_equivalent_to(self, other: "ReasoningStateNode") -> bool:
        return self.merge_compatibility(other).equivalent

    def merge_readiness_against(self, other: "ReasoningStateNode") -> bool:
        comp = self.merge_compatibility(other)
        return comp.compatible and comp.dominance in {"self", "other", "equal"}

    def export_state(self) -> Mapping[str, Any]:
        return {
            "node_id": self.node_id,
            "state_fingerprint": self.state_fingerprint,
            "problem_id": self.provenance.problem_id,
            "branch_id": self.provenance.branch_id,
            "depth": self.provenance.depth,
            "parent_node_ids": list(self.provenance.parent_node_ids),
            "merged_from_node_ids": list(self.provenance.merged_from_node_ids),
            "repair_from_node_id": self.provenance.repair_from_node_id,
            "lineage": list(self.provenance.lineage),
            "source_tags": list(self.provenance.source_tags),
            "constraints": [
                {
                    "constraint_id": c.constraint_id,
                    "kind": c.kind,
                    "lhs": c.lhs,
                    "relation": c.relation,
                    "rhs": c.rhs,
                    "normalized_lhs": c.normalized_lhs,
                    "normalized_relation": c.normalized_relation,
                    "normalized_rhs": c.normalized_rhs,
                    "scope": c.scope,
                    "origin": c.origin,
                    "strength": c.strength,
                    "confidence": c.confidence,
                    "domain_assumptions": list(c.domain_assumptions),
                    "metadata": dict(c.metadata),
                }
                for c in self.constraints
            ],
            "invariants": [
                {
                    "invariant_id": inv.invariant_id,
                    "invariant_type": inv.invariant_type,
                    "expression": inv.expression,
                    "normalized_expression": inv.normalized_expression,
                    "scope": inv.scope,
                    "strength": inv.strength,
                    "proof_source": inv.proof_source,
                    "confidence": inv.confidence,
                    "metadata": dict(inv.metadata),
                }
                for inv in self.invariants
            ],
            "goals": [
                {
                    "goal_id": g.goal_id,
                    "kind": g.kind,
                    "text": g.text,
                    "normalized_text": g.normalized_text,
                    "priority": g.priority,
                    "status": g.status,
                    "confidence": g.confidence,
                    "metadata": dict(g.metadata),
                }
                for g in self.goals
            ],
            "proof_obligations": [
                {
                    "obligation_id": item.obligation_id,
                    "originating_node_id": item.originating_node_id,
                    "claim": item.claim,
                    "normalized_claim": item.normalized_claim,
                    "evidence_kind_required": item.evidence_kind_required.value,
                    "status": item.status.value,
                    "discharged_by": list(item.discharged_by),
                    "contradiction_provenance": None
                    if item.contradiction_provenance is None
                    else {
                        "source": item.contradiction_provenance.source,
                        "summary": item.contradiction_provenance.summary,
                        "evidence_id": item.contradiction_provenance.evidence_id,
                        "failure_type": item.contradiction_provenance.failure_type,
                        "metadata": dict(item.contradiction_provenance.metadata),
                    },
                    "notes": list(item.notes),
                    "target_goal_id": item.target_goal_id,
                    "source_constraint_ids": list(item.source_constraint_ids),
                    "metadata": dict(item.metadata),
                }
                for item in self.proof_obligations
            ],
            "operator_history": [
                {
                    "step_id": op.step_id,
                    "operator_name": op.operator_name,
                    "rationale": op.rationale,
                    "pre_state_fingerprint": op.pre_state_fingerprint,
                    "post_state_fingerprint": op.post_state_fingerprint,
                    "success": op.success,
                    "metadata": dict(op.metadata),
                }
                for op in self.operator_history
            ],
            "evidence": [
                {
                    "evidence_id": ev.evidence_id,
                    "kind": ev.kind,
                    "source": ev.source,
                    "summary": ev.summary,
                    "score": ev.score,
                    "supports": list(ev.supports),
                    "refutes": list(ev.refutes),
                    "payload": dict(ev.payload),
                }
                for ev in self.evidence
            ],
            "extracted_objects": dict(self.extracted_objects),
            "partial_solution": self.partial_solution,
            "summary_text": self.summary_text,
            "confidence": self.confidence,
            "verifier_score": self.verifier_score,
            "symbolic_valid": self.symbolic_valid,
            "merge_ready": self.merge_ready,
            "is_terminal": self.is_terminal,
            "proof_obligation_status_counts": obligation_status_counts(self.proof_obligations),
            "metadata": dict(self.metadata),
        }


StateGraphNode = ReasoningStateNode
ReasoningState = ReasoningStateNode

__all__ = [
    "ConstraintRecord",
    "InvariantRecord",
    "GoalRecord",
    "OperatorApplicationRecord",
    "EvidenceRecord",
    "NodeProvenance",
    "MergeCompatibility",
    "ProofObligation",
    "ReasoningState",
    "ReasoningStateNode",
    "StateGraphNode",
]
