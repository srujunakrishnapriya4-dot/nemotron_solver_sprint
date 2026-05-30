from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.common.schemas import ParsedProblem, RouteDecision, render_constraint_text
from src.state_graph.node import ConstraintRecord, GoalRecord, InvariantRecord, ReasoningStateNode
from src.state_graph.graph_store import StateGraphStore
from src.state_graph.proof_obligations import ProofEvidenceKind, ProofObligation, obligation_status_counts


class InitialStateMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problem_id: str
    root_node_id: str
    canonical_state_hash: str
    top_problem_type: str
    top_problem_type_probability: float
    top_archetypes: list[str] = Field(default_factory=list)
    branch_budget: int
    retrieval_enabled: bool
    retrieval_depth_hint: int
    symbolic_enabled: bool
    brute_force_enabled: bool
    routing_uncertainty: float
    operator_compatibility_hints: dict[str, float] = Field(default_factory=dict)
    proof_obligation_count: int = 0
    proof_obligation_status_counts: dict[str, int] = Field(default_factory=dict)
    control_metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)


class StateGraphInitResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    graph: StateGraphStore
    root_node: ReasoningStateNode
    root_edge: Any | None = None
    metadata: InitialStateMetadata


class StateGraphInitializer:
    def initialize(self, problem: ParsedProblem, route: RouteDecision) -> StateGraphInitResult:
        constraints = self._extract_initial_constraints(problem)
        goals = self._initialize_goals(problem)
        invariants = self._initialize_invariants(problem, route)
        proof_obligations = self._initialize_proof_obligations(problem, constraints, goals)
        operator_hints = self._initialize_operator_hints(route)
        retrieval_depth_hint = self._infer_retrieval_depth_hint(route)

        control_metadata = self._build_control_metadata(
            problem=problem,
            route=route,
            constraints=constraints,
            goals=goals,
            invariants=invariants,
            proof_obligations=proof_obligations,
            operator_hints=operator_hints,
            retrieval_depth_hint=retrieval_depth_hint,
        )
        state_summary = self._build_state_summary(
            problem=problem,
            route=route,
            constraints=constraints,
            goals=goals,
            invariants=invariants,
            proof_obligations=proof_obligations,
            operator_hints=operator_hints,
        )
        canonical_state_hash = self._compute_canonical_state_hash(
            problem=problem,
            constraints=constraints,
            goals=goals,
            invariants=invariants,
            proof_obligations=proof_obligations,
            operator_hints=operator_hints,
        )

        root_node = ReasoningStateNode.create(
            problem_id=problem.problem_id,
            branch_id="root",
            constraints=constraints,
            invariants=invariants,
            goals=goals,
            proof_obligations=proof_obligations,
            extracted_objects={
                "knowns": list(getattr(problem, "knowns", []) or []),
                "unknowns": list(getattr(problem, "unknowns", []) or []),
                "domain": getattr(getattr(problem, "domain", None), "value", str(getattr(problem, "domain", ""))),
                "answer_type": getattr(problem, "answer_type", ""),
                "route_problem_type": dict(getattr(route, "problem_type", {}) or {}),
                "route_archetypes": dict(getattr(route, "archetypes", {}) or {}),
                "objective": None if getattr(problem, "objective", None) is None else problem.objective.model_dump(mode="json"),
                "typed_constraint_ids": [constraint.constraint_id for constraint in constraints],
                "proof_obligation_ids": [item.obligation_id for item in proof_obligations],
            },
            partial_solution="",
            summary_text=state_summary,
            confidence=self._root_confidence(problem, route),
            verifier_score=0.0,
            symbolic_valid=True,
            merge_ready=True,
            is_terminal=False,
            parent_node_ids=(),
            merged_from_node_ids=(),
            source_tags=("root", "state_init"),
            depth=0,
            lineage=(),
            metadata={
                "canonical_state_hash": canonical_state_hash,
                "operator_compatibility_hints": operator_hints,
                "proof_obligation_ids": [item.obligation_id for item in proof_obligations],
                "proof_obligation_status_counts": obligation_status_counts(proof_obligations),
                "control_metadata": control_metadata,
                "provenance": {
                    "source": "state_init",
                    "problem_id": problem.problem_id,
                    "derived_from": ["ParsedProblem", "RouteDecision"],
                },
            },
            node_id=f"root::{problem.problem_id}",
        )

        graph = StateGraphStore()
        graph.insert_node(root_node)

        metadata = InitialStateMetadata(
            problem_id=problem.problem_id,
            root_node_id=root_node.node_id,
            canonical_state_hash=canonical_state_hash,
            top_problem_type=self._top_problem_type(route),
            top_problem_type_probability=self._top_problem_type_probability(route),
            top_archetypes=self._top_archetypes(route),
            branch_budget=int(getattr(route, "branch_budget", 32)),
            retrieval_enabled=bool(getattr(route, "use_retrieval", True)),
            retrieval_depth_hint=retrieval_depth_hint,
            symbolic_enabled=bool(getattr(route, "use_symbolic", True)),
            brute_force_enabled=bool(getattr(route, "use_brute_force", False)),
            routing_uncertainty=self._routing_uncertainty(route),
            operator_compatibility_hints=operator_hints,
            proof_obligation_count=len(proof_obligations),
            proof_obligation_status_counts=obligation_status_counts(proof_obligations),
            control_metadata=control_metadata,
            provenance={
                "source": "state_init",
                "problem_id": problem.problem_id,
                "state_hash_policy": "canonical-json-sha1",
            },
        )
        return StateGraphInitResult(graph=graph, root_node=root_node, root_edge=None, metadata=metadata)

    def build_initial_graph(self, problem: ParsedProblem, route: RouteDecision) -> StateGraphInitResult:
        return self.initialize(problem, route)

    def from_parsed_problem(self, problem: ParsedProblem, route: RouteDecision) -> StateGraphInitResult:
        return self.initialize(problem, route)

    def _extract_initial_constraints(self, problem: ParsedProblem) -> list[ConstraintRecord]:
        out: list[ConstraintRecord] = []
        for item in list(getattr(problem, "constraints", []) or []):
            record = self._coerce_constraint_record(item)
            if record is not None:
                out.append(record)
        for text in list(getattr(problem, "integrality_constraints", []) or []):
            out.append(
                ConstraintRecord.create(
                    kind="integrality",
                    lhs=str(text),
                    relation=None,
                    rhs=None,
                    origin="explicit",
                    strength="hard",
                    metadata={"source": "integrality_constraints"},
                )
            )

        domain = getattr(problem, "domain", None)
        domain_value = getattr(domain, "value", str(domain)) if domain is not None else ""
        for variable in getattr(problem, "unknowns", []) or []:
            if domain_value:
                out.append(
                    ConstraintRecord.create(
                        kind="domain_membership",
                        lhs=str(variable),
                        relation="in",
                        rhs=domain_value,
                        origin="explicit",
                        strength="hard",
                        domain_assumptions=[domain_value],
                    )
                )

        dedup = {c.canonical_key(): c for c in out}
        return [dedup[k] for k in sorted(dedup.keys())]

    def _coerce_constraint_record(self, item: Any) -> ConstraintRecord | None:
        if item is None:
            return None
        if isinstance(item, ConstraintRecord):
            return item
        if isinstance(item, str):
            text = render_constraint_text(item)
            if not text:
                return None
            return ConstraintRecord.create(
                kind="other",
                lhs=text,
                relation=None,
                rhs=None,
                origin="explicit",
                strength="hard",
                metadata={"source": "text_constraint"},
            )

        relation_type = getattr(getattr(item, "relation_type", None), "value", None) or "other"
        kind = relation_type if relation_type in {
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
        } else "other"
        lhs = getattr(getattr(item, "lhs", None), "normalized_text", None) or getattr(getattr(item, "lhs", None), "raw_text", None)
        rhs = getattr(getattr(item, "rhs", None), "normalized_text", None) or getattr(getattr(item, "rhs", None), "raw_text", None)
        relation = getattr(item, "operator", None)
        if lhs is None and rhs is None:
            lhs = render_constraint_text(item)
        provenance = getattr(item, "provenance", None)
        metadata = {
            "typed_constraint": True,
            "constraint_id": getattr(item, "constraint_id", None),
            "origin": getattr(getattr(item, "origin", None), "value", getattr(item, "origin", None)),
            "explicit": bool(getattr(item, "explicit", True)),
            "inferred_ready": bool(getattr(item, "inferred_ready", False)),
            "quantified_variables": [
                getattr(var, "raw_text", "")
                for var in list(getattr(item, "quantified_variables", []) or [])
                if getattr(var, "raw_text", "")
            ],
            "provenance": None
            if provenance is None
            else {
                "source_text": getattr(provenance, "source_text", ""),
                "normalized_source_text": getattr(provenance, "normalized_source_text", ""),
                "span_start": getattr(provenance, "span_start", None),
                "span_end": getattr(provenance, "span_end", None),
                "stage": getattr(provenance, "stage", ""),
                "source_channel": getattr(provenance, "source_channel", ""),
            },
        }
        return ConstraintRecord.create(
            kind=kind,
            lhs=str(lhs) if lhs is not None else None,
            relation=str(relation) if relation is not None else None,
            rhs=str(rhs) if rhs is not None else None,
            origin=str(metadata["origin"] or "explicit"),
            strength=getattr(getattr(item, "strength", None), "value", getattr(item, "strength", "hard")),
            confidence=1.0 if bool(getattr(item, "explicit", True)) else 0.75,
            domain_assumptions=list(getattr(item, "domain_assumptions", []) or []),
            metadata=metadata,
        )

    def _initialize_goals(self, problem: ParsedProblem) -> list[GoalRecord]:
        goals: list[GoalRecord] = []
        target = (getattr(problem, "target", "") or "").strip()
        if target:
            goals.append(
                GoalRecord.create(
                    kind="final_answer",
                    text=target,
                    priority=10,
                    metadata={
                        "target": target,
                        "answer_type": str(getattr(problem, "answer_type", "")),
                    },
                )
            )
        unknowns = getattr(problem, "unknowns", []) or []
        if unknowns:
            goals.append(GoalRecord.create(kind="subgoal", text=f"resolve_unknowns::{','.join(sorted(str(u) for u in unknowns))}", priority=20))
        answer_type = (getattr(problem, "answer_type", "") or "").strip()
        if answer_type:
            goals.append(GoalRecord.create(kind="verification", text=f"answer_type::{answer_type}", priority=30))
        if not goals:
            goals.append(GoalRecord.create(kind="final_answer", text="determine_final_answer", priority=10))
        dedup = {g.canonical_key(): g for g in goals}
        return [dedup[k] for k in sorted(dedup.keys())]

    def _initialize_proof_obligations(
        self,
        problem: ParsedProblem,
        constraints: list[ConstraintRecord],
        goals: list[GoalRecord],
    ) -> list[ProofObligation]:
        root_node_id = f"root::{problem.problem_id}"
        obligations: list[ProofObligation] = []

        for constraint in constraints:
            claim = constraint.summary()
            if not claim:
                continue
            obligations.append(
                ProofObligation.create(
                    originating_node_id=root_node_id,
                    claim=claim,
                    evidence_kind_required=ProofEvidenceKind.SYMBOLIC_CHECK,
                    source_constraint_ids=(constraint.constraint_id,),
                    metadata={
                        "constraint_kind": constraint.kind,
                        "constraint_origin": constraint.origin,
                    },
                )
            )

        for goal in goals:
            obligations.append(
                ProofObligation.create(
                    originating_node_id=root_node_id,
                    claim=goal.normalized_text or goal.text,
                    evidence_kind_required=(
                        ProofEvidenceKind.VERIFIER if goal.kind == "final_answer" else ProofEvidenceKind.GOAL
                    ),
                    target_goal_id=goal.goal_id,
                    metadata={
                        "goal_kind": goal.kind,
                        "goal_priority": goal.priority,
                    },
                )
            )

        for hint in list(getattr(problem, "proof_obligation_hints", []) or []):
            if str(hint).strip():
                obligations.append(
                    ProofObligation.create(
                        originating_node_id=root_node_id,
                        claim=str(hint),
                        evidence_kind_required=ProofEvidenceKind.OTHER,
                        metadata={"source": "parsed_problem.proof_obligation_hints"},
                    )
                )

        dedup = {item.canonical_key(): item for item in obligations}
        return [dedup[key] for key in sorted(dedup.keys())]

    def _initialize_invariants(self, problem: ParsedProblem, route: RouteDecision) -> list[InvariantRecord]:
        invariants: list[InvariantRecord] = []
        for item in getattr(problem, "symmetries", []) or []:
            invariants.append(
                InvariantRecord.create(
                    invariant_type="symmetry",
                    expression=str(item),
                    strength="seed",
                    proof_source="parser",
                    confidence=0.7,
                )
            )
        for item in getattr(problem, "parity_cues", []) or []:
            invariants.append(
                InvariantRecord.create(
                    invariant_type="parity",
                    expression=str(item),
                    strength="seed",
                    proof_source="parser",
                    confidence=0.7,
                )
            )

        archetypes = getattr(route, "archetypes", {}) or {}
        top_archetypes = sorted(archetypes.items(), key=lambda kv: (-float(kv[1]), kv[0]))[:4]
        for name, score in top_archetypes:
            if float(score) >= 0.20 and name in {"invariant", "symmetry", "modular", "bounding", "extremal"}:
                invariants.append(
                    InvariantRecord.create(
                        invariant_type="route_seed",
                        expression=name,
                        strength="seed",
                        proof_source="router",
                        confidence=float(score),
                    )
                )

        dedup = {i.canonical_key(): i for i in invariants}
        return [dedup[k] for k in sorted(dedup.keys())]

    def _initialize_operator_hints(self, route: RouteDecision) -> dict[str, float]:
        operator_prior = getattr(route, "operator_prior", {}) or {}
        ranked = sorted(((str(k), float(v)) for k, v in operator_prior.items()), key=lambda kv: (-kv[1], kv[0]))[:8]
        total = sum(v for _, v in ranked) or 1.0
        return {name: round(val / total, 6) for name, val in ranked}

    def _build_control_metadata(
        self,
        *,
        problem: ParsedProblem,
        route: RouteDecision,
        constraints: list[ConstraintRecord],
        goals: list[GoalRecord],
        invariants: list[InvariantRecord],
        operator_hints: dict[str, float],
        proof_obligations: list[ProofObligation],
        retrieval_depth_hint: int,
    ) -> dict[str, Any]:
        return {
            "branch_budget": int(getattr(route, "branch_budget", 32)),
            "retrieval_enabled": bool(getattr(route, "use_retrieval", True)),
            "retrieval_depth_hint": retrieval_depth_hint,
            "symbolic_enabled": bool(getattr(route, "use_symbolic", True)),
            "brute_force_enabled": bool(getattr(route, "use_brute_force", False)),
            "operator_hints": operator_hints,
            "top_problem_type": self._top_problem_type(route),
            "top_archetypes": self._top_archetypes(route),
            "routing_uncertainty": self._routing_uncertainty(route),
            "constraint_count": len(constraints),
            "goal_count": len(goals),
            "invariant_count": len(invariants),
            "proof_obligation_count": len(proof_obligations),
            "proof_obligation_status_counts": obligation_status_counts(proof_obligations),
            "answer_type": getattr(problem, "answer_type", ""),
        }

    def _build_state_summary(
        self,
        *,
        problem: ParsedProblem,
        route: RouteDecision,
        constraints: list[ConstraintRecord],
        goals: list[GoalRecord],
        invariants: list[InvariantRecord],
        proof_obligations: list[ProofObligation],
        operator_hints: dict[str, float],
    ) -> str:
        unknowns = ", ".join(sorted(str(u) for u in (getattr(problem, "unknowns", []) or []))[:6]) or "unknown"
        domain = getattr(getattr(problem, "domain", None), "value", str(getattr(problem, "domain", "unknown")))
        top_ops = ", ".join(name for name in list(operator_hints.keys())[:4]) or "none"
        top_arch = ", ".join(self._top_archetypes(route)[:3]) or "none"
        lines = [
            f"problem_id={problem.problem_id}",
            f"domain={domain}",
            f"unknowns={unknowns}",
            f"constraints={len(constraints)}",
            f"goals={len(goals)}",
            f"invariants={len(invariants)}",
            f"proof_obligations={len(proof_obligations)}",
            f"top_problem_type={self._top_problem_type(route)}",
            f"top_archetypes={top_arch}",
            f"preferred_operators={top_ops}",
        ]
        if constraints:
            lines.append(f"constraint_head={constraints[0].normalized_lhs or constraints[0].kind}")
        if goals:
            lines.append(f"goal_head={goals[0].normalized_text}")
        return " | ".join(lines)

    def _compute_canonical_state_hash(
        self,
        *,
        problem: ParsedProblem,
        constraints: list[ConstraintRecord],
        goals: list[GoalRecord],
        invariants: list[InvariantRecord],
        proof_obligations: list[ProofObligation],
        operator_hints: dict[str, float],
    ) -> str:
        payload = {
            "problem_id": problem.problem_id,
            "constraints": [c.canonical_key() for c in constraints],
            "goals": [g.canonical_key() for g in goals],
            "invariants": [i.canonical_key() for i in invariants],
            "proof_obligations": [item.canonical_key() for item in proof_obligations],
            "operator_hints": operator_hints,
            "answer_type": getattr(problem, "answer_type", ""),
            "domain": getattr(getattr(problem, "domain", None), "value", str(getattr(problem, "domain", ""))),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()

    def _root_confidence(self, problem: ParsedProblem, route: RouteDecision) -> float:
        difficulty_score = float(getattr(route, "difficulty_score", 0.5))
        parse_seed = float(getattr(problem, "difficulty_seed", 0.5))
        confidence = 0.65 - 0.20 * difficulty_score - 0.10 * max(0.0, parse_seed - 0.5)
        return round(max(0.15, min(0.75, confidence)), 6)

    def _infer_retrieval_depth_hint(self, route: RouteDecision) -> int:
        budget = int(getattr(route, "branch_budget", 32))
        enabled = bool(getattr(route, "use_retrieval", True))
        if not enabled:
            return 0
        if budget >= 96:
            return 5
        if budget >= 48:
            return 4
        if budget >= 24:
            return 3
        return 2

    def _top_problem_type(self, route: RouteDecision) -> str:
        distribution = getattr(route, "problem_type", {}) or {}
        if not distribution:
            return "unknown"
        return sorted(distribution.items(), key=lambda kv: (-float(kv[1]), kv[0]))[0][0]

    def _top_problem_type_probability(self, route: RouteDecision) -> float:
        distribution = getattr(route, "problem_type", {}) or {}
        if not distribution:
            return 0.0
        return float(sorted(distribution.items(), key=lambda kv: (-float(kv[1]), kv[0]))[0][1])

    def _top_archetypes(self, route: RouteDecision) -> list[str]:
        distribution = getattr(route, "archetypes", {}) or {}
        return [name for name, _ in sorted(distribution.items(), key=lambda kv: (-float(kv[1]), kv[0]))[:5]]

    def _routing_uncertainty(self, route: RouteDecision) -> float:
        return round(
            max(
                self._distribution_entropy(getattr(route, "problem_type", {}) or {}),
                self._distribution_entropy(getattr(route, "archetypes", {}) or {}),
            ),
            6,
        )

    def _distribution_entropy(self, distribution: dict[str, float]) -> float:
        if not distribution:
            return 0.0
        probs = [max(0.0, float(v)) for v in distribution.values()]
        total = sum(probs)
        if total <= 0:
            return 0.0
        probs = [p / total for p in probs if p > 0]
        if len(probs) <= 1:
            return 0.0
        entropy = -sum(p * math.log(p) for p in probs)
        return entropy / math.log(len(probs))


def build_initial_state_graph(problem: ParsedProblem, route: RouteDecision) -> StateGraphInitResult:
    return StateGraphInitializer().initialize(problem, route)


def initialize_state_graph(problem: ParsedProblem, route: RouteDecision) -> StateGraphInitResult:
    return StateGraphInitializer().initialize(problem, route)


__all__ = [
    "InitialStateMetadata",
    "StateGraphInitResult",
    "StateGraphInitializer",
    "build_initial_state_graph",
    "initialize_state_graph",
]
