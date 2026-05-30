"""
Local branch repair planning and execution.

This module keeps repair local by default:
- preserve the verified prefix whenever possible
- prefer targeted mutation or operator-neighbor substitution
- support rollback to a safe checkpoint instead of full restart
- emit explicit typed repair plans/candidates for branch-controller use

The implementation is intentionally deterministic and retrieval-ready.
"""
from __future__ import annotations

from enum import Enum
from hashlib import sha1
import re
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.common.constants import MAX_REPAIR_ATTEMPTS
from src.common.schemas import BranchTrace, FailureType, ReasoningStep, RouteDecision
from src.operators.operator_library import OperatorLibrary
from src.operators.operator_types import ApplicabilityStatus, OperatorNeighbor
from src.state_graph.node import (
    ConstraintRecord,
    EvidenceRecord,
    GoalRecord,
    InvariantRecord,
    OperatorApplicationRecord,
    ReasoningStateNode,
)
from src.state_graph.proof_obligations import ProofObligation


class RepairActionType(str, Enum):
    PATCH_STEP = "patch_step"
    MUTATE_WINDOW = "mutate_window"
    SUBSTITUTE_OPERATOR = "substitute_operator"
    ROLLBACK = "rollback"
    ROLLBACK_AND_SUBSTITUTE = "rollback_and_substitute"


class RepairStrategy(str, Enum):
    PREFIX_PATCH = "prefix_patch"
    TARGETED_MUTATION = "targeted_mutation"
    NEIGHBOR_SUBSTITUTION = "neighbor_substitution"
    SAFE_ROLLBACK = "safe_rollback"


class RepairOutcome(str, Enum):
    APPLIED = "applied"
    SKIPPED = "skipped"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_CANDIDATE = "no_candidate"


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )


class RepairBudget(StrictModel):
    max_attempts: int = Field(default=MAX_REPAIR_ATTEMPTS, ge=0)
    max_candidates: int = Field(default=8, ge=1)
    max_mutation_window: int = Field(default=2, ge=1)
    max_rollback_steps: int = Field(default=3, ge=0)
    local_window_radius: int = Field(default=1, ge=0)
    allow_operator_substitution: bool = True
    allow_mutation: bool = True
    allow_rollback: bool = True
    allow_full_restart: bool = False
    deterministic: bool = True
    retrieval_hints: tuple[str, ...] = Field(default_factory=tuple)


class RepairAction(StrictModel):
    action_id: str
    action_type: RepairActionType
    strategy: RepairStrategy
    description: str
    target_step_index: int | None = Field(default=None, ge=0)
    target_step_id: str | None = None
    target_node_id: str | None = None
    target_obligation_id: str | None = None
    preserve_prefix_until: int = Field(default=0, ge=0)
    rollback_to_step_index: int | None = Field(default=None, ge=0)
    replacement_operator: str | None = None
    mutated_steps: list[ReasoningStep] = Field(default_factory=list)
    added_constraints: list[str] = Field(default_factory=list)
    added_invariants: list[str] = Field(default_factory=list)
    added_goals: list[str] = Field(default_factory=list)
    retrieval_hints_used: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RepairCheckpoint(StrictModel):
    step_index: int = Field(ge=0)
    preserved_steps: int = Field(ge=0)
    rationale: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class RepairCandidate(StrictModel):
    candidate_id: str
    action: RepairAction
    score: float = Field(default=0.0, ge=0.0)
    expected_gain: float = Field(default=0.0)
    prefix_preservation_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    route_alignment: float = Field(default=0.0, ge=0.0, le=1.0)
    operator_alignment: float = Field(default=0.0, ge=0.0, le=1.0)
    rollback_cost: float = Field(default=0.0, ge=0.0)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class RepairPlan(StrictModel):
    branch_id: str
    failure_type: FailureType | None = None
    safe_prefix_end: int = Field(default=0, ge=0)
    checkpoints: list[RepairCheckpoint] = Field(default_factory=list)
    candidates: list[RepairCandidate] = Field(default_factory=list)
    selected_candidate_id: str | None = None
    budget: RepairBudget = Field(default_factory=RepairBudget)
    retrieval_context: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class RepairResult(StrictModel):
    outcome: RepairOutcome
    plan: RepairPlan
    selected_candidate: RepairCandidate | None = None
    repaired_branch: BranchTrace | None = None
    repaired_node: ReasoningStateNode | None = None
    history_update: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


def _stable_id(prefix: str, payload: Any) -> str:
    raw = repr(payload).encode("utf-8")
    return f"{prefix}_{sha1(raw).hexdigest()[:16]}"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _safe_copy_steps(steps: Sequence[ReasoningStep]) -> list[ReasoningStep]:
    return [
        ReasoningStep(
            step_num=step.step_num,
            description=step.description,
            operator_used=step.operator_used,
            symbolic_expression=step.symbolic_expression,
            symbolic_valid=step.symbolic_valid,
            python_code=step.python_code,
            python_result=step.python_result,
        )
        for step in steps
    ]


def _synthesized_steps_from_reasoning(branch: BranchTrace) -> list[ReasoningStep]:
    text = (branch.full_reasoning or "").strip()
    if not text:
        return []

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return [
        ReasoningStep(
            step_num=index + 1,
            description=line,
            operator_used=(
                branch.operator_sequence[index]
                if index < len(branch.operator_sequence)
                else None
            ),
            symbolic_valid=True,
        )
        for index, line in enumerate(lines)
    ]


def _branch_steps(branch: BranchTrace) -> list[ReasoningStep]:
    structured = list(branch.steps or [])
    if structured:
        return _safe_copy_steps(structured)
    return _synthesized_steps_from_reasoning(branch)


def _normalize_step_numbers(steps: Sequence[ReasoningStep]) -> list[ReasoningStep]:
    return [
        ReasoningStep(
            step_num=index + 1,
            description=step.description,
            operator_used=step.operator_used,
            symbolic_expression=step.symbolic_expression,
            symbolic_valid=step.symbolic_valid,
            python_code=step.python_code,
            python_result=step.python_result,
        )
        for index, step in enumerate(steps)
    ]


def _render_reasoning(steps: Sequence[ReasoningStep]) -> str:
    parts: list[str] = []
    for step in steps:
        parts.append(f"STEP {step.step_num}: {(step.description or '').strip()}")
    return "\n".join(parts)


def _parse_failure_step(branch: BranchTrace, steps: Sequence[ReasoningStep]) -> int:
    typed_index = getattr(branch, "failure_step_index", None)
    if isinstance(typed_index, int) and typed_index >= 0:
        return min(typed_index, max(len(steps) - 1, 0))

    if branch.failure_location:
        match = re.search(r"(\d+)", branch.failure_location)
        if match:
            idx = max(0, int(match.group(1)) - 1)
            return min(idx, max(len(steps) - 1, 0))

    for idx, step in enumerate(steps):
        if not step.symbolic_valid:
            return idx

    if steps:
        return max(len(steps) - 1, 0)
    return 0


def _proof_obligations_from_branch(branch: BranchTrace) -> list[ProofObligation]:
    obligations: list[ProofObligation] = []
    for item in list(getattr(branch, "proof_obligations", []) or []):
        if isinstance(item, ProofObligation):
            obligations.append(item)
            continue
        if not isinstance(item, dict):
            continue
        try:
            obligations.append(
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
    return obligations


def _typed_failure_obligation(branch: BranchTrace) -> ProofObligation | None:
    target_id = getattr(branch, "failure_obligation_id", None)
    obligations = _proof_obligations_from_branch(branch)
    if isinstance(target_id, str) and target_id.strip():
        for item in obligations:
            if item.obligation_id == target_id.strip():
                return item
    for item in obligations:
        if item.status.value == "contradicted":
            return item
    for item in obligations:
        if item.is_open():
            return item
    return None


def _typed_failure_reference(
    branch: BranchTrace,
    node: ReasoningStateNode | None,
    steps: Sequence[ReasoningStep],
) -> dict[str, Any]:
    obligation = _typed_failure_obligation(branch)
    return {
        "step_index": _parse_failure_step(branch, steps),
        "step_id": _clean_text(getattr(branch, "failure_step_id", "")) or None,
        "node_id": _clean_text(getattr(branch, "failure_node_id", "")) or getattr(node, "node_id", None),
        "obligation_id": obligation.obligation_id if obligation is not None else (_clean_text(getattr(branch, "failure_obligation_id", "")) or None),
        "obligation_claim": obligation.claim if obligation is not None else None,
        "source": "typed"
        if obligation is not None or getattr(branch, "failure_step_index", None) is not None
        else "fallback",
    }


def _safe_prefix_end(
    branch: BranchTrace,
    steps: Sequence[ReasoningStep],
    failure_step_index: int,
) -> int:
    if failure_step_index <= 0:
        return 0

    prefix = 0
    for idx, step in enumerate(steps[:failure_step_index]):
        if step.symbolic_valid:
            prefix = idx + 1
        else:
            break

    if prefix == 0 and failure_step_index > 0:
        prefix = max(0, failure_step_index)
    return min(prefix, len(steps))


def _failure_operator(
    branch: BranchTrace,
    steps: Sequence[ReasoningStep],
    failure_step_index: int,
) -> str | None:
    if 0 <= failure_step_index < len(steps) and steps[failure_step_index].operator_used:
        return steps[failure_step_index].operator_used

    sequence = list(branch.operator_sequence or [])
    if 0 <= failure_step_index < len(sequence):
        return sequence[failure_step_index]
    if sequence:
        return sequence[-1]
    return None


def _top_route_operators(route: RouteDecision | None, limit: int = 4) -> list[str]:
    if route is None:
        return []
    prior = dict(route.operator_prior or {})
    if not prior:
        return []
    ordered = sorted(prior.items(), key=lambda item: (-float(item[1]), item[0]))
    return [name for name, _ in ordered[:limit]]


def _failure_preferences(failure_type: FailureType | None) -> tuple[RepairStrategy, ...]:
    mapping: dict[FailureType, tuple[RepairStrategy, ...]] = {
        FailureType.ARITHMETIC_ERROR: (
            RepairStrategy.TARGETED_MUTATION,
            RepairStrategy.PREFIX_PATCH,
            RepairStrategy.SAFE_ROLLBACK,
        ),
        FailureType.SYMBOLIC_MISMATCH: (
            RepairStrategy.NEIGHBOR_SUBSTITUTION,
            RepairStrategy.TARGETED_MUTATION,
            RepairStrategy.SAFE_ROLLBACK,
        ),
        FailureType.MISSING_CASE: (
            RepairStrategy.TARGETED_MUTATION,
            RepairStrategy.SAFE_ROLLBACK,
        ),
        FailureType.FALSE_ASSUMPTION: (
            RepairStrategy.SAFE_ROLLBACK,
            RepairStrategy.NEIGHBOR_SUBSTITUTION,
            RepairStrategy.TARGETED_MUTATION,
        ),
        FailureType.LOGIC_ERROR: (
            RepairStrategy.NEIGHBOR_SUBSTITUTION,
            RepairStrategy.SAFE_ROLLBACK,
            RepairStrategy.TARGETED_MUTATION,
        ),
        FailureType.COVERAGE_GAP: (
            RepairStrategy.TARGETED_MUTATION,
            RepairStrategy.SAFE_ROLLBACK,
        ),
        FailureType.TIMEOUT: (
            RepairStrategy.PREFIX_PATCH,
            RepairStrategy.SAFE_ROLLBACK,
        ),
    }
    if failure_type is None:
        return (
            RepairStrategy.PREFIX_PATCH,
            RepairStrategy.TARGETED_MUTATION,
            RepairStrategy.NEIGHBOR_SUBSTITUTION,
            RepairStrategy.SAFE_ROLLBACK,
        )
    return mapping.get(
        failure_type,
        (
            RepairStrategy.PREFIX_PATCH,
            RepairStrategy.TARGETED_MUTATION,
            RepairStrategy.NEIGHBOR_SUBSTITUTION,
            RepairStrategy.SAFE_ROLLBACK,
        ),
    )


class BranchRepairPlanner:
    """
    Deterministic local repair planner.

    The planner produces controller-ready repair plans and can also apply the
    highest-ranked candidate to a BranchTrace / ReasoningStateNode pair.
    """

    def __init__(self, operator_library: OperatorLibrary | None = None) -> None:
        self.operator_library = operator_library or OperatorLibrary()

    def plan_repair(
        self,
        branch: BranchTrace,
        *,
        node: ReasoningStateNode | None = None,
        route: RouteDecision | None = None,
        failure_type: FailureType | None = None,
        budget: RepairBudget | None = None,
        retrieval_hints: Sequence[str] | None = None,
    ) -> RepairPlan:
        active_budget = budget or RepairBudget()
        active_failure = failure_type or branch.failure_type
        steps = _branch_steps(branch)
        typed_failure = _typed_failure_reference(branch, node, steps)
        failure_step_index = int(typed_failure["step_index"])
        safe_prefix_end = _safe_prefix_end(branch, steps, failure_step_index)
        checkpoints = self._build_checkpoints(
            steps=steps,
            safe_prefix_end=safe_prefix_end,
            failure_step_index=failure_step_index,
            budget=active_budget,
        )

        retrieval_context = list(retrieval_hints or active_budget.retrieval_hints)
        candidates = self._generate_candidates(
            branch=branch,
            steps=steps,
            node=node,
            route=route,
            failure_type=active_failure,
            failure_step_index=failure_step_index,
            safe_prefix_end=safe_prefix_end,
            checkpoints=checkpoints,
            budget=active_budget,
            typed_failure=typed_failure,
            retrieval_hints=retrieval_context,
        )

        selected_candidate_id = candidates[0].candidate_id if candidates else None
        return RepairPlan(
            branch_id=branch.branch_id,
            failure_type=active_failure,
            safe_prefix_end=safe_prefix_end,
            checkpoints=checkpoints,
            candidates=candidates,
            selected_candidate_id=selected_candidate_id,
            budget=active_budget,
            retrieval_context=retrieval_context,
            diagnostics={
                "failure_step_index": failure_step_index,
                "failure_step_id": typed_failure["step_id"],
                "failure_node_id": typed_failure["node_id"],
                "failure_obligation_id": typed_failure["obligation_id"],
                "failure_reference_source": typed_failure["source"],
                "structured_step_count": len(steps),
                "repair_count": branch.repair_count,
            },
        )

    def apply_plan(
        self,
        branch: BranchTrace,
        plan: RepairPlan,
        *,
        node: ReasoningStateNode | None = None,
    ) -> RepairResult:
        if branch.repair_count >= plan.budget.max_attempts:
            return RepairResult(
                outcome=RepairOutcome.BUDGET_EXHAUSTED,
                plan=plan,
                diagnostics={"reason": "repair_budget_exhausted"},
            )

        if not plan.candidates:
            return RepairResult(
                outcome=RepairOutcome.NO_CANDIDATE,
                plan=plan,
                diagnostics={"reason": "no_repair_candidate"},
            )

        selected = plan.candidates[0]
        repaired_branch = self._apply_candidate_to_branch(branch, selected)
        repaired_node = self._apply_candidate_to_node(node, selected, repaired_branch) if node is not None else None

        return RepairResult(
            outcome=RepairOutcome.APPLIED,
            plan=plan.model_copy(update={"selected_candidate_id": selected.candidate_id}),
            selected_candidate=selected,
            repaired_branch=repaired_branch,
            repaired_node=repaired_node,
            history_update=self._history_update(selected, repaired_branch, repaired_node),
            diagnostics={"selected_score": selected.score},
        )

    def repair_branch(
        self,
        branch: BranchTrace,
        *,
        node: ReasoningStateNode | None = None,
        route: RouteDecision | None = None,
        failure_type: FailureType | None = None,
        budget: RepairBudget | None = None,
        retrieval_hints: Sequence[str] | None = None,
    ) -> RepairResult:
        plan = self.plan_repair(
            branch,
            node=node,
            route=route,
            failure_type=failure_type,
            budget=budget,
            retrieval_hints=retrieval_hints,
        )
        return self.apply_plan(branch, plan, node=node)

    def _build_checkpoints(
        self,
        *,
        steps: Sequence[ReasoningStep],
        safe_prefix_end: int,
        failure_step_index: int,
        budget: RepairBudget,
    ) -> list[RepairCheckpoint]:
        checkpoints: list[RepairCheckpoint] = []
        checkpoints.append(
            RepairCheckpoint(
                step_index=safe_prefix_end,
                preserved_steps=safe_prefix_end,
                rationale="largest verified prefix",
                confidence=0.85 if safe_prefix_end < failure_step_index else 0.65,
            )
        )

        for rollback in range(1, budget.max_rollback_steps + 1):
            step_index = max(0, safe_prefix_end - rollback)
            if checkpoints and checkpoints[-1].step_index == step_index:
                continue
            checkpoints.append(
                RepairCheckpoint(
                    step_index=step_index,
                    preserved_steps=step_index,
                    rationale=f"rollback_{rollback}_step",
                    confidence=max(0.30, 0.80 - 0.15 * rollback),
                )
            )

        return checkpoints

    def _generate_candidates(
        self,
        *,
        branch: BranchTrace,
        steps: Sequence[ReasoningStep],
        node: ReasoningStateNode | None,
        route: RouteDecision | None,
        failure_type: FailureType | None,
        failure_step_index: int,
        safe_prefix_end: int,
        checkpoints: Sequence[RepairCheckpoint],
        budget: RepairBudget,
        typed_failure: Mapping[str, Any],
        retrieval_hints: Sequence[str],
    ) -> list[RepairCandidate]:
        raw: list[RepairCandidate] = []
        preferred = _failure_preferences(failure_type)
        failure_operator = _failure_operator(branch, steps, failure_step_index)

        for strategy in preferred:
            if strategy is RepairStrategy.NEIGHBOR_SUBSTITUTION and budget.allow_operator_substitution:
                raw.extend(
                    self._neighbor_substitution_candidates(
                        branch=branch,
                        steps=steps,
                        node=node,
                        route=route,
                        failure_type=failure_type,
                        failure_step_index=failure_step_index,
                        safe_prefix_end=safe_prefix_end,
                        failure_operator=failure_operator,
                        typed_failure=typed_failure,
                        retrieval_hints=retrieval_hints,
                    )
                )
            elif strategy in {RepairStrategy.PREFIX_PATCH, RepairStrategy.TARGETED_MUTATION} and budget.allow_mutation:
                raw.extend(
                    self._mutation_candidates(
                        branch=branch,
                        steps=steps,
                        route=route,
                        failure_type=failure_type,
                        failure_step_index=failure_step_index,
                        safe_prefix_end=safe_prefix_end,
                        strategy=strategy,
                        budget=budget,
                        typed_failure=typed_failure,
                        retrieval_hints=retrieval_hints,
                    )
                )
            elif strategy is RepairStrategy.SAFE_ROLLBACK and budget.allow_rollback:
                raw.extend(
                    self._rollback_candidates(
                        branch=branch,
                        steps=steps,
                        route=route,
                        failure_type=failure_type,
                        checkpoints=checkpoints,
                        failure_step_index=failure_step_index,
                        failure_operator=failure_operator,
                        typed_failure=typed_failure,
                        retrieval_hints=retrieval_hints,
                    )
                )

        dedup: dict[tuple[Any, ...], RepairCandidate] = {}
        for candidate in raw:
            key = (
                candidate.action.action_type,
                candidate.action.preserve_prefix_until,
                candidate.action.rollback_to_step_index,
                candidate.action.replacement_operator,
                candidate.action.target_obligation_id,
                tuple(step.description for step in candidate.action.mutated_steps),
            )
            if key not in dedup or candidate.score > dedup[key].score:
                dedup[key] = candidate

        ordered = sorted(
            dedup.values(),
            key=lambda c: (
                -c.score,
                -c.expected_gain,
                -c.prefix_preservation_ratio,
                c.rollback_cost,
                c.candidate_id,
            ),
        )
        return ordered[: budget.max_candidates]

    def _mutation_candidates(
        self,
        *,
        branch: BranchTrace,
        steps: Sequence[ReasoningStep],
        route: RouteDecision | None,
        failure_type: FailureType | None,
        failure_step_index: int,
        safe_prefix_end: int,
        strategy: RepairStrategy,
        budget: RepairBudget,
        typed_failure: Mapping[str, Any],
        retrieval_hints: Sequence[str],
    ) -> list[RepairCandidate]:
        if not steps:
            return []

        window_start = max(safe_prefix_end, failure_step_index - budget.local_window_radius)
        window_end = min(
            len(steps),
            max(failure_step_index + 1, window_start + budget.max_mutation_window),
        )
        preserved_prefix = _safe_copy_steps(steps[:safe_prefix_end])
        window = _safe_copy_steps(steps[window_start:window_end])
        suffix = _safe_copy_steps(steps[window_end:])
        target = window[0] if window else steps[min(failure_step_index, len(steps) - 1)]
        top_route = _top_route_operators(route, limit=1)
        operator_hint = target.operator_used or (top_route[0] if top_route else None)

        templates = self._mutation_templates(
            failure_type=failure_type,
            step=target,
            operator_hint=operator_hint,
            retrieval_hints=retrieval_hints,
        )

        candidates: list[RepairCandidate] = []
        for variant_index, description in enumerate(templates):
            mutated_core = ReasoningStep(
                step_num=target.step_num,
                description=description,
                operator_used=operator_hint or target.operator_used,
                symbolic_expression=target.symbolic_expression,
                symbolic_valid=False,
                python_code=target.python_code,
                python_result=target.python_result,
            )
            mutated_steps = _normalize_step_numbers(preserved_prefix + [mutated_core] + suffix)
            action_type = (
                RepairActionType.PATCH_STEP
                if strategy is RepairStrategy.PREFIX_PATCH
                else RepairActionType.MUTATE_WINDOW
            )
            action = RepairAction(
                action_id=_stable_id(
                    "repair_action",
                    (branch.branch_id, strategy.value, variant_index, description),
                ),
                action_type=action_type,
                strategy=strategy,
                description=f"{strategy.value} on step {target.step_num}",
                target_step_index=failure_step_index,
                target_step_id=str(typed_failure.get("step_id") or "") or None,
                target_node_id=str(typed_failure.get("node_id") or "") or None,
                target_obligation_id=str(typed_failure.get("obligation_id") or "") or None,
                preserve_prefix_until=safe_prefix_end,
                replacement_operator=operator_hint,
                mutated_steps=mutated_steps,
                retrieval_hints_used=list(retrieval_hints),
                metadata={
                    "window_start": window_start,
                    "window_end": window_end,
                    "failure_hint": getattr(failure_type, "value", None),
                },
            )
            candidates.append(
                self._rank_candidate(
                    action=action,
                    route=route,
                    failure_type=failure_type,
                    safe_prefix_end=safe_prefix_end,
                    total_steps=len(steps),
                    rollback_cost=0.0,
                )
            )
        return candidates

    def _neighbor_substitution_candidates(
        self,
        *,
        branch: BranchTrace,
        steps: Sequence[ReasoningStep],
        node: ReasoningStateNode | None,
        route: RouteDecision | None,
        failure_type: FailureType | None,
        failure_step_index: int,
        safe_prefix_end: int,
        failure_operator: str | None,
        typed_failure: Mapping[str, Any],
        retrieval_hints: Sequence[str],
    ) -> list[RepairCandidate]:
        if not failure_operator:
            return []

        neighbors = self.operator_library.fallback_neighbors(failure_operator)
        if not neighbors and route is not None:
            neighbors = [
                OperatorNeighbor(
                    operator_name=name,
                    reason="route_prior_fallback",
                    weight=0.50,
                )
                for name in _top_route_operators(route, limit=3)
                if name != failure_operator
            ]

        candidates: list[RepairCandidate] = []
        prefix = _safe_copy_steps(steps[:safe_prefix_end])
        suffix = _safe_copy_steps(steps[failure_step_index + 1 :])
        failed_step = steps[min(failure_step_index, len(steps) - 1)] if steps else None

        for index, neighbor in enumerate(neighbors):
            replacement = neighbor.operator_name
            route_alignment, diagnostics = self._operator_alignment(
                replacement,
                node=node,
                route=route,
            )

            description = (
                failed_step.description
                if failed_step is not None
                else "re-evaluate failed local step"
            )
            substituted = ReasoningStep(
                step_num=failure_step_index + 1,
                description=f"{description} [repaired using {replacement}: {neighbor.reason}]",
                operator_used=replacement,
                symbolic_expression=getattr(failed_step, "symbolic_expression", None),
                symbolic_valid=False,
                python_code=getattr(failed_step, "python_code", None),
                python_result=getattr(failed_step, "python_result", None),
            )
            action = RepairAction(
                action_id=_stable_id(
                    "repair_action",
                    (
                        branch.branch_id,
                        "neighbor",
                        failure_operator,
                        replacement,
                        index,
                    ),
                ),
                action_type=RepairActionType.SUBSTITUTE_OPERATOR,
                strategy=RepairStrategy.NEIGHBOR_SUBSTITUTION,
                description=f"swap {failure_operator} -> {replacement}",
                target_step_index=failure_step_index,
                target_step_id=str(typed_failure.get("step_id") or "") or None,
                target_node_id=str(typed_failure.get("node_id") or "") or None,
                target_obligation_id=str(typed_failure.get("obligation_id") or "") or None,
                preserve_prefix_until=safe_prefix_end,
                replacement_operator=replacement,
                mutated_steps=_normalize_step_numbers(prefix + [substituted] + suffix),
                retrieval_hints_used=list(retrieval_hints),
                metadata={
                    "failed_operator": failure_operator,
                    "neighbor_reason": neighbor.reason,
                    "neighbor_weight": neighbor.weight,
                    "applicability": diagnostics,
                },
            )
            candidates.append(
                self._rank_candidate(
                    action=action,
                    route=route,
                    failure_type=failure_type,
                    safe_prefix_end=safe_prefix_end,
                    total_steps=len(steps),
                    rollback_cost=0.0,
                    route_alignment_override=route_alignment,
                    operator_alignment_override=route_alignment,
                    bonus=0.12 * float(neighbor.weight),
                )
            )
        return candidates

    def _rollback_candidates(
        self,
        *,
        branch: BranchTrace,
        steps: Sequence[ReasoningStep],
        route: RouteDecision | None,
        failure_type: FailureType | None,
        checkpoints: Sequence[RepairCheckpoint],
        failure_step_index: int,
        failure_operator: str | None,
        typed_failure: Mapping[str, Any],
        retrieval_hints: Sequence[str],
    ) -> list[RepairCandidate]:
        top_route = _top_route_operators(route, limit=3)
        candidates: list[RepairCandidate] = []

        for index, checkpoint in enumerate(checkpoints):
            prefix = _safe_copy_steps(steps[: checkpoint.step_index])
            fallback_operator = None
            if top_route:
                for name in top_route:
                    if name != failure_operator:
                        fallback_operator = name
                        break

            action_type = (
                RepairActionType.ROLLBACK_AND_SUBSTITUTE
                if fallback_operator
                else RepairActionType.ROLLBACK
            )
            patch_step = ReasoningStep(
                step_num=checkpoint.step_index + 1,
                description=(
                    f"Rollback to safe checkpoint after step {checkpoint.step_index} "
                    f"and re-open local search"
                    + (f" using {fallback_operator}" if fallback_operator else "")
                ),
                operator_used=fallback_operator,
                symbolic_valid=False,
            )
            action = RepairAction(
                action_id=_stable_id(
                    "repair_action",
                    (
                        branch.branch_id,
                        "rollback",
                        checkpoint.step_index,
                        fallback_operator,
                        index,
                    ),
                ),
                action_type=action_type,
                strategy=RepairStrategy.SAFE_ROLLBACK,
                description=checkpoint.rationale,
                target_step_index=failure_step_index,
                target_step_id=str(typed_failure.get("step_id") or "") or None,
                target_node_id=str(typed_failure.get("node_id") or "") or None,
                target_obligation_id=str(typed_failure.get("obligation_id") or "") or None,
                preserve_prefix_until=checkpoint.step_index,
                rollback_to_step_index=checkpoint.step_index,
                replacement_operator=fallback_operator,
                mutated_steps=_normalize_step_numbers(prefix + [patch_step]),
                retrieval_hints_used=list(retrieval_hints),
                metadata={"checkpoint_confidence": checkpoint.confidence},
            )
            rollback_cost = max(0.0, len(steps) - checkpoint.step_index)
            bonus = 0.08 if failure_type in {FailureType.FALSE_ASSUMPTION, FailureType.LOGIC_ERROR} else 0.0
            candidates.append(
                self._rank_candidate(
                    action=action,
                    route=route,
                    failure_type=failure_type,
                    safe_prefix_end=checkpoint.step_index,
                    total_steps=len(steps),
                    rollback_cost=rollback_cost,
                    bonus=bonus,
                )
            )
        return candidates

    def _mutation_templates(
        self,
        *,
        failure_type: FailureType | None,
        step: ReasoningStep,
        operator_hint: str | None,
        retrieval_hints: Sequence[str],
    ) -> list[str]:
        hint_text = f" with retrieval hints: {', '.join(retrieval_hints[:2])}" if retrieval_hints else ""
        operator_text = f" via {operator_hint}" if operator_hint else ""
        base = step.description or "repair the local derivation"

        if failure_type is FailureType.ARITHMETIC_ERROR:
            return [
                f"Recompute the arithmetic in the previous relation{operator_text}{hint_text}; keep the algebraic setup unchanged.",
                f"Preserve the prior equalities and correct the numeric simplification{operator_text}.",
            ]
        if failure_type is FailureType.MISSING_CASE:
            return [
                f"{base} Add the missing local case split before continuing{hint_text}.",
                f"{base} Preserve the prefix and enumerate the omitted case explicitly.",
            ]
        if failure_type is FailureType.SYMBOLIC_MISMATCH:
            return [
                f"{base} Re-express the failing transformation using a symbolic-safe rewrite{operator_text}.",
                f"{base} Preserve the prefix and normalize the local symbolic step before using it.",
            ]
        if failure_type is FailureType.FALSE_ASSUMPTION:
            return [
                f"{base} Remove the unsupported assumption and restart only from this local point{hint_text}.",
                f"{base} Replace the unsupported assumption with a route-compatible operator{operator_text}.",
            ]
        if failure_type is FailureType.COVERAGE_GAP:
            return [
                f"{base} Add the missing coverage argument while preserving the valid prefix.",
                f"{base} Refine the local reasoning so all remaining constraints are addressed.",
            ]
        return [
            f"{base} Repair only this local step{operator_text}{hint_text}.",
            f"{base} Keep the prefix and mutate the failing local inference, not the whole branch.",
        ]

    def _rank_candidate(
        self,
        *,
        action: RepairAction,
        route: RouteDecision | None,
        failure_type: FailureType | None,
        safe_prefix_end: int,
        total_steps: int,
        rollback_cost: float,
        route_alignment_override: float | None = None,
        operator_alignment_override: float | None = None,
        bonus: float = 0.0,
    ) -> RepairCandidate:
        prefix_ratio = 1.0 if total_steps <= 0 else safe_prefix_end / max(total_steps, 1)
        route_alignment = (
            route_alignment_override
            if route_alignment_override is not None
            else self._route_alignment(action.replacement_operator, route)
        )
        operator_alignment = (
            operator_alignment_override
            if operator_alignment_override is not None
            else route_alignment
        )
        failure_bonus = self._failure_bonus(action.strategy, failure_type)
        rollback_penalty = min(0.35, rollback_cost * 0.05)
        expected_gain = (
            0.42 * prefix_ratio
            + 0.22 * route_alignment
            + 0.18 * operator_alignment
            + failure_bonus
            + bonus
            - rollback_penalty
        )
        score = max(0.0, expected_gain)

        return RepairCandidate(
            candidate_id=_stable_id(
                "repair_candidate",
                (
                    action.action_id,
                    action.strategy.value,
                    action.replacement_operator,
                    action.rollback_to_step_index,
                ),
            ),
            action=action,
            score=round(score, 6),
            expected_gain=round(expected_gain, 6),
            prefix_preservation_ratio=round(prefix_ratio, 6),
            route_alignment=round(route_alignment, 6),
            operator_alignment=round(operator_alignment, 6),
            rollback_cost=round(rollback_cost, 6),
            diagnostics={
                "failure_type": getattr(failure_type, "value", None),
                "strategy": action.strategy.value,
                "rollback_penalty": round(rollback_penalty, 6),
                "bonus": round(bonus, 6),
            },
        )

    def _failure_bonus(
        self,
        strategy: RepairStrategy,
        failure_type: FailureType | None,
    ) -> float:
        if failure_type is None:
            return 0.05

        table: dict[tuple[RepairStrategy, FailureType], float] = {
            (RepairStrategy.TARGETED_MUTATION, FailureType.ARITHMETIC_ERROR): 0.18,
            (RepairStrategy.PREFIX_PATCH, FailureType.ARITHMETIC_ERROR): 0.12,
            (RepairStrategy.NEIGHBOR_SUBSTITUTION, FailureType.SYMBOLIC_MISMATCH): 0.18,
            (RepairStrategy.SAFE_ROLLBACK, FailureType.FALSE_ASSUMPTION): 0.18,
            (RepairStrategy.TARGETED_MUTATION, FailureType.MISSING_CASE): 0.16,
            (RepairStrategy.TARGETED_MUTATION, FailureType.COVERAGE_GAP): 0.14,
            (RepairStrategy.SAFE_ROLLBACK, FailureType.LOGIC_ERROR): 0.12,
        }
        return table.get((strategy, failure_type), 0.06)

    def _route_alignment(
        self,
        operator_name: str | None,
        route: RouteDecision | None,
    ) -> float:
        if operator_name is None or route is None:
            return 0.50
        return _clamp01(float((route.operator_prior or {}).get(operator_name, 0.35)))

    def _operator_alignment(
        self,
        operator_name: str,
        *,
        node: ReasoningStateNode | None,
        route: RouteDecision | None,
    ) -> tuple[float, dict[str, Any]]:
        if node is None:
            return self._route_alignment(operator_name, route), {"source": "route_only"}

        applicability = self.operator_library.check_applicability(
            operator_name,
            node,
            route,
        )
        if applicability.status is ApplicabilityStatus.INAPPLICABLE:
            return 0.05, applicability.model_dump(mode="json")
        base = 0.30 if applicability.status is ApplicabilityStatus.WEAK else 0.55
        return _clamp01(base + 0.45 * applicability.score), applicability.model_dump(mode="json")

    def _apply_candidate_to_branch(
        self,
        branch: BranchTrace,
        candidate: RepairCandidate,
    ) -> BranchTrace:
        steps = _normalize_step_numbers(candidate.action.mutated_steps)
        sequence = [step.operator_used for step in steps if step.operator_used]
        reasoning = _render_reasoning(steps)
        prefix_note = (
            f"[REPAIR:{candidate.action.strategy.value}] "
            f"preserved_prefix={candidate.action.preserve_prefix_until}"
        )
        full_reasoning = reasoning if not branch.full_reasoning else f"{reasoning}\n{prefix_note}"

        repaired = branch.model_copy(
            update={
                "steps": steps,
                "full_reasoning": full_reasoning,
                "operator_sequence": sequence,
                "repaired": True,
                "repair_count": branch.repair_count + 1,
                "failure_step_index": candidate.action.target_step_index,
                "failure_step_id": candidate.action.target_step_id,
                "failure_node_id": candidate.action.target_node_id,
                "failure_obligation_id": candidate.action.target_obligation_id,
                "failure_location": (
                    f"step_{candidate.action.target_step_index + 1}"
                    if candidate.action.target_step_index is not None
                    else branch.failure_location
                ),
                "branch_score": max(branch.branch_score, candidate.score),
            }
        )
        return repaired

    def _apply_candidate_to_node(
        self,
        node: ReasoningStateNode,
        candidate: RepairCandidate,
        repaired_branch: BranchTrace,
    ) -> ReasoningStateNode:
        action = candidate.action
        repair_op = OperatorApplicationRecord.create(
            operator_name=action.replacement_operator or action.action_type.value,
            rationale=action.description,
            pre_state_fingerprint=node.state_fingerprint,
            post_state_fingerprint=None,
            success=True,
            metadata={
                "repair_action_type": action.action_type.value,
                "repair_strategy": action.strategy.value,
                "preserve_prefix_until": action.preserve_prefix_until,
                "rollback_to_step_index": action.rollback_to_step_index,
            },
        )
        evidence = EvidenceRecord.create(
            kind="repair",
            source="branch_repair",
            summary=action.description,
            score=min(1.0, candidate.score),
            supports=[action.replacement_operator] if action.replacement_operator else [],
            payload={
                "candidate_id": candidate.candidate_id,
                "branch_id": repaired_branch.branch_id,
            },
        )

        constraints = list(node.constraints)
        invariants = list(node.invariants)
        goals = list(node.goals)
        proof_obligations = list(node.proof_obligations)

        for item in action.added_constraints:
            constraints.append(
                ConstraintRecord.create(
                    kind="other",
                    lhs=item,
                    relation=None,
                    rhs=None,
                    origin="repair",
                    strength="soft",
                    confidence=0.60,
                )
            )
        for item in action.added_invariants:
            invariants.append(
                InvariantRecord.create(
                    invariant_type="repair",
                    expression=item,
                    strength="candidate",
                    proof_source="repair",
                    confidence=0.55,
                )
            )
        for item in action.added_goals:
            goals.append(
                GoalRecord.create(
                    kind="subgoal",
                    text=item,
                    priority=40,
                    status="open",
                    confidence=0.40,
                )
            )

        updated_metadata = dict(node.metadata)
        history = list(updated_metadata.get("repair_history", []))
        history.append(
            {
                "candidate_id": candidate.candidate_id,
                "action_type": action.action_type.value,
                "strategy": action.strategy.value,
                "rollback_to_step_index": action.rollback_to_step_index,
                "replacement_operator": action.replacement_operator,
                "target_obligation_id": action.target_obligation_id,
                "score": candidate.score,
            }
        )
        updated_metadata["repair_history"] = history
        updated_metadata["latest_repair_plan"] = {
            "candidate_id": candidate.candidate_id,
            "preserve_prefix_until": action.preserve_prefix_until,
            "target_obligation_id": action.target_obligation_id,
        }
        updated_metadata["branch_trace_repair_count"] = repaired_branch.repair_count

        if action.target_obligation_id:
            proof_obligations = [
                item.add_note(f"repair_candidate::{candidate.candidate_id}")
                if item.obligation_id == action.target_obligation_id
                else item
                for item in proof_obligations
            ]

        partial_solution = "\n".join(
            step.description for step in repaired_branch.steps[: action.preserve_prefix_until]
        )
        return node.with_updates(
            constraints=tuple(constraints),
            invariants=tuple(invariants),
            goals=tuple(goals),
            proof_obligations=tuple(proof_obligations),
            partial_solution=partial_solution,
            summary_text=f"{node.summary_text} | repaired via {action.strategy.value}",
            operator_history=node.operator_history + (repair_op,),
            evidence=node.evidence + (evidence,),
            confidence=max(node.confidence, min(1.0, 0.45 + candidate.score * 0.35)),
            verifier_score=max(node.verifier_score, min(1.0, candidate.score)),
            symbolic_valid=False,
            merge_ready=False,
            metadata=updated_metadata,
            repair_from_node_id=node.node_id,
            parent_node_ids=(node.node_id,),
            source_tags=tuple(sorted(set(node.provenance.source_tags + ("repair",)))),
            depth=node.provenance.depth + 1,
            lineage=node.provenance.lineage + (node.node_id,),
            node_id=_stable_id("repair_node", (node.node_id, candidate.candidate_id)),
        )

    def _history_update(
        self,
        candidate: RepairCandidate,
        repaired_branch: BranchTrace,
        repaired_node: ReasoningStateNode | None,
    ) -> dict[str, Any]:
        return {
            "branch_id": repaired_branch.branch_id,
            "repair_count": repaired_branch.repair_count,
            "selected_candidate_id": candidate.candidate_id,
            "action_type": candidate.action.action_type.value,
            "strategy": candidate.action.strategy.value,
            "target_step_id": candidate.action.target_step_id,
            "replacement_operator": candidate.action.replacement_operator,
            "rollback_to_step_index": candidate.action.rollback_to_step_index,
            "target_obligation_id": candidate.action.target_obligation_id,
            "node_id": repaired_node.node_id if repaired_node is not None else None,
        }


def plan_branch_repair(
    branch: BranchTrace,
    *,
    node: ReasoningStateNode | None = None,
    route: RouteDecision | None = None,
    failure_type: FailureType | None = None,
    budget: RepairBudget | None = None,
    retrieval_hints: Sequence[str] | None = None,
    operator_library: OperatorLibrary | None = None,
) -> RepairPlan:
    planner = BranchRepairPlanner(operator_library=operator_library)
    return planner.plan_repair(
        branch,
        node=node,
        route=route,
        failure_type=failure_type,
        budget=budget,
        retrieval_hints=retrieval_hints,
    )


def repair_branch(
    branch: BranchTrace,
    *,
    node: ReasoningStateNode | None = None,
    route: RouteDecision | None = None,
    failure_type: FailureType | None = None,
    budget: RepairBudget | None = None,
    retrieval_hints: Sequence[str] | None = None,
    operator_library: OperatorLibrary | None = None,
) -> RepairResult:
    planner = BranchRepairPlanner(operator_library=operator_library)
    return planner.repair_branch(
        branch,
        node=node,
        route=route,
        failure_type=failure_type,
        budget=budget,
        retrieval_hints=retrieval_hints,
    )


__all__ = [
    "BranchRepairPlanner",
    "RepairAction",
    "RepairActionType",
    "RepairBudget",
    "RepairCandidate",
    "RepairCheckpoint",
    "RepairOutcome",
    "RepairPlan",
    "RepairResult",
    "RepairStrategy",
    "plan_branch_repair",
    "repair_branch",
]
