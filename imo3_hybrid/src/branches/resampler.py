# src/branches/resampler.py
"""
Bounded, diversity-aware branch resampling.

This module owns controlled candidate regeneration for self-consistency-first
search. It does not generate random retry noise; instead it emits typed,
budget-aware resampling plans and can materialize deterministic child branches
annotated with the profile/hints that a branch controller or solver can use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha1
import json
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.branches.branch_state import BranchPhase, BranchState, BranchStepKind
from src.branches.failure_classifier import BranchFailureDiagnosis, FailureCategory
from src.common.schemas import FailureType, RouteDecision


class ResampleProfileKind(str, Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    DIVERSE = "diverse"


class ResampleReason(str, Enum):
    SELF_CONSISTENCY_GAP = "self_consistency_gap"
    FAILURE_RECOVERY = "failure_recovery"
    DIVERSITY_RECOVERY = "diversity_recovery"
    BUDGET_FILL = "budget_fill"


class ResampleOutcome(str, Enum):
    MATERIALIZED = "materialized"
    NO_BUDGET = "no_budget"
    NO_NOVELTY = "no_novelty"
    EMPTY = "empty"


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )


class ResampleBudget(StrictModel):
    max_new_branches: int = Field(default=4, ge=0)
    max_total_branches: int = Field(default=32, ge=1)
    max_attempts_per_focus: int = Field(default=1, ge=1)
    max_duplicates_per_signature: int = Field(default=0, ge=0)
    novelty_floor: float = Field(default=0.20, ge=0.0, le=1.0)
    deterministic_seed: int = Field(default=0, ge=0)
    prefer_self_consistency_fill: bool = True
    suppress_answer_duplicates: bool = True


class FailureResamplingHint(StrictModel):
    category: str
    failure_type: str | None = None
    recoverability: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str
    suggested_operators: list[str] = Field(default_factory=list)
    preferred_profile: ResampleProfileKind = ResampleProfileKind.BALANCED
    rollback_first: bool = False


class ResampleProfile(StrictModel):
    profile_id: str
    profile_kind: ResampleProfileKind
    reason: ResampleReason
    focus_operators: list[str] = Field(default_factory=list)
    avoided_operators: list[str] = Field(default_factory=list)
    temperature: float = Field(default=0.5, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    seed: int = Field(default=0, ge=0)
    rationale: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResampleCandidate(StrictModel):
    candidate_id: str
    parent_branch_id: str
    branch_label: str
    profile: ResampleProfile
    novelty_score: float = Field(default=0.0, ge=0.0, le=1.0)
    duplicate_score: float = Field(default=0.0, ge=0.0, le=1.0)
    route_alignment: float = Field(default=0.0, ge=0.0, le=1.0)
    budget_cost: int = Field(default=1, ge=0)
    suppressed: bool = False
    suppression_reason: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class ResamplePlan(StrictModel):
    problem_id: str
    active_branch_count: int = Field(default=0, ge=0)
    branch_budget: int = Field(default=0, ge=0)
    remaining_budget: int = Field(default=0, ge=0)
    target_sample_count: int = Field(default=0, ge=0)
    budget: ResampleBudget
    failure_hints: list[FailureResamplingHint] = Field(default_factory=list)
    selected_candidates: list[ResampleCandidate] = Field(default_factory=list)
    suppressed_candidates: list[ResampleCandidate] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


@dataclass
class ResampleResult:
    outcome: ResampleOutcome
    plan: ResamplePlan
    created_branches: list[BranchState] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def model_dump(self, mode: str = "python", **_: Any) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value if mode == "json" else self.outcome,
            "plan": self.plan.model_dump(mode=mode),
            "created_branches": list(self.created_branches),
            "diagnostics": dict(self.diagnostics),
        }

    def dict(self) -> dict[str, Any]:
        return self.model_dump()


def _stable_json(payload: Mapping[str, Any] | list[Any] | tuple[Any, ...]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _stable_hash(prefix: str, payload: Mapping[str, Any] | list[Any] | tuple[Any, ...]) -> str:
    raw = f"{prefix}::{_stable_json(payload)}".encode("utf-8")
    return f"{prefix}_{sha1(raw).hexdigest()[:16]}"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_text(text: str | None) -> str:
    return " ".join((text or "").strip().split())


class BranchResampler:
    """
    Deterministic resampling planner for bounded self-consistency replenishment.

    The resampler plans which new branches should be forked, why they should
    exist, and what operator/profile hints they should carry. It does not
    generate new reasoning text itself.
    """

    def plan_resampling(
        self,
        branches: Sequence[BranchState],
        *,
        route: RouteDecision,
        failure_diagnoses: Sequence[BranchFailureDiagnosis | FailureType | None] | None = None,
        budget: ResampleBudget | None = None,
    ) -> ResamplePlan:
        active_branches = list(branches)
        active_count = len(active_branches)
        active_budget = self._derive_budget(route, active_count, budget)
        remaining_budget = max(
            0,
            min(
                active_budget.max_new_branches,
                active_budget.max_total_branches - active_count,
                max(0, int(getattr(route, "branch_budget", 0)) - active_count),
            ),
        )
        failure_hints = self._build_failure_hints(failure_diagnoses or ())
        target_count = self._target_sample_count(route, active_count, active_budget)

        if active_count <= 0:
            return ResamplePlan(
                problem_id=str(getattr(route, "problem_id", "")),
                active_branch_count=0,
                branch_budget=int(getattr(route, "branch_budget", 0)),
                remaining_budget=0,
                target_sample_count=target_count,
                budget=active_budget,
                failure_hints=failure_hints,
                diagnostics={"reason": "no_source_branches"},
            )

        if remaining_budget <= 0:
            return ResamplePlan(
                problem_id=str(getattr(route, "problem_id", "")),
                active_branch_count=active_count,
                branch_budget=int(getattr(route, "branch_budget", 0)),
                remaining_budget=0,
                target_sample_count=target_count,
                budget=active_budget,
                failure_hints=failure_hints,
                diagnostics={"reason": "no_remaining_budget"},
            )

        operator_usage = self._operator_usage(active_branches)
        desired_focus = self._desired_focus_operators(
            route=route,
            operator_usage=operator_usage,
            failure_hints=failure_hints,
            target_count=target_count,
        )

        selected: list[ResampleCandidate] = []
        suppressed: list[ResampleCandidate] = []
        seen_signatures: dict[str, int] = {}

        for slot, operator_name in enumerate(desired_focus):
            if len(selected) >= remaining_budget:
                break

            profile = self._profile_for_operator(
                route=route,
                operator_name=operator_name,
                failure_hints=failure_hints,
                seed_offset=slot,
                base_seed=active_budget.deterministic_seed,
            )
            parent = self._select_parent_branch(active_branches, operator_name)
            novelty = self._novelty_score(parent, operator_name, active_branches)
            duplicate_score = 1.0 - novelty
            route_alignment = _clamp01(float((getattr(route, "operator_prior", {}) or {}).get(operator_name, 0.0)))
            signature = self._candidate_signature(parent, operator_name, profile.profile_kind)

            suppression_reason = None
            suppressed_flag = False
            if novelty < active_budget.novelty_floor:
                suppressed_flag = True
                suppression_reason = "novelty_below_floor"
            elif (
                active_budget.suppress_answer_duplicates
                and self._answer_duplicate_pressure(parent, active_branches) >= 0.95
            ):
                suppressed_flag = True
                suppression_reason = "duplicate_answer_cluster"
            elif seen_signatures.get(signature, 0) > active_budget.max_duplicates_per_signature:
                suppressed_flag = True
                suppression_reason = "duplicate_signature"

            candidate = ResampleCandidate(
                candidate_id=_stable_hash(
                    "resample_candidate",
                    {
                        "problem_id": getattr(route, "problem_id", ""),
                        "parent": parent.branch_id,
                        "operator": operator_name,
                        "profile": profile.profile_kind.value,
                        "slot": slot,
                    },
                ),
                parent_branch_id=parent.branch_id,
                branch_label=f"resample::{operator_name}::{slot}",
                profile=profile,
                novelty_score=round(novelty, 6),
                duplicate_score=round(duplicate_score, 6),
                route_alignment=round(route_alignment, 6),
                budget_cost=1,
                suppressed=suppressed_flag,
                suppression_reason=suppression_reason,
                diagnostics={
                    "signature": signature,
                    "parent_score": float(parent.composite_score()),
                    "parent_phase": parent.phase.value,
                    "usage_count": operator_usage.get(operator_name, 0),
                },
            )

            if suppressed_flag:
                suppressed.append(candidate)
                continue

            selected.append(candidate)
            seen_signatures[signature] = seen_signatures.get(signature, 0) + 1

        return ResamplePlan(
            problem_id=str(getattr(route, "problem_id", "")),
            active_branch_count=active_count,
            branch_budget=int(getattr(route, "branch_budget", 0)),
            remaining_budget=remaining_budget,
            target_sample_count=target_count,
            budget=active_budget,
            failure_hints=failure_hints,
            selected_candidates=selected,
            suppressed_candidates=suppressed,
            diagnostics={
                "desired_focus": desired_focus,
                "active_operator_usage": operator_usage,
                "self_consistency_samples": int(
                    getattr(getattr(route, "budget_plan", None), "self_consistency_samples", 0)
                ),
            },
        )

    def materialize_plan(
        self,
        plan: ResamplePlan,
        *,
        branches: Sequence[BranchState],
    ) -> ResampleResult:
        if plan.remaining_budget <= 0:
            return ResampleResult(
                outcome=ResampleOutcome.NO_BUDGET,
                plan=plan,
                diagnostics={"reason": "no_remaining_budget"},
            )
        if not plan.selected_candidates:
            outcome = (
                ResampleOutcome.NO_NOVELTY
                if plan.suppressed_candidates
                else ResampleOutcome.EMPTY
            )
            return ResampleResult(
                outcome=outcome,
                plan=plan,
                diagnostics={"reason": "no_selected_candidates"},
            )

        parent_by_id = {branch.branch_id: branch for branch in branches}
        created: list[BranchState] = []
        for candidate in plan.selected_candidates:
            parent = parent_by_id.get(candidate.parent_branch_id)
            if parent is None:
                continue
            child = self._materialize_candidate(parent, candidate)
            created.append(child)

        return ResampleResult(
            outcome=ResampleOutcome.MATERIALIZED if created else ResampleOutcome.EMPTY,
            plan=plan,
            created_branches=created,
            diagnostics={"created_count": len(created)},
        )

    def resample(
        self,
        branches: Sequence[BranchState],
        *,
        route: RouteDecision,
        failure_diagnoses: Sequence[BranchFailureDiagnosis | FailureType | None] | None = None,
        budget: ResampleBudget | None = None,
    ) -> ResampleResult:
        plan = self.plan_resampling(
            branches,
            route=route,
            failure_diagnoses=failure_diagnoses,
            budget=budget,
        )
        return self.materialize_plan(plan, branches=branches)

    def _derive_budget(
        self,
        route: RouteDecision,
        active_count: int,
        budget: ResampleBudget | None,
    ) -> ResampleBudget:
        if budget is not None:
            return budget

        route_budget = max(1, int(getattr(route, "branch_budget", 1)))
        self_consistency = int(
            getattr(getattr(route, "budget_plan", None), "self_consistency_samples", 0)
        )
        route_uncertainty = float(getattr(route, "route_uncertainty", 0.0))

        max_new = min(6, max(1, route_budget - active_count))
        if self_consistency >= 16:
            max_new = min(max_new + 1, 8)
        novelty_floor = 0.15 if route_uncertainty >= 0.45 else 0.20
        max_total = max(route_budget, active_count + max_new)

        return ResampleBudget(
            max_new_branches=max_new,
            max_total_branches=max_total,
            max_attempts_per_focus=1,
            max_duplicates_per_signature=0,
            novelty_floor=novelty_floor,
            deterministic_seed=max(0, int(route_budget + self_consistency)),
            prefer_self_consistency_fill=True,
            suppress_answer_duplicates=True,
        )

    def _target_sample_count(
        self,
        route: RouteDecision,
        active_count: int,
        budget: ResampleBudget,
    ) -> int:
        desired = int(getattr(getattr(route, "budget_plan", None), "self_consistency_samples", active_count))
        desired = max(desired, active_count)
        desired = min(desired, budget.max_total_branches, int(getattr(route, "branch_budget", desired)))
        return desired

    def _build_failure_hints(
        self,
        failure_diagnoses: Sequence[BranchFailureDiagnosis | FailureType | None],
    ) -> list[FailureResamplingHint]:
        hints: list[FailureResamplingHint] = []

        for item in failure_diagnoses:
            if item is None:
                continue

            if isinstance(item, FailureType):
                category = item.value
                preferred = self._preferred_profile_for_failure_type(item)
                operators = self._suggested_operators_for_failure_type(item)
                hints.append(
                    FailureResamplingHint(
                        category=category,
                        failure_type=item.value,
                        recoverability=0.5,
                        reason=f"failure_type::{item.value}",
                        suggested_operators=operators,
                        preferred_profile=preferred,
                        rollback_first=item in {
                            FailureType.SYMBOLIC_MISMATCH,
                            FailureType.INVALID_INVARIANT,
                            FailureType.BAD_CASE_SPLIT,
                        },
                    )
                )
                continue

            raw_category = getattr(item, "category", None)
            if isinstance(raw_category, FailureCategory):
                category = raw_category.value
            elif raw_category is None:
                category = "failure_recovery"
            else:
                category = _normalize_text(str(raw_category)) or "failure_recovery"

            failure_type_value = getattr(item, "failure_type", None)
            schema_failure_type_value = getattr(item, "schema_failure_type", None)
            selected_failure_type = failure_type_value if failure_type_value is not None else schema_failure_type_value
            if isinstance(selected_failure_type, FailureType):
                failure_type_str = selected_failure_type.value
            elif selected_failure_type is None:
                failure_type_str = None
            else:
                failure_type_str = _normalize_text(str(selected_failure_type)) or None

            preferred = self._preferred_profile_for_failure_category(category)
            suggested = getattr(item, "suggested_operators", None)
            if not suggested:
                suggested = getattr(item, "suggested_operator_bias", None)
            operators = [
                str(op)
                for op in (suggested or ())
                if str(op).strip()
            ]

            rollback_first = bool(getattr(item, "rollback_first", False))
            if hasattr(item, "can_repair_in_place"):
                rollback_first = not bool(getattr(item, "can_repair_in_place"))

            hints.append(
                FailureResamplingHint(
                    category=category,
                    failure_type=failure_type_str,
                    recoverability=_clamp01(float(getattr(item, "recoverability", 0.5))),
                    reason=(
                        _normalize_text(str(getattr(item, "reason", "")))
                        or _normalize_text(str(getattr(item, "summary", "")))
                        or f"category::{category}"
                    ),
                    suggested_operators=operators,
                    preferred_profile=preferred,
                    rollback_first=rollback_first,
                )
            )

        return hints

    def _preferred_profile_for_failure_type(self, failure_type: FailureType) -> ResampleProfileKind:
        if failure_type in {
            FailureType.SYMBOLIC_MISMATCH,
            FailureType.INVALID_INVARIANT,
            FailureType.BAD_CASE_SPLIT,
        }:
            return ResampleProfileKind.CONSERVATIVE
        if failure_type in {
            FailureType.RETRIEVAL_MISLEAD,
            FailureType.INCOMPLETE_PROOF,
        }:
            return ResampleProfileKind.DIVERSE
        return ResampleProfileKind.BALANCED

    def _preferred_profile_for_failure_category(self, category: str) -> ResampleProfileKind:
        normalized = _normalize_text(category).lower()
        if normalized in {"symbolic_mismatch", "invalid_invariant", "bad_case_split"}:
            return ResampleProfileKind.CONSERVATIVE
        if normalized in {"diversity_recovery", "retrieval_mislead", "coverage_gap", "missing_case", "verifier_rejection"}:
            return ResampleProfileKind.DIVERSE
        return ResampleProfileKind.BALANCED

    def _suggested_operators_for_failure_type(self, failure_type: FailureType) -> list[str]:
        mapping: dict[FailureType, list[str]] = {
            FailureType.SYMBOLIC_MISMATCH: ["substitute", "factor", "modular_reduce"],
            FailureType.ARITHMETIC_ERROR: ["brute_force_small", "substitute"],
            FailureType.LOGIC_ERROR: ["case_split", "contradiction"],
            FailureType.INVALID_INVARIANT: ["invariant_extract", "symmetry_quotient"],
            FailureType.BAD_CASE_SPLIT: ["case_split", "bound"],
            FailureType.INCOMPLETE_PROOF: ["bound", "extremal", "contradiction"],
            FailureType.RETRIEVAL_MISLEAD: ["symmetry_quotient", "invariant_extract"],
            FailureType.TOOL_EXECUTION_FAILURE: ["substitute"],
            FailureType.UNKNOWN: [],
        }
        return list(mapping.get(failure_type, []))

    def _operator_usage(self, branches: Sequence[BranchState]) -> dict[str, int]:
        usage: dict[str, int] = {}
        for branch in branches:
            for step in branch.active_steps():
                operator_name = _normalize_text(step.operator_name)
                if not operator_name:
                    continue
                usage[operator_name] = usage.get(operator_name, 0) + 1
        return usage

    def _desired_focus_operators(
        self,
        *,
        route: RouteDecision,
        operator_usage: Mapping[str, int],
        failure_hints: Sequence[FailureResamplingHint],
        target_count: int,
    ) -> list[str]:
        desired: list[str] = []

        for hint in failure_hints:
            for op in hint.suggested_operators:
                normalized = _normalize_text(op)
                if normalized and normalized not in desired:
                    desired.append(normalized)

        operator_prior = getattr(route, "operator_prior", {}) or {}
        ordered_priors = [
            name
            for name, _ in sorted(
                operator_prior.items(),
                key=lambda item: (-float(item[1]), item[0]),
            )
            if _normalize_text(name)
        ]
        for name in ordered_priors:
            normalized = _normalize_text(name)
            if normalized not in desired:
                desired.append(normalized)

        if not desired:
            desired = ["substitute", "factor", "modular_reduce", "invariant_extract"]

        original_order = {op: i for i, op in enumerate(desired)}
        desired.sort(key=lambda op: (operator_usage.get(op, 0), original_order.get(op, 10**9)))
        limit = max(1, min(target_count, len(desired)))
        return desired[:limit]

    def _profile_for_operator(
        self,
        *,
        route: RouteDecision,
        operator_name: str,
        failure_hints: Sequence[FailureResamplingHint],
        seed_offset: int,
        base_seed: int,
    ) -> ResampleProfile:
        preferred_kind = ResampleProfileKind.BALANCED
        rationale: list[str] = []
        if failure_hints:
            preferred_kind = failure_hints[0].preferred_profile
            rationale.extend(h.reason for h in failure_hints[:2])
        else:
            route_uncertainty = float(getattr(route, "route_uncertainty", 0.0))
            if route_uncertainty >= 0.55:
                preferred_kind = ResampleProfileKind.DIVERSE
                rationale.append("route uncertainty favors diversity recovery")
            else:
                preferred_kind = ResampleProfileKind.BALANCED
                rationale.append("self-consistency replenishment under bounded uncertainty")

        seed = int(base_seed + seed_offset)
        temperature, top_p = self._temperature_controls(preferred_kind)
        return ResampleProfile(
            profile_id=_stable_hash(
                "resample_profile",
                {
                    "problem_id": getattr(route, "problem_id", ""),
                    "operator": operator_name,
                    "kind": preferred_kind.value,
                    "seed": seed,
                },
            ),
            profile_kind=preferred_kind,
            reason=self._reason_for_profile(failure_hints),
            focus_operators=[operator_name],
            avoided_operators=self._avoided_operators(route, operator_name),
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            rationale=rationale,
            metadata={
                "difficulty": getattr(getattr(route, "difficulty", None), "value", None),
                "route_uncertainty": float(getattr(route, "route_uncertainty", 0.0)),
                "branch_budget": int(getattr(route, "branch_budget", 0)),
            },
        )

    def _temperature_controls(self, profile_kind: ResampleProfileKind) -> tuple[float, float]:
        if profile_kind is ResampleProfileKind.CONSERVATIVE:
            return 0.35, 0.82
        if profile_kind is ResampleProfileKind.DIVERSE:
            return 0.80, 0.96
        return 0.55, 0.90

    def _reason_for_profile(
        self,
        failure_hints: Sequence[FailureResamplingHint],
    ) -> ResampleReason:
        if failure_hints:
            return ResampleReason.FAILURE_RECOVERY
        return ResampleReason.SELF_CONSISTENCY_GAP

    def _avoided_operators(self, route: RouteDecision, focus_operator: str) -> list[str]:
        ordered = [
            name
            for name, _ in sorted(
                (getattr(route, "operator_prior", {}) or {}).items(),
                key=lambda item: (float(item[1]), item[0]),
            )
        ]
        return [name for name in ordered[:2] if name != focus_operator]

    def _select_parent_branch(
        self,
        branches: Sequence[BranchState],
        operator_name: str,
    ) -> BranchState:
        ranked = sorted(
            branches,
            key=lambda branch: (
                self._parent_penalty(branch, operator_name),
                float(branch.composite_score()),
                branch.branch_id,
            ),
        )
        return ranked[0]

    def _parent_penalty(self, branch: BranchState, operator_name: str) -> tuple[float, float, str]:
        used = any(
            _normalize_text(step.operator_name) == operator_name
            for step in branch.active_steps()
        )
        novelty_penalty = 1.0 if used else 0.0
        return (novelty_penalty, float(branch.composite_score()), branch.branch_id)

    def _branch_operator_similarity(
        self,
        branch: BranchState,
        operator_name: str,
    ) -> float:
        normalized = _normalize_text(operator_name)
        if not normalized:
            return 0.0

        steps = list(branch.active_steps())
        if not steps:
            return 0.0

        used = sum(1 for step in steps if _normalize_text(step.operator_name) == normalized)
        if used > 0:
            return 1.0

        last_operator = _normalize_text(getattr(steps[-1], "operator_name", None))
        if last_operator and last_operator[:4] == normalized[:4]:
            return 0.5
        return 0.0

    def _novelty_score(
        self,
        parent: BranchState,
        operator_name: str,
        branches: Sequence[BranchState],
    ) -> float:
        similarities = [self._branch_operator_similarity(branch, operator_name) for branch in branches]
        max_similarity = max(similarities) if similarities else 0.0
        candidate_bonus = 0.0
        current_candidate = parent.current_candidate()
        if current_candidate is None:
            candidate_bonus = 0.15
        elif getattr(current_candidate, "canonical_answer", None):
            answer_reuse = sum(
                1
                for branch in branches
                if branch.current_candidate() is not None
                and branch.current_candidate().canonical_answer == current_candidate.canonical_answer
            )
            if answer_reuse <= 1:
                candidate_bonus = 0.10
        return _clamp01(1.0 - 0.75 * max_similarity + candidate_bonus)

    def _answer_duplicate_pressure(
        self,
        parent: BranchState,
        branches: Sequence[BranchState],
    ) -> float:
        candidate = parent.current_candidate()
        if candidate is None or not getattr(candidate, "canonical_answer", None):
            return 0.0
        same = sum(
            1
            for branch in branches
            if branch.current_candidate() is not None
            and branch.current_candidate().canonical_answer == candidate.canonical_answer
        )
        return _clamp01(same / max(len(branches), 1))

    def _candidate_signature(
        self,
        parent: BranchState,
        operator_name: str,
        profile_kind: ResampleProfileKind,
    ) -> str:
        candidate = parent.current_candidate()
        payload = {
            "parent_branch_id": parent.branch_id,
            "operator_name": operator_name,
            "profile_kind": profile_kind.value,
            "latest_state_fingerprint": getattr(parent, "latest_state_fingerprint", None),
            "canonical_answer": candidate.canonical_answer if candidate else None,
        }
        return _stable_hash("resample_signature", payload)

    def _materialize_candidate(
        self,
        parent: BranchState,
        candidate: ResampleCandidate,
    ) -> BranchState:
        child = parent.fork(branch_label=candidate.branch_label)

        child.metadata = dict(child.metadata)
        existing = list(child.metadata.get("resample_history", ()))
        existing.append(
            {
                "candidate_id": candidate.candidate_id,
                "profile_id": candidate.profile.profile_id,
                "seed": candidate.profile.seed,
                "focus_operators": list(candidate.profile.focus_operators),
                "reason": candidate.profile.reason.value,
            }
        )
        child.metadata["resample_history"] = existing
        child.metadata["resample_profile"] = candidate.profile.model_dump(mode="json")
        child.metadata["resample_candidate_id"] = candidate.candidate_id
        child.metadata["resample_duplicate_score"] = candidate.duplicate_score
        child.metadata["resample_novelty_score"] = candidate.novelty_score

        child.set_phase(
            BranchPhase.REASONING,
            reason=f"resampled via {candidate.profile.profile_kind.value} profile",
        )
        child.add_step(
            kind=BranchStepKind.OPERATOR,
            phase=BranchPhase.REASONING,
            description=(
                "resample prepared with focus operators "
                + ", ".join(candidate.profile.focus_operators)
            ),
            operator_name=(
                candidate.profile.focus_operators[0]
                if candidate.profile.focus_operators
                else None
            ),
            summary_text="deterministic resampling directive",
        )
        child.update_score_breakdown(branch_novelty=candidate.novelty_score)
        return child


def plan_branch_resampling(
    branches: Sequence[BranchState],
    *,
    route: RouteDecision,
    failure_diagnoses: Sequence[BranchFailureDiagnosis | FailureType | None] | None = None,
    budget: ResampleBudget | None = None,
) -> ResamplePlan:
    return BranchResampler().plan_resampling(
        branches,
        route=route,
        failure_diagnoses=failure_diagnoses,
        budget=budget,
    )


def resample_branches(
    branches: Sequence[BranchState],
    *,
    route: RouteDecision,
    failure_diagnoses: Sequence[BranchFailureDiagnosis | FailureType | None] | None = None,
    budget: ResampleBudget | None = None,
) -> ResampleResult:
    return BranchResampler().resample(
        branches,
        route=route,
        failure_diagnoses=failure_diagnoses,
        budget=budget,
    )


__all__ = [
    "ResampleProfileKind",
    "ResampleReason",
    "ResampleOutcome",
    "ResampleBudget",
    "FailureResamplingHint",
    "ResampleProfile",
    "ResampleCandidate",
    "ResamplePlan",
    "ResampleResult",
    "BranchResampler",
    "plan_branch_resampling",
    "resample_branches",
]