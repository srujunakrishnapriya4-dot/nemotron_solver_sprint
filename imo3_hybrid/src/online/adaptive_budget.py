from __future__ import annotations

"""
Runtime budget planning and adaptation for the online solve loop.

Focused repair pass:
- sharper difficulty-conditioned compute economy
- uncertainty / obligation / PRM / verifier / retrieval aware widening
- bounded, deterministic adaptation
- backward-compatible RuntimeBudgetBundle handoff to inference_engine.py
"""

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping

from src.common.constants import (
    DEFAULT_BRANCH_BUDGETS,
    DEFAULT_MAX_SEARCH_DEPTHS,
    DEFAULT_MAX_SEARCH_NODES,
    DEFAULT_RETRIEVAL_DEPTHS,
    ROUTING_HIGH_ENTROPY_THRESHOLD,
    SELF_CRITIQUE_TOP_K,
    TIME_LIMIT_PER_PROBLEM_SEC,
)
from src.common.schemas import BudgetPlan, Difficulty, RouteDecision


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _route_compute_signals(route: RouteDecision) -> dict[str, float]:
    raw = getattr(route, "compute_signals", {}) or {}
    if not isinstance(raw, Mapping):
        raw = {}
    out: dict[str, float] = {}
    for key in (
        "difficulty_intensity",
        "route_uncertainty",
        "proof_burden",
        "retrieval_need",
        "repair_need",
        "critique_aggressiveness",
        "repair_aggressiveness",
        "resample_aggressiveness",
    ):
        out[key] = _clamp01(raw.get(key, 0.0))
    if out["route_uncertainty"] <= 0.0:
        out["route_uncertainty"] = _clamp01(getattr(route, "route_uncertainty", 0.0))
    if out["difficulty_intensity"] <= 0.0:
        out["difficulty_intensity"] = _clamp01(getattr(route, "difficulty_score", 0.0))
    plan = getattr(route, "budget_plan", None)
    if plan is not None:
        out["critique_aggressiveness"] = max(
            out["critique_aggressiveness"],
            _clamp01(getattr(plan, "critique_aggressiveness", 0.0)),
        )
        out["repair_aggressiveness"] = max(
            out["repair_aggressiveness"],
            _clamp01(getattr(plan, "repair_aggressiveness", 0.0)),
        )
        out["resample_aggressiveness"] = max(
            out["resample_aggressiveness"],
            _clamp01(getattr(plan, "resample_aggressiveness", 0.0)),
        )
    return out


class RuntimeMode(str, Enum):
    FULL = "full"
    DEGRADED = "degraded"
    PANIC_SAVE = "panic_save"


@dataclass(frozen=True)
class GlobalBudgetTracker:
    total_budget_s: float
    reserve_s: float = 300.0
    spent_s: float = 0.0

    def remaining_s(self) -> float:
        return max(0.0, float(self.total_budget_s) - float(self.reserve_s) - float(self.spent_s))

    def per_problem_cap(self, remaining_problems: int) -> float:
        if remaining_problems <= 0:
            return 0.0
        return max(0.0, self.remaining_s() / float(remaining_problems))

    def with_spent(self, additional_spent_s: float) -> "GlobalBudgetTracker":
        return replace(self, spent_s=max(0.0, self.spent_s + max(0.0, float(additional_spent_s))))


@dataclass(frozen=True)
class RuntimeCeilings:
    wall_clock_s: float
    token_budget: int
    branch_ceiling: int
    search_node_ceiling: int
    search_depth_ceiling: int
    retrieval_depth_ceiling: int
    critique_ceiling: int
    repair_ceiling: int
    resample_ceiling: int


@dataclass(frozen=True)
class EarlyExitPolicy:
    min_branches_before_exit: int
    min_solved_branches: int
    confidence_threshold: float
    verifier_threshold: float
    symbolic_threshold: float
    entropy_threshold: float
    cluster_ratio_threshold: float


@dataclass(frozen=True)
class WideningPolicy:
    enabled: bool
    max_widen_steps: int
    branch_growth_factor: float
    search_node_growth_factor: float
    retrieval_growth_step: int
    critique_growth_step: int
    repair_growth_step: int
    uncertainty_trigger: float
    entropy_trigger: float
    disagreement_trigger: float
    obligation_trigger: float
    prm_trigger: float
    retrieval_trigger: float


@dataclass(frozen=True)
class RuntimeBudgetBundle:
    problem_id: str
    mode: RuntimeMode
    difficulty: Difficulty
    route_uncertainty: float
    initial_plan: BudgetPlan
    effective_plan: BudgetPlan
    ceilings: RuntimeCeilings
    early_exit: EarlyExitPolicy
    widening: WideningPolicy
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def branch_controller_overrides(self) -> dict[str, int]:
        return {
            "self_consistency_samples": int(self.effective_plan.self_consistency_samples),
            "frontier_width": int(min(self.effective_plan.branch_budget, self.effective_plan.max_search_nodes)),
            "max_search_depth": int(self.effective_plan.max_search_depth),
            "max_search_nodes": int(self.effective_plan.max_search_nodes),
            "resample_budget": max(
                1,
                min(
                    int(self.ceilings.resample_ceiling),
                    int(self.effective_plan.branch_budget) // 16 or 1,
                ),
            ),
            "repair_budget": int(self.effective_plan.repair_budget),
            "critique_top_k": int(self.effective_plan.critique_top_k),
        }

    def retrieval_kwargs(self) -> dict[str, int | bool]:
        return {
            "retrieval_depth": int(self.effective_plan.retrieval_depth),
            "use_retrieval": bool(self.effective_plan.use_retrieval),
        }


@dataclass(frozen=True)
class RuntimeObservation:
    elapsed_s: float = 0.0
    estimated_tokens_used: int = 0
    branches_generated: int = 0
    solved_branches: int = 0
    active_frontier: int = 0
    top_answer_confidence: float = 0.0
    verifier_agreement: float = 0.0
    symbolic_pass_rate: float = 0.0
    answer_entropy: float = 1.0
    best_cluster_ratio: float = 0.0
    repair_attempts: int = 0
    critique_count: int = 0
    retrieval_hits: int = 0
    search_nodes_expanded: int = 0
    max_depth_reached: int = 0
    final_candidate_available: bool = False
    force_stop: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def open_obligation_burden(self) -> float:
        return _clamp01(self.metadata.get("open_obligation_burden", 0.0))

    @property
    def prm_prefix_quality(self) -> float:
        return _clamp01(self.metadata.get("prm_prefix_quality", 0.0))

    @property
    def verifier_repairability(self) -> float:
        return _clamp01(self.metadata.get("verifier_repairability", 0.0))

    @property
    def logical_consistency(self) -> float:
        return _clamp01(self.metadata.get("logical_consistency", 0.0))

    @property
    def completeness(self) -> float:
        return _clamp01(self.metadata.get("completeness", 0.0))

    @property
    def retrieval_compatibility(self) -> float:
        return _clamp01(self.metadata.get("retrieval_compatibility", 0.0))

    @property
    def retrieval_support(self) -> float:
        return _clamp01(self.metadata.get("retrieval_support", 0.0))

    @property
    def operator_reliability(self) -> float:
        return _clamp01(self.metadata.get("operator_reliability", 0.0))

    @property
    def branch_disagreement(self) -> float:
        explicit = self.metadata.get("branch_disagreement")
        if explicit is not None:
            return _clamp01(explicit)
        return _clamp01(self.answer_entropy)

    @property
    def answer_entropy_proxy(self) -> float:
        return _clamp01(self.answer_entropy)

    @property
    def widen_step_count(self) -> int:
        return _safe_int(self.metadata.get("widen_step_count", 0), 0)


@dataclass(frozen=True)
class AdaptationDecision:
    updated_bundle: RuntimeBudgetBundle
    stop: bool
    stop_reason: str | None
    widen: bool
    degrade: bool
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class AdaptiveBudgetPlanner:
    _MODE_TIME_PRESSURE = {
        RuntimeMode.FULL: 1.00,
        RuntimeMode.DEGRADED: 0.78,
        RuntimeMode.PANIC_SAVE: 0.50,
    }

    def build_initial_budget(
        self,
        route: RouteDecision,
        *,
        remaining_problems: int | None = None,
        tracker: GlobalBudgetTracker | None = None,
        requested_mode: str | RuntimeMode | None = None,
        per_problem_time_limit_s: float | None = None,
        per_problem_token_limit: int | None = None,
    ) -> RuntimeBudgetBundle:
        base_plan = self._normalize_plan(route.budget_plan, route)
        mode = self._resolve_mode(
            route=route,
            tracker=tracker,
            remaining_problems=remaining_problems,
            requested_mode=requested_mode,
        )
        time_cap_s = self._wall_clock_cap(
            route=route,
            mode=mode,
            tracker=tracker,
            remaining_problems=remaining_problems,
            explicit_cap_s=per_problem_time_limit_s,
        )
        token_budget = int(per_problem_token_limit or self._estimate_token_budget(base_plan, route, mode))
        ceilings = self._compute_ceilings(
            base_plan=base_plan,
            route=route,
            mode=mode,
            wall_clock_s=time_cap_s,
            token_budget=token_budget,
        )
        effective_plan = self._apply_mode_and_ceiling_constraints(base_plan, ceilings, mode)
        early_exit = self._early_exit_policy(route=route, plan=effective_plan)
        widening = self._widening_policy(route=route, mode=mode)

        return RuntimeBudgetBundle(
            problem_id=route.problem_id,
            mode=mode,
            difficulty=route.difficulty,
            route_uncertainty=float(getattr(route, "route_uncertainty", 0.0)),
            initial_plan=base_plan,
            effective_plan=effective_plan,
            ceilings=ceilings,
            early_exit=early_exit,
            widening=widening,
            diagnostics={
                "difficulty": route.difficulty.value,
                "difficulty_score": float(getattr(route, "difficulty_score", 0.0)),
                "route_uncertainty": float(getattr(route, "route_uncertainty", 0.0)),
                "mode": mode.value,
                "time_cap_s": round(time_cap_s, 4),
                "token_budget": int(token_budget),
                "remaining_problems": remaining_problems,
                "tracker_remaining_s": None if tracker is None else round(tracker.remaining_s(), 4),
                "route_compute_signals": _route_compute_signals(route),
            },
        )

    def adapt(self, bundle: RuntimeBudgetBundle, observation: RuntimeObservation) -> AdaptationDecision:
        if observation.force_stop:
            return AdaptationDecision(
                updated_bundle=bundle,
                stop=True,
                stop_reason="forced_stop",
                widen=False,
                degrade=False,
                diagnostics={"trigger": "force_stop"},
            )

        if self.should_stop_early(bundle, observation):
            return AdaptationDecision(
                updated_bundle=bundle,
                stop=True,
                stop_reason="high_confidence_early_exit",
                widen=False,
                degrade=False,
                diagnostics={"trigger": "early_exit"},
            )

        pressure = self._pressure(bundle, observation)
        near_ceiling = self._near_ceiling(bundle, observation)
        ambiguous = self._is_ambiguous(bundle, observation)

        if pressure >= 0.92 or near_ceiling:
            degraded = self._degrade(bundle, observation)
            must_stop = self._must_stop(degraded, observation)
            return AdaptationDecision(
                updated_bundle=degraded,
                stop=must_stop,
                stop_reason="runtime_ceiling_exhausted" if must_stop else None,
                widen=False,
                degrade=True,
                diagnostics={
                    "trigger": "degrade",
                    "pressure": round(pressure, 4),
                    "near_ceiling": bool(near_ceiling),
                },
            )

        if ambiguous and bundle.widening.enabled:
            widened = self._widen(bundle, observation)
            if widened != bundle:
                return AdaptationDecision(
                    updated_bundle=widened,
                    stop=False,
                    stop_reason=None,
                    widen=True,
                    degrade=False,
                    diagnostics={
                        "trigger": "widen",
                        "pressure": round(pressure, 4),
                        "ambiguity": self._ambiguity_signature(bundle, observation),
                    },
                )

        must_stop = self._must_stop(bundle, observation)
        return AdaptationDecision(
            updated_bundle=bundle,
            stop=must_stop,
            stop_reason="runtime_ceiling_exhausted" if must_stop else None,
            widen=False,
            degrade=False,
            diagnostics={
                "trigger": "steady",
                "pressure": round(pressure, 4),
                "ambiguity": self._ambiguity_signature(bundle, observation),
            },
        )

    def should_stop_early(self, bundle: RuntimeBudgetBundle, observation: RuntimeObservation) -> bool:
        policy = bundle.early_exit
        enough_branches = observation.branches_generated >= policy.min_branches_before_exit
        enough_solutions = observation.solved_branches >= policy.min_solved_branches
        high_conf = observation.top_answer_confidence >= policy.confidence_threshold
        verifier_ok = observation.verifier_agreement >= policy.verifier_threshold
        symbolic_ok = observation.symbolic_pass_rate >= policy.symbolic_threshold
        entropy_ok = observation.answer_entropy <= policy.entropy_threshold
        cluster_ok = observation.best_cluster_ratio >= policy.cluster_ratio_threshold
        obligation_ok = observation.open_obligation_burden <= 0.20
        prm_ok = observation.prm_prefix_quality >= 0.72
        return bool(
            observation.final_candidate_available
            and enough_branches
            and enough_solutions
            and high_conf
            and verifier_ok
            and symbolic_ok
            and entropy_ok
            and cluster_ok
            and obligation_ok
            and prm_ok
        )

    def hard_stop_required(self, bundle: RuntimeBudgetBundle, observation: RuntimeObservation) -> bool:
        return self._must_stop(bundle, observation)

    def update_tracker(self, tracker: GlobalBudgetTracker, observation: RuntimeObservation) -> GlobalBudgetTracker:
        return tracker.with_spent(observation.elapsed_s)

    def _resolve_mode(
        self,
        *,
        route: RouteDecision,
        tracker: GlobalBudgetTracker | None,
        remaining_problems: int | None,
        requested_mode: str | RuntimeMode | None,
    ) -> RuntimeMode:
        if isinstance(requested_mode, RuntimeMode):
            return requested_mode
        if isinstance(requested_mode, str):
            normalized = requested_mode.strip().lower()
            for mode in RuntimeMode:
                if mode.value == normalized:
                    return mode

        if tracker is not None and remaining_problems is not None and remaining_problems > 0:
            cap = tracker.per_problem_cap(remaining_problems)
            if cap < 25.0:
                return RuntimeMode.PANIC_SAVE
            if cap < 45.0:
                return RuntimeMode.DEGRADED

        return RuntimeMode.FULL

    def _normalize_plan(self, plan: BudgetPlan | None, route: RouteDecision) -> BudgetPlan:
        difficulty = route.difficulty
        compute_signals = _route_compute_signals(route)

        base_branch = int(DEFAULT_BRANCH_BUDGETS[difficulty])
        base_retrieval = int(DEFAULT_RETRIEVAL_DEPTHS[difficulty])
        base_depth = int(DEFAULT_MAX_SEARCH_DEPTHS[difficulty])
        base_nodes = int(DEFAULT_MAX_SEARCH_NODES[difficulty])

        difficulty_score = _clamp01(getattr(route, "difficulty_score", 0.0))
        uncertainty = _clamp01(getattr(route, "route_uncertainty", 0.0))
        operator_prior = getattr(route, "operator_prior", {}) or {}
        archetype_probs = getattr(route, "archetype_probs", {}) or {}

        archetype_spread = min(1.0, 0.20 * max(0, len(archetype_probs) - 1))
        operator_spread = min(1.0, 0.08 * max(0, len(operator_prior) - 3))

        intensity = (
            0.38 * difficulty_score
            + 0.27 * uncertainty
            + 0.20 * archetype_spread
            + 0.15 * operator_spread
        )
        intensity = _clamp01(intensity)

        branch_budget = max(base_branch, int(round(base_branch * (1.0 + 0.70 * intensity))))
        retrieval_depth = max(base_retrieval, int(round(base_retrieval + 2.0 * uncertainty + 1.2 * archetype_spread)))
        max_search_depth = max(base_depth, int(round(base_depth + 1.5 * difficulty_score + 1.5 * uncertainty)))
        max_search_nodes = max(base_nodes, int(round(base_nodes * (1.0 + 0.55 * intensity))))

        critique_top_k = max(1, min(SELF_CRITIQUE_TOP_K, max(1, branch_budget // 16)))
        repair_budget = 1 + int(difficulty in {Difficulty.HARD, Difficulty.VERY_HARD}) + int(
            uncertainty >= ROUTING_HIGH_ENTROPY_THRESHOLD
        )
        self_consistency_samples = max(branch_budget, int(round(branch_budget * (1.0 + 0.12 * uncertainty))))

        merged = {
            "branch_budget": branch_budget,
            "retrieval_depth": retrieval_depth,
            "max_search_depth": max_search_depth,
            "max_search_nodes": max_search_nodes,
            "repair_budget": repair_budget,
            "self_consistency_samples": self_consistency_samples,
            "critique_top_k": critique_top_k,
            "widen_on_uncertainty": uncertainty >= 0.28,
            "use_retrieval": retrieval_depth > 0,
            "use_symbolic": True,
            "use_brute_force": difficulty in {Difficulty.EASY, Difficulty.MEDIUM},
            "critique_aggressiveness": _clamp01(
                max(
                    0.5,
                    0.42
                    + 0.30 * compute_signals.get("route_uncertainty", 0.0)
                    + 0.20 * compute_signals.get("proof_burden", 0.0)
                    + 0.10 * compute_signals.get("difficulty_intensity", 0.0),
                    compute_signals.get("critique_aggressiveness", 0.0),
                )
            ),
            "repair_aggressiveness": _clamp01(
                max(
                    0.5,
                    0.44
                    + 0.34 * compute_signals.get("repair_need", 0.0)
                    + 0.16 * compute_signals.get("proof_burden", 0.0)
                    + 0.10 * compute_signals.get("route_uncertainty", 0.0),
                    compute_signals.get("repair_aggressiveness", 0.0),
                )
            ),
            "resample_aggressiveness": _clamp01(
                max(
                    0.5,
                    0.42
                    + 0.28 * compute_signals.get("route_uncertainty", 0.0)
                    + 0.18 * compute_signals.get("retrieval_need", 0.0)
                    + 0.12 * compute_signals.get("repair_need", 0.0),
                    compute_signals.get("resample_aggressiveness", 0.0),
                )
            ),
        }

        if plan is not None:
            merged.update(
                {
                    "branch_budget": max(merged["branch_budget"], int(plan.branch_budget)),
                    "retrieval_depth": max(merged["retrieval_depth"], int(plan.retrieval_depth)),
                    "max_search_depth": max(merged["max_search_depth"], int(plan.max_search_depth)),
                    "max_search_nodes": max(merged["max_search_nodes"], int(plan.max_search_nodes)),
                    "repair_budget": max(merged["repair_budget"], int(plan.repair_budget)),
                    "self_consistency_samples": max(merged["self_consistency_samples"], int(plan.self_consistency_samples)),
                    "critique_top_k": max(merged["critique_top_k"], int(plan.critique_top_k)),
                    "widen_on_uncertainty": bool(
                        getattr(plan, "widen_on_uncertainty", merged["widen_on_uncertainty"])
                    ),
                    "use_retrieval": bool(getattr(plan, "use_retrieval", merged["use_retrieval"])),
                    "use_symbolic": bool(getattr(plan, "use_symbolic", merged["use_symbolic"])),
                    "use_brute_force": bool(getattr(plan, "use_brute_force", merged["use_brute_force"])),
                    "critique_aggressiveness": max(
                        merged["critique_aggressiveness"],
                        _clamp01(getattr(plan, "critique_aggressiveness", 0.0)),
                    ),
                    "repair_aggressiveness": max(
                        merged["repair_aggressiveness"],
                        _clamp01(getattr(plan, "repair_aggressiveness", 0.0)),
                    ),
                    "resample_aggressiveness": max(
                        merged["resample_aggressiveness"],
                        _clamp01(getattr(plan, "resample_aggressiveness", 0.0)),
                    ),
                }
            )

        return BudgetPlan(**merged)

    def _estimate_token_budget(self, plan: BudgetPlan, route: RouteDecision, mode: RuntimeMode) -> int:
        retrieval_tokens = 60 * max(0, plan.retrieval_depth)
        critique_tokens = 120 * max(1, plan.critique_top_k)
        repair_tokens = 150 * max(1, plan.repair_budget)
        search_tokens = 30 * max(1, plan.max_search_nodes)
        self_consistency_tokens = 75 * max(1, plan.self_consistency_samples)
        fixed_overhead = 320
        raw = fixed_overhead + retrieval_tokens + critique_tokens + repair_tokens + search_tokens + self_consistency_tokens
        if route.difficulty in {Difficulty.HARD, Difficulty.VERY_HARD}:
            raw = int(round(raw * 1.10))
        raw = int(round(raw * self._MODE_TIME_PRESSURE[mode]))
        return max(768, raw)

    def _wall_clock_cap(
        self,
        *,
        route: RouteDecision,
        mode: RuntimeMode,
        tracker: GlobalBudgetTracker | None,
        remaining_problems: int | None,
        explicit_cap_s: float | None,
    ) -> float:
        base = float(TIME_LIMIT_PER_PROBLEM_SEC)
        if route.difficulty is Difficulty.EASY:
            base *= 0.70
        elif route.difficulty is Difficulty.MEDIUM:
            base *= 0.88
        elif route.difficulty is Difficulty.HARD:
            base *= 1.03
        else:
            base *= 1.15
        base *= self._MODE_TIME_PRESSURE[mode]
        cap = base
        if explicit_cap_s is not None:
            cap = min(cap, float(explicit_cap_s))
        if tracker is not None and remaining_problems is not None and remaining_problems > 0:
            cap = min(cap, tracker.per_problem_cap(remaining_problems))
        return max(20.0, cap)

    def _compute_ceilings(
        self,
        *,
        base_plan: BudgetPlan,
        route: RouteDecision,
        mode: RuntimeMode,
        wall_clock_s: float,
        token_budget: int,
    ) -> RuntimeCeilings:
        uncertainty = _clamp01(getattr(route, "route_uncertainty", 0.0))
        difficulty_score = _clamp01(getattr(route, "difficulty_score", 0.0))
        branch_ceiling = min(
            self._difficulty_branch_ceiling(route.difficulty),
            self._scaled(base_plan.branch_budget, 1.15 + 0.45 * max(uncertainty, difficulty_score)),
        )
        search_node_ceiling = min(
            self._difficulty_node_ceiling(route.difficulty),
            self._scaled(base_plan.max_search_nodes, 1.15 + 0.40 * max(uncertainty, difficulty_score)),
        )
        search_depth_ceiling = min(
            max(int(base_plan.max_search_depth), 2)
            + int(route.difficulty in {Difficulty.HARD, Difficulty.VERY_HARD})
            + int(uncertainty >= ROUTING_HIGH_ENTROPY_THRESHOLD),
            12,
        )
        retrieval_depth_ceiling = min(max(0, int(base_plan.retrieval_depth) + 2), 6)
        critique_ceiling = min(max(int(base_plan.critique_top_k), 1) + 2, max(SELF_CRITIQUE_TOP_K + 2, 7))
        repair_ceiling = min(max(int(base_plan.repair_budget), 1) + 2, 5)
        resample_ceiling = min(max(1, branch_ceiling // 8), 5)

        if mode is RuntimeMode.DEGRADED:
            branch_ceiling = min(branch_ceiling, max(8, int(base_plan.branch_budget * 0.90)))
            search_node_ceiling = min(search_node_ceiling, max(16, int(base_plan.max_search_nodes * 0.88)))
        elif mode is RuntimeMode.PANIC_SAVE:
            branch_ceiling = min(branch_ceiling, max(6, int(base_plan.branch_budget * 0.65)))
            search_node_ceiling = min(search_node_ceiling, max(12, int(base_plan.max_search_nodes * 0.60)))
            critique_ceiling = min(critique_ceiling, 2)
            retrieval_depth_ceiling = min(retrieval_depth_ceiling, 2)
            repair_ceiling = min(repair_ceiling, 1)
            resample_ceiling = 1

        return RuntimeCeilings(
            wall_clock_s=round(max(20.0, wall_clock_s), 4),
            token_budget=max(512, int(token_budget)),
            branch_ceiling=max(4, int(branch_ceiling)),
            search_node_ceiling=max(8, int(search_node_ceiling)),
            search_depth_ceiling=max(2, int(search_depth_ceiling)),
            retrieval_depth_ceiling=max(0, int(retrieval_depth_ceiling)),
            critique_ceiling=max(1, int(critique_ceiling)),
            repair_ceiling=max(1, int(repair_ceiling)),
            resample_ceiling=max(1, int(resample_ceiling)),
        )

    def _apply_mode_and_ceiling_constraints(
        self,
        plan: BudgetPlan,
        ceilings: RuntimeCeilings,
        mode: RuntimeMode,
    ) -> BudgetPlan:
        branch_budget = min(int(plan.branch_budget), ceilings.branch_ceiling)
        retrieval_depth = min(int(plan.retrieval_depth), ceilings.retrieval_depth_ceiling)
        max_search_depth = min(int(plan.max_search_depth), ceilings.search_depth_ceiling)
        max_search_nodes = min(int(plan.max_search_nodes), ceilings.search_node_ceiling)
        repair_budget = min(int(plan.repair_budget), ceilings.repair_ceiling)
        self_consistency_samples = min(int(plan.self_consistency_samples), ceilings.branch_ceiling)
        critique_top_k = min(int(plan.critique_top_k), ceilings.critique_ceiling)

        if mode is RuntimeMode.DEGRADED:
            branch_budget = max(8, int(round(branch_budget * 0.82)))
            self_consistency_samples = max(8, int(round(self_consistency_samples * 0.82)))
            max_search_nodes = max(16, int(round(max_search_nodes * 0.82)))
            retrieval_depth = max(0, retrieval_depth - 1)
            critique_top_k = max(2, min(critique_top_k, 4))
        elif mode is RuntimeMode.PANIC_SAVE:
            branch_budget = max(6, int(round(branch_budget * 0.60)))
            self_consistency_samples = max(6, int(round(self_consistency_samples * 0.60)))
            max_search_nodes = max(12, int(round(max_search_nodes * 0.55)))
            max_search_depth = max(2, min(max_search_depth, 4))
            retrieval_depth = min(retrieval_depth, 1)
            critique_top_k = 1
            repair_budget = 1

        use_retrieval = bool(plan.use_retrieval and retrieval_depth > 0)
        return BudgetPlan(
            branch_budget=max(4, branch_budget),
            retrieval_depth=max(0, retrieval_depth),
            max_search_depth=max(2, max_search_depth),
            max_search_nodes=max(8, max_search_nodes),
            repair_budget=max(1, repair_budget),
            self_consistency_samples=max(4, self_consistency_samples),
            critique_top_k=max(1, critique_top_k),
            widen_on_uncertainty=bool(plan.widen_on_uncertainty),
            use_retrieval=use_retrieval,
            use_symbolic=bool(plan.use_symbolic),
            use_brute_force=bool(plan.use_brute_force),
            critique_aggressiveness=_clamp01(getattr(plan, "critique_aggressiveness", 0.5)),
            repair_aggressiveness=_clamp01(getattr(plan, "repair_aggressiveness", 0.5)),
            resample_aggressiveness=_clamp01(getattr(plan, "resample_aggressiveness", 0.5)),
        )

    def _early_exit_policy(self, *, route: RouteDecision, plan: BudgetPlan) -> EarlyExitPolicy:
        uncertainty = _clamp01(getattr(route, "route_uncertainty", 0.0))
        harder = route.difficulty in {Difficulty.HARD, Difficulty.VERY_HARD}
        return EarlyExitPolicy(
            min_branches_before_exit=max(4, min(plan.branch_budget, 8 if not harder else 12)),
            min_solved_branches=1 if not harder else 2,
            confidence_threshold=0.95 if not harder else 0.97,
            verifier_threshold=0.84 if not harder else 0.88,
            symbolic_threshold=0.84 if not harder else 0.88,
            entropy_threshold=0.12 if uncertainty < 0.20 else 0.08,
            cluster_ratio_threshold=0.72 if not harder else 0.78,
        )

    def _widening_policy(self, *, route: RouteDecision, mode: RuntimeMode) -> WideningPolicy:
        if mode is RuntimeMode.PANIC_SAVE:
            return WideningPolicy(
                enabled=False,
                max_widen_steps=0,
                branch_growth_factor=1.0,
                search_node_growth_factor=1.0,
                retrieval_growth_step=0,
                critique_growth_step=0,
                repair_growth_step=0,
                uncertainty_trigger=1.0,
                entropy_trigger=1.0,
                disagreement_trigger=1.0,
                obligation_trigger=1.0,
                prm_trigger=0.0,
                retrieval_trigger=0.0,
            )
        return WideningPolicy(
            enabled=bool(getattr(route.budget_plan, "widen_on_uncertainty", True)),
            max_widen_steps=2 if route.difficulty in {Difficulty.EASY, Difficulty.MEDIUM} else 3,
            branch_growth_factor=1.22 if mode is RuntimeMode.FULL else 1.12,
            search_node_growth_factor=1.20 if mode is RuntimeMode.FULL else 1.10,
            retrieval_growth_step=1,
            critique_growth_step=1,
            repair_growth_step=1,
            uncertainty_trigger=max(0.28, float(getattr(route, "route_uncertainty", 0.0))),
            entropy_trigger=0.34 if route.difficulty in {Difficulty.EASY, Difficulty.MEDIUM} else 0.26,
            disagreement_trigger=0.32,
            obligation_trigger=0.35,
            prm_trigger=0.46,
            retrieval_trigger=0.44,
        )

    def _pressure(self, bundle: RuntimeBudgetBundle, observation: RuntimeObservation) -> float:
        ceilings = bundle.ceilings
        time_pressure = _clamp01(observation.elapsed_s / max(1.0, ceilings.wall_clock_s))
        token_pressure = _clamp01(observation.estimated_tokens_used / max(1, ceilings.token_budget))
        branch_pressure = _clamp01(observation.branches_generated / max(1, bundle.effective_plan.branch_budget))
        search_pressure = _clamp01(observation.search_nodes_expanded / max(1, bundle.effective_plan.max_search_nodes))
        return _clamp01(0.34 * time_pressure + 0.24 * token_pressure + 0.22 * branch_pressure + 0.20 * search_pressure)

    def _near_ceiling(self, bundle: RuntimeBudgetBundle, observation: RuntimeObservation) -> bool:
        return bool(
            observation.elapsed_s >= bundle.ceilings.wall_clock_s * 0.96
            or observation.estimated_tokens_used >= bundle.ceilings.token_budget * 0.96
            or observation.search_nodes_expanded >= bundle.ceilings.search_node_ceiling * 0.96
            or observation.branches_generated >= bundle.ceilings.branch_ceiling * 0.96
        )

    def _ambiguity_signature(self, bundle: RuntimeBudgetBundle, observation: RuntimeObservation) -> dict[str, float]:
        return {
            "uncertainty": round(bundle.route_uncertainty, 4),
            "entropy": round(observation.answer_entropy_proxy, 4),
            "disagreement": round(observation.branch_disagreement, 4),
            "open_obligation_burden": round(observation.open_obligation_burden, 4),
            "prm_prefix_quality": round(observation.prm_prefix_quality, 4),
            "retrieval_compatibility": round(observation.retrieval_compatibility, 4),
            "verifier_repairability": round(observation.verifier_repairability, 4),
        }

    def _is_ambiguous(self, bundle: RuntimeBudgetBundle, observation: RuntimeObservation) -> bool:
        wp = bundle.widening
        too_uncertain = bundle.route_uncertainty >= wp.uncertainty_trigger
        high_entropy = observation.answer_entropy_proxy >= wp.entropy_trigger
        disagreement = observation.branch_disagreement >= wp.disagreement_trigger
        open_obligations = observation.open_obligation_burden >= wp.obligation_trigger
        weak_prm = observation.prm_prefix_quality <= wp.prm_trigger
        weak_retrieval = (
            observation.retrieval_hits > 0
            and observation.retrieval_compatibility <= wp.retrieval_trigger
        )
        deceptive_consensus = (
            observation.best_cluster_ratio >= 0.65
            and observation.verifier_agreement <= 0.58
        )
        return bool(
            too_uncertain
            or high_entropy
            or disagreement
            or open_obligations
            or weak_prm
            or weak_retrieval
            or deceptive_consensus
        )

    def _widen(self, bundle: RuntimeBudgetBundle, observation: RuntimeObservation) -> RuntimeBudgetBundle:
        widen_steps = observation.widen_step_count
        if widen_steps >= bundle.widening.max_widen_steps:
            return bundle

        ceilings = bundle.ceilings
        plan = bundle.effective_plan
        branch_budget = min(
            ceilings.branch_ceiling,
            self._scaled(plan.branch_budget, bundle.widening.branch_growth_factor),
        )
        search_nodes = min(
            ceilings.search_node_ceiling,
            self._scaled(plan.max_search_nodes, bundle.widening.search_node_growth_factor),
        )
        retrieval_depth = min(
            ceilings.retrieval_depth_ceiling,
            int(plan.retrieval_depth) + int(bundle.widening.retrieval_growth_step),
        )
        critique_top_k = min(
            ceilings.critique_ceiling,
            int(plan.critique_top_k) + int(bundle.widening.critique_growth_step),
        )
        repair_budget = min(
            ceilings.repair_ceiling,
            int(plan.repair_budget) + int(bundle.widening.repair_growth_step),
        )
        self_consistency_samples = min(ceilings.branch_ceiling, max(int(plan.self_consistency_samples), branch_budget))

        updated_plan = BudgetPlan(
            branch_budget=branch_budget,
            retrieval_depth=retrieval_depth,
            max_search_depth=plan.max_search_depth,
            max_search_nodes=search_nodes,
            repair_budget=repair_budget,
            self_consistency_samples=self_consistency_samples,
            critique_top_k=critique_top_k,
            widen_on_uncertainty=plan.widen_on_uncertainty,
            use_retrieval=bool(plan.use_retrieval or retrieval_depth > 0),
            use_symbolic=plan.use_symbolic,
            use_brute_force=plan.use_brute_force,
            critique_aggressiveness=_clamp01(getattr(plan, "critique_aggressiveness", 0.5)),
            repair_aggressiveness=_clamp01(getattr(plan, "repair_aggressiveness", 0.5)),
            resample_aggressiveness=_clamp01(getattr(plan, "resample_aggressiveness", 0.5)),
        )
        diagnostics = dict(bundle.diagnostics)
        diagnostics["last_widen_reason"] = self._ambiguity_signature(bundle, observation)
        diagnostics["widen_step_count"] = widen_steps + 1
        return replace(bundle, effective_plan=updated_plan, diagnostics=diagnostics)

    def _degrade(self, bundle: RuntimeBudgetBundle, observation: RuntimeObservation) -> RuntimeBudgetBundle:
        mode = RuntimeMode.PANIC_SAVE if observation.elapsed_s >= bundle.ceilings.wall_clock_s * 0.98 else RuntimeMode.DEGRADED
        degraded = self._apply_mode_and_ceiling_constraints(bundle.effective_plan, bundle.ceilings, mode)
        diagnostics = dict(bundle.diagnostics)
        diagnostics["last_degrade_pressure"] = round(self._pressure(bundle, observation), 4)
        return replace(bundle, mode=mode, effective_plan=degraded, diagnostics=diagnostics)

    def _must_stop(self, bundle: RuntimeBudgetBundle, observation: RuntimeObservation) -> bool:
        return bool(
            observation.elapsed_s >= bundle.ceilings.wall_clock_s
            or observation.estimated_tokens_used >= bundle.ceilings.token_budget
            or observation.search_nodes_expanded >= bundle.ceilings.search_node_ceiling
            or observation.branches_generated >= bundle.ceilings.branch_ceiling
        )

    @staticmethod
    def _difficulty_branch_ceiling(difficulty: Difficulty) -> int:
        return {
            Difficulty.EASY: 24,
            Difficulty.MEDIUM: 48,
            Difficulty.HARD: 96,
            Difficulty.VERY_HARD: 160,
        }[difficulty]

    @staticmethod
    def _difficulty_node_ceiling(difficulty: Difficulty) -> int:
        return {
            Difficulty.EASY: 40,
            Difficulty.MEDIUM: 80,
            Difficulty.HARD: 144,
            Difficulty.VERY_HARD: 224,
        }[difficulty]

    @staticmethod
    def _scaled(value: int, factor: float) -> int:
        return max(1, int(round(float(value) * float(factor))))


_DEFAULT_PLANNER = AdaptiveBudgetPlanner()


def build_initial_budget(
    route: RouteDecision,
    *,
    remaining_problems: int | None = None,
    tracker: GlobalBudgetTracker | None = None,
    requested_mode: str | RuntimeMode | None = None,
    per_problem_time_limit_s: float | None = None,
    per_problem_token_limit: int | None = None,
) -> RuntimeBudgetBundle:
    return _DEFAULT_PLANNER.build_initial_budget(
        route,
        remaining_problems=remaining_problems,
        tracker=tracker,
        requested_mode=requested_mode,
        per_problem_time_limit_s=per_problem_time_limit_s,
        per_problem_token_limit=per_problem_token_limit,
    )


def adapt_budget(bundle: RuntimeBudgetBundle, observation: RuntimeObservation) -> AdaptationDecision:
    return _DEFAULT_PLANNER.adapt(bundle, observation)


def should_stop_early(bundle: RuntimeBudgetBundle, observation: RuntimeObservation) -> bool:
    return _DEFAULT_PLANNER.should_stop_early(bundle, observation)


def hard_stop_required(bundle: RuntimeBudgetBundle, observation: RuntimeObservation) -> bool:
    return _DEFAULT_PLANNER.hard_stop_required(bundle, observation)


__all__ = [
    "AdaptationDecision",
    "AdaptiveBudgetPlanner",
    "EarlyExitPolicy",
    "GlobalBudgetTracker",
    "RuntimeBudgetBundle",
    "RuntimeCeilings",
    "RuntimeMode",
    "RuntimeObservation",
    "WideningPolicy",
    "adapt_budget",
    "build_initial_budget",
    "hard_stop_required",
    "should_stop_early",
]