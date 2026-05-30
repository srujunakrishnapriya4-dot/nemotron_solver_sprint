"""
Dual Router — canonical routing control-plane combiner.
Fuses problem-type, archetype, and difficulty into a real RouteDecision.
"""
from __future__ import annotations
import math


from src.common.metrics import clamp01, normalize_probabilities, normalized_entropy
from src.common.schemas import (
    ParsedProblem,
    RouteDecision,
    BudgetPlan,
    ProblemTypePrediction,
    ArchetypePrediction,
    DifficultyEstimate,
)
from src.common.constants import (
    DEFAULT_OPERATOR_PRIOR,
    DOMAIN_OPERATOR_PRIORS,
    ARCHETYPE_OPERATOR_PRIORS,
    ROUTING_HIGH_ENTROPY_THRESHOLD,
    ROUTING_SHARPEN_TEMPERATURE,
    ROUTING_FLATTEN_TEMPERATURE,
)
from src.routing.problem_type import predict_problem_type
from src.routing.archetype_predictor import predict_archetypes
from src.routing.difficulty_estimator import estimate_difficulty

class DualRouter:
    """Canonical routing entrypoint."""

    def route(
        self,
        problem: ParsedProblem,
        *,
        problem_type_prediction: ProblemTypePrediction | None = None,
        archetype_prediction: ArchetypePrediction | None = None,
        difficulty_estimate: DifficultyEstimate | None = None,
    ) -> RouteDecision:
        if problem_type_prediction is None:
            problem_type_prediction = predict_problem_type(problem)
        if archetype_prediction is None:
            archetype_prediction = predict_archetypes(
                problem,
                problem_type_prediction=problem_type_prediction,
            )
        if difficulty_estimate is None:
            difficulty_estimate = estimate_difficulty(
                problem,
                problem_type_prediction=problem_type_prediction,
                archetype_prediction=archetype_prediction,
            )

        estimator_signals = self._estimator_compute_signals(difficulty_estimate)
        route_uncertainty = self._route_uncertainty(
            problem_type_prediction,
            archetype_prediction,
            difficulty_estimate,
            estimator_signals=estimator_signals,
        )
        route_uncertainty = max(
            route_uncertainty,
            clamp01(float(getattr(difficulty_estimate, "uncertainty", 0.0)) * 0.88),
        )
        budget_plan = self._widen_budget_if_needed(difficulty_estimate.budget_plan, route_uncertainty)
        compute_signals = self._route_compute_signals(
            difficulty_estimate,
            budget_plan=budget_plan,
            route_uncertainty=route_uncertainty,
        )
        operator_prior = self._build_operator_prior(
            problem_type_prediction.problem_type_probs,
            archetype_prediction.archetype_probs,
            difficulty_estimate.difficulty.value,
            route_uncertainty,
            compute_signals=compute_signals,
        )

        retrieval_tags = self._retrieval_tags(problem_type_prediction.problem_type_probs, archetype_prediction.archetype_probs)
        repair_neighbors = self._repair_neighbors(archetype_prediction.archetype_probs)
        route_rationale = self._route_rationale(
            problem_type_prediction,
            archetype_prediction,
            difficulty_estimate,
            route_uncertainty,
            compute_signals=compute_signals,
        )

        repair_threshold = clamp01(
            0.42
            + 0.26 * route_uncertainty
            + 0.18 * compute_signals.get("repair_need", 0.0)
            + 0.08 * compute_signals.get("proof_burden", 0.0)
        )

        return RouteDecision(
            problem_id=problem.problem_id,
            problem_type_probs=problem_type_prediction.problem_type_probs,
            archetype_probs=archetype_prediction.archetype_probs,
            difficulty=difficulty_estimate.difficulty,
            difficulty_score=difficulty_estimate.difficulty_score,
            operator_prior=operator_prior,
            budget_plan=budget_plan,
            retrieval_depth=budget_plan.retrieval_depth,
            branch_budget=budget_plan.branch_budget,
            repair_threshold=round(repair_threshold, 4),
            route_uncertainty=round(route_uncertainty, 4),
            verifier_mode="strict" if difficulty_estimate.difficulty.value in {"hard", "very_hard"} else "default",
            problem_type=problem_type_prediction.problem_type_probs,
            archetypes=archetype_prediction.archetype_probs,
            use_retrieval=budget_plan.use_retrieval,
            use_symbolic=budget_plan.use_symbolic,
            use_brute_force=budget_plan.use_brute_force,
            retrieval_tags=retrieval_tags,
            repair_neighbors=repair_neighbors,
            route_rationale=route_rationale,
            compute_signals=compute_signals,
            calibration_metadata={
                "routing_version": "dual_router.compute_signal_repair.v1",
                "difficulty_conditioned": bool(getattr(budget_plan, "difficulty_conditioned", True)),
                "widen_on_uncertainty": bool(budget_plan.widen_on_uncertainty),
            },
            diagnostics={
                "problem_type_prediction": problem_type_prediction.model_dump(),
                "archetype_prediction": archetype_prediction.model_dump(),
                "difficulty_estimate": difficulty_estimate.model_dump(),
                "route_uncertainty": round(route_uncertainty, 4),
                "compute_signals": compute_signals,
                "budget_plan": budget_plan.model_dump(),
            },
        )

    def _build_operator_prior(
        self,
        domain_probs: dict[str, float],
        archetype_probs: dict[str, float],
        difficulty_key: str,
        route_uncertainty: float,
        *,
        compute_signals: dict[str, float],
    ) -> dict[str, float]:
        blended = dict(DEFAULT_OPERATOR_PRIOR)

        for domain, dp in domain_probs.items():
            for op, op_p in DOMAIN_OPERATOR_PRIORS.get(domain, {}).items():
                blended[op] = blended.get(op, 0.0) + 0.45 * dp * op_p

        for archetype, ap in archetype_probs.items():
            for op, op_p in ARCHETYPE_OPERATOR_PRIORS.get(archetype, {}).items():
                blended[op] = blended.get(op, 0.0) + 0.55 * ap * op_p

        if difficulty_key in {"hard", "very_hard"}:
            blended["symbolic_execution"] = blended.get("symbolic_execution", 0.0) + 0.03
            blended["case_split"] = blended.get("case_split", 0.0) + 0.02
        if compute_signals.get("proof_burden", 0.0) >= 0.40:
            blended["case_split"] = blended.get("case_split", 0.0) + 0.025
            blended["contradiction"] = blended.get("contradiction", 0.0) + 0.015
        if compute_signals.get("retrieval_need", 0.0) >= 0.35:
            blended["retrieve_trace"] = blended.get("retrieve_trace", 0.0) + 0.02
        if compute_signals.get("repair_need", 0.0) >= 0.35:
            blended["self_critique"] = blended.get("self_critique", 0.0) + 0.015
            blended["repair_local"] = blended.get("repair_local", 0.0) + 0.02

        temperature = (
            ROUTING_FLATTEN_TEMPERATURE
            if route_uncertainty >= ROUTING_HIGH_ENTROPY_THRESHOLD
            else ROUTING_SHARPEN_TEMPERATURE
        )

        adjusted = {k: max(v, 1e-6) ** (1.0 / temperature) for k, v in blended.items()}
        normalized = normalize_probabilities(adjusted)
        out = {k: round(v, 6) for k, v in normalized.items()}
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    def _route_uncertainty(
        self,
        problem_type_prediction: ProblemTypePrediction,
        archetype_prediction: ArchetypePrediction,
        difficulty_estimate: DifficultyEstimate,
        *,
        estimator_signals: dict[str, float],
    ) -> float:
        domain_conf = problem_type_prediction.confidence
        archetype_conf = archetype_prediction.confidence
        domain_entropy_norm = normalized_entropy(problem_type_prediction.problem_type_probs)
        archetype_entropy_norm = normalized_entropy(archetype_prediction.archetype_probs)
        uncertainty = (
            0.22 * difficulty_estimate.uncertainty
            + 0.18 * (1.0 - domain_conf)
            + 0.18 * (1.0 - archetype_conf)
            + 0.13 * domain_entropy_norm
            + 0.16 * archetype_entropy_norm
            + 0.08 * estimator_signals.get("proof_burden", 0.0)
            + 0.05 * estimator_signals.get("retrieval_need", 0.0)
        )
        return clamp01(uncertainty)
    
    def _widen_budget_if_needed(self, budget_plan: BudgetPlan, route_uncertainty: float) -> BudgetPlan:
        uncertainty = clamp01(route_uncertainty)

        base_branch_budget = int(budget_plan.branch_budget)
        base_retrieval_depth = int(budget_plan.retrieval_depth)
        base_max_search_depth = int(budget_plan.max_search_depth)
        base_max_search_nodes = int(budget_plan.max_search_nodes)
        base_repair_budget = int(budget_plan.repair_budget)
        base_self_consistency = int(budget_plan.self_consistency_samples)
        base_critique_top_k = int(budget_plan.critique_top_k)

        should_widen = uncertainty >= 0.60 or bool(getattr(budget_plan, "widen_on_uncertainty", False))
        if not should_widen:
            return budget_plan

        widen_factor = 1.0 + 0.25 * uncertainty

        widened_branch_budget = max(
            base_branch_budget + 1,
            int(math.ceil(base_branch_budget * widen_factor)),
        )
        widened_retrieval_depth = max(
            base_retrieval_depth + 1,
            int(math.ceil(base_retrieval_depth * (1.0 + 0.15 * uncertainty))),
        )
        widened_max_search_depth = max(
            base_max_search_depth + 1,
            int(math.ceil(base_max_search_depth * (1.0 + 0.10 * uncertainty))),
        )
        widened_max_search_nodes = max(
            base_max_search_nodes + 1,
            int(math.ceil(base_max_search_nodes * (1.0 + 0.35 * uncertainty))),
        )
        widened_repair_budget = max(
            base_repair_budget + 1,
            int(math.ceil(base_repair_budget * (1.0 + 0.20 * uncertainty))),
        )
        widened_self_consistency = max(
            base_self_consistency,
            widened_branch_budget,
            int(math.ceil(base_self_consistency * (1.0 + 0.20 * uncertainty))),
        )
        widened_critique_top_k = max(
            base_critique_top_k,
            int(math.ceil(base_critique_top_k * (1.0 + 0.10 * uncertainty))),
        )

        data = budget_plan.model_dump()
        data["branch_budget"] = widened_branch_budget
        data["retrieval_depth"] = widened_retrieval_depth
        data["max_search_depth"] = widened_max_search_depth
        data["max_search_nodes"] = widened_max_search_nodes
        data["repair_budget"] = widened_repair_budget
        data["self_consistency_samples"] = widened_self_consistency
        data["critique_top_k"] = widened_critique_top_k
        data["widen_on_uncertainty"] = True
        data["repair_aggressiveness"] = clamp01(
            float(data.get("repair_aggressiveness", 0.5)) + 0.08 * uncertainty
        )
        data["resample_aggressiveness"] = clamp01(
            float(data.get("resample_aggressiveness", 0.5)) + 0.08 * uncertainty
        )
        data["critique_aggressiveness"] = clamp01(
            float(data.get("critique_aggressiveness", 0.5)) + 0.05 * uncertainty
        )
        return BudgetPlan(**data)

    @staticmethod
    def _estimator_compute_signals(difficulty_estimate: DifficultyEstimate) -> dict[str, float]:
        diagnostics = getattr(difficulty_estimate, "diagnostics", {}) or {}
        signals = diagnostics.get("compute_signals", {}) if isinstance(diagnostics, dict) else {}
        if not isinstance(signals, dict):
            signals = {}
        out: dict[str, float] = {}
        for key in (
            "difficulty_intensity",
            "route_uncertainty",
            "proof_burden",
            "retrieval_need",
            "repair_need",
        ):
            if key in signals:
                out[key] = round(clamp01(signals.get(key, 0.0)), 4)
        if "route_uncertainty" not in out:
            out["route_uncertainty"] = round(clamp01(getattr(difficulty_estimate, "uncertainty", 0.0)), 4)
        return out

    @staticmethod
    def _route_compute_signals(
        difficulty_estimate: DifficultyEstimate,
        *,
        budget_plan: BudgetPlan,
        route_uncertainty: float,
    ) -> dict[str, float]:
        signals = DualRouter._estimator_compute_signals(difficulty_estimate)
        signals["route_uncertainty"] = round(clamp01(route_uncertainty), 4)
        signals.setdefault("difficulty_intensity", round(clamp01(difficulty_estimate.difficulty_score), 4))
        signals.setdefault("proof_burden", 0.0)
        signals.setdefault("retrieval_need", round(clamp01(budget_plan.retrieval_depth / max(1.0, budget_plan.retrieval_depth + 2.0)), 4))
        signals.setdefault("repair_need", round(clamp01(budget_plan.repair_budget / max(1.0, budget_plan.repair_budget + 2.0)), 4))
        signals["critique_aggressiveness"] = round(clamp01(getattr(budget_plan, "critique_aggressiveness", 0.5)), 4)
        signals["repair_aggressiveness"] = round(clamp01(getattr(budget_plan, "repair_aggressiveness", 0.5)), 4)
        signals["resample_aggressiveness"] = round(clamp01(getattr(budget_plan, "resample_aggressiveness", 0.5)), 4)
        return dict(sorted(signals.items()))

    @staticmethod
    def _retrieval_tags(domain_probs: dict[str, float], archetype_probs: dict[str, float]) -> list[str]:
        tags: list[str] = []
        if domain_probs.get("number_theory", 0.0) >= 0.30:
            tags.extend(["modular_patterns", "divisibility_traces"])
        if domain_probs.get("geometry", 0.0) >= 0.30:
            tags.extend(["diagram_structure", "construction_patterns"])
        if domain_probs.get("combinatorics", 0.0) >= 0.30:
            tags.extend(["counting_patterns", "extremal_counting"])
        if domain_probs.get("functional_equation", 0.0) >= 0.25:
            tags.extend(["functional_equation_patterns", "symbolic_function_traces"])
        if archetype_probs.get("invariant", 0.0) >= 0.12:
            tags.append("invariant_traces")
        if archetype_probs.get("construction", 0.0) >= 0.12:
            tags.append("construction_traces")
        if archetype_probs.get("case_work", 0.0) >= 0.10:
            tags.append("case_split_examples")
        if archetype_probs.get("symbolic_manipulation", 0.0) >= 0.10:
            tags.append("symbolic_rewrite_traces")
        return sorted(set(tags))

    @staticmethod
    def _repair_neighbors(archetype_probs: dict[str, float]) -> list[str]:
        neighbors: list[str] = []
        if archetype_probs.get("symmetry", 0.0) >= 0.10:
            neighbors.extend(["invariant", "case_work"])
        if archetype_probs.get("construction", 0.0) >= 0.10:
            neighbors.extend(["extremal", "case_work"])
        if archetype_probs.get("modular", 0.0) >= 0.10:
            neighbors.extend(["invariant", "contradiction", "parity"])
        if archetype_probs.get("symbolic_manipulation", 0.0) >= 0.10:
            neighbors.extend(["bounding", "induction"])
        return sorted(set(neighbors))

    @staticmethod
    def _route_rationale(
        problem_type_prediction: ProblemTypePrediction,
        archetype_prediction: ArchetypePrediction,
        difficulty_estimate: DifficultyEstimate,
        route_uncertainty: float,
        *,
        compute_signals: dict[str, float],
    ) -> list[str]:
        domain_top = max(problem_type_prediction.problem_type_probs.items(), key=lambda kv: kv[1])[0]
        arch_top = max(archetype_prediction.archetype_probs.items(), key=lambda kv: kv[1])[0]
        rationale = [
            f"top_domain={domain_top}",
            f"top_archetype={arch_top}",
            f"difficulty={difficulty_estimate.difficulty.value}",
            f"route_uncertainty={route_uncertainty:.4f}",
            f"difficulty_intensity={compute_signals.get('difficulty_intensity', 0.0):.4f}",
            f"proof_burden={compute_signals.get('proof_burden', 0.0):.4f}",
            f"retrieval_need={compute_signals.get('retrieval_need', 0.0):.4f}",
            f"repair_need={compute_signals.get('repair_need', 0.0):.4f}",
        ]
        if problem_type_prediction.is_mixed:
            rationale.append("mixed_domain_handling_enabled")
        if difficulty_estimate.budget_plan.widen_on_uncertainty or route_uncertainty >= ROUTING_HIGH_ENTROPY_THRESHOLD:
            rationale.append("budget_widened_on_uncertainty")
        if difficulty_estimate.budget_plan.use_brute_force:
            rationale.append("bruteforce_enabled")
        return rationale


def dual_router(
    problem: ParsedProblem,
    *,
    problem_type_prediction: ProblemTypePrediction | None = None,
    archetype_prediction: ArchetypePrediction | None = None,
    difficulty_estimate: DifficultyEstimate | None = None,
) -> RouteDecision:
    return DualRouter().route(
        problem,
        problem_type_prediction=problem_type_prediction,
        archetype_prediction=archetype_prediction,
        difficulty_estimate=difficulty_estimate,
    )
