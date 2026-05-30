
from src.common.schemas import BudgetPlan, Difficulty, RouteDecision
from src.online.adaptive_budget import AdaptiveBudgetPlanner, RuntimeObservation


def _route() -> RouteDecision:
    plan = BudgetPlan(
        branch_budget=24,
        retrieval_depth=2,
        max_search_depth=4,
        max_search_nodes=40,
        repair_budget=1,
        self_consistency_samples=24,
        critique_top_k=2,
        widen_on_uncertainty=True,
        use_retrieval=True,
        use_symbolic=True,
        use_brute_force=False,
        repair_aggressiveness=0.72,
        resample_aggressiveness=0.66,
        critique_aggressiveness=0.61,
    )
    return RouteDecision(
        problem_id="p",
        difficulty=Difficulty.HARD,
        difficulty_score=0.67,
        budget_plan=plan,
        route_uncertainty=0.31,
        compute_signals={
            "difficulty_intensity": 0.73,
            "route_uncertainty": 0.31,
            "proof_burden": 0.62,
            "retrieval_need": 0.58,
            "repair_need": 0.64,
            "critique_aggressiveness": 0.61,
            "repair_aggressiveness": 0.72,
            "resample_aggressiveness": 0.66,
        },
    )


def test_build_initial_budget_consumes_compute_signals_nontrivially() -> None:
    planner = AdaptiveBudgetPlanner()
    bundle = planner.build_initial_budget(_route())
    assert bundle.initial_plan.repair_aggressiveness >= 0.72
    assert bundle.initial_plan.critique_aggressiveness >= 0.61
    assert bundle.initial_plan.retrieval_depth >= 3
    assert bundle.diagnostics["route_compute_signals"]["proof_burden"] == 0.62


def test_adaptation_uses_richer_signals_for_widening() -> None:
    planner = AdaptiveBudgetPlanner()
    bundle = planner.build_initial_budget(_route())
    observation = RuntimeObservation(
        answer_entropy=0.41,
        best_cluster_ratio=0.68,
        verifier_agreement=0.52,
        retrieval_hits=2,
        metadata={
            "open_obligation_burden": 0.59,
            "prm_prefix_quality": 0.32,
            "retrieval_compatibility": 0.21,
            "retrieval_support": 0.40,
            "verifier_repairability": 0.63,
            "logical_consistency": 0.55,
            "completeness": 0.48,
        },
    )
    decision = planner.adapt(bundle, observation)
    assert decision.widen or decision.degrade
