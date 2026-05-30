from __future__ import annotations

from src.aggregation.entropy_weighting import WeightedClusterScore, EntropyWeightingBundle
from src.aggregation.final_selector import select_final_answer
from src.common.schemas import (
    ArchetypePrediction,
    BudgetPlan,
    Difficulty,
    DifficultyEstimate,
    ParsedProblem,
    ProblemDomain,
    ProblemTypePrediction,
)
from src.routing.dual_router import dual_router


def _problem() -> ParsedProblem:
    return ParsedProblem(
        problem_id="p_route_selector",
        raw_text=(
            "Find all positive integers n such that a modular condition holds, "
            "prove completeness, and justify every eliminated case."
        ),
        constraints=["n is integer", "modular condition", "prove all cases are covered"],
        knowns=["n is positive integer"],
        unknowns=["n"],
        domain=ProblemDomain.NUMBER_THEORY,
        target="find all n",
        answer_type="positive_integer",
        likely_archetypes=["modular", "case_work", "contradiction"],
        parity_cues=["parity"],
        integrality_constraints=["n is integer"],
        proof_targets=["justify modular pruning", "prove completeness"],
        proof_obligation_hints=["symbolic_check", "case_split"],
        parse_quality={"overall_confidence": 0.76},
        difficulty_seed=0.61,
    )


def test_route_decision_threads_structured_compute_signals() -> None:
    problem = _problem()
    decision = dual_router(
        problem,
        problem_type_prediction=ProblemTypePrediction(
            problem_id=problem.problem_id,
            problem_type_probs={
                "number_theory": 0.61,
                "algebra": 0.12,
                "combinatorics": 0.08,
                "geometry": 0.04,
                "functional_equation": 0.02,
                "mixed": 0.09,
                "unknown": 0.04,
            },
            top_domain=ProblemDomain.NUMBER_THEORY,
            confidence=0.56,
            entropy=1.12,
            is_mixed=False,
        ),
        archetype_prediction=ArchetypePrediction(
            problem_id=problem.problem_id,
            archetype_probs={
                "modular": 0.34,
                "case_work": 0.18,
                "contradiction": 0.15,
                "parity": 0.10,
                "invariant": 0.08,
                "bounding": 0.07,
                "construction": 0.03,
                "induction": 0.03,
                "symbolic_manipulation": 0.02,
            },
            top_archetypes=["modular", "case_work", "contradiction"],
            confidence=0.44,
            entropy=1.36,
        ),
        difficulty_estimate=DifficultyEstimate(
            problem_id=problem.problem_id,
            difficulty=Difficulty.HARD,
            difficulty_score=0.67,
            uncertainty=0.49,
            budget_plan=BudgetPlan(
                branch_budget=48,
                retrieval_depth=4,
                max_search_depth=7,
                max_search_nodes=96,
                repair_budget=2,
                self_consistency_samples=52,
                critique_top_k=4,
                repair_aggressiveness=0.63,
                resample_aggressiveness=0.58,
                critique_aggressiveness=0.61,
            ),
            diagnostics={
                "compute_signals": {
                    "difficulty_intensity": 0.71,
                    "proof_burden": 0.56,
                    "retrieval_need": 0.49,
                    "repair_need": 0.52,
                }
            },
        ),
    )

    assert set(decision.compute_signals) >= {
        "difficulty_intensity",
        "route_uncertainty",
        "proof_burden",
        "retrieval_need",
        "repair_need",
        "critique_aggressiveness",
        "repair_aggressiveness",
        "resample_aggressiveness",
    }
    assert all(0.0 <= float(value) <= 1.0 for value in decision.compute_signals.values())
    assert decision.compute_signals["repair_aggressiveness"] == decision.budget_plan.repair_aggressiveness
    assert decision.compute_signals["resample_aggressiveness"] == decision.budget_plan.resample_aggressiveness
    assert decision.compute_signals["critique_aggressiveness"] == decision.budget_plan.critique_aggressiveness
    assert decision.compute_signals["difficulty_intensity"] >= 0.71
    assert decision.repair_threshold > 0.5


def test_final_selector_consumes_richer_decomposed_evidence() -> None:
    strong = WeightedClusterScore(
        cluster_id="c_strong",
        answer="42",
        answer_canonical="42",
        branch_ids=("b1", "b2", "b3"),
        cluster_size=3,
        support_share=0.62,
        diversity_weighted_support=0.66,
        unique_branch_families=3,
        family_diversity=0.82,
        verifier_score=0.84,
        weak_symbolic_support=0.76,
        exact_symbolic_support=0.81,
        retrieval_support=0.57,
        branch_novelty=0.41,
        weak_symbolic_only_support=0.18,
        symbolic_dominance_bonus=0.11,
        uncertainty=0.15,
        entropy_penalty=0.08,
        outlier_penalty=0.02,
        composite_score=0.79,
        mean_provenance_strength=0.71,
        mean_retrieval_relevance=0.68,
        mean_evidence_quality=0.77,
        best_branch_id="b1",
        best_member_score=0.86,
        answer_variants=("42",),
        caution_flags=(),
    )
    weak = WeightedClusterScore(
        cluster_id="c_weak",
        answer="41",
        answer_canonical="41",
        branch_ids=("b4",),
        cluster_size=1,
        support_share=0.18,
        diversity_weighted_support=0.17,
        unique_branch_families=1,
        family_diversity=0.20,
        verifier_score=0.74,
        weak_symbolic_support=0.72,
        exact_symbolic_support=0.74,
        retrieval_support=0.05,
        branch_novelty=0.19,
        weak_symbolic_only_support=0.11,
        symbolic_dominance_bonus=0.04,
        uncertainty=0.41,
        entropy_penalty=0.15,
        outlier_penalty=0.09,
        composite_score=0.80,
        mean_provenance_strength=0.09,
        mean_retrieval_relevance=0.04,
        mean_evidence_quality=0.43,
        best_branch_id="b4",
        best_member_score=0.75,
        answer_variants=("41",),
        caution_flags=("weak_evidence",),
    )

    result = select_final_answer(
        "p_route_selector",
        EntropyWeightingBundle(
            winner=strong,
            ranked_clusters=(strong, weak),
            ranked_candidates=(strong.as_candidate_answer(), weak.as_candidate_answer()),
            global_entropy=0.31,
            total_branches=4,
        ),
    )

    assert result.prediction.final_answer == 42
    assert result.prediction.signal_decomposition["logical_consistency"] >= 0.84
    assert result.prediction.signal_decomposition["retrieval_compatibility"] >= 0.68
    assert result.prediction.signal_decomposition["operator_reliability"] >= 0.71
    assert result.prediction.signal_decomposition["prm_prefix_quality"] >= 0.77
    assert result.prediction.signal_decomposition["open_obligation_burden"] <= 0.15
    assert result.prediction.calibration_summary["selector_status"] == "ok"
    assert result.prediction.provenance["winner_cluster_id"] == "c_strong"
