"""Difficulty estimator: produces continuous difficulty score and compute budget plan.
This drives compute allocation.
"""
from __future__ import annotations

from typing import Any, Mapping

from src.common.metrics import clamp01, normalized_entropy
from src.common.schemas import (
    ArchetypePrediction,
    BudgetPlan,
    Difficulty,
    DifficultyEstimate,
    ParsedProblem,
    ProblemTypePrediction,
)
from src.common.constants import (
    DEFAULT_BRANCH_BUDGETS,
    DEFAULT_MAX_SEARCH_DEPTHS,
    DEFAULT_MAX_SEARCH_NODES,
    DEFAULT_RETRIEVAL_DEPTHS,
    ROUTING_HIGH_ENTROPY_THRESHOLD,
)


class DifficultyEstimator:
    HARD_KEYWORDS = [
        "for all", "there exists", "infinite", "necessary and sufficient",
        "prove that", "show that", "characterize all", "find all",
        "maximum possible", "minimum possible", "for every",
        "non-negative integer", "positive integer solutions", "determine whether",
    ]
    EASY_SIGNALS = [
        "compute", "evaluate", "calculate", "what is", "find the value",
        "simplify", "solve for",
    ]

    def _parse_confidence(self, problem: ParsedProblem) -> float:
        raw = problem.parse_quality
        if isinstance(raw, Mapping):
            for key in ("overall_confidence", "confidence", "parse_confidence"):
                if key in raw:
                    return clamp01(raw[key])
            return 0.70
        for key in ("overall_confidence", "confidence", "parse_confidence"):
            value = getattr(raw, key, None)
            if value is not None:
                return clamp01(value)
        return 0.70

    def extract_features(
        self,
        problem: ParsedProblem,
        *,
        problem_type_prediction: ProblemTypePrediction | None = None,
        archetype_prediction: ArchetypePrediction | None = None,
    ) -> dict[str, Any]:
        text = (problem.raw_text or "").lower()
        length = len(problem.raw_text or "")
        hard_hits = sum(1 for kw in self.HARD_KEYWORDS if kw in text)
        easy_hits = sum(1 for kw in self.EASY_SIGNALS if kw in text)

        domain_entropy = 0.0
        domain_probs: dict[str, float] = {}
        if problem_type_prediction is not None:
            domain_probs = dict(problem_type_prediction.problem_type_probs)
            domain_entropy = normalized_entropy(domain_probs)

        archetype_entropy = 0.0
        archetype_probs: dict[str, float] = {}
        if archetype_prediction is not None:
            archetype_probs = dict(archetype_prediction.archetype_probs)
            archetype_entropy = normalized_entropy(archetype_probs)

        parse_conf = self._parse_confidence(problem)
        proof_targets = len(getattr(problem, "proof_targets", []) or [])
        obligation_hints = len(getattr(problem, "proof_obligation_hints", []) or [])
        theorem_like = sum(1 for kw in ("prove", "show", "find all", "characterize", "determine whether") if kw in text)
        answer_type = str(getattr(problem, "answer_type", "") or "")
        graph = getattr(problem, "canonical_problem_graph", None)
        graph_nodes = len(getattr(graph, "nodes", []) or []) if graph is not None else 0
        graph_edges = len(getattr(graph, "edges", []) or []) if graph is not None else 0

        return {
            "hard_hits": hard_hits,
            "easy_hits": easy_hits,
            "length": length,
            "unknowns": len(problem.unknowns),
            "constraints": len(problem.constraints),
            "likely_archetypes": len(problem.likely_archetypes),
            "archetype_cues": len(getattr(problem, "archetype_cues", []) or []),
            "symmetries": len(problem.symmetries),
            "parity_cues": len(problem.parity_cues),
            "integrality_constraints": len(problem.integrality_constraints),
            "proof_targets": proof_targets,
            "proof_obligation_hints": obligation_hints,
            "theorem_like": theorem_like,
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
            "answer_type": answer_type,
            "parse_confidence": parse_conf,
            "domain_probs": domain_probs,
            "domain_entropy": domain_entropy,
            "archetype_probs": archetype_probs,
            "archetype_entropy": archetype_entropy,
            "archetype_ambiguous": bool(archetype_prediction is not None and archetype_prediction.confidence < 0.22),
        }

    def estimate(
        self,
        problem: ParsedProblem,
        *,
        problem_type_prediction: ProblemTypePrediction | None = None,
        archetype_prediction: ArchetypePrediction | None = None,
    ) -> DifficultyEstimate:
        features = self.extract_features(problem, problem_type_prediction=problem_type_prediction, archetype_prediction=archetype_prediction)
        score = float(problem.difficulty_seed)
        evidence: list[str] = []

        hard_bonus = min(features["hard_hits"] * 0.07, 0.21)
        easy_penalty = min(features["easy_hits"] * 0.05, 0.15)
        score += hard_bonus
        score -= easy_penalty
        if hard_bonus:
            evidence.append(f"hard_hits:+{hard_bonus:.3f}")
        if easy_penalty:
            evidence.append(f"easy_hits:-{easy_penalty:.3f}")

        for threshold, bonus in ((300, 0.07), (600, 0.07), (1000, 0.05)):
            if features["length"] > threshold:
                score += bonus
                evidence.append(f"length>{threshold}:+{bonus:.3f}")

        score += min(features["unknowns"] * 0.03, 0.12)
        score += min(features["constraints"] * 0.025, 0.15)
        score += min(features["likely_archetypes"] * 0.02, 0.08)
        score += min(features["archetype_cues"] * 0.015, 0.06)
        score += min(features["proof_targets"] * 0.03, 0.09)
        score += min(features["proof_obligation_hints"] * 0.02, 0.08)
        score += min(features["theorem_like"] * 0.05, 0.15)
        score += min(features["symmetries"] * 0.015, 0.05)
        score += min(features["parity_cues"] * 0.0125, 0.05)
        score += min(features["integrality_constraints"] * 0.015, 0.05)
        score += min(features["graph_nodes"] * 0.004, 0.04)
        score += min(features["graph_edges"] * 0.003, 0.04)
        if "integer" in features["answer_type"]:
            score += 0.01

        uncertainty = clamp01(
            0.38 * features["domain_entropy"]
            + 0.35 * features["archetype_entropy"]
            + 0.27 * (1.0 - features["parse_confidence"])
        )
        if features["archetype_ambiguous"]:
            uncertainty = clamp01(uncertainty + 0.05)
            evidence.append("archetype_ambiguous:+0.050_uncertainty")

        structure_complexity = clamp01(
            0.32 * min(features["constraints"] / 6.0, 1.0)
            + 0.18 * min(features["unknowns"] / 4.0, 1.0)
            + 0.16 * min(features["likely_archetypes"] / 4.0, 1.0)
            + 0.12 * min(features["symmetries"] / 3.0, 1.0)
            + 0.10 * min(features["graph_nodes"] / 8.0, 1.0)
            + 0.12 * min(features["graph_edges"] / 10.0, 1.0)
        )
        proof_burden = clamp01(
            0.45 * min(features["proof_targets"] / 3.0, 1.0)
            + 0.35 * min(features["proof_obligation_hints"] / 4.0, 1.0)
            + 0.20 * min(features["theorem_like"] / 2.0, 1.0)
        )
        retrieval_need = clamp01(
            0.40 * uncertainty
            + 0.25 * min(features["likely_archetypes"] / 4.0, 1.0)
            + 0.20 * min(features["symmetries"] / 3.0, 1.0)
            + 0.15 * min(features["constraints"] / 6.0, 1.0)
        )
        repair_need = clamp01(
            0.35 * uncertainty + 0.35 * proof_burden + 0.30 * (1.0 - features["parse_confidence"])
        )
        difficulty_intensity = clamp01(
            0.48 * float(problem.difficulty_seed)
            + 0.22 * structure_complexity
            + 0.18 * proof_burden
            + 0.12 * uncertainty
        )

        score += 0.10 * uncertainty + 0.06 * structure_complexity + 0.04 * proof_burden
        score = clamp01(score)
        difficulty = self._band(score)
        budget_plan = self._budget_plan(
            difficulty,
            uncertainty,
            features,
            difficulty_intensity=difficulty_intensity,
            proof_burden=proof_burden,
            retrieval_need=retrieval_need,
            repair_need=repair_need,
        )

        return DifficultyEstimate(
            problem_id=problem.problem_id,
            difficulty=difficulty,
            difficulty_score=round(score, 4),
            uncertainty=round(uncertainty, 4),
            budget_plan=budget_plan,
            diagnostics={
                "features": features,
                "evidence": evidence,
                "compute_signals": {
                    "difficulty_intensity": round(difficulty_intensity, 4),
                    "route_uncertainty": round(uncertainty, 4),
                    "structure_complexity": round(structure_complexity, 4),
                    "proof_burden": round(proof_burden, 4),
                    "retrieval_need": round(retrieval_need, 4),
                    "repair_need": round(repair_need, 4),
                },
            },
        )

    def _band(self, score: float) -> Difficulty:
        if score < 0.20:
            return Difficulty.EASY
        if score < 0.45:
            return Difficulty.MEDIUM
        if score < 0.70:
            return Difficulty.HARD
        return Difficulty.VERY_HARD

    def _budget_plan(self, difficulty: Difficulty, uncertainty: float, features: dict[str, Any], *, difficulty_intensity: float, proof_burden: float, retrieval_need: float, repair_need: float) -> BudgetPlan:
        key = difficulty.value
        base_branch = int(DEFAULT_BRANCH_BUDGETS[key])
        base_retrieval = int(DEFAULT_RETRIEVAL_DEPTHS[key])
        base_depth = int(DEFAULT_MAX_SEARCH_DEPTHS[key])
        base_nodes = int(DEFAULT_MAX_SEARCH_NODES[key])

        ambiguity = clamp01(0.42 * uncertainty + 0.24 * proof_burden + 0.18 * retrieval_need + 0.16 * repair_need)
        compute_pressure = clamp01(0.58 * difficulty_intensity + 0.42 * ambiguity)

        branch_budget = max(base_branch, int(round(base_branch * (1.0 + 0.65 * compute_pressure))))
        retrieval_depth = max(base_retrieval, int(round(base_retrieval + 2.1 * retrieval_need + 0.8 * proof_burden)))
        max_search_depth = max(base_depth, int(round(base_depth + 1.5 * difficulty_intensity + 0.9 * proof_burden)))
        max_search_nodes = max(base_nodes, int(round(base_nodes * (1.0 + 0.58 * compute_pressure))))
        self_consistency_samples = max(branch_budget, int(round(branch_budget * (1.0 + 0.10 * ambiguity))))
        critique_top_k = max(1, min(5, max(1, branch_budget // 16) + int(proof_burden >= 0.40)))
        repair_budget = 1 + int(difficulty in {Difficulty.HARD, Difficulty.VERY_HARD}) + int(repair_need >= 0.45)

        domain_probs = features["domain_probs"]
        arch_probs = features["archetype_probs"]
        if domain_probs.get("geometry", 0.0) >= 0.35 and arch_probs.get("construction", 0.0) >= 0.15:
            max_search_depth += 1
            repair_budget += 1
        if domain_probs.get("number_theory", 0.0) >= 0.35 and arch_probs.get("modular", 0.0) >= 0.20:
            self_consistency_samples = int(round(self_consistency_samples * 1.10))
            retrieval_depth += 1
        if domain_probs.get("combinatorics", 0.0) >= 0.35 and arch_probs.get("extremal", 0.0) >= 0.15:
            retrieval_depth += 1
            critique_top_k = min(5, critique_top_k + 1)
        if domain_probs.get("functional_equation", 0.0) >= 0.30:
            max_search_depth += 1
            branch_budget = max(branch_budget, int(round(branch_budget * 1.08)))
        if uncertainty >= 0.60:
            branch_budget = int(round(branch_budget * 1.12))
            max_search_nodes = int(round(max_search_nodes * 1.10))
        if proof_burden >= 0.55:
            critique_top_k = min(5, critique_top_k + 1)
            repair_budget += 1

        # Keep self-consistency aligned with the final widened branch budget.
        self_consistency_samples = max(
            self_consistency_samples,
            int(round(branch_budget * (1.0 + 0.10 * ambiguity))),
        )

        # Hard problems must never allocate fewer self-consistency samples than branches.
        if difficulty in {Difficulty.HARD, Difficulty.VERY_HARD}:
            self_consistency_samples = max(self_consistency_samples, branch_budget)

        return BudgetPlan(
            branch_budget=branch_budget,
            retrieval_depth=retrieval_depth,
            max_search_depth=max_search_depth,
            max_search_nodes=max_search_nodes,
            repair_budget=repair_budget,
            self_consistency_samples=self_consistency_samples,
            critique_top_k=critique_top_k,
            widen_on_uncertainty=uncertainty >= ROUTING_HIGH_ENTROPY_THRESHOLD,
            use_retrieval=retrieval_depth > 0,
            use_symbolic=True,
            use_brute_force=domain_probs.get("number_theory", 0.0) >= 0.40 or arch_probs.get("brute_force_smallspace", 0.0) >= 0.15,
            repair_aggressiveness=clamp01(0.35 + 0.45 * repair_need),
            resample_aggressiveness=clamp01(0.30 + 0.35 * uncertainty + 0.10 * retrieval_need),
            critique_aggressiveness=clamp01(0.30 + 0.40 * proof_burden + 0.20 * uncertainty),
            verifier_disagreement_widening=True,
            entropy_widening=True,
        )


def estimate_difficulty(problem: ParsedProblem, *, problem_type_prediction: ProblemTypePrediction | None = None, archetype_prediction: ArchetypePrediction | None = None) -> DifficultyEstimate:
    return DifficultyEstimator().estimate(problem, problem_type_prediction=problem_type_prediction, archetype_prediction=archetype_prediction)