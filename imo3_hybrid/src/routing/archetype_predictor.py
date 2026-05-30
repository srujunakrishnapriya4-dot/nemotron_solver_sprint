"""
Archetype predictor: multi-label classification over olympiad reasoning archetypes.
Canonical schema-backed output, with explicit supported registry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from src.common.schemas import ParsedProblem, ArchetypePrediction
from src.common.constants import ROUTING_ARCHETYPE_TEXT_BACKOFF_WEIGHT
from src.routing.problem_type import ProblemTypePrediction


@dataclass(frozen=True)
class ArchetypeSpec:
    name: str
    operator_tags: tuple[str, ...]
    retrieval_tags: tuple[str, ...]
    neighbors: tuple[str, ...]
    compatible_domains: tuple[str, ...]
    description: str


ARCHETYPE_REGISTRY: dict[str, ArchetypeSpec] = {
    "parity": ArchetypeSpec(
        "parity",
        ("mod2_check", "mod4_check", "contradiction_check"),
        ("parity", "even_odd"),
        ("modular", "invariant", "contradiction"),
        ("number_theory", "combinatorics", "algebra"),
        "Even/odd and parity-preservation reasoning.",
    ),
    "invariant": ArchetypeSpec(
        "invariant",
        ("invariant_extract", "state_preservation_check"),
        ("invariant", "monovariant"),
        ("extremal", "symmetry", "parity"),
        ("combinatorics", "number_theory", "algebra", "geometry"),
        "Preserved quantity or structured transition reasoning.",
    ),
    "extremal": ArchetypeSpec(
        "extremal",
        ("extremal_pick", "minimal_counterexample", "max_min_argument"),
        ("extremal", "minimum", "maximum"),
        ("bounding", "contradiction", "invariant"),
        ("combinatorics", "algebra", "number_theory", "geometry"),
        "Choose smallest/largest object and reason from it.",
    ),
    "pigeonhole": ArchetypeSpec(
        "pigeonhole",
        ("bucket_argument", "collision_count"),
        ("pigeonhole", "forced_repetition"),
        ("bijection", "case_work", "bounding"),
        ("combinatorics",),
        "Forced repetition/collision by occupancy counting.",
    ),
    "modular": ArchetypeSpec(
        "modular",
        ("mod_reduction", "residue_check", "divisibility_prune"),
        ("modular", "congruence", "residue"),
        ("parity", "invariant", "contradiction"),
        ("number_theory", "algebra"),
        "Congruence and residue-class reasoning.",
    ),
    "symmetry": ArchetypeSpec(
        "symmetry",
        ("swap_variables", "canonical_order", "symmetry_reduce"),
        ("symmetry", "wlog", "exchange"),
        ("invariant", "construction", "symbolic_manipulation"),
        ("geometry", "algebra", "combinatorics"),
        "Exploit symmetry and exchangeability.",
    ),
    "bounding": ArchetypeSpec(
        "bounding",
        ("upper_bound", "lower_bound", "estimate"),
        ("bound", "estimate", "inequality"),
        ("extremal", "symbolic_manipulation", "case_work"),
        ("algebra", "number_theory", "combinatorics", "geometry"),
        "Upper/lower bounds and inequality reasoning.",
    ),
    "contradiction": ArchetypeSpec(
        "contradiction",
        ("assume_negation", "derive_conflict"),
        ("contradiction", "impossible"),
        ("parity", "extremal", "case_work"),
        ("algebra", "number_theory", "combinatorics", "geometry", "functional_equation"),
        "Assume the contrary and derive impossibility.",
    ),
    "construction": ArchetypeSpec(
        "construction",
        ("construct_witness", "explicit_example", "geometric_build"),
        ("construct", "witness", "exists"),
        ("case_work", "symmetry", "brute_force_smallspace"),
        ("geometry", "combinatorics", "number_theory", "algebra"),
        "Build an explicit witness/configuration.",
    ),
    "induction": ArchetypeSpec(
        "induction",
        ("base_case", "inductive_step", "recursive_unfold"),
        ("induction", "recursive"),
        ("invariant", "symbolic_manipulation", "contradiction"),
        ("number_theory", "combinatorics", "algebra", "functional_equation"),
        "Recursive or inductive reasoning.",
    ),
    "bijection": ArchetypeSpec(
        "bijection",
        ("map_objects", "count_two_ways", "injective_surjective_check"),
        ("bijection", "double_count"),
        ("pigeonhole", "case_work", "construction"),
        ("combinatorics",),
        "Correspondence / double-counting reasoning.",
    ),
    "case_work": ArchetypeSpec(
        "case_work",
        ("split_cases", "partition_domain", "merge_case_results"),
        ("cases", "casework", "split"),
        ("contradiction", "brute_force_smallspace", "construction"),
        ("algebra", "number_theory", "combinatorics", "geometry"),
        "Partition the problem into cases.",
    ),
    "symbolic_manipulation": ArchetypeSpec(
        "symbolic_manipulation",
        ("substitute", "factor", "expand", "rearrange", "eliminate"),
        ("algebraic_manipulation", "substitute", "factor"),
        ("bounding", "symmetry", "induction"),
        ("algebra", "functional_equation", "number_theory"),
        "Heavy algebraic rewriting and elimination.",
    ),
    "brute_force_smallspace": ArchetypeSpec(
        "brute_force_smallspace",
        ("enumerate_small_space", "small_case_check", "program_search"),
        ("small_space", "enumerate", "bruteforce"),
        ("case_work", "construction", "pigeonhole"),
        ("combinatorics", "number_theory", "algebra"),
        "Small finite search space / explicit enumeration.",
    ),
    "generating_function": ArchetypeSpec(
        "generating_function",
        ("series_encode", "coefficient_extract"),
        ("generating_function", "power_series", "coefficient"),
        ("bijection", "counting_reasoning"),
        ("combinatorics",),
        "Formal power-series counting.",
    ),
}


class ArchetypePredictor:
    def supported_archetypes(self) -> tuple[str, ...]:
        return tuple(ARCHETYPE_REGISTRY.keys())

    def extract_features(
        self,
        problem: ParsedProblem,
        *,
        problem_type_prediction: ProblemTypePrediction | None = None,
    ) -> dict[str, Any]:
        text = problem.raw_text.lower()
        constraints_text = " ".join(problem.constraint_texts()).lower()

        parser_cues = {name: 0.0 for name in ARCHETYPE_REGISTRY}
        for cue in problem.likely_archetypes:
            if cue in parser_cues:
                parser_cues[cue] += 1.0

        domain_probs = (
            problem_type_prediction.problem_type_probs
            if problem_type_prediction is not None
            else {}
        )

        return {
            "text": text,
            "constraints_text": constraints_text,
            "parser_cues": parser_cues,
            "domain_probs": domain_probs,
            "has_parity": bool(problem.parity_cues or problem.integrality_constraints),
            "has_symmetry": bool(problem.symmetries),
            "constraint_count": len(problem.constraints),
            "parse_confidence": problem.parse_quality.get("overall_confidence", 0.7) if problem.parse_quality else 0.7,
        }

    def predict(
        self,
        problem: ParsedProblem,
        problem_type_prediction: ProblemTypePrediction | None = None,
    ) -> ArchetypePrediction:
        features = self.extract_features(problem, problem_type_prediction=problem_type_prediction)
        text = features["text"]
        constraints_text = features["constraints_text"]
        domain_probs = features["domain_probs"]

        scores: dict[str, float] = {name: 0.2 for name in ARCHETYPE_REGISTRY}
        reasons: dict[str, list[str]] = {name: [] for name in ARCHETYPE_REGISTRY}

        def add(name: str, weight: float, reason: str) -> None:
            if weight == 0.0:
                return
            scores[name] += weight
            reasons[name].append(f"{reason}:{weight:.3f}")

        # parser cues
        for name, value in features["parser_cues"].items():
            add(name, 1.55 * value, "parser_archetype_cue")

        # structured hints
        if features["has_parity"]:
            add("parity", 1.2, "parser_parity")
            add("modular", 0.9, "parser_parity")
        if features["has_symmetry"]:
            add("symmetry", 1.0, "parser_symmetry")

        if len(problem.constraints) >= 4:
            add("case_work", 0.3, "constraint_count")
            add("bounding", 0.2, "constraint_count")
            add("symbolic_manipulation", 0.25, "constraint_count")

        # weak text backoff
        keyword_map = {
            "parity": ["even", "odd", "parity"],
            "invariant": ["invariant", "preserved", "unchanged"],
            "extremal": ["smallest", "largest", "minimum", "maximum", "extremal"],
            "pigeonhole": ["pigeonhole", "forced repetition", "boxes", "drawers"],
            "modular": ["mod", "modulo", "congruent", "residue"],
            "symmetry": ["symmetric", "symmetry", "wlog", "without loss"],
            "bounding": ["bound", "estimate", "at most", "at least", "inequality"],
            "contradiction": ["contradiction", "impossible", "assume not"],
            "construction": ["construct", "example", "there exists"],
            "induction": ["induction", "recursive", "for all n"],
            "bijection": ["bijection", "double count", "count two ways"],
            "case_work": ["case 1", "case 2", "consider cases", "split into cases"],
            "symbolic_manipulation": ["factor", "expand", "substitute", "rearrange"],
            "brute_force_smallspace": ["enumerate", "check all", "small cases", "finite search"],
            "generating_function": ["generating function", "power series", "coefficient"],
        }
        for name, kws in keyword_map.items():
            add(name, ROUTING_ARCHETYPE_TEXT_BACKOFF_WEIGHT * sum(1.0 for kw in kws if kw in text), "weak_text_backoff")

        # constraint hints
        if any(kw in constraints_text for kw in ["divisible", "congruent", "mod", "prime"]):
            add("modular", 1.1, "constraint_modularity")
            add("parity", 0.4, "constraint_modularity")
        if any(kw in constraints_text for kw in ["<", ">", "inequality", "at most", "at least"]):
            add("bounding", 0.9, "constraint_inequality")
        if any(kw in text for kw in ["function", "f(", "g(", "h("]):
            add("symbolic_manipulation", 0.8, "function_structure")
            add("induction", 0.3, "function_structure")

        # domain compatibility shaping
        for name, spec in ARCHETYPE_REGISTRY.items():
            compat_mass = sum(float(domain_probs.get(d, 0.0)) for d in spec.compatible_domains)
            incompat_mass = sum(
                float(v) for d, v in domain_probs.items()
                if d not in spec.compatible_domains and d not in {"mixed", "unknown"}
            )
            add(name, 0.35 * compat_mass, "domain_compatibility")
            if incompat_mass > 0.0:
                add(name, -0.20 * incompat_mass, "domain_incompatibility")

        # parse penalty softening
        parse_penalty = max(0.0, 0.65 - float(features["parse_confidence"]))
        if parse_penalty > 0.0:
            for name in scores:
                add(name, -0.55 * parse_penalty, "parse_quality_penalty")

        probs = self._normalize(scores)
        ranked = sorted(probs.items(), key=lambda x: (-x[1], x[0]))
        entropy = self._entropy(list(probs.values()))
        entropy_norm = entropy / math.log(max(len(probs), 2))
        margin = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)
        confidence = max(0.0, min(1.0, 0.60 * margin + 0.40 * (1.0 - entropy_norm)))

        top_archetypes = [a for a, p in ranked if p >= 0.08][:5] or [ranked[0][0]]

        return ArchetypePrediction(
            problem_id=problem.problem_id,
            archetype_probs=probs,
            top_archetypes=top_archetypes,
            confidence=round(confidence, 4),
            entropy=round(entropy, 6),
            diagnostics={
                "features": features,
                "raw_scores": {k: round(v, 4) for k, v in scores.items()},
                "reasons": reasons,
                "registry": {
                    name: {
                        "operator_tags": spec.operator_tags,
                        "retrieval_tags": spec.retrieval_tags,
                        "neighbors": spec.neighbors,
                        "compatible_domains": spec.compatible_domains,
                    }
                    for name, spec in ARCHETYPE_REGISTRY.items()
                },
            },
        )

    def top_archetypes(self, problem: ParsedProblem, k: int = 3) -> list[str]:
        return self.predict(problem).top_archetypes[:k]

    @staticmethod
    def _normalize(scores: dict[str, float]) -> dict[str, float]:
        positive = {k: max(v, 1e-6) for k, v in scores.items()}
        total = sum(positive.values()) or 1.0
        return {k: round(v / total, 6) for k, v in positive.items()}

    @staticmethod
    def _entropy(values: list[float]) -> float:
        total = 0.0
        for p in values:
            if p > 0.0:
                total -= p * math.log(p)
        return total


def get_supported_archetypes() -> tuple[str, ...]:
    return tuple(ARCHETYPE_REGISTRY.keys())


def get_archetype_registry() -> dict[str, ArchetypeSpec]:
    return dict(ARCHETYPE_REGISTRY)


_DEFAULT_PREDICTOR = ArchetypePredictor()


def extract_archetype_features(
    problem: ParsedProblem,
    *,
    problem_type_prediction: ProblemTypePrediction | None = None,
) -> dict[str, Any]:
    return _DEFAULT_PREDICTOR.extract_features(problem, problem_type_prediction=problem_type_prediction)


def predict_archetypes(
    problem: ParsedProblem,
    *,
    problem_type_prediction: ProblemTypePrediction | None = None,
) -> ArchetypePrediction:
    return _DEFAULT_PREDICTOR.predict(problem, problem_type_prediction=problem_type_prediction)


def predict_archetype_probs(
    problem: ParsedProblem,
    *,
    problem_type_prediction: ProblemTypePrediction | None = None,
) -> dict[str, float]:
    return _DEFAULT_PREDICTOR.predict(problem, problem_type_prediction=problem_type_prediction).archetype_probs
