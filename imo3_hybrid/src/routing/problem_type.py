"""
Problem type classifier: algebra / number_theory / combinatorics / geometry / functional_equation / mixed.
Schema-backed and deterministic.
"""
from __future__ import annotations

from typing import Any

from src.common.metrics import distribution_summary, normalize_probabilities
from src.common.schemas import ParsedProblem, ProblemDomain, ProblemTypePrediction
from src.common.constants import (
    ROUTING_MIXED_DOMAIN_MARGIN,
    ROUTING_LOW_EVIDENCE_THRESHOLD,
    ROUTING_DOMAIN_TEXT_BACKOFF_WEIGHT,
)


DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "algebra": [
        "polynomial", "equation", "inequality", "roots", "solve", "factor",
        "simplify", "quadratic", "cubic", "coefficient", "expression",
    ],
    "number_theory": [
        "prime", "divisor", "mod", "remainder", "gcd", "lcm", "integer",
        "modulo", "congruent", "coprime", "divisible", "diophantine",
    ],
    "combinatorics": [
        "count", "ways", "arrange", "permutation", "combination", "subset",
        "graph", "coloring", "path", "cycle", "tree", "partition",
    ],
    "geometry": [
        "triangle", "circle", "angle", "parallel", "perpendicular", "area",
        "tangent", "radius", "chord", "polygon", "midpoint", "centroid",
        "similar", "congruent",
    ],
    "functional_equation": [
        "function", "functional equation", "f(", "g(", "h(",
    ],
}


class ProblemTypeClassifier:
    """
    Structured deterministic domain classifier.
    Consumes ParsedProblem and emits a schema-backed probabilistic prediction.
    """

    def extract_features(self, problem: ParsedProblem) -> dict[str, Any]:
        text = problem.raw_text.lower()
        constraints_text = " ".join(problem.constraint_texts()).lower()
        graph_blob = " ".join(f"{k}:{v}" for k, v in sorted(problem.metadata.items())).lower()

        relation_counts = {
            "equation_like": sum(1 for kw in ["=", "equation", "inequality", "<", ">"] if kw in constraints_text),
            "divisibility_like": sum(1 for kw in ["mod", "divisible", "congruent", "prime", "integer"] if kw in constraints_text),
            "geometry_like": sum(1 for kw in ["angle", "triangle", "circle", "parallel", "perpendicular"] if kw in constraints_text),
            "set_like": sum(1 for kw in ["subset", "set", "graph", "path", "cycle", "tree"] if kw in constraints_text),
            "function_like": sum(1 for kw in ["function", "f(", "g(", "h("] if kw in text),
        }

        archetype_features = {
            "combinatorics_bias": float(any(a in {"bijection", "pigeonhole", "generating_function"} for a in problem.likely_archetypes)),
            "number_theory_bias": float(any(a in {"modular", "induction", "parity"} for a in problem.likely_archetypes)),
            "algebra_bias": float(any(a in {"bounding", "symmetry", "extremal", "symbolic_manipulation"} for a in problem.likely_archetypes)),
        }

        text_backoff = {
            domain: float(sum(1 for kw in kws if kw in text))
            for domain, kws in DOMAIN_KEYWORDS.items()
        }

        parse_conf = problem.parse_quality.get("overall_confidence", 0.7) if problem.parse_quality else 0.7

        return {
            "relation_counts": relation_counts,
            "archetype_features": archetype_features,
            "text_backoff": text_backoff,
            "parse_confidence": parse_conf,
            "has_symmetry": bool(problem.symmetries),
            "has_parity": bool(problem.parity_cues or problem.integrality_constraints),
            "constraint_count": len(problem.constraints),
            "unknown_count": len(problem.unknowns),
            "graph_blob": graph_blob,
        }

    def predict(self, problem: ParsedProblem) -> ProblemTypePrediction:
        features = self.extract_features(problem)

        raw_scores = {
            "algebra": 0.0,
            "number_theory": 0.0,
            "combinatorics": 0.0,
            "geometry": 0.0,
            "functional_equation": 0.0,
            "mixed": 0.0,
            "unknown": 0.0,
        }

        reasons: dict[str, list[str]] = {k: [] for k in raw_scores}

        def add(domain: str, weight: float, reason: str) -> None:
            if weight == 0.0:
                return
            raw_scores[domain] += weight
            reasons[domain].append(f"{reason}:{weight:.3f}")

        # parser-owned explicit domain seed
        if problem.domain in {
            ProblemDomain.ALGEBRA,
            ProblemDomain.NUMBER_THEORY,
            ProblemDomain.COMBINATORICS,
            ProblemDomain.GEOMETRY,
            ProblemDomain.FUNCTIONAL_EQUATION,
        }:
            add(problem.domain.value, 2.5, "parser_domain_seed")

        rc = features["relation_counts"]
        add("algebra", 1.00 * rc["equation_like"], "equation_like_constraints")
        add("number_theory", 1.35 * rc["divisibility_like"], "divisibility_like_constraints")
        add("geometry", 1.60 * rc["geometry_like"], "geometry_like_constraints")
        add("combinatorics", 1.15 * rc["set_like"], "set_like_constraints")
        add("functional_equation", 1.70 * rc["function_like"], "function_like_structure")

        if features["has_symmetry"]:
            add("geometry", 0.35, "parser_symmetry")
            add("algebra", 0.25, "parser_symmetry")
        if features["has_parity"]:
            add("number_theory", 0.75, "parser_parity")

        af = features["archetype_features"]
        add("combinatorics", 0.80 * af["combinatorics_bias"], "archetype_bias")
        add("number_theory", 0.65 * af["number_theory_bias"], "archetype_bias")
        add("algebra", 0.45 * af["algebra_bias"], "archetype_bias")

        for domain, value in features["text_backoff"].items():
            add(domain, ROUTING_DOMAIN_TEXT_BACKOFF_WEIGHT * value, "weak_text_backoff")

        total_base = sum(raw_scores[d] for d in ["algebra", "number_theory", "combinatorics", "geometry", "functional_equation"])
        if total_base < ROUTING_LOW_EVIDENCE_THRESHOLD:
            add("unknown", 0.35 + (ROUTING_LOW_EVIDENCE_THRESHOLD - total_base), "low_total_evidence")

        parse_penalty = max(0.0, 0.65 - float(features["parse_confidence"]))
        if parse_penalty > 0.0:
            add("unknown", 0.5 * parse_penalty, "parse_quality_penalty")

        base_domains = ["algebra", "number_theory", "combinatorics", "geometry", "functional_equation"]
        ranked_base = sorted(((d, raw_scores[d]) for d in base_domains), key=lambda kv: (-kv[1], kv[0]))
        top_name, top_score = ranked_base[0]
        second_name, second_score = ranked_base[1]
        if top_score >= 1.10 and second_score >= 1.10 and abs(top_score - second_score) < 0.75:
            add("mixed", 0.10 + 0.45 * min(top_score, second_score) / max(top_score, 1e-6), f"top_domains:{top_name}+{second_name}")

        normalized = normalize_probabilities(raw_scores)
        probs = {key: round(value, 6) for key, value in normalized.items()}
        summary = distribution_summary(probs)
        top_name = summary.top_label or "unknown"
        entropy = summary.entropy
        confidence = summary.confidence
        is_mixed = top_name == "mixed" or summary.margin < ROUTING_MIXED_DOMAIN_MARGIN

        top_domain = ProblemDomain.MIXED if is_mixed and top_name != "unknown" else ProblemDomain(top_name)

        return ProblemTypePrediction(
            problem_id=problem.problem_id,
            problem_type_probs=probs,
            top_domain=top_domain,
            confidence=round(confidence, 4),
            entropy=round(entropy, 6),
            is_mixed=is_mixed,
            diagnostics={
                "features": features,
                "raw_scores": {k: round(v, 4) for k, v in raw_scores.items()},
                "reasons": reasons,
            },
        )

    def classify(self, problem: ParsedProblem) -> dict[str, float]:
        return self.predict(problem).problem_type_probs

    def top_domain(self, problem: ParsedProblem) -> ProblemDomain:
        return self.predict(problem).top_domain


_DEFAULT_CLASSIFIER = ProblemTypeClassifier()


def extract_problem_type_features(problem: ParsedProblem) -> dict[str, Any]:
    return _DEFAULT_CLASSIFIER.extract_features(problem)


def predict_problem_type(problem: ParsedProblem) -> ProblemTypePrediction:
    return _DEFAULT_CLASSIFIER.predict(problem)


def predict_problem_type_probs(problem: ParsedProblem) -> dict[str, float]:
    return _DEFAULT_CLASSIFIER.classify(problem)


def predict_top_problem_domain(problem: ParsedProblem) -> ProblemDomain:
    return _DEFAULT_CLASSIFIER.top_domain(problem)
