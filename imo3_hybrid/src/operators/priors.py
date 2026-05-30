from __future__ import annotations

import math
from typing import Any, Iterable

from src.common.constants import (
    ARCHETYPE_OPERATOR_PRIORS,
    DEFAULT_OPERATOR_PRIOR,
    DOMAIN_OPERATOR_PRIORS,
)
from src.common.schemas import RetrievedTrace, RouteDecision
from src.state_graph.node import ReasoningStateNode

from .operator_library import OperatorLibrary
from .operator_types import (
    ApplicabilityStatus,
    OperatorFamily,
    OperatorPolicyScore,
)


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    clean = {k: max(0.0, float(v)) for k, v in scores.items()}
    total = sum(clean.values())
    if total <= 0:
        n = max(len(clean), 1)
        return {k: 1.0 / n for k in sorted(clean)}
    return {k: clean[k] / total for k in sorted(clean)}


def _entropy(distribution: dict[str, float]) -> float:
    probs = [p for p in distribution.values() if p > 0]
    if len(probs) <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in probs)
    return h / math.log(len(probs))


class OperatorPriorShaper:
    """
    Route + retrieval + state fused operator prior shaping.
    """

    def __init__(self, library: OperatorLibrary | None = None) -> None:
        self.library = library or OperatorLibrary()
        self.family_domain_weights: dict[str, dict[OperatorFamily, float]] = {
            "algebra": {
                OperatorFamily.ALGEBRAIC: 1.15,
                OperatorFamily.STRUCTURAL: 1.00,
                OperatorFamily.VERIFICATION: 0.90,
            },
            "number_theory": {
                OperatorFamily.NUMBER_THEORETIC: 1.20,
                OperatorFamily.ALGEBRAIC: 0.95,
                OperatorFamily.VERIFICATION: 1.00,
            },
            "combinatorics": {
                OperatorFamily.COMBINATORIAL: 1.20,
                OperatorFamily.STRUCTURAL: 1.05,
                OperatorFamily.SEARCH: 1.00,
            },
            "geometry": {
                OperatorFamily.GEOMETRIC: 1.25,
                OperatorFamily.STRUCTURAL: 0.95,
                OperatorFamily.ALGEBRAIC: 0.90,
            },
            "mixed": {
                OperatorFamily.STRUCTURAL: 1.05,
                OperatorFamily.ALGEBRAIC: 1.00,
                OperatorFamily.COMBINATORIAL: 1.00,
                OperatorFamily.NUMBER_THEORETIC: 1.00,
            },
        }

    def shape_priors(
        self,
        *,
        node: ReasoningStateNode,
        route: RouteDecision,
        retrieved_operator_hints: dict[str, float] | None = None,
        retrieved_traces: Iterable[RetrievedTrace] | None = None,
        retrieval_context: dict[str, Any] | None = None,
        available_tools: Iterable[str] | None = None,
    ) -> tuple[dict[str, float], list[OperatorPolicyScore]]:
        base_route = self._route_prior(route)
        retrieval = _normalize(dict(retrieved_operator_hints or {})) if retrieved_operator_hints else {}
        resolved_context = self._retrieval_context(retrieved_traces=retrieved_traces, retrieval_context=retrieval_context)
        uncertainty = self._routing_uncertainty(route)

        policy_rows: list[OperatorPolicyScore] = []
        raw_scores: dict[str, float] = {}

        for name in self.library.all_operator_names():
            desc = self.library.get(name)
            assert desc is not None

            applicability = self.library.check_applicability(
                name,
                node,
                route,
                available_tools=available_tools,
                retrieval_context=resolved_context,
            )
            stats = self.library.usage_stats(name)

            base_route_score = base_route.get(name, 0.0)
            family_score = self._family_weight(desc.family, route)
            applicability_score = self._applicability_weight(applicability.status, applicability.score)
            retrieval_score = min(
                1.0,
                0.75 * retrieval.get(name, 0.0) + 0.25 * applicability.retrieval_compatibility_score,
            )
            success_history_score = self._history_weight(stats)
            uncertainty_smoothing = self._uncertainty_mix(uncertainty)

            final_score = (
                0.38 * base_route_score
                + 0.14 * family_score
                + 0.26 * applicability_score
                + 0.10 * retrieval_score
                + 0.12 * success_history_score
            )

            final_score = (
                (1.0 - uncertainty_smoothing) * final_score
                + uncertainty_smoothing * self._default_prior().get(name, 0.0)
            )

            if applicability.status is ApplicabilityStatus.INAPPLICABLE:
                final_score *= 0.05

            raw_scores[name] = max(0.0, final_score)
            policy_rows.append(
                OperatorPolicyScore(
                    operator_name=name,
                    family=desc.family,
                    base_route_score=round(base_route_score, 6),
                    family_score=round(family_score, 6),
                    applicability_score=round(applicability_score, 6),
                    retrieval_score=round(retrieval_score, 6),
                    success_history_score=round(success_history_score, 6),
                    uncertainty_smoothing=round(uncertainty_smoothing, 6),
                    final_score=round(final_score, 6),
                    diagnostics={
                        "applicability_status": applicability.status.value,
                        "applicability_passed": applicability.passed_conditions,
                        "applicability_failed": applicability.failed_conditions,
                        "warnings": applicability.warnings,
                        "retrieval_compatibility_score": applicability.retrieval_compatibility_score,
                        "retrieval_signals": applicability.compatible_retrieval_signals,
                    },
                )
            )

        normalized = _normalize(raw_scores)
        row_by_name = {row.operator_name: row for row in policy_rows}
        for name, prob in normalized.items():
            row_by_name[name].normalized_score = round(prob, 6)

        ordered_rows = sorted(
            policy_rows,
            key=lambda row: (-row.normalized_score, -row.final_score, row.operator_name),
        )
        return normalized, ordered_rows

    def top_k(
        self,
        *,
        node: ReasoningStateNode,
        route: RouteDecision,
        retrieved_operator_hints: dict[str, float] | None = None,
        retrieved_traces: Iterable[RetrievedTrace] | None = None,
        retrieval_context: dict[str, Any] | None = None,
        available_tools: Iterable[str] | None = None,
        k: int = 8,
    ) -> list[OperatorPolicyScore]:
        _, rows = self.shape_priors(
            node=node,
            route=route,
            retrieved_operator_hints=retrieved_operator_hints,
            retrieved_traces=retrieved_traces,
            retrieval_context=retrieval_context,
            available_tools=available_tools,
        )
        return rows[:k]

    @staticmethod
    def _retrieval_context(
        *,
        retrieved_traces: Iterable[RetrievedTrace] | None,
        retrieval_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        resolved = dict(retrieval_context or {})
        traces = list(retrieved_traces or ())
        if not traces:
            return resolved

        def _merge_unique(key: str, values: list[str], limit: int) -> None:
            existing = [str(item) for item in list(resolved.get(key, []) or []) if str(item).strip()]
            for item in values:
                token = str(item).strip()
                if token and token not in existing:
                    existing.append(token)
                if len(existing) >= limit:
                    break
            resolved[key] = existing[:limit]

        _merge_unique("archetypes", [item for trace in traces for item in trace.archetypes], 6)
        _merge_unique("evidence_kinds", [item for trace in traces for item in getattr(trace, "relevant_evidence_kinds", [])], 4)
        _merge_unique("failure_modes", [item for trace in traces for item in getattr(trace, "failure_mode_support", [])], 4)
        _merge_unique("tags", [item for trace in traces for item in getattr(trace, "compatibility_reasons", [])], 6)
        _merge_unique("repair_operators", [item for trace in traces for item in getattr(trace, "repair_operator_hints", [])], 4)
        _merge_unique("operator_prefix", [item for trace in traces for item in getattr(trace, "operators_used", [])[:2]], 4)
        return resolved

    def _route_prior(self, route: RouteDecision) -> dict[str, float]:
        route_prior = dict(getattr(route, "operator_prior", {}) or {})
        if route_prior:
            return _normalize(route_prior)

        blended = dict(DEFAULT_OPERATOR_PRIOR)
        for domain, p_domain in (getattr(route, "problem_type", {}) or {}).items():
            for op, p_op in DOMAIN_OPERATOR_PRIORS.get(domain, {}).items():
                blended[op] = blended.get(op, 0.0) + 0.40 * p_domain * p_op

        for archetype, p_arch in (getattr(route, "archetypes", {}) or {}).items():
            for op, p_op in ARCHETYPE_OPERATOR_PRIORS.get(archetype, {}).items():
                blended[op] = blended.get(op, 0.0) + 0.60 * p_arch * p_op

        return _normalize(blended)

    def _default_prior(self) -> dict[str, float]:
        return _normalize(dict(DEFAULT_OPERATOR_PRIOR))

    def _family_weight(self, family: OperatorFamily, route: RouteDecision) -> float:
        top_domain = self._top_domain(route)
        domain_table = self.family_domain_weights.get(top_domain or "mixed", {})
        return float(domain_table.get(family, 1.0))

    @staticmethod
    def _applicability_weight(status: ApplicabilityStatus, score: float) -> float:
        if status is ApplicabilityStatus.APPLICABLE:
            return 0.60 + 0.40 * score
        if status is ApplicabilityStatus.WEAK:
            return 0.20 + 0.50 * score
        return 0.02

    @staticmethod
    def _history_weight(stats: Any) -> float:
        success = float(getattr(stats, "success_rate", 0.5))
        recent = float(getattr(stats, "recent_success_rate", 0.5))
        symbolic = float(getattr(stats, "symbolic_success_rate", 0.5))
        gain = float(getattr(stats, "verifier_gain_mean", 0.0))
        gain_term = max(0.0, min(1.0, 0.5 + gain))
        return 0.35 * success + 0.30 * recent + 0.20 * symbolic + 0.15 * gain_term

    @staticmethod
    def _routing_uncertainty(route: RouteDecision) -> float:
        domain_h = _entropy(dict(getattr(route, "problem_type", {}) or {}))
        arch_h = _entropy(dict(getattr(route, "archetypes", {}) or {}))
        return max(domain_h, arch_h)

    @staticmethod
    def _uncertainty_mix(uncertainty: float) -> float:
        return max(0.0, min(0.35, 0.35 * uncertainty))

    @staticmethod
    def _top_domain(route: RouteDecision) -> str | None:
        distribution = getattr(route, "problem_type", {}) or {}
        if not distribution:
            return None
        return sorted(distribution.items(), key=lambda kv: (-float(kv[1]), kv[0]))[0][0]


__all__ = ["OperatorPriorShaper"]
