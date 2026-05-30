from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.common.constants import RETRIEVAL_TOP_K_EASY, RETRIEVAL_TOP_K_HARD
from src.common.schemas import RetrievedTrace, RouteDecision
from src.retrieval.embedder import MathEmbedder
from src.retrieval.index_builder import IndexedTraceRecord, RetrievalIndexBuilder
from src.retrieval.query import RetrievalQuery


def _normalize_score_dict(values: dict[str, float]) -> dict[str, float]:
    clipped = {k: max(0.0, float(v)) for k, v in values.items()}
    total = sum(clipped.values())
    if total <= 0.0:
        return {}
    return {k: v / total for k, v in clipped.items()}


@dataclass(frozen=True)
class RetrievalHit:
    trace_id: str
    problem_id: str
    score_dense: float
    score_lexical: float
    score_structural: float
    fused_score: float
    compatibility_score: float
    confidence: float
    retrieval_method: str
    domain: str
    archetypes: tuple[str, ...]
    operators_used: tuple[str, ...]
    problem: str
    solution: str
    answer: str
    source: str
    operator_support: dict[str, float] = field(default_factory=dict)
    repair_operator_hints: tuple[str, ...] = ()
    branch_continuation_bias: dict[str, float] = field(default_factory=dict)
    relevant_obligation_claims: tuple[str, ...] = ()
    relevant_evidence_kinds: tuple[str, ...] = ()
    failure_mode_support: tuple[str, ...] = ()
    compatibility_reasons: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    diagnostics: dict[str, object] = field(default_factory=dict)


class RetrievalPolicy:
    """
    Hybrid retrieval policy:
    - dense + lexical + structural fusion
    - confidence gating
    - MMR-style diversity
    - repair-time narrowing
    - operator-support shaping
    """

    def should_retrieve(self, route: RouteDecision, query: Optional[RetrievalQuery] = None) -> bool:
        if not route.use_retrieval:
            return False
        if route.difficulty_score < 0.20 and (query is None or not query.repair_mode):
            return False
        return True

    def get_top_k(self, route: RouteDecision, *, repair_mode: bool = False) -> int:
        if repair_mode:
            return 3
        return RETRIEVAL_TOP_K_HARD if route.difficulty_score >= 0.6 else RETRIEVAL_TOP_K_EASY

    def retrieve(
        self,
        *,
        query: RetrievalQuery,
        route: RouteDecision,
        index: RetrievalIndexBuilder,
        embedder: MathEmbedder,
        top_k: Optional[int] = None,
    ) -> list[RetrievalHit]:
        if not self.should_retrieve(route, query) or not index.is_loaded:
            return []

        k = top_k or self.get_top_k(route, repair_mode=query.repair_mode)
        query_text = query.lexical_text()
        candidate_budget = min(
            48,
            max(k * 4, 12, int(getattr(route, "retrieval_depth", 0) or 0) * 6),
        )
        dense_weight, lexical_weight, structural_weight = self._fusion_weights(query)

        dense_candidates = index.search_dense(embedder.embed_single(query_text), top_k=candidate_budget)
        lexical_candidates = index.search_lexical(query_text, top_k=candidate_budget)
        structural_candidates = index.search_structural(
            domain=query.domain,
            archetypes=query.archetypes,
            operators=query.operator_hints,
            tags=query.tags,
            proof_obligations=query.required_evidence_kinds,
            failure_modes=query.failure_modes,
            branch_operators=query.branch_operator_prefix,
            top_k=candidate_budget,
        )

        dense_map = {r.trace_id: (score, r) for score, r in dense_candidates}
        lexical_map = {r.trace_id: (score, r) for score, r in lexical_candidates}
        structural_map = {r.trace_id: (score, r) for score, r in structural_candidates}

        all_ids = set(dense_map) | set(lexical_map) | set(structural_map)
        raw_hits: list[RetrievalHit] = []
        min_confidence = self._minimum_confidence(query)

        for tid in sorted(all_ids):
            record = (
                dense_map.get(tid, (0.0, None))[1]
                or lexical_map.get(tid, (0.0, None))[1]
                or structural_map.get(tid, (0.0, None))[1]
            )
            assert record is not None

            score_dense = dense_map.get(tid, (0.0, record))[0]
            score_lexical = lexical_map.get(tid, (0.0, record))[0]
            score_structural = structural_map.get(tid, (0.0, record))[0]

            fused = (
                dense_weight * max(0.0, score_dense)
                + lexical_weight * max(0.0, score_lexical)
                + structural_weight * max(0.0, score_structural)
            )

            compatibility_score, compatibility_details = self._compatibility_score(record=record, query=query)

            if query.repair_mode and record.repair_snippets:
                fused += 0.08
            if query.operator_hints:
                overlap = len(set(query.operator_hints) & set(record.operators_used))
                fused += 0.04 * overlap
            if query.archetypes:
                overlap = len(set(query.archetypes) & set(record.archetypes))
                fused += 0.03 * overlap
            fused += 0.24 * compatibility_score

            confidence = self._confidence(fused, score_dense, score_lexical, score_structural, compatibility_score)
            if confidence < min_confidence:
                continue

            operator_support = self._operator_support(record, confidence, query, compatibility_details)
            continuation_bias = self._continuation_bias(record=record, query=query, confidence=confidence)
            repair_hints = self._repair_operator_hints(record=record, query=query, confidence=confidence)
            raw_hits.append(
                RetrievalHit(
                    trace_id=record.trace_id,
                    problem_id=record.problem_id,
                    score_dense=round(score_dense, 6),
                    score_lexical=round(score_lexical, 6),
                    score_structural=round(score_structural, 6),
                    fused_score=round(fused, 6),
                    compatibility_score=round(compatibility_score, 6),
                    confidence=round(confidence, 6),
                    retrieval_method="hybrid",
                    domain=record.domain,
                    archetypes=tuple(record.archetypes),
                    operators_used=tuple(record.operators_used),
                    problem=record.problem_text,
                    solution=record.solution_text,
                    answer=record.answer,
                    source=record.source,
                    operator_support=operator_support,
                    repair_operator_hints=repair_hints,
                    branch_continuation_bias=continuation_bias,
                    relevant_obligation_claims=tuple(compatibility_details["obligation_claims"]),
                    relevant_evidence_kinds=tuple(compatibility_details["evidence_kinds"]),
                    failure_mode_support=tuple(compatibility_details["failure_modes"]),
                    compatibility_reasons=tuple(compatibility_details["reasons"]),
                    tags=tuple(record.tags),
                    diagnostics={
                        "repair_mode": query.repair_mode,
                        "query_source": query.source,
                        "query_tags": list(query.tags),
                        "query_ops": list(query.operator_hints),
                        "signal_count": sum(
                            1 for value in (score_dense, score_lexical, score_structural) if value > 0.0
                        ),
                        "matched_operator_overlap": len(set(query.operator_hints) & set(record.operators_used)),
                        "matched_archetype_overlap": len(set(query.archetypes) & set(record.archetypes)),
                        "matched_obligation_overlap": compatibility_details["obligation_overlap"],
                        "matched_failure_overlap": compatibility_details["failure_overlap"],
                        "branch_operator_overlap": compatibility_details["continuation_overlap"],
                        "compatibility_score": round(compatibility_score, 6),
                    },
                )
            )

        return self._mmr_diversify(raw_hits, top_k=k)

    def operator_support_signal(self, hits: list[RetrievalHit]) -> dict[str, float]:
        accum: dict[str, float] = {}
        for hit in hits:
            for op, score in hit.operator_support.items():
                accum[op] = accum.get(op, 0.0) + score
        return _normalize_score_dict(accum)

    def compute_prior_adjustment(
        self,
        retrieved_traces: list[RetrievedTrace],
        current_prior: dict[str, float],
        blend_weight: float = 0.30,
    ) -> dict[str, float]:
        if not retrieved_traces:
            return current_prior

        op_counts: dict[str, float] = {}
        total = 0.0
        for trace in retrieved_traces:
            direct_support = getattr(trace, "operator_support", None)
            if isinstance(direct_support, dict) and direct_support:
                for op, weight in direct_support.items():
                    score = max(0.0, float(weight))
                    if score <= 0.0:
                        continue
                    op_counts[op] = op_counts.get(op, 0.0) + score
                    total += score
                continuation = getattr(trace, "branch_continuation_bias", None)
                if isinstance(continuation, dict):
                    for op, weight in continuation.items():
                        score = max(0.0, 0.45 * float(weight))
                        if score <= 0.0:
                            continue
                        op_counts[op] = op_counts.get(op, 0.0) + score
                        total += score
                continue
            similarity = max(0.0, float(getattr(trace, "similarity_score", 0.0)))
            for op in getattr(trace, "operators_used", []):
                op_counts[op] = op_counts.get(op, 0.0) + similarity
                total += similarity

        if total <= 0.0:
            return current_prior

        retrieval_signal = {op: count / total for op, count in op_counts.items()}
        all_ops = set(current_prior) | set(retrieval_signal)
        blended = {}
        for op in all_ops:
            curr = current_prior.get(op, 0.0)
            ret = retrieval_signal.get(op, 0.0)
            blended[op] = (1.0 - blend_weight) * curr + blend_weight * ret

        s = sum(blended.values())
        if s > 0:
            blended = {k: v / s for k, v in blended.items()}
        return blended

    def hit_to_trace(self, hit: RetrievalHit) -> RetrievedTrace:
        return RetrievedTrace(
            trace_id=hit.trace_id,
            problem=hit.problem,
            solution=hit.solution,
            answer=hit.answer,
            domain=hit.domain,
            archetypes=list(hit.archetypes),
            operators_used=list(hit.operators_used),
            similarity_score=hit.confidence,
            source=hit.source,
            operator_support=dict(hit.operator_support),
            repair_operator_hints=list(hit.repair_operator_hints),
            branch_continuation_bias=dict(hit.branch_continuation_bias),
            compatibility_score=hit.compatibility_score,
            compatibility_reasons=list(hit.compatibility_reasons),
            relevant_obligation_claims=list(hit.relevant_obligation_claims),
            relevant_evidence_kinds=list(hit.relevant_evidence_kinds),
            failure_mode_support=list(hit.failure_mode_support),
            strategy_summary=self._strategy_summary(hit),
            diagnostics=dict(hit.diagnostics),
        )

    def select_best_trace_for_conditioning(
        self,
        hits_or_traces,
        min_similarity: float = 0.70,
    ):
        if not hits_or_traces:
            return None
        first = hits_or_traces[0]
        if isinstance(first, RetrievalHit):
            ranked = sorted(
                hits_or_traces,
                key=lambda h: (
                    -(0.65 * h.confidence + 0.35 * h.compatibility_score),
                    -h.fused_score,
                    h.trace_id,
                ),
            )
            good = [h for h in ranked if h.confidence >= min_similarity]
            return (good or ranked)[0]
        ranked_traces = sorted(
            hits_or_traces,
            key=lambda t: (
                -(0.60 * float(getattr(t, "similarity_score", 0.0)) + 0.40 * float(getattr(t, "compatibility_score", 0.0))),
                str(getattr(t, "trace_id", "")),
            ),
        )
        good = [t for t in ranked_traces if t.similarity_score >= min_similarity]
        good = good or ranked_traces[:1]
        good_with_solution = [t for t in good if len(t.solution) > 100]
        return good_with_solution[0] if good_with_solution else good[0]

    @staticmethod
    def _confidence(fused: float, dense: float, lexical: float, structural: float, compatibility: float) -> float:
        active = sum(1 for x in (dense, lexical, structural) if x > 0.0)
        agreement_bonus = 0.05 * max(0, active - 1)
        if dense >= 0.18 and structural >= 0.20:
            agreement_bonus += 0.03
        if lexical >= 0.05 and structural >= 0.20:
            agreement_bonus += 0.02
        agreement_bonus += 0.10 * compatibility
        return max(0.0, min(1.0, fused + agreement_bonus))

    @staticmethod
    def _operator_support(
        record: IndexedTraceRecord,
        confidence: float,
        query: RetrievalQuery,
        compatibility_details: dict[str, object],
    ) -> dict[str, float]:
        if not record.operators_used:
            return {}
        hint_set = set(query.operator_hints)
        continuation_set = set(query.branch_operator_prefix)
        repair_neighbors = set(record.repair_neighbors)
        weights: dict[str, float] = {}
        for idx, op in enumerate(record.operators_used[:6]):
            position_weight = 1.0 / (1.0 + idx)
            overlap_bonus = 0.35 if op in hint_set else 0.0
            continuation_bonus = 0.20 if op in continuation_set else 0.0
            repair_bonus = 0.24 if query.repair_mode and op in repair_neighbors else 0.0
            weights[op] = position_weight + overlap_bonus + continuation_bonus + repair_bonus
        total = sum(weights.values()) or 1.0
        compatibility_boost = 1.0 + 0.35 * float(compatibility_details.get("score", 0.0) or 0.0)
        return {op: round(confidence * compatibility_boost * (weight / total), 6) for op, weight in weights.items()}

    def _mmr_diversify(self, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        if len(hits) <= top_k:
            return sorted(hits, key=lambda h: (-h.confidence, -h.compatibility_score, -h.fused_score, h.trace_id))

        remaining = sorted(hits, key=lambda h: (-h.confidence, -h.compatibility_score, -h.fused_score, h.trace_id))
        selected: list[RetrievalHit] = [remaining.pop(0)]

        def sim(a: RetrievalHit, b: RetrievalHit) -> float:
            a_set = set(a.tags) | set(a.archetypes) | set(a.operators_used)
            b_set = set(b.tags) | set(b.archetypes) | set(b.operators_used)
            if not a_set and not b_set:
                return 0.0
            return len(a_set & b_set) / max(len(a_set | b_set), 1)

        while remaining and len(selected) < top_k:
            best_idx = 0
            best_score = -1e9
            for idx, cand in enumerate(remaining):
                novelty_penalty = max(sim(cand, s) for s in selected) if selected else 0.0
                mmr = 0.60 * cand.confidence + 0.20 * cand.compatibility_score - 0.20 * novelty_penalty
                if mmr > best_score:
                    best_score = mmr
                    best_idx = idx
            selected.append(remaining.pop(best_idx))

        return selected

    @staticmethod
    def _fusion_weights(query: RetrievalQuery) -> tuple[float, float, float]:
        dense_weight = 0.42
        lexical_weight = 0.23
        structural_weight = 0.35
        if query.repair_mode:
            return 0.28, 0.30, 0.42
        if query.operator_hints:
            dense_weight -= 0.02
            structural_weight += 0.02
        return dense_weight, lexical_weight, structural_weight

    @staticmethod
    def _minimum_confidence(query: RetrievalQuery) -> float:
        return 0.12 if query.repair_mode else 0.10

    @staticmethod
    def _compatibility_score(
        *,
        record: IndexedTraceRecord,
        query: RetrievalQuery,
    ) -> tuple[float, dict[str, object]]:
        record_arch = set(record.archetypes)
        record_ops = set(record.operators_used)
        record_tags = set(record.tags)
        query_arch = set(query.archetypes)
        query_ops = set(query.operator_hints)
        query_tags = set(query.tags)
        query_failure = {item for item in query.failure_modes if item}
        record_failure = set(record.failure_modes)
        query_evidence = {item for item in query.required_evidence_kinds if item}
        record_obligations = set(record.proof_obligation_hints)
        branch_prefix = {item for item in query.branch_operator_prefix if item}

        structure_overlap = (
            0.55 * (len(query_arch & record_arch) / max(len(query_arch | record_arch), 1))
            + 0.45 * (len(query_tags & record_tags) / max(len(query_tags | record_tags), 1))
        ) if (query_arch or query_tags) else 0.0
        operator_overlap = len(query_ops & record_ops) / max(len(query_ops | record_ops), 1) if (query_ops or record_ops) else 0.0
        continuation_overlap = len(branch_prefix & set(record.continuation_operators or record.operators_used)) / max(
            len(branch_prefix | set(record.continuation_operators or record.operators_used)),
            1,
        ) if branch_prefix else 0.0
        failure_overlap = len(query_failure & record_failure) / max(len(query_failure | record_failure), 1) if query_failure else 0.0
        obligation_overlap = len(query_evidence & record_obligations) / max(len(query_evidence | record_obligations), 1) if query_evidence else 0.0
        repair_overlap = len(set(record.repair_neighbors) & query_ops) / max(len(set(record.repair_neighbors) | query_ops), 1) if query.repair_mode and query_ops else 0.0
        branch_phase_bonus = 0.12 if query.repair_mode and record.repair_snippets else 0.0

        score = max(
            0.0,
            min(
                1.0,
                0.26 * structure_overlap
                + 0.22 * operator_overlap
                + 0.18 * continuation_overlap
                + 0.16 * obligation_overlap
                + 0.10 * failure_overlap
                + 0.08 * repair_overlap
                + branch_phase_bonus,
            ),
        )

        reasons: list[str] = []
        if structure_overlap > 0.0:
            reasons.append("problem_structure_match")
        if operator_overlap > 0.0:
            reasons.append("operator_pattern_support")
        if continuation_overlap > 0.0:
            reasons.append("branch_prefix_compatible")
        if obligation_overlap > 0.0:
            reasons.append("proof_obligation_support")
        if failure_overlap > 0.0:
            reasons.append("failure_context_support")
        if repair_overlap > 0.0:
            reasons.append("repair_neighbor_support")
        if query.repair_mode and record.repair_snippets:
            reasons.append("repair_trace")

        obligation_claims = [
            item for item in record.proof_obligation_hints
            if item and not item.startswith("status::") and item not in query.required_evidence_kinds
        ][:3]
        evidence_kinds = [item for item in record.proof_obligation_hints if item in query.required_evidence_kinds][:3]
        failure_modes = [item for item in record.failure_modes if item in query.failure_modes][:3]

        return score, {
            "score": score,
            "reasons": reasons,
            "obligation_overlap": len(query_evidence & record_obligations),
            "failure_overlap": len(query_failure & record_failure),
            "continuation_overlap": len(branch_prefix & set(record.continuation_operators or record.operators_used)),
            "obligation_claims": obligation_claims,
            "evidence_kinds": evidence_kinds or list(query.required_evidence_kinds[:2]),
            "failure_modes": failure_modes,
        }

    @staticmethod
    def _continuation_bias(
        *,
        record: IndexedTraceRecord,
        query: RetrievalQuery,
        confidence: float,
    ) -> dict[str, float]:
        ordered = list(record.continuation_operators or record.operators_used)
        if not ordered:
            return {}
        weights: dict[str, float] = {}
        prefix = set(query.branch_operator_prefix)
        total = 0.0
        for index, operator_name in enumerate(ordered[:4]):
            weight = (1.0 / (index + 1)) + (0.20 if operator_name in prefix else 0.0)
            weights[operator_name] = weight
            total += weight
        if total <= 0.0:
            return {}
        return {name: round(confidence * (value / total), 6) for name, value in weights.items()}

    @staticmethod
    def _repair_operator_hints(
        *,
        record: IndexedTraceRecord,
        query: RetrievalQuery,
        confidence: float,
    ) -> tuple[str, ...]:
        hints: list[str] = []
        if query.repair_mode:
            hints.extend(record.repair_neighbors[:3])
        for operator_name in record.operators_used[:4]:
            if operator_name not in hints:
                hints.append(operator_name)
        return tuple(hints[:4])

    @staticmethod
    def _strategy_summary(hit: RetrievalHit) -> str:
        parts: list[str] = []
        if hit.compatibility_reasons:
            parts.append("why=" + ",".join(hit.compatibility_reasons[:3]))
        if hit.operator_support:
            top_ops = sorted(hit.operator_support.items(), key=lambda kv: (-float(kv[1]), kv[0]))[:2]
            parts.append("ops=" + "->".join(name for name, _ in top_ops))
        if hit.relevant_evidence_kinds:
            parts.append("evidence=" + ",".join(hit.relevant_evidence_kinds[:2]))
        if hit.failure_mode_support:
            parts.append("repair=" + ",".join(hit.failure_mode_support[:2]))
        return " | ".join(parts)
