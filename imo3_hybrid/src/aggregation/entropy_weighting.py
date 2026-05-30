# src/aggregation/entropy_weighting.py
"""Entropy-aware evidence fusion over canonical answer clusters.

This module preserves the integrated public contract expected by the
controller/final-selector stack while merging in richer evidence-aware
scoring from the repair pass.

Preferred path:
    canonicalization -> clustering -> entropy weighting -> final selector

Compatibility path:
    candidate answers / branch traces can still be scored, but that path is
    weaker and only retained as a bounded fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from src.aggregation.clustering import AnswerCluster, ClusteringResult
from src.common.constants import (
    W_ANSWER_AGREEMENT,
    W_BRANCH_NOVELTY,
    W_SYMBOLIC_CHECK,
    W_TOOL_CONSISTENCY,
    W_VERIFIER,
)
from src.common.schemas import BranchTrace, CandidateAnswer


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _normalize_entropy(probabilities: Sequence[float]) -> float:
    probs = [max(0.0, float(p)) for p in probabilities if float(p) > 0.0]
    if not probs:
        return 0.0
    total = sum(probs)
    if total <= 0.0:
        return 0.0
    normalized = [p / total for p in probs]
    raw = -sum(p * math.log(p, 2) for p in normalized if p > 0.0)
    max_entropy = math.log(len(normalized), 2) if len(normalized) > 1 else 1.0
    if max_entropy <= 0.0:
        return 0.0
    return _clamp01(raw / max_entropy)


@dataclass(frozen=True)
class EntropyWeightingConfig:
    verifier_weight: float = W_VERIFIER
    weak_symbolic_weight: float = W_TOOL_CONSISTENCY
    agreement_weight: float = W_ANSWER_AGREEMENT
    novelty_weight: float = W_BRANCH_NOVELTY
    exact_symbolic_weight: float = W_SYMBOLIC_CHECK

    retrieval_weight: float = 0.12
    process_quality_weight: float = 0.14
    retrieval_compatibility_weight: float = 0.08
    operator_reliability_weight: float = 0.08
    critique_support_weight: float = 0.08
    exact_symbolic_dominance_weight: float = 0.18
    symbolic_separation_weight: float = 0.08

    uncertainty_penalty_weight: float = 0.18
    outlier_penalty_weight: float = 0.12
    obligation_penalty_weight: float = 0.12


@dataclass(frozen=True)
class BranchEvidence:
    branch_id: str
    answer_canonical: str
    verifier_score: float
    logical_consistency: float
    completeness: float
    prefix_quality: float
    prm_prefix_quality: float
    weak_symbolic_support: float
    exact_symbolic_support: float
    retrieval_support: float
    retrieval_compatibility: float
    operator_reliability: float
    novelty: float
    repair_pressure: float
    open_obligation_burden: float
    discharge_fraction: float
    critique_count: int
    critique_adjustment: float
    confidence: float
    support_score: float


@dataclass(frozen=True)
class AnswerClusterInput:
    answer: str
    answer_canonical: str
    branches: tuple[BranchTrace, ...]


@dataclass(frozen=True)
class WeightedClusterScore:
    cluster_id: str
    answer: str
    answer_canonical: str
    branch_ids: tuple[str, ...]
    cluster_size: int

    support_share: float
    diversity_weighted_support: float
    unique_branch_families: int
    family_diversity: float

    verifier_score: float
    weak_symbolic_support: float
    exact_symbolic_support: float
    retrieval_support: float
    branch_novelty: float
    weak_symbolic_only_support: float
    symbolic_dominance_bonus: float

    uncertainty: float
    entropy_penalty: float
    outlier_penalty: float
    composite_score: float

    mean_provenance_strength: float = 0.0
    mean_retrieval_relevance: float = 0.0
    mean_evidence_quality: float = 0.0
    best_branch_id: str | None = None
    best_member_score: float = 0.0
    answer_variants: tuple[str, ...] = ()
    caution_flags: tuple[str, ...] = ()
    decomposed_signals: Mapping[str, float] | None = None

    def as_candidate_answer(self) -> CandidateAnswer:
        return CandidateAnswer(
            answer=self.answer,
            answer_canonical=self.answer_canonical,
            branch_ids=list(self.branch_ids),
            verifier_score=self.verifier_score,
            tool_consistency=self.weak_symbolic_support,
            answer_agreement=self.diversity_weighted_support,
            branch_novelty=self.branch_novelty,
            symbolic_check=self.exact_symbolic_support,
            composite_score=self.composite_score,
            cluster_size=self.cluster_size,
            entropy_penalty=self.entropy_penalty,
        )


@dataclass(frozen=True)
class EntropyWeightingBundle:
    winner: WeightedClusterScore | None
    ranked_clusters: tuple[WeightedClusterScore, ...]
    ranked_candidates: tuple[CandidateAnswer, ...]
    global_entropy: float
    total_branches: int


def _extract_support_float(trace: BranchTrace, attribute_names: Sequence[str]) -> float | None:
    metadata = getattr(trace, "metadata", None)
    for name in attribute_names:
        raw = getattr(trace, name, None)
        if isinstance(raw, (int, float)):
            return _clamp01(raw)

        if isinstance(metadata, Mapping):
            sources = (
                metadata,
                metadata.get("signal_decomposition", {}) if isinstance(metadata.get("signal_decomposition"), Mapping) else {},
                metadata.get("verifier_decomposition", {}) if isinstance(metadata.get("verifier_decomposition"), Mapping) else {},
                metadata.get("proof_state", {}) if isinstance(metadata.get("proof_state"), Mapping) else {},
                metadata.get("retrieval", {}) if isinstance(metadata.get("retrieval"), Mapping) else {},
                metadata.get("operator", {}) if isinstance(metadata.get("operator"), Mapping) else {},
                metadata.get("self_critique", {}).get("trigger_signals", {})
                if isinstance(metadata.get("self_critique"), Mapping)
                else {},
            )
            for source in sources:
                raw = source.get(name) if isinstance(source, Mapping) else None
                if isinstance(raw, (int, float)):
                    return _clamp01(raw)
    return None


def _retrieval_support(trace: BranchTrace) -> float:
    explicit = _extract_support_float(trace, ("retrieval_support", "retrieval_score"))
    if explicit is not None:
        return explicit
    return 1.0 if bool(getattr(trace, "retrieval_used", False)) else 0.0


def _retrieval_compatibility(trace: BranchTrace) -> float:
    explicit = _extract_support_float(trace, ("retrieval_compatibility", "retrieval_support", "retrieval_score"))
    if explicit is not None:
        return explicit
    return _retrieval_support(trace)


def _exact_symbolic_support(trace: BranchTrace) -> float:
    explicit = _extract_support_float(trace, ("exact_symbolic_support", "exact_symbolic_check"))
    if explicit is not None:
        return explicit
    return 1.0 if bool(getattr(trace, "symbolic_valid", False)) else 0.0


def _tool_consistency(trace: BranchTrace) -> float:
    explicit = _extract_support_float(trace, ("tool_consistency",))
    if explicit is not None:
        return explicit
    return 1.0 if bool(getattr(trace, "symbolic_valid", False)) else 0.0


def _branch_novelty(trace: BranchTrace) -> float:
    explicit = _extract_support_float(trace, ("branch_novelty", "novelty"))
    if explicit is not None:
        return explicit
    sequence = tuple(str(item).strip() for item in getattr(trace, "operator_sequence", ()) if str(item).strip())
    if not sequence:
        return 0.0
    unique_ratio = _safe_ratio(len(set(sequence)), len(sequence))
    repetition_penalty = _safe_ratio(sum(1 for item in sequence if sequence.count(item) > 1), len(sequence))
    critique_bonus = 0.10 if bool(getattr(trace, "self_critiqued", False)) else 0.0
    repair_bonus = 0.08 if bool(getattr(trace, "repaired", False)) else 0.0
    retrieval_bonus = 0.07 if bool(getattr(trace, "retrieval_used", False)) else 0.0
    return _clamp01(0.65 * unique_ratio + 0.20 * (1.0 - repetition_penalty) + critique_bonus + repair_bonus + retrieval_bonus)


def _repair_pressure(trace: BranchTrace) -> float:
    explicit = _extract_support_float(trace, ("repair_pressure",))
    if explicit is not None:
        return explicit
    return _clamp01(_safe_ratio(max(0, int(getattr(trace, "repair_count", 0) or 0)), 3.0))


def _operator_reliability(trace: BranchTrace) -> float:
    explicit = _extract_support_float(trace, ("operator_reliability",))
    if explicit is not None:
        return explicit
    return _extract_support_float(trace, ("provenance_strength", "confidence")) or 0.0


def _logical_consistency(trace: BranchTrace) -> float:
    return _extract_support_float(trace, ("logical_consistency",)) or _clamp01(getattr(trace, "verifier_score", 0.0) or 0.0)


def _completeness(trace: BranchTrace) -> float:
    return _extract_support_float(trace, ("completeness",)) or 0.0


def _prefix_quality(trace: BranchTrace) -> float:
    return _extract_support_float(trace, ("prefix_quality",)) or 0.0


def _prm_prefix_quality(trace: BranchTrace) -> float:
    return _extract_support_float(trace, ("prm_prefix_quality", "prefix_quality", "step_quality")) or 0.0


def _open_obligation_burden(trace: BranchTrace) -> float:
    return _extract_support_float(trace, ("open_obligation_burden",)) or 0.0


def _discharge_fraction(trace: BranchTrace) -> float:
    explicit = _extract_support_float(trace, ("discharge_fraction",))
    if explicit is not None:
        return explicit
    return _exact_symbolic_support(trace)


def _critique_count(trace: BranchTrace) -> int:
    explicit = _extract_support_float(trace, ("critique_count",))
    if explicit is not None:
        return int(round(explicit))
    metadata = getattr(trace, "metadata", None)
    if isinstance(metadata, Mapping):
        critique = metadata.get("self_critique")
        if isinstance(critique, Mapping):
            issue_categories = critique.get("issue_categories")
            if isinstance(issue_categories, Sequence) and not isinstance(issue_categories, (str, bytes)):
                return len(issue_categories)
    return 1 if bool(getattr(trace, "self_critiqued", False)) else 0


def _critique_adjustment(trace: BranchTrace) -> float:
    explicit = _extract_support_float(trace, ("critique_adjustment",))
    if explicit is not None:
        return explicit
    metadata = getattr(trace, "metadata", None)
    if isinstance(metadata, Mapping):
        critique = metadata.get("self_critique")
        if isinstance(critique, Mapping):
            confidence_delta = critique.get("confidence_delta")
            if isinstance(confidence_delta, Mapping):
                delta = confidence_delta.get("delta")
                if isinstance(delta, (int, float)):
                    return _clamp01(max(0.0, float(delta) + 0.5) / 1.5)
    return 0.08 if bool(getattr(trace, "self_critiqued", False)) else 0.0


def _branch_confidence(trace: BranchTrace) -> float:
    return _clamp01(max(getattr(trace, "branch_score", 0.0) or 0.0, getattr(trace, "verifier_score", 0.0) or 0.0))


def _branch_support_score(trace: BranchTrace) -> float:
    verifier = _clamp01(getattr(trace, "verifier_score", 0.0) or 0.0)
    logical = _logical_consistency(trace)
    completeness = _completeness(trace)
    prefix = _prefix_quality(trace)
    prm_prefix = _prm_prefix_quality(trace)
    weak_symbolic = _tool_consistency(trace)
    retrieval = _retrieval_support(trace)
    retrieval_compat = _retrieval_compatibility(trace)
    novelty = _branch_novelty(trace)
    exact_symbolic = _exact_symbolic_support(trace)
    operator_rel = _operator_reliability(trace)
    discharge = _discharge_fraction(trace)
    critique_adj = _critique_adjustment(trace)
    obligation_penalty = 0.08 * _open_obligation_burden(trace)
    repair_penalty = 0.05 * _repair_pressure(trace)
    return _clamp01(
        0.18 * verifier
        + 0.10 * logical
        + 0.06 * completeness
        + 0.06 * prefix
        + 0.08 * prm_prefix
        + 0.14 * weak_symbolic
        + 0.08 * retrieval
        + 0.06 * retrieval_compat
        + 0.08 * novelty
        + 0.10 * exact_symbolic
        + 0.04 * operator_rel
        + 0.04 * discharge
        + 0.04 * critique_adj
        - obligation_penalty
        - repair_penalty
    )


def cluster_traces_by_answer(traces: Sequence[BranchTrace]) -> tuple[AnswerClusterInput, ...]:
    """Compatibility fallback only.

    This is not the preferred path. When used, it remains canonical-aware
    and prefers answer_canonical over raw answer text.
    """
    groups: dict[str, list[BranchTrace]] = {}
    for trace in traces:
        answer = str(getattr(trace, "answer_canonical", "") or getattr(trace, "answer", "") or "").strip()
        if not answer:
            continue
        groups.setdefault(answer, []).append(trace)
    return tuple(
        AnswerClusterInput(answer=answer, answer_canonical=answer, branches=tuple(branches))
        for answer, branches in sorted(groups.items())
    )


def _coerce_answer_clusters(
    clusters_or_traces: Sequence[object] | Mapping[str, Sequence[BranchTrace]] | ClusteringResult,
) -> tuple[AnswerCluster, ...] | None:
    if isinstance(clusters_or_traces, ClusteringResult):
        return tuple(clusters_or_traces.clusters)
    if isinstance(clusters_or_traces, Mapping):
        return None
    seq = tuple(clusters_or_traces)
    if not seq:
        return ()
    return tuple(seq) if isinstance(seq[0], AnswerCluster) else None


def _coerce_candidate_answers(
    clusters_or_traces: Sequence[object] | Mapping[str, Sequence[BranchTrace]] | ClusteringResult,
) -> tuple[CandidateAnswer, ...] | None:
    if isinstance(clusters_or_traces, (ClusteringResult, Mapping)):
        return None
    seq = tuple(clusters_or_traces)
    if not seq:
        return ()
    return tuple(seq) if isinstance(seq[0], CandidateAnswer) else None


def _coerce_cluster_inputs(
    clusters_or_traces: Sequence[object] | Mapping[str, Sequence[BranchTrace]] | ClusteringResult,
) -> tuple[AnswerClusterInput, ...]:
    if isinstance(clusters_or_traces, Mapping):
        return tuple(
            AnswerClusterInput(answer=key, answer_canonical=key, branches=tuple(value))
            for key, value in sorted(clusters_or_traces.items())
        )
    if isinstance(clusters_or_traces, ClusteringResult):
        return ()
    seq = tuple(clusters_or_traces)
    if not seq:
        return ()
    if isinstance(seq[0], AnswerClusterInput):
        return tuple(seq)
    if isinstance(seq[0], BranchTrace):
        return cluster_traces_by_answer(seq)  # type: ignore[arg-type]
    return ()


def _candidate_support_weight(candidate: CandidateAnswer) -> int:
    return max(1, int(getattr(candidate, "cluster_size", 1) or 1), len(getattr(candidate, "branch_ids", []) or []))


def _rank_weighted_clusters(clusters: Sequence[WeightedClusterScore]) -> tuple[WeightedClusterScore, ...]:
    return tuple(
        sorted(
            clusters,
            key=lambda item: (
                item.composite_score,
                item.support_share,
                item.exact_symbolic_support,
                1.0 - item.uncertainty,
                item.answer_canonical,
            ),
            reverse=True,
        )
    )


def _candidate_cautions(candidate: CandidateAnswer, *, support_share: float) -> tuple[str, ...]:
    cautions: list[str] = []
    if int(getattr(candidate, "cluster_size", 1) or 1) <= 1 and support_share < 0.35:
        cautions.append("singleton")
    if float(getattr(candidate, "entropy_penalty", 0.0) or 0.0) > 0.45:
        cautions.append("weak_evidence")
    return tuple(sorted(set(cautions)))


def _score_candidate_answers(
    candidates: Sequence[CandidateAnswer],
    *,
    config: EntropyWeightingConfig,
) -> EntropyWeightingBundle:
    """Compatibility fallback.

    This remains intentionally weaker than scoring real AnswerCluster summaries.
    """
    if not candidates:
        return EntropyWeightingBundle(
            winner=None,
            ranked_clusters=(),
            ranked_candidates=(),
            global_entropy=0.0,
            total_branches=0,
        )

    total_support = sum(_candidate_support_weight(candidate) for candidate in candidates)
    probabilities = [_candidate_support_weight(candidate) / max(1, total_support) for candidate in candidates]
    global_entropy = _normalize_entropy(probabilities)
    scored_clusters: list[WeightedClusterScore] = []

    for candidate in candidates:
        support_weight = _candidate_support_weight(candidate)
        support_share = _safe_ratio(support_weight, total_support)
        entropy_penalty = _clamp01(getattr(candidate, "entropy_penalty", 0.0) or 0.0)
        uncertainty = _clamp01(1.0 - (getattr(candidate, "composite_score", 0.0) or 0.0))
        composite_score = _clamp01(
            config.verifier_weight * (getattr(candidate, "verifier_score", 0.0) or 0.0)
            + config.weak_symbolic_weight * (getattr(candidate, "tool_consistency", 0.0) or 0.0)
            + config.agreement_weight * (getattr(candidate, "answer_agreement", 0.0) or 0.0)
            + config.novelty_weight * (getattr(candidate, "branch_novelty", 0.0) or 0.0)
            + config.exact_symbolic_weight * (getattr(candidate, "symbolic_check", 0.0) or 0.0)
            - config.uncertainty_penalty_weight * uncertainty
            - config.outlier_penalty_weight * entropy_penalty
        )
        signals = {
            "logical_consistency": _clamp01(getattr(candidate, "verifier_score", 0.0) or 0.0),
            "symbolic_agreement": _clamp01(getattr(candidate, "symbolic_check", 0.0) or 0.0),
            "completeness": _clamp01(getattr(candidate, "composite_score", 0.0) or 0.0),
            "repairability": _clamp01(getattr(candidate, "tool_consistency", 0.0) or 0.0),
            "answer_correctness_likelihood": _clamp01(getattr(candidate, "verifier_score", 0.0) or 0.0),
            "step_quality": _clamp01(getattr(candidate, "composite_score", 0.0) or 0.0),
            "prefix_quality": _clamp01(getattr(candidate, "composite_score", 0.0) or 0.0),
            "prm_prefix_quality": _clamp01(getattr(candidate, "composite_score", 0.0) or 0.0),
            "retrieval_compatibility": 0.0,
            "retrieval_support": 0.0,
            "operator_reliability": 0.0,
            "open_obligation_burden": _clamp01(1.0 - (getattr(candidate, "symbolic_check", 0.0) or 0.0)),
            "discharge_fraction": _clamp01(getattr(candidate, "symbolic_check", 0.0) or 0.0),
            "critique_support": 0.0,
            "confidence_dispersion": entropy_penalty,
            "family_diversity": _safe_ratio(len(set(candidate.branch_ids)), max(1, len(candidate.branch_ids))),
        }
        scored_clusters.append(
            WeightedClusterScore(
                cluster_id=f"candidate::{candidate.answer_canonical}",
                answer=candidate.answer,
                answer_canonical=candidate.answer_canonical,
                branch_ids=tuple(candidate.branch_ids),
                cluster_size=support_weight,
                support_share=support_share,
                diversity_weighted_support=_clamp01(getattr(candidate, "answer_agreement", 0.0) or 0.0),
                unique_branch_families=max(1, len(set(candidate.branch_ids))),
                family_diversity=_safe_ratio(len(set(candidate.branch_ids)), max(1, len(candidate.branch_ids))),
                verifier_score=_clamp01(getattr(candidate, "verifier_score", 0.0) or 0.0),
                weak_symbolic_support=_clamp01(getattr(candidate, "tool_consistency", 0.0) or 0.0),
                exact_symbolic_support=_clamp01(getattr(candidate, "symbolic_check", 0.0) or 0.0),
                retrieval_support=0.0,
                branch_novelty=_clamp01(getattr(candidate, "branch_novelty", 0.0) or 0.0),
                weak_symbolic_only_support=_clamp01(
                    max(
                        0.0,
                        (getattr(candidate, "tool_consistency", 0.0) or 0.0)
                        - (getattr(candidate, "symbolic_check", 0.0) or 0.0),
                    )
                ),
                symbolic_dominance_bonus=_clamp01(getattr(candidate, "symbolic_check", 0.0) or 0.0),
                uncertainty=uncertainty,
                entropy_penalty=entropy_penalty,
                outlier_penalty=entropy_penalty,
                composite_score=composite_score,
                mean_provenance_strength=0.0,
                mean_retrieval_relevance=0.0,
                mean_evidence_quality=_clamp01(getattr(candidate, "composite_score", 0.0) or 0.0),
                best_branch_id=(candidate.branch_ids[0] if candidate.branch_ids else None),
                best_member_score=_clamp01(getattr(candidate, "composite_score", 0.0) or 0.0),
                answer_variants=(candidate.answer,),
                caution_flags=_candidate_cautions(candidate, support_share=support_share),
                decomposed_signals=signals,
            )
        )

    ranked_clusters = _rank_weighted_clusters(scored_clusters)
    ranked_candidates = tuple(cluster.as_candidate_answer() for cluster in ranked_clusters)
    return EntropyWeightingBundle(
        winner=ranked_clusters[0] if ranked_clusters else None,
        ranked_clusters=ranked_clusters,
        ranked_candidates=ranked_candidates,
        global_entropy=global_entropy,
        total_branches=total_support,
    )


def _cluster_member_signal_float(member: object, names: Sequence[str]) -> float:
    for name in names:
        raw = getattr(member, name, None)
        if isinstance(raw, (int, float)):
            return _clamp01(raw)
    return 0.0


def _score_answer_clusters_from_clusters(
    clusters: Sequence[AnswerCluster],
    *,
    config: EntropyWeightingConfig,
) -> EntropyWeightingBundle:
    """Preferred scoring path using real AnswerCluster evidence summaries."""
    if not clusters:
        return EntropyWeightingBundle(
            winner=None,
            ranked_clusters=(),
            ranked_candidates=(),
            global_entropy=0.0,
            total_branches=0,
        )

    total_branches = sum(max(1, int(cluster.evidence_summary.support_size)) for cluster in clusters)
    global_entropy = compute_entropy(clusters)
    scored_clusters: list[WeightedClusterScore] = []

    for cluster in clusters:
        summary = cluster.evidence_summary
        members = cluster.members

        support_share = _safe_ratio(summary.support_size, total_branches)
        verifier = _clamp01(summary.mean_verifier_score)
        weak_symbolic = _clamp01(summary.mean_tool_consistency)
        exact_symbolic = _clamp01(summary.exact_symbolic_rate)
        retrieval_support = _clamp01(summary.mean_retrieval_support or summary.mean_retrieval_relevance)
        retrieval_compatibility = _clamp01(summary.mean_retrieval_relevance)
        novelty = _clamp01(summary.mean_branch_novelty)
        family_diversity = _clamp01(summary.family_diversity)
        evidence_quality = _clamp01(summary.mean_evidence_quality)
        provenance_strength = _clamp01(summary.mean_provenance_strength)
        entropy_penalty = _clamp01(summary.entropy_penalty)

        logical = _clamp01(summary.mean_logical_consistency)
        completeness = _clamp01(summary.mean_completeness)
        prefix_quality = _clamp01(summary.mean_prefix_quality)
        prm_prefix_quality = _clamp01(summary.mean_prm_prefix_quality)
        operator_reliability = _clamp01(summary.mean_operator_reliability)
        obligation_burden = _clamp01(summary.mean_open_obligation_burden)
        discharge_fraction = _clamp01(summary.mean_discharge_fraction)
        critique_support = _clamp01(summary.critique_adjustment)

        # Use member-level evidence if available to capture dispersion in a richer way.
        confidence_dispersion = _normalize_entropy(
            [max(1e-9, _cluster_member_signal_float(member, ("evidence_quality", "verifier_score"))) for member in members]
        ) if members else 0.0

        weak_symbolic_only_support = _clamp01(max(0.0, weak_symbolic - exact_symbolic))
        symbolic_dominance_bonus = _clamp01(max(0.0, exact_symbolic - weak_symbolic_only_support))

        uncertainty = _clamp01(
            0.30 * (1.0 - verifier)
            + 0.20 * (1.0 - exact_symbolic)
            + 0.20 * obligation_burden
            + 0.15 * confidence_dispersion
            + 0.15 * entropy_penalty
        )

        caution_flags = tuple(
            caution.value if hasattr(caution, "value") else str(caution)
            for caution in cluster.cautions
        )
        outlier_penalty = 0.0
        if "low_support_outlier" in caution_flags:
            outlier_penalty += 0.08
        if "singleton" in caution_flags:
            outlier_penalty += 0.04
        if "weak_evidence" in caution_flags:
            outlier_penalty += 0.06
        if "low_diversity_support" in caution_flags:
            outlier_penalty += 0.04

        process_quality = _clamp01(_mean((logical, completeness, prefix_quality, prm_prefix_quality)))

        positive_score = (
            config.verifier_weight * verifier
            + config.weak_symbolic_weight * weak_symbolic
            + config.agreement_weight * _clamp01(summary.diversity_weighted_support)
            + config.novelty_weight * novelty
            + config.exact_symbolic_weight * exact_symbolic
            + config.retrieval_weight * retrieval_support
            + config.process_quality_weight * process_quality
            + config.retrieval_compatibility_weight * retrieval_compatibility
            + config.operator_reliability_weight * operator_reliability
            + config.critique_support_weight * critique_support
            + config.exact_symbolic_dominance_weight * symbolic_dominance_bonus
            + config.symbolic_separation_weight * discharge_fraction
        )
        penalties = (
            config.obligation_penalty_weight * obligation_burden
            + config.uncertainty_penalty_weight * uncertainty
            + config.outlier_penalty_weight * outlier_penalty
        )
        composite_score = _clamp01(positive_score - penalties)

        decomposed_signals = {
            "logical_consistency": logical,
            "symbolic_agreement": exact_symbolic,
            "completeness": completeness,
            "repairability": _clamp01(max(0.0, critique_support + weak_symbolic_only_support)),
            "answer_correctness_likelihood": verifier,
            "step_quality": process_quality,
            "prefix_quality": prefix_quality,
            "prm_prefix_quality": prm_prefix_quality,
            "retrieval_compatibility": retrieval_compatibility,
            "retrieval_support": retrieval_support,
            "operator_reliability": operator_reliability,
            "open_obligation_burden": obligation_burden,
            "discharge_fraction": discharge_fraction,
            "critique_support": critique_support,
            "confidence_dispersion": confidence_dispersion,
            "family_diversity": family_diversity,
        }

        scored_clusters.append(
            WeightedClusterScore(
                cluster_id=cluster.cluster_id,
                answer=cluster.display_answer,
                answer_canonical=cluster.canonical_answer,
                branch_ids=tuple(cluster.branch_ids),
                cluster_size=int(summary.support_size),
                support_share=_clamp01(support_share),
                diversity_weighted_support=_clamp01(summary.diversity_weighted_support),
                unique_branch_families=int(summary.unique_branch_families),
                family_diversity=family_diversity,
                verifier_score=verifier,
                weak_symbolic_support=weak_symbolic,
                exact_symbolic_support=exact_symbolic,
                retrieval_support=retrieval_support,
                branch_novelty=novelty,
                weak_symbolic_only_support=weak_symbolic_only_support,
                symbolic_dominance_bonus=symbolic_dominance_bonus,
                uncertainty=uncertainty,
                entropy_penalty=entropy_penalty,
                outlier_penalty=_clamp01(outlier_penalty),
                composite_score=composite_score,
                mean_provenance_strength=provenance_strength,
                mean_retrieval_relevance=retrieval_compatibility,
                mean_evidence_quality=evidence_quality,
                best_branch_id=summary.best_branch_id,
                best_member_score=_clamp01(summary.best_member_score),
                answer_variants=tuple(summary.answer_variants),
                caution_flags=tuple(sorted(set(caution_flags))),
                decomposed_signals=decomposed_signals,
            )
        )

    ranked_clusters = _rank_weighted_clusters(scored_clusters)
    ranked_candidates = tuple(cluster.as_candidate_answer() for cluster in ranked_clusters)
    return EntropyWeightingBundle(
        winner=ranked_clusters[0] if ranked_clusters else None,
        ranked_clusters=ranked_clusters,
        ranked_candidates=ranked_candidates,
        global_entropy=global_entropy,
        total_branches=total_branches,
    )


def _score_cluster_inputs(
    cluster_inputs: Sequence[AnswerClusterInput],
    *,
    config: EntropyWeightingConfig,
) -> EntropyWeightingBundle:
    """Fallback path for raw grouped branch traces.

    This remains canonical-aware and preserves rich signal extraction, but it is
    secondary to scoring real AnswerCluster summaries.
    """
    if not cluster_inputs:
        return EntropyWeightingBundle(
            winner=None,
            ranked_clusters=(),
            ranked_candidates=(),
            global_entropy=0.0,
            total_branches=0,
        )

    total_support = sum(max(1, len(cluster.branches)) for cluster in cluster_inputs)
    scored_clusters: list[WeightedClusterScore] = []
    all_family_names: set[str] = set()

    for cluster in cluster_inputs:
        for trace in cluster.branches:
            all_family_names.add(str(getattr(trace, "branch_family", None) or getattr(trace, "branch_id", "")))

    for cluster in cluster_inputs:
        evidences = [
            BranchEvidence(
                branch_id=str(getattr(trace, "branch_id", "")),
                answer_canonical=cluster.answer_canonical,
                verifier_score=_clamp01(getattr(trace, "verifier_score", 0.0) or 0.0),
                logical_consistency=_logical_consistency(trace),
                completeness=_completeness(trace),
                prefix_quality=_prefix_quality(trace),
                prm_prefix_quality=_prm_prefix_quality(trace),
                weak_symbolic_support=_tool_consistency(trace),
                exact_symbolic_support=_exact_symbolic_support(trace),
                retrieval_support=_retrieval_support(trace),
                retrieval_compatibility=_retrieval_compatibility(trace),
                operator_reliability=_operator_reliability(trace),
                novelty=_branch_novelty(trace),
                repair_pressure=_repair_pressure(trace),
                open_obligation_burden=_open_obligation_burden(trace),
                discharge_fraction=_discharge_fraction(trace),
                critique_count=_critique_count(trace),
                critique_adjustment=_critique_adjustment(trace),
                confidence=_branch_confidence(trace),
                support_score=_branch_support_score(trace),
            )
            for trace in cluster.branches
        ]
        if not evidences:
            continue

        support_size = len(evidences)
        support_share = _safe_ratio(support_size, total_support)
        family_names = tuple(
            sorted(
                set(str(getattr(trace, "branch_family", None) or getattr(trace, "branch_id", "")) for trace in cluster.branches)
            )
        )
        unique_families = max(1, len(family_names))
        family_diversity = _safe_ratio(unique_families, max(1, len(all_family_names)))
        answer_variants = tuple(
            sorted(
                set(
                    str(getattr(trace, "answer_canonical", "") or getattr(trace, "answer", "") or "").strip()
                    for trace in cluster.branches
                    if str(getattr(trace, "answer_canonical", "") or getattr(trace, "answer", "") or "").strip()
                )
            )
        )

        verifier = _mean([e.verifier_score for e in evidences])
        logical = _mean([e.logical_consistency for e in evidences])
        completeness = _mean([e.completeness for e in evidences])
        prefix_quality = _mean([e.prefix_quality for e in evidences])
        prm_prefix_quality = _mean([e.prm_prefix_quality for e in evidences])
        weak_symbolic = _mean([e.weak_symbolic_support for e in evidences])
        exact_symbolic = _mean([e.exact_symbolic_support for e in evidences])
        retrieval_support = _mean([e.retrieval_support for e in evidences])
        retrieval_compatibility = _mean([e.retrieval_compatibility for e in evidences])
        operator_reliability = _mean([e.operator_reliability for e in evidences])
        novelty = _mean([e.novelty for e in evidences])
        obligation_burden = _mean([e.open_obligation_burden for e in evidences])
        discharge_fraction = _mean([e.discharge_fraction for e in evidences])
        critique_support = _mean([e.critique_adjustment for e in evidences])
        evidence_quality = _mean([e.support_score for e in evidences])
        confidence_dispersion = _normalize_entropy([max(1e-9, e.confidence) for e in evidences])

        agreement = _clamp01(
            0.55 * support_share
            + 0.25 * family_diversity
            + 0.20 * _safe_ratio(sum(1 for e in evidences if e.exact_symbolic_support >= 0.5), support_size)
        )
        entropy_penalty = _clamp01(
            0.45 * confidence_dispersion
            + 0.30 * _mean([e.repair_pressure for e in evidences])
            + 0.25 * max(0.0, 1.0 - agreement)
        )
        weak_symbolic_only_support = _clamp01(max(0.0, weak_symbolic - exact_symbolic))
        symbolic_dominance_bonus = _clamp01(max(0.0, exact_symbolic - weak_symbolic_only_support))
        outlier_penalty = _clamp01(max(0.0, 0.16 - support_share) / 0.16) if support_share < 0.16 else 0.0
        uncertainty = _clamp01(
            0.30 * (1.0 - verifier)
            + 0.20 * (1.0 - exact_symbolic)
            + 0.15 * obligation_burden
            + 0.20 * confidence_dispersion
            + 0.15 * outlier_penalty
        )
        process_quality = _clamp01(_mean((logical, completeness, prefix_quality, prm_prefix_quality)))

        composite_score = _clamp01(
            config.verifier_weight * verifier
            + config.weak_symbolic_weight * weak_symbolic
            + config.agreement_weight * agreement
            + config.novelty_weight * novelty
            + config.exact_symbolic_weight * exact_symbolic
            + config.retrieval_weight * retrieval_support
            + config.process_quality_weight * process_quality
            + config.retrieval_compatibility_weight * retrieval_compatibility
            + config.operator_reliability_weight * operator_reliability
            + config.critique_support_weight * critique_support
            + config.exact_symbolic_dominance_weight * symbolic_dominance_bonus
            + config.symbolic_separation_weight * discharge_fraction
            - config.obligation_penalty_weight * obligation_burden
            - config.uncertainty_penalty_weight * uncertainty
            - config.outlier_penalty_weight * outlier_penalty
        )

        caution_flags: list[str] = []
        if support_size <= 1 and support_share < 0.35:
            caution_flags.append("singleton")
        if family_diversity < 0.34 and support_size > 1:
            caution_flags.append("low_diversity_support")
        if outlier_penalty > 0.0:
            caution_flags.append("low_support_outlier")
        if evidence_quality < 0.40 or verifier < 0.35:
            caution_flags.append("weak_evidence")

        best_evidence = max(evidences, key=lambda item: (item.support_score, item.confidence, item.branch_id))
        decomposed_signals = {
            "logical_consistency": _clamp01(logical),
            "symbolic_agreement": _clamp01(exact_symbolic),
            "completeness": _clamp01(completeness),
            "repairability": _clamp01(max(0.0, critique_support + weak_symbolic_only_support)),
            "answer_correctness_likelihood": _clamp01(verifier),
            "step_quality": process_quality,
            "prefix_quality": _clamp01(prefix_quality),
            "prm_prefix_quality": _clamp01(prm_prefix_quality),
            "retrieval_compatibility": _clamp01(retrieval_compatibility),
            "retrieval_support": _clamp01(retrieval_support),
            "operator_reliability": _clamp01(operator_reliability),
            "open_obligation_burden": _clamp01(obligation_burden),
            "discharge_fraction": _clamp01(discharge_fraction),
            "critique_support": _clamp01(critique_support),
            "confidence_dispersion": _clamp01(confidence_dispersion),
            "family_diversity": _clamp01(family_diversity),
        }

        scored_clusters.append(
            WeightedClusterScore(
                cluster_id=f"trace_cluster::{cluster.answer_canonical}",
                answer=cluster.answer,
                answer_canonical=cluster.answer_canonical,
                branch_ids=tuple(e.branch_id for e in evidences),
                cluster_size=support_size,
                support_share=_clamp01(support_share),
                diversity_weighted_support=_clamp01(agreement),
                unique_branch_families=unique_families,
                family_diversity=_clamp01(family_diversity),
                verifier_score=_clamp01(verifier),
                weak_symbolic_support=_clamp01(weak_symbolic),
                exact_symbolic_support=_clamp01(exact_symbolic),
                retrieval_support=_clamp01(retrieval_support),
                branch_novelty=_clamp01(novelty),
                weak_symbolic_only_support=_clamp01(weak_symbolic_only_support),
                symbolic_dominance_bonus=_clamp01(symbolic_dominance_bonus),
                uncertainty=_clamp01(uncertainty),
                entropy_penalty=_clamp01(entropy_penalty),
                outlier_penalty=_clamp01(outlier_penalty),
                composite_score=_clamp01(composite_score),
                mean_provenance_strength=_clamp01(_mean([e.operator_reliability for e in evidences])),
                mean_retrieval_relevance=_clamp01(retrieval_compatibility),
                mean_evidence_quality=_clamp01(evidence_quality),
                best_branch_id=best_evidence.branch_id,
                best_member_score=_clamp01(best_evidence.support_score),
                answer_variants=answer_variants,
                caution_flags=tuple(sorted(set(caution_flags))),
                decomposed_signals=decomposed_signals,
            )
        )

    ranked_clusters = _rank_weighted_clusters(scored_clusters)
    ranked_candidates = tuple(cluster.as_candidate_answer() for cluster in ranked_clusters)
    return EntropyWeightingBundle(
        winner=ranked_clusters[0] if ranked_clusters else None,
        ranked_clusters=ranked_clusters,
        ranked_candidates=ranked_candidates,
        global_entropy=compute_entropy(cluster_inputs),
        total_branches=total_support,
    )


def compute_entropy(
    clusters_or_traces: Sequence[object] | Mapping[str, Sequence[BranchTrace]] | ClusteringResult,
) -> float:
    answer_clusters = _coerce_answer_clusters(clusters_or_traces)
    if answer_clusters is not None:
        total = sum(cluster.evidence_summary.support_size for cluster in answer_clusters)
        if total <= 0:
            return 0.0
        return _normalize_entropy(
            [cluster.evidence_summary.support_size / total for cluster in answer_clusters]
        )

    candidate_inputs = _coerce_candidate_answers(clusters_or_traces)
    if candidate_inputs is not None:
        total = sum(_candidate_support_weight(candidate) for candidate in candidate_inputs)
        if total <= 0:
            return 0.0
        return _normalize_entropy(
            [_candidate_support_weight(candidate) / total for candidate in candidate_inputs]
        )

    cluster_inputs = _coerce_cluster_inputs(clusters_or_traces)
    total = sum(len(cluster.branches) for cluster in cluster_inputs)
    if total <= 0:
        return 0.0
    return _normalize_entropy([len(cluster.branches) / total for cluster in cluster_inputs])


def score_answer_clusters(
    clusters_or_traces: Sequence[object] | Mapping[str, Sequence[BranchTrace]] | ClusteringResult,
    *,
    config: EntropyWeightingConfig | None = None,
) -> EntropyWeightingBundle:
    """Integrated public entrypoint.

    Preferred order:
        1. real AnswerCluster / ClusteringResult
        2. CandidateAnswer compatibility path
        3. canonical-aware grouped BranchTrace fallback
    """
    weights = config or EntropyWeightingConfig()

    answer_clusters = _coerce_answer_clusters(clusters_or_traces)
    if answer_clusters is not None:
        return _score_answer_clusters_from_clusters(answer_clusters, config=weights)

    candidate_inputs = _coerce_candidate_answers(clusters_or_traces)
    if candidate_inputs is not None:
        return _score_candidate_answers(candidate_inputs, config=weights)

    cluster_inputs = _coerce_cluster_inputs(clusters_or_traces)
    if cluster_inputs:
        return _score_cluster_inputs(cluster_inputs, config=weights)

    return EntropyWeightingBundle(
        winner=None,
        ranked_clusters=(),
        ranked_candidates=(),
        global_entropy=0.0,
        total_branches=0,
    )


def weighted_select(
    clusters_or_traces: Sequence[object] | Mapping[str, Sequence[BranchTrace]] | ClusteringResult,
    *,
    config: EntropyWeightingConfig | None = None,
) -> WeightedClusterScore | None:
    return score_answer_clusters(clusters_or_traces, config=config).winner


__all__ = [
    "AnswerClusterInput",
    "BranchEvidence",
    "EntropyWeightingBundle",
    "EntropyWeightingConfig",
    "WeightedClusterScore",
    "cluster_traces_by_answer",
    "compute_entropy",
    "score_answer_clusters",
    "weighted_select",
]