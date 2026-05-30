# src/aggregation/clustering.py
"""Typed answer clustering for post-canonicalization aggregation."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Callable, Sequence

from src.aggregation.canonicalize import canonicalize_competition_answer
from src.common.constants import (
    W_ANSWER_AGREEMENT,
    W_BRANCH_NOVELTY,
    W_SYMBOLIC_CHECK,
    W_TOOL_CONSISTENCY,
    W_VERIFIER,
)
from src.common.schemas import CandidateAnswer


class ClusterCaution(str, Enum):
    """Non-fatal warnings attached to clusters."""

    NONE = "none"
    SINGLETON = "singleton"
    LOW_SUPPORT_OUTLIER = "low_support_outlier"
    LOW_DIVERSITY_SUPPORT = "low_diversity_support"
    WEAK_EVIDENCE = "weak_evidence"
    HIGH_CONFIDENCE_DISPERSION = "high_confidence_dispersion"
    LOW_DISCHARGE = "low_discharge"
    CRITIQUE_HEAVY = "critique_heavy"


@dataclass(frozen=True)
class BranchFamilySupport:
    branch_family: str
    count: int


@dataclass(frozen=True)
class ClusterMember:
    branch_id: str
    branch_ids: tuple[str, ...]
    branch_family: str
    raw_answer: str
    canonical_answer: str

    verifier_score: float
    logical_consistency: float
    completeness: float
    prefix_quality: float
    prm_prefix_quality: float

    tool_consistency: float
    symbolic_check: float
    discharge_fraction: float
    open_obligation_burden: float

    branch_novelty: float
    provenance_strength: float
    retrieval_relevance: float
    retrieval_support: float
    operator_reliability: float

    repair_pressure: float
    critique_count: int
    critique_adjustment: float
    confidence: float
    evidence_quality: float
    support_weight: int
    source_index: int


@dataclass(frozen=True)
class ClusterEvidenceSummary:
    support_size: int
    support_share: float
    diversity_weighted_support: float
    unique_branch_families: int
    family_diversity: float

    mean_verifier_score: float
    mean_logical_consistency: float
    mean_completeness: float
    mean_prefix_quality: float
    mean_prm_prefix_quality: float

    mean_tool_consistency: float
    mean_branch_novelty: float
    exact_symbolic_rate: float
    mean_provenance_strength: float
    mean_retrieval_relevance: float
    mean_retrieval_support: float
    mean_operator_reliability: float
    mean_open_obligation_burden: float
    mean_discharge_fraction: float

    mean_evidence_quality: float
    mean_confidence: float
    confidence_dispersion: float
    critique_ratio: float
    critique_adjustment: float

    best_branch_id: str | None
    best_member_score: float
    entropy_penalty: float
    branch_family_support: tuple[BranchFamilySupport, ...]
    answer_variants: tuple[str, ...]


@dataclass(frozen=True)
class AnswerCluster:
    cluster_id: str
    canonical_answer: str
    display_answer: str
    members: tuple[ClusterMember, ...]
    branch_ids: tuple[str, ...]
    evidence_summary: ClusterEvidenceSummary
    cautions: tuple[ClusterCaution, ...]
    preliminary_score: float
    answer_family_id: str
    answer_family_metadata: dict[str, Any]

    def to_candidate_answer(self) -> CandidateAnswer:
        summary = self.evidence_summary
        return CandidateAnswer(
            answer=self.display_answer,
            answer_canonical=self.canonical_answer,
            branch_ids=list(self.branch_ids),
            verifier_score=summary.mean_verifier_score,
            tool_consistency=summary.mean_tool_consistency,
            answer_agreement=summary.diversity_weighted_support,
            branch_novelty=summary.mean_branch_novelty,
            symbolic_check=summary.exact_symbolic_rate,
            composite_score=self.preliminary_score,
            cluster_size=summary.support_size,
            entropy_penalty=summary.entropy_penalty,
        )


@dataclass(frozen=True)
class ExcludedCandidate:
    branch_id: str
    raw_answer: str
    reason: str
    source_index: int


@dataclass(frozen=True)
class ClusteringResult:
    clusters: tuple[AnswerCluster, ...]
    excluded: tuple[ExcludedCandidate, ...]
    total_candidates: int
    clustered_candidates: int
    clustered_support: int

    def candidate_answers(self) -> tuple[CandidateAnswer, ...]:
        return tuple(cluster.to_candidate_answer() for cluster in self.clusters)


def cluster_candidates(
    candidates: Sequence[object],
    *,
    canonicalizer: Callable[[str], str | None] | None = None,
    outlier_support_threshold: float = 0.15,
    weak_evidence_threshold: float = 0.45,
    high_dispersion_threshold: float = 0.22,
) -> ClusteringResult:
    resolved_canonicalizer = canonicalizer or canonicalize_competition_answer
    normalized: list[ClusterMember] = []
    excluded: list[ExcludedCandidate] = []

    for index, candidate in enumerate(candidates):
        member, exclusion_reason = _normalize_candidate(
            candidate,
            source_index=index,
            canonicalizer=resolved_canonicalizer,
        )
        if member is None:
            excluded.append(
                ExcludedCandidate(
                    branch_id=_candidate_branch_id(candidate, index),
                    raw_answer=_candidate_raw_answer(candidate),
                    reason=exclusion_reason or "missing_canonical_answer",
                    source_index=index,
                )
            )
            continue
        normalized.append(member)

    if not normalized:
        return ClusteringResult(
            clusters=(),
            excluded=tuple(sorted(excluded, key=lambda item: (item.source_index, item.branch_id))),
            total_candidates=len(candidates),
            clustered_candidates=0,
            clustered_support=0,
        )

    grouped: dict[str, list[ClusterMember]] = {}
    for member in normalized:
        grouped.setdefault(member.canonical_answer, []).append(member)

    total_support = sum(member.support_weight for member in normalized)
    clusters = [
        _build_cluster(
            canonical_answer=canonical_answer,
            members=members,
            total_clustered=total_support,
            outlier_support_threshold=outlier_support_threshold,
            weak_evidence_threshold=weak_evidence_threshold,
            high_dispersion_threshold=high_dispersion_threshold,
        )
        for canonical_answer, members in grouped.items()
    ]
    clusters.sort(key=_cluster_sort_key)
    return ClusteringResult(
        clusters=tuple(clusters),
        excluded=tuple(sorted(excluded, key=lambda item: (item.source_index, item.branch_id))),
        total_candidates=len(candidates),
        clustered_candidates=len(normalized),
        clustered_support=total_support,
    )


def clusters_to_candidate_answers(clusters: Sequence[AnswerCluster]) -> tuple[CandidateAnswer, ...]:
    return tuple(cluster.to_candidate_answer() for cluster in clusters)


def _normalize_candidate(
    candidate: object,
    *,
    source_index: int,
    canonicalizer: Callable[[str], str | None],
) -> tuple[ClusterMember | None, str | None]:
    raw_answer = _candidate_raw_answer(candidate)
    canonical_answer = _candidate_canonical_answer(candidate, raw_answer, canonicalizer)
    if not canonical_answer:
        return None, "missing_canonical_answer"

    branch_id = _candidate_branch_id(candidate, source_index)
    branch_ids = _candidate_branch_ids(candidate, branch_id)
    branch_family = _candidate_branch_family(candidate, branch_id)

    verifier_score = _clamp01(_first_numeric_attr(candidate, ("verifier_score",), default=0.0))
    logical_consistency = _clamp01(
        _first_numeric_attr(candidate, ("logical_consistency",), default=verifier_score)
    )
    completeness = _clamp01(_first_numeric_attr(candidate, ("completeness",), default=0.0))
    prefix_quality = _clamp01(_first_numeric_attr(candidate, ("prefix_quality",), default=0.0))
    prm_prefix_quality = _clamp01(
        _first_numeric_attr(candidate, ("prm_prefix_quality", "prefix_quality"), default=prefix_quality)
    )

    tool_consistency = _clamp01(
        _first_numeric_attr(candidate, ("tool_consistency", "symbolic_valid"), default=0.0)
    )
    symbolic_check = _clamp01(
        _first_numeric_attr(candidate, ("symbolic_check", "exact_symbolic_check", "symbolic_valid"), default=0.0)
    )
    discharge_fraction = _clamp01(
        _first_numeric_attr(candidate, ("discharge_fraction", "symbolic_check"), default=symbolic_check)
    )
    open_obligation_burden = _clamp01(
        _first_numeric_attr(candidate, ("open_obligation_burden",), default=max(0.0, 1.0 - discharge_fraction))
    )

    branch_novelty = _clamp01(_first_numeric_attr(candidate, ("branch_novelty", "novelty_score"), default=0.0))
    provenance_strength = _derive_provenance_strength(candidate, canonical_answer, symbolic_check, tool_consistency)
    retrieval_relevance = _clamp01(_first_numeric_attr(candidate, ("retrieval_relevance",), default=0.0))
    retrieval_support = _clamp01(
        _first_numeric_attr(candidate, ("retrieval_support", "retrieval_relevance"), default=retrieval_relevance)
    )
    operator_reliability = _clamp01(_first_numeric_attr(candidate, ("operator_reliability",), default=provenance_strength))

    repair_pressure = _clamp01(_first_numeric_attr(candidate, ("repair_pressure", "repair_count"), default=0.0))
    critique_count = int(max(0, round(_first_numeric_attr(candidate, ("critique_count",), default=0.0))))
    critique_adjustment = _clamp01(_first_numeric_attr(candidate, ("critique_adjustment",), default=0.0))
    confidence = _clamp01(
        _first_numeric_attr(
            candidate,
            ("confidence", "branch_score", "verifier_score"),
            default=verifier_score,
        )
    )

    evidence_quality = _support_quality(
        verifier_score=verifier_score,
        logical_consistency=logical_consistency,
        completeness=completeness,
        prefix_quality=prefix_quality,
        prm_prefix_quality=prm_prefix_quality,
        tool_consistency=tool_consistency,
        symbolic_check=symbolic_check,
        provenance_strength=provenance_strength,
        retrieval_relevance=retrieval_relevance,
        retrieval_support=retrieval_support,
        operator_reliability=operator_reliability,
        open_obligation_burden=open_obligation_burden,
        discharge_fraction=discharge_fraction,
        critique_adjustment=critique_adjustment,
        confidence=confidence,
    )

    return ClusterMember(
        branch_id=branch_id,
        branch_ids=branch_ids,
        branch_family=branch_family,
        raw_answer=raw_answer,
        canonical_answer=canonical_answer,
        verifier_score=verifier_score,
        logical_consistency=logical_consistency,
        completeness=completeness,
        prefix_quality=prefix_quality,
        prm_prefix_quality=prm_prefix_quality,
        tool_consistency=tool_consistency,
        symbolic_check=symbolic_check,
        discharge_fraction=discharge_fraction,
        open_obligation_burden=open_obligation_burden,
        branch_novelty=branch_novelty,
        provenance_strength=provenance_strength,
        retrieval_relevance=retrieval_relevance,
        retrieval_support=retrieval_support,
        operator_reliability=operator_reliability,
        repair_pressure=repair_pressure,
        critique_count=critique_count,
        critique_adjustment=critique_adjustment,
        confidence=confidence,
        evidence_quality=evidence_quality,
        support_weight=_candidate_support_weight(candidate, branch_ids),
        source_index=source_index,
    ), None


def _build_cluster(
    *,
    canonical_answer: str,
    members: Sequence[ClusterMember],
    total_clustered: int,
    outlier_support_threshold: float,
    weak_evidence_threshold: float,
    high_dispersion_threshold: float,
) -> AnswerCluster:
    ordered_members = tuple(sorted(members, key=_member_sort_key))
    support_size = sum(member.support_weight for member in ordered_members)
    support_share = support_size / max(1, total_clustered)

    family_counts = _family_counts(ordered_members)
    unique_branch_families = len(family_counts)
    family_diversity = unique_branch_families / max(1, support_size)
    diversity_weighted_support = support_share * (0.50 + 0.50 * family_diversity)

    mean_verifier_score = _weighted_mean((member.verifier_score, member.support_weight) for member in ordered_members)
    mean_logical_consistency = _weighted_mean((member.logical_consistency, member.support_weight) for member in ordered_members)
    mean_completeness = _weighted_mean((member.completeness, member.support_weight) for member in ordered_members)
    mean_prefix_quality = _weighted_mean((member.prefix_quality, member.support_weight) for member in ordered_members)
    mean_prm_prefix_quality = _weighted_mean((member.prm_prefix_quality, member.support_weight) for member in ordered_members)

    mean_tool_consistency = _weighted_mean((member.tool_consistency, member.support_weight) for member in ordered_members)
    mean_branch_novelty = _weighted_mean((member.branch_novelty, member.support_weight) for member in ordered_members)
    exact_symbolic_rate = _weighted_mean((member.symbolic_check, member.support_weight) for member in ordered_members)
    mean_provenance_strength = _weighted_mean((member.provenance_strength, member.support_weight) for member in ordered_members)
    mean_retrieval_relevance = _weighted_mean((member.retrieval_relevance, member.support_weight) for member in ordered_members)
    mean_retrieval_support = _weighted_mean((member.retrieval_support, member.support_weight) for member in ordered_members)
    mean_operator_reliability = _weighted_mean((member.operator_reliability, member.support_weight) for member in ordered_members)
    mean_open_obligation_burden = _weighted_mean((member.open_obligation_burden, member.support_weight) for member in ordered_members)
    mean_discharge_fraction = _weighted_mean((member.discharge_fraction, member.support_weight) for member in ordered_members)

    mean_evidence_quality = _weighted_mean((member.evidence_quality, member.support_weight) for member in ordered_members)
    mean_confidence = _weighted_mean((member.confidence, member.support_weight) for member in ordered_members)
    confidence_dispersion = _weighted_std(
        ((member.confidence, member.support_weight) for member in ordered_members),
        mean=mean_confidence,
    )
    critique_ratio = _safe_ratio(sum(1 for member in ordered_members if member.critique_count > 0), len(ordered_members))
    critique_adjustment = _weighted_mean((member.critique_adjustment, member.support_weight) for member in ordered_members)

    answer_variants = tuple(dict.fromkeys(member.raw_answer for member in ordered_members if member.raw_answer))
    entropy_penalty = _clamp01(1.0 - diversity_weighted_support)

    best_member = ordered_members[0] if ordered_members else None
    summary = ClusterEvidenceSummary(
        support_size=support_size,
        support_share=support_share,
        diversity_weighted_support=diversity_weighted_support,
        unique_branch_families=unique_branch_families,
        family_diversity=family_diversity,
        mean_verifier_score=mean_verifier_score,
        mean_logical_consistency=mean_logical_consistency,
        mean_completeness=mean_completeness,
        mean_prefix_quality=mean_prefix_quality,
        mean_prm_prefix_quality=mean_prm_prefix_quality,
        mean_tool_consistency=mean_tool_consistency,
        mean_branch_novelty=mean_branch_novelty,
        exact_symbolic_rate=exact_symbolic_rate,
        mean_provenance_strength=mean_provenance_strength,
        mean_retrieval_relevance=mean_retrieval_relevance,
        mean_retrieval_support=mean_retrieval_support,
        mean_operator_reliability=mean_operator_reliability,
        mean_open_obligation_burden=mean_open_obligation_burden,
        mean_discharge_fraction=mean_discharge_fraction,
        mean_evidence_quality=mean_evidence_quality,
        mean_confidence=mean_confidence,
        confidence_dispersion=confidence_dispersion,
        critique_ratio=critique_ratio,
        critique_adjustment=critique_adjustment,
        best_branch_id=best_member.branch_id if best_member is not None else None,
        best_member_score=best_member.evidence_quality if best_member is not None else 0.0,
        entropy_penalty=entropy_penalty,
        branch_family_support=tuple(
            BranchFamilySupport(branch_family=branch_family, count=count)
            for branch_family, count in sorted(family_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        answer_variants=answer_variants,
    )

    cautions = _cluster_cautions(
        summary=summary,
        total_clustered=total_clustered,
        outlier_support_threshold=outlier_support_threshold,
        weak_evidence_threshold=weak_evidence_threshold,
        high_dispersion_threshold=high_dispersion_threshold,
    )
    preliminary_score = diversity_weighted_support
    family_metadata = {
        "support_share": round(summary.support_share, 6),
        "confidence_dispersion": round(summary.confidence_dispersion, 6),
        "mean_open_obligation_burden": round(summary.mean_open_obligation_burden, 6),
        "mean_discharge_fraction": round(summary.mean_discharge_fraction, 6),
        "critique_ratio": round(summary.critique_ratio, 6),
    }
    return AnswerCluster(
        cluster_id=f"answer::{canonical_answer}",
        canonical_answer=canonical_answer,
        display_answer=canonical_answer,
        members=ordered_members,
        branch_ids=_flatten_branch_ids(ordered_members),
        evidence_summary=summary,
        cautions=cautions,
        preliminary_score=preliminary_score,
        answer_family_id=f"family::{canonical_answer}",
        answer_family_metadata=family_metadata,
    )


def _cluster_cautions(
    *,
    summary: ClusterEvidenceSummary,
    total_clustered: int,
    outlier_support_threshold: float,
    weak_evidence_threshold: float,
    high_dispersion_threshold: float,
) -> tuple[ClusterCaution, ...]:
    cautions: list[ClusterCaution] = []
    if summary.support_size == 1 and total_clustered > 1:
        cautions.append(ClusterCaution.SINGLETON)
    if total_clustered > 1 and summary.support_share < outlier_support_threshold:
        cautions.append(ClusterCaution.LOW_SUPPORT_OUTLIER)
    if summary.support_size > 1 and summary.family_diversity < 0.5:
        cautions.append(ClusterCaution.LOW_DIVERSITY_SUPPORT)
    if summary.mean_evidence_quality < weak_evidence_threshold:
        cautions.append(ClusterCaution.WEAK_EVIDENCE)
    if summary.confidence_dispersion > high_dispersion_threshold:
        cautions.append(ClusterCaution.HIGH_CONFIDENCE_DISPERSION)
    if summary.mean_discharge_fraction < 0.40 and summary.mean_open_obligation_burden > 0.40:
        cautions.append(ClusterCaution.LOW_DISCHARGE)
    if summary.critique_ratio > 0.60:
        cautions.append(ClusterCaution.CRITIQUE_HEAVY)
    return tuple(cautions) if cautions else (ClusterCaution.NONE,)


def _cluster_sort_key(cluster: AnswerCluster) -> tuple[float, float, float, str]:
    summary = cluster.evidence_summary
    return (
        summary.support_share,
        summary.diversity_weighted_support,
        summary.mean_verifier_score,
        cluster.canonical_answer,
    )


def _member_sort_key(member: ClusterMember) -> tuple[float, float, float, str]:
    return (
        member.evidence_quality,
        member.verifier_score,
        member.confidence,
        member.branch_id,
    )


def _candidate_raw_answer(candidate: object) -> str:
    return str(getattr(candidate, "answer", "") or "").strip()


def _candidate_canonical_answer(
    candidate: object,
    raw_answer: str,
    canonicalizer: Callable[[str], str | None],
) -> str | None:
    explicit = getattr(candidate, "answer_canonical", None)
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if not raw_answer:
        return None
    try:
        return canonicalizer(raw_answer)
    except Exception:
        return None


def _candidate_branch_id(candidate: object, source_index: int) -> str:
    raw = getattr(candidate, "branch_id", None)
    if raw:
        return str(raw)
    branch_ids = getattr(candidate, "branch_ids", None)
    if isinstance(branch_ids, Sequence) and branch_ids and not isinstance(branch_ids, (str, bytes)):
        return str(branch_ids[0])
    return f"candidate-{source_index}"


def _candidate_branch_ids(candidate: object, branch_id: str) -> tuple[str, ...]:
    branch_ids = getattr(candidate, "branch_ids", None)
    if isinstance(branch_ids, Sequence) and not isinstance(branch_ids, (str, bytes)):
        normalized = tuple(str(item) for item in branch_ids if str(item).strip())
        if normalized:
            return normalized
    return (branch_id,)


def _candidate_branch_family(candidate: object, branch_id: str) -> str:
    explicit = getattr(candidate, "branch_family", None)
    if explicit:
        return str(explicit)
    metadata = getattr(candidate, "metadata", None)
    if isinstance(metadata, dict):
        family = metadata.get("branch_family")
        if family:
            return str(family)
    return branch_id


def _first_numeric_attr(candidate: object, names: Sequence[str], *, default: float) -> float:
    metadata = getattr(candidate, "metadata", None)
    for name in names:
        raw = getattr(candidate, name, None)
        if isinstance(raw, bool):
            return 1.0 if raw else 0.0
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(metadata, dict):
            for source_name in (
                None,
                "signal_decomposition",
                "verifier_decomposition",
                "retrieval",
                "operator",
                "proof_state",
                "self_critique",
            ):
                source = metadata if source_name is None else metadata.get(source_name)
                if isinstance(source, dict):
                    value = source.get(name)
                    if isinstance(value, bool):
                        return 1.0 if value else 0.0
                    if isinstance(value, (int, float)):
                        return float(value)
            critique = metadata.get("self_critique")
            if isinstance(critique, dict):
                trigger_signals = critique.get("trigger_signals")
                if isinstance(trigger_signals, dict):
                    value = trigger_signals.get(name)
                    if isinstance(value, bool):
                        return 1.0 if value else 0.0
                    if isinstance(value, (int, float)):
                        return float(value)
    return float(default)


def _candidate_support_weight(candidate: object, branch_ids: tuple[str, ...]) -> int:
    explicit = getattr(candidate, "cluster_size", None)
    if isinstance(explicit, int) and explicit > 0:
        return explicit
    return max(1, len(branch_ids))


def _derive_provenance_strength(
    candidate: object,
    canonical_answer: str,
    symbolic_check: float,
    tool_consistency: float,
) -> float:
    explicit = _first_numeric_attr(candidate, ("provenance_strength",), default=-1.0)
    if explicit >= 0.0:
        return _clamp01(explicit)
    return _clamp01(
        0.45 * symbolic_check
        + 0.35 * tool_consistency
        + 0.20 * (1.0 if canonical_answer else 0.0)
    )


def _support_quality(
    *,
    verifier_score: float,
    logical_consistency: float,
    completeness: float,
    prefix_quality: float,
    prm_prefix_quality: float,
    tool_consistency: float,
    symbolic_check: float,
    provenance_strength: float,
    retrieval_relevance: float,
    retrieval_support: float,
    operator_reliability: float,
    open_obligation_burden: float,
    discharge_fraction: float,
    critique_adjustment: float,
    confidence: float,
) -> float:
    raw = (
        W_VERIFIER * verifier_score
        + 0.06 * logical_consistency
        + 0.05 * completeness
        + 0.05 * prefix_quality
        + 0.07 * prm_prefix_quality
        + W_TOOL_CONSISTENCY * tool_consistency
        + W_SYMBOLIC_CHECK * symbolic_check
        + 0.05 * provenance_strength
        + 0.04 * retrieval_relevance
        + 0.05 * retrieval_support
        + 0.05 * operator_reliability
        + 0.05 * discharge_fraction
        + 0.03 * critique_adjustment
        + 0.04 * confidence
        - 0.08 * open_obligation_burden
    )
    return _clamp01(raw)


def _family_counts(members: Sequence[ClusterMember]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for member in members:
        counts[member.branch_family] = counts.get(member.branch_family, 0) + member.support_weight
    return counts


def _flatten_branch_ids(members: Sequence[ClusterMember]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for member in members:
        for branch_id in member.branch_ids:
            if branch_id not in seen:
                ordered.append(branch_id)
                seen.add(branch_id)
    return tuple(ordered)


def _weighted_mean(items: Sequence[tuple[float, int]] | Any) -> float:
    total_weight = 0.0
    total_value = 0.0
    for value, weight in items:
        total_weight += max(0.0, float(weight))
        total_value += float(value) * max(0.0, float(weight))
    if total_weight <= 0.0:
        return 0.0
    return total_value / total_weight


def _weighted_std(items: Sequence[tuple[float, int]] | Any, *, mean: float) -> float:
    total_weight = 0.0
    total_value = 0.0
    for value, weight in items:
        w = max(0.0, float(weight))
        total_weight += w
        total_value += ((float(value) - mean) ** 2) * w
    if total_weight <= 0.0:
        return 0.0
    return math.sqrt(total_value / total_weight)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))