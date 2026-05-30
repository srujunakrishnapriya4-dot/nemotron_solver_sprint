# src/aggregation/final_selector.py
"""Deterministic final answer selection over weighted answer clusters.

This module is intentionally the last aggregation stage:
canonicalization -> clustering -> weighting -> final selection.

It consumes richer decomposed evidence when available, while remaining
backward-compatible with older weighted-cluster and candidate-only callers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Mapping, Sequence

from src.aggregation.clustering import AnswerCluster, ClusteringResult
from src.aggregation.entropy_weighting import EntropyWeightingBundle, WeightedClusterScore
from src.common.constants import ANSWER_MAX, ANSWER_MIN
from src.common.schemas import CandidateAnswer, FinalPrediction

_INTEGER_RE = re.compile(r"^[+-]?\d+$")


class SelectionWarning(str, Enum):
    NONE = "none"
    NO_VALID_CANDIDATES = "no_valid_candidates"
    LOW_CONFIDENCE = "low_confidence"
    LOW_MARGIN = "low_margin"
    HIGH_DISAGREEMENT = "high_disagreement"
    FRAGILE_WINNER = "fragile_winner"
    OUTLIER_WINNER = "outlier_winner"
    WEAK_EVIDENCE = "weak_evidence"
    CLUSTER_CAUTION = "cluster_caution"
    LOW_DISCHARGE = "low_discharge"
    HIGH_OBLIGATION_BURDEN = "high_obligation_burden"
    CRITIQUE_HEAVY_FAMILY = "critique_heavy_family"
    FRAGMENTED_SUPPORT = "fragmented_support"
    STRONG_EVIDENCE_FRAGMENTED_SUPPORT = "strong_evidence_fragmented_support"
    NON_INTEGER_WINNER = "non_integer_winner"
    OUT_OF_RANGE_WINNER = "out_of_range_winner"
    ABSTENTION_RECOMMENDED = "abstention_recommended"


@dataclass(frozen=True)
class FinalSelectorConfig:
    minimum_final_score: float = 0.42
    minimum_margin: float = 0.05
    minimum_support_share: float = 0.18
    minimum_discharge_fraction: float = 0.40
    maximum_obligation_burden: float = 0.55
    caution_penalty_singleton: float = 0.03
    caution_penalty_outlier: float = 0.07
    caution_penalty_low_diversity: float = 0.03
    caution_penalty_weak_evidence: float = 0.06
    disagreement_penalty_weight: float = 0.08
    ambiguity_penalty_weight: float = 0.08
    fragility_penalty_weight: float = 0.07
    obligation_penalty_weight: float = 0.08
    critique_penalty_weight: float = 0.04
    family_bonus_weight: float = 0.05
    quality_bonus_weight: float = 0.05
    evidence_bonus_weight: float = 0.05
    invalid_answer_penalty: float = 0.10
    allow_abstention: bool = True
    default_submission_answer: int = ANSWER_MIN


@dataclass(frozen=True)
class FinalScoreBreakdown:
    weighted_evidence_score: float
    upstream_weighted_score: float
    evidence_bonus: float
    family_bonus: float
    quality_bonus: float
    disagreement_penalty: float
    fragility_penalty: float
    caution_penalty: float
    invalid_answer_penalty: float
    final_score: float


@dataclass(frozen=True)
class SelectedClusterEvidence:
    support_size: int
    support_share: float
    diversity_weighted_support: float
    unique_branch_families: int
    family_diversity: float
    mean_verifier_score: float
    mean_tool_consistency: float
    mean_branch_novelty: float
    exact_symbolic_rate: float
    mean_provenance_strength: float
    mean_retrieval_relevance: float
    mean_evidence_quality: float
    mean_discharge_fraction: float
    mean_open_obligation_burden: float
    critique_ratio: float
    confidence_dispersion: float
    best_branch_id: str | None
    best_member_score: float
    entropy_penalty: float
    branch_ids: tuple[str, ...]
    answer_variants: tuple[str, ...]
    caution_flags: tuple[str, ...]
    answer_family_metadata: Mapping[str, Any] = field(default_factory=dict)
    decomposed_signals: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectionRationale:
    summary: str
    winning_factors: tuple[str, ...]
    cautionary_factors: tuple[str, ...]
    tie_break_explanation: str
    margin_to_runner_up: float
    evidence: SelectedClusterEvidence
    score_breakdown: FinalScoreBreakdown


@dataclass(frozen=True)
class RankedCluster:
    rank: int
    cluster_id: str
    display_answer: str
    candidate: CandidateAnswer
    evidence: SelectedClusterEvidence
    cautions: tuple[str, ...]
    score_breakdown: FinalScoreBreakdown
    final_score: float
    submission_answer: int | None
    tie_break_key: tuple[float, float, float, float, float, float, float, str, str]


@dataclass(frozen=True)
class FinalSelectionResult:
    prediction: FinalPrediction
    selected_cluster: RankedCluster
    ranked_clusters: tuple[RankedCluster, ...]
    rationale: SelectionRationale
    warnings: tuple[SelectionWarning, ...]
    abstain_recommended: bool
    submission_answer: int
    metadata: Mapping[str, object] = field(default_factory=dict)

    def submission_record(self) -> dict[str, int | str]:
        return {"id": self.prediction.problem_id, "answer": self.submission_answer}


@dataclass(frozen=True)
class _ClusterRecord:
    cluster_id: str
    display_answer: str
    candidate: CandidateAnswer
    evidence: SelectedClusterEvidence
    cautions: tuple[str, ...]
    upstream_weighted_score: float
    raw_item: object | None = None


def rank_weighted_clusters(
    weighted_clusters: Sequence[AnswerCluster | CandidateAnswer | WeightedClusterScore] | ClusteringResult | EntropyWeightingBundle,
    *,
    config: FinalSelectorConfig | None = None,
) -> tuple[RankedCluster, ...]:
    resolved_config = config or FinalSelectorConfig()
    records = [_coerce_cluster_record(item) for item in _coerce_cluster_sequence(weighted_clusters)]
    if not records:
        return ()

    disagreement_level = _global_disagreement(records)
    ranked: list[RankedCluster] = []
    for record in records:
        submission_answer = _parse_submission_answer(record.candidate.answer_canonical or record.candidate.answer)
        breakdown = _score_cluster(
            record,
            submission_answer=submission_answer,
            disagreement_level=disagreement_level,
            config=resolved_config,
        )
        ranked.append(
            RankedCluster(
                rank=0,
                cluster_id=record.cluster_id,
                display_answer=record.display_answer,
                candidate=record.candidate,
                evidence=record.evidence,
                cautions=record.cautions,
                score_breakdown=breakdown,
                final_score=breakdown.final_score,
                submission_answer=submission_answer,
                tie_break_key=_tie_break_key(record, breakdown),
            )
        )

    ranked.sort(key=lambda item: item.tie_break_key, reverse=True)
    return tuple(
        RankedCluster(
            rank=index + 1,
            cluster_id=item.cluster_id,
            display_answer=item.display_answer,
            candidate=item.candidate,
            evidence=item.evidence,
            cautions=item.cautions,
            score_breakdown=item.score_breakdown,
            final_score=item.final_score,
            submission_answer=item.submission_answer,
            tie_break_key=item.tie_break_key,
        )
        for index, item in enumerate(ranked)
    )


def select_final_answer(
    problem_id: str,
    weighted_clusters: Sequence[AnswerCluster | CandidateAnswer | WeightedClusterScore] | ClusteringResult | EntropyWeightingBundle,
    *,
    num_branches_generated: int | None = None,
    num_branches_survived: int | None = None,
    solve_time_sec: float = 0.0,
    method_used: str = "weighted_cluster_final_selector",
    config: FinalSelectorConfig | None = None,
) -> FinalSelectionResult:
    resolved_config = config or FinalSelectorConfig()
    ranked = rank_weighted_clusters(weighted_clusters, config=resolved_config)

    if not ranked:
        fallback_candidate = CandidateAnswer(
            answer=str(resolved_config.default_submission_answer),
            answer_canonical=str(resolved_config.default_submission_answer),
            branch_ids=[],
            verifier_score=0.0,
            tool_consistency=0.0,
            answer_agreement=0.0,
            branch_novelty=0.0,
            symbolic_check=0.0,
            composite_score=0.0,
            cluster_size=0,
            entropy_penalty=1.0,
        )
        fallback_evidence = SelectedClusterEvidence(
            support_size=0,
            support_share=0.0,
            diversity_weighted_support=0.0,
            unique_branch_families=0,
            family_diversity=0.0,
            mean_verifier_score=0.0,
            mean_tool_consistency=0.0,
            mean_branch_novelty=0.0,
            exact_symbolic_rate=0.0,
            mean_provenance_strength=0.0,
            mean_retrieval_relevance=0.0,
            mean_evidence_quality=0.0,
            mean_discharge_fraction=0.0,
            mean_open_obligation_burden=1.0,
            critique_ratio=0.0,
            confidence_dispersion=1.0,
            best_branch_id=None,
            best_member_score=0.0,
            entropy_penalty=1.0,
            branch_ids=(),
            answer_variants=(),
            caution_flags=(),
            answer_family_metadata={},
            decomposed_signals=_empty_signal_map(),
        )
        fallback_breakdown = FinalScoreBreakdown(
            weighted_evidence_score=0.0,
            upstream_weighted_score=0.0,
            evidence_bonus=0.0,
            family_bonus=0.0,
            quality_bonus=0.0,
            disagreement_penalty=0.0,
            fragility_penalty=0.0,
            caution_penalty=0.0,
            invalid_answer_penalty=0.0,
            final_score=0.0,
        )
        fallback_ranked = RankedCluster(
            rank=1,
            cluster_id="answer::fallback",
            display_answer=str(resolved_config.default_submission_answer),
            candidate=fallback_candidate,
            evidence=fallback_evidence,
            cautions=(),
            score_breakdown=fallback_breakdown,
            final_score=0.0,
            submission_answer=resolved_config.default_submission_answer,
            tie_break_key=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "answer::fallback", ""),
        )
        warnings = (
            SelectionWarning.NO_VALID_CANDIDATES,
            SelectionWarning.ABSTENTION_RECOMMENDED,
        )
        prediction = FinalPrediction(
            problem_id=problem_id,
            final_answer=resolved_config.default_submission_answer,
            confidence=0.0,
            winning_cluster=fallback_candidate,
            num_branches_generated=num_branches_generated or 0,
            num_branches_survived=num_branches_survived or 0,
            solve_time_sec=solve_time_sec,
            method_used=method_used,
            signal_decomposition={
                **_empty_signal_map(),
                "selector_confidence": 0.0,
                "disagreement_level": 1.0,
                "abstention_pressure": 1.0,
                "warning_pressure": 1.0,
            },
            calibration_summary={
                "selector_status": "fallback",
                "warnings": [warning.value for warning in warnings],
                "abstain_recommended": True,
                "winner_score": 0.0,
                "runner_up_score": None,
                "margin_to_runner_up": 0.0,
                "cluster_count": 0,
                "support_share": 0.0,
            },
            provenance={
                "selector": "final_selector.v3",
                "winner_cluster_id": "answer::fallback",
                "runner_up_cluster_id": None,
                "branch_ids": [],
                "caution_flags": [],
            },
        )
        rationale = SelectionRationale(
            summary="No valid weighted answer families were available; emitted deterministic fallback answer.",
            winning_factors=(),
            cautionary_factors=("no_valid_candidates", "fallback_submission_answer"),
            tie_break_explanation="Fallback path due to empty ranked cluster set.",
            margin_to_runner_up=0.0,
            evidence=fallback_evidence,
            score_breakdown=fallback_breakdown,
        )
        return FinalSelectionResult(
            prediction=prediction,
            selected_cluster=fallback_ranked,
            ranked_clusters=(fallback_ranked,),
            rationale=rationale,
            warnings=warnings,
            abstain_recommended=True,
            submission_answer=resolved_config.default_submission_answer,
            metadata={
                "cluster_count": 0,
                "selector_status": "fallback",
                "winner_cluster_id": "answer::fallback",
                "runner_up_cluster_id": None,
                "winner_score": 0.0,
                "runner_up_score": None,
                "margin_to_runner_up": 0.0,
                "submission_answer": resolved_config.default_submission_answer,
            },
        )

    winner = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    margin = winner.final_score - runner_up.final_score if runner_up is not None else winner.final_score
    warnings = _selection_warnings(winner, runner_up=runner_up, config=resolved_config)
    abstain_recommended = resolved_config.allow_abstention and SelectionWarning.ABSTENTION_RECOMMENDED in warnings
    submission_answer = (
        winner.submission_answer
        if winner.submission_answer is not None
        else resolved_config.default_submission_answer
    )

    signal_decomposition = _prediction_signal_decomposition(
        winner,
        runner_up=runner_up,
        margin=margin,
        warnings=warnings,
        config=resolved_config,
    )
    confidence = _prediction_confidence(
        winner,
        runner_up=runner_up,
        margin=margin,
        warnings=warnings,
        config=resolved_config,
    )
    prediction = FinalPrediction(
        problem_id=problem_id,
        final_answer=submission_answer,
        confidence=confidence,
        winning_cluster=winner.candidate,
        num_branches_generated=num_branches_generated or _infer_total_branch_count(ranked),
        num_branches_survived=num_branches_survived or _infer_surviving_branch_count(ranked),
        solve_time_sec=solve_time_sec,
        method_used=method_used,
        signal_decomposition=signal_decomposition,
        calibration_summary={
            "selector_status": "ok",
            "warnings": [warning.value for warning in warnings],
            "abstain_recommended": abstain_recommended,
            "winner_score": round(winner.final_score, 4),
            "runner_up_score": round(runner_up.final_score, 4) if runner_up is not None else None,
            "margin_to_runner_up": round(margin, 4),
            "cluster_count": len(ranked),
            "support_share": round(winner.evidence.support_share, 4),
            "family_diversity": round(winner.evidence.family_diversity, 4),
            "confidence_dispersion": round(winner.evidence.confidence_dispersion, 4),
            "discharge_fraction": round(winner.evidence.mean_discharge_fraction, 4),
            "open_obligation_burden": round(winner.evidence.mean_open_obligation_burden, 4),
            "disagreement_level": round(signal_decomposition["disagreement_level"], 4),
            "abstention_pressure": round(signal_decomposition["abstention_pressure"], 4),
        },
        provenance={
            "selector": "final_selector.v3",
            "winner_cluster_id": winner.cluster_id,
            "runner_up_cluster_id": runner_up.cluster_id if runner_up is not None else None,
            "branch_ids": list(winner.evidence.branch_ids),
            "caution_flags": list(winner.evidence.caution_flags),
            "answer_variants": list(winner.evidence.answer_variants),
        },
    )
    rationale = _build_rationale(winner, runner_up=runner_up, margin=margin, warnings=warnings)
    metadata = {
        "cluster_count": len(ranked),
        "selector_status": "ok",
        "winner_cluster_id": winner.cluster_id,
        "runner_up_cluster_id": runner_up.cluster_id if runner_up is not None else None,
        "winner_score": winner.final_score,
        "runner_up_score": runner_up.final_score if runner_up is not None else None,
        "margin_to_runner_up": margin,
        "submission_answer": submission_answer,
        "abstain_recommended": abstain_recommended,
    }
    return FinalSelectionResult(
        prediction=prediction,
        selected_cluster=winner,
        ranked_clusters=ranked,
        rationale=rationale,
        warnings=warnings,
        abstain_recommended=abstain_recommended,
        submission_answer=submission_answer,
        metadata=metadata,
    )


def _coerce_cluster_sequence(
    weighted_clusters: Sequence[AnswerCluster | CandidateAnswer | WeightedClusterScore] | ClusteringResult | EntropyWeightingBundle,
) -> tuple[AnswerCluster | CandidateAnswer | WeightedClusterScore, ...]:
    if isinstance(weighted_clusters, EntropyWeightingBundle):
        return tuple(weighted_clusters.ranked_clusters)
    if isinstance(weighted_clusters, ClusteringResult):
        return tuple(weighted_clusters.clusters)
    return tuple(weighted_clusters)


def _coerce_cluster_record(item: AnswerCluster | CandidateAnswer | WeightedClusterScore) -> _ClusterRecord:
    if isinstance(item, WeightedClusterScore):
        candidate = item.as_candidate_answer()
        signals = _normalize_signal_map(
            {
                **(item.decomposed_signals or {}),
                "logical_consistency": item.decomposed_signals.get("logical_consistency", item.verifier_score)
                if item.decomposed_signals
                else item.verifier_score,
                "symbolic_agreement": item.decomposed_signals.get("symbolic_agreement", item.exact_symbolic_support)
                if item.decomposed_signals
                else item.exact_symbolic_support,
                "completeness": item.decomposed_signals.get("completeness", item.mean_evidence_quality)
                if item.decomposed_signals
                else item.mean_evidence_quality,
                "repairability": item.decomposed_signals.get("repairability", 1.0 - item.uncertainty)
                if item.decomposed_signals
                else 1.0 - item.uncertainty,
                "answer_correctness_likelihood": item.decomposed_signals.get("answer_correctness_likelihood", item.verifier_score)
                if item.decomposed_signals
                else item.verifier_score,
                "step_quality": item.decomposed_signals.get("step_quality", item.mean_evidence_quality)
                if item.decomposed_signals
                else item.mean_evidence_quality,
                "prefix_quality": item.decomposed_signals.get("prefix_quality", item.mean_evidence_quality)
                if item.decomposed_signals
                else item.mean_evidence_quality,
                "prm_prefix_quality": item.decomposed_signals.get("prm_prefix_quality", item.mean_evidence_quality)
                if item.decomposed_signals
                else item.mean_evidence_quality,
                "retrieval_compatibility": item.decomposed_signals.get("retrieval_compatibility", item.mean_retrieval_relevance)
                if item.decomposed_signals
                else item.mean_retrieval_relevance,
                "retrieval_support": item.decomposed_signals.get("retrieval_support", item.retrieval_support)
                if item.decomposed_signals
                else item.retrieval_support,
                "operator_reliability": item.decomposed_signals.get("operator_reliability", item.mean_provenance_strength)
                if item.decomposed_signals
                else item.mean_provenance_strength,
                "open_obligation_burden": item.decomposed_signals.get("open_obligation_burden", item.uncertainty)
                if item.decomposed_signals
                else item.uncertainty,
                "discharge_fraction": item.decomposed_signals.get("discharge_fraction", item.exact_symbolic_support)
                if item.decomposed_signals
                else item.exact_symbolic_support,
                "critique_support": item.decomposed_signals.get("critique_support", 0.0)
                if item.decomposed_signals
                else 0.0,
                "confidence_dispersion": item.decomposed_signals.get("confidence_dispersion", item.uncertainty)
                if item.decomposed_signals
                else item.uncertainty,
            }
        )
        evidence = SelectedClusterEvidence(
            support_size=int(item.cluster_size),
            support_share=_clamp01(item.support_share),
            diversity_weighted_support=_clamp01(item.diversity_weighted_support),
            unique_branch_families=int(item.unique_branch_families),
            family_diversity=_clamp01(item.family_diversity),
            mean_verifier_score=_clamp01(item.verifier_score),
            mean_tool_consistency=_clamp01(item.weak_symbolic_support),
            mean_branch_novelty=_clamp01(item.branch_novelty),
            exact_symbolic_rate=_clamp01(item.exact_symbolic_support),
            mean_provenance_strength=_clamp01(item.mean_provenance_strength),
            mean_retrieval_relevance=_clamp01(item.mean_retrieval_relevance),
            mean_evidence_quality=_clamp01(item.mean_evidence_quality),
            mean_discharge_fraction=_clamp01(signals.get("discharge_fraction", item.exact_symbolic_support)),
            mean_open_obligation_burden=_clamp01(signals.get("open_obligation_burden", item.uncertainty)),
            critique_ratio=_clamp01(1.0 if "critique_heavy" in item.caution_flags else 0.0),
            confidence_dispersion=_clamp01(signals.get("confidence_dispersion", item.uncertainty)),
            best_branch_id=item.best_branch_id,
            best_member_score=_clamp01(item.best_member_score),
            entropy_penalty=_clamp01(item.entropy_penalty),
            branch_ids=tuple(item.branch_ids),
            answer_variants=tuple(item.answer_variants),
            caution_flags=tuple(item.caution_flags),
            answer_family_metadata={"family_diversity": _clamp01(item.family_diversity)},
            decomposed_signals=signals,
        )
        return _ClusterRecord(
            cluster_id=item.cluster_id,
            display_answer=item.answer,
            candidate=candidate,
            evidence=evidence,
            cautions=tuple(item.caution_flags),
            upstream_weighted_score=_clamp01(item.composite_score),
            raw_item=item,
        )

    if isinstance(item, AnswerCluster):
        summary = item.evidence_summary
        candidate = item.to_candidate_answer()
        signals = _normalize_signal_map(
            {
                "logical_consistency": summary.mean_logical_consistency,
                "symbolic_agreement": summary.exact_symbolic_rate,
                "completeness": summary.mean_completeness,
                "repairability": _clamp01(max(0.0, summary.critique_adjustment + summary.mean_discharge_fraction)),
                "answer_correctness_likelihood": summary.mean_verifier_score,
                "step_quality": _clamp01(_mean((summary.mean_prefix_quality, summary.mean_prm_prefix_quality, summary.mean_evidence_quality))),
                "prefix_quality": summary.mean_prefix_quality,
                "prm_prefix_quality": summary.mean_prm_prefix_quality,
                "retrieval_compatibility": summary.mean_retrieval_relevance,
                "retrieval_support": summary.mean_retrieval_support,
                "operator_reliability": summary.mean_operator_reliability,
                "open_obligation_burden": summary.mean_open_obligation_burden,
                "discharge_fraction": summary.mean_discharge_fraction,
                "critique_support": summary.critique_adjustment,
                "confidence_dispersion": summary.confidence_dispersion,
                "family_diversity": summary.family_diversity,
            }
        )
        evidence = SelectedClusterEvidence(
            support_size=int(summary.support_size),
            support_share=_clamp01(summary.support_share),
            diversity_weighted_support=_clamp01(summary.diversity_weighted_support),
            unique_branch_families=int(summary.unique_branch_families),
            family_diversity=_clamp01(summary.family_diversity),
            mean_verifier_score=_clamp01(summary.mean_verifier_score),
            mean_tool_consistency=_clamp01(summary.mean_tool_consistency),
            mean_branch_novelty=_clamp01(summary.mean_branch_novelty),
            exact_symbolic_rate=_clamp01(summary.exact_symbolic_rate),
            mean_provenance_strength=_clamp01(summary.mean_provenance_strength),
            mean_retrieval_relevance=_clamp01(summary.mean_retrieval_relevance),
            mean_evidence_quality=_clamp01(summary.mean_evidence_quality),
            mean_discharge_fraction=_clamp01(summary.mean_discharge_fraction),
            mean_open_obligation_burden=_clamp01(summary.mean_open_obligation_burden),
            critique_ratio=_clamp01(summary.critique_ratio),
            confidence_dispersion=_clamp01(summary.confidence_dispersion),
            best_branch_id=summary.best_branch_id,
            best_member_score=_clamp01(summary.best_member_score),
            entropy_penalty=_clamp01(summary.entropy_penalty),
            branch_ids=tuple(item.branch_ids),
            answer_variants=tuple(summary.answer_variants),
            caution_flags=tuple(
                caution.value if hasattr(caution, "value") else str(caution) for caution in item.cautions
            ),
            answer_family_metadata=dict(item.answer_family_metadata),
            decomposed_signals=signals,
        )
        return _ClusterRecord(
            cluster_id=item.cluster_id,
            display_answer=item.display_answer,
            candidate=candidate,
            evidence=evidence,
            cautions=tuple(
                caution.value if hasattr(caution, "value") else str(caution) for caution in item.cautions
            ),
            upstream_weighted_score=_clamp01(item.preliminary_score),
            raw_item=item,
        )

    candidate = item
    answer = candidate.answer_canonical or candidate.answer
    evidence = SelectedClusterEvidence(
        support_size=max(0, int(getattr(candidate, "cluster_size", 0) or 0)),
        support_share=_clamp01(getattr(candidate, "answer_agreement", 0.0) or 0.0),
        diversity_weighted_support=_clamp01(getattr(candidate, "answer_agreement", 0.0) or 0.0),
        unique_branch_families=len(set(candidate.branch_ids)),
        family_diversity=_clamp01(
            len(set(candidate.branch_ids)) / max(1, int(getattr(candidate, "cluster_size", 1) or 1))
        ),
        mean_verifier_score=_clamp01(candidate.verifier_score),
        mean_tool_consistency=_clamp01(candidate.tool_consistency),
        mean_branch_novelty=_clamp01(candidate.branch_novelty),
        exact_symbolic_rate=_clamp01(candidate.symbolic_check),
        mean_provenance_strength=_clamp01(candidate.composite_score),
        mean_retrieval_relevance=_clamp01(getattr(candidate, "retrieval_support", 0.0) or 0.0),
        mean_evidence_quality=_clamp01(candidate.composite_score),
        mean_discharge_fraction=_clamp01(candidate.symbolic_check),
        mean_open_obligation_burden=_clamp01(1.0 - candidate.symbolic_check),
        critique_ratio=0.0,
        confidence_dispersion=_clamp01(candidate.entropy_penalty),
        best_branch_id=candidate.branch_ids[0] if candidate.branch_ids else None,
        best_member_score=_clamp01(candidate.composite_score),
        entropy_penalty=_clamp01(candidate.entropy_penalty),
        branch_ids=tuple(candidate.branch_ids),
        answer_variants=(answer,),
        caution_flags=(),
        answer_family_metadata={},
        decomposed_signals=_normalize_signal_map(
            {
                "logical_consistency": candidate.verifier_score,
                "symbolic_agreement": candidate.symbolic_check,
                "completeness": candidate.composite_score,
                "repairability": candidate.tool_consistency,
                "answer_correctness_likelihood": candidate.verifier_score,
                "step_quality": candidate.composite_score,
                "prefix_quality": candidate.composite_score,
                "prm_prefix_quality": candidate.composite_score,
                "retrieval_compatibility": getattr(candidate, "retrieval_support", 0.0) or 0.0,
                "retrieval_support": getattr(candidate, "retrieval_support", 0.0) or 0.0,
                "operator_reliability": candidate.composite_score,
                "open_obligation_burden": 1.0 - candidate.symbolic_check,
                "discharge_fraction": candidate.symbolic_check,
                "critique_support": 0.0,
                "confidence_dispersion": candidate.entropy_penalty,
                "family_diversity": _clamp01(
                    len(set(candidate.branch_ids)) / max(1, int(getattr(candidate, "cluster_size", 1) or 1))
                ),
            }
        ),
    )
    return _ClusterRecord(
        cluster_id=f"answer::{answer}",
        display_answer=answer,
        candidate=candidate,
        evidence=evidence,
        cautions=(),
        upstream_weighted_score=_clamp01(candidate.composite_score),
        raw_item=item,
    )


def _score_cluster(
    record: _ClusterRecord,
    *,
    submission_answer: int | None,
    disagreement_level: float,
    config: FinalSelectorConfig,
) -> FinalScoreBreakdown:
    evidence = record.evidence
    signals = evidence.decomposed_signals
    upstream_weighted_score = _clamp01(record.upstream_weighted_score)

    evidence_quality = _clamp01(
        _mean(
            (
                signals.get("logical_consistency", evidence.mean_verifier_score),
                signals.get("symbolic_agreement", evidence.exact_symbolic_rate),
                signals.get("completeness", evidence.mean_evidence_quality),
                signals.get("answer_correctness_likelihood", evidence.mean_verifier_score),
                signals.get("step_quality", evidence.mean_evidence_quality),
                signals.get("prm_prefix_quality", evidence.mean_evidence_quality),
                signals.get("retrieval_support", evidence.mean_retrieval_relevance),
                signals.get("operator_reliability", evidence.mean_provenance_strength),
            )
        )
    )
    family_bonus = config.family_bonus_weight * _clamp01(
        0.45 * evidence.support_share
        + 0.30 * evidence.diversity_weighted_support
        + 0.25 * evidence.family_diversity
    )
    quality_bonus = config.quality_bonus_weight * _clamp01(
        0.40 * signals.get("logical_consistency", evidence.mean_verifier_score)
        + 0.30 * signals.get("symbolic_agreement", evidence.exact_symbolic_rate)
        + 0.30 * signals.get("completeness", evidence.mean_evidence_quality)
    )
    evidence_bonus = config.evidence_bonus_weight * _clamp01(
        0.35 * signals.get("prm_prefix_quality", evidence.mean_evidence_quality)
        + 0.25 * signals.get("retrieval_compatibility", evidence.mean_retrieval_relevance)
        + 0.20 * signals.get("operator_reliability", evidence.mean_provenance_strength)
        + 0.20 * signals.get("discharge_fraction", evidence.mean_discharge_fraction)
    )

    caution_penalty = 0.0
    if "singleton" in record.cautions:
        caution_penalty += config.caution_penalty_singleton
    if "low_support_outlier" in record.cautions:
        caution_penalty += config.caution_penalty_outlier
    if "low_diversity_support" in record.cautions:
        caution_penalty += config.caution_penalty_low_diversity
    if "weak_evidence" in record.cautions:
        caution_penalty += config.caution_penalty_weak_evidence

    obligation_penalty = config.obligation_penalty_weight * _clamp01(
        0.60 * signals.get("open_obligation_burden", evidence.mean_open_obligation_burden)
        + 0.40 * max(0.0, 1.0 - signals.get("discharge_fraction", evidence.mean_discharge_fraction))
    )
    critique_penalty = config.critique_penalty_weight * _clamp01(
        0.65 * evidence.critique_ratio + 0.35 * max(0.0, 1.0 - signals.get("critique_support", 0.0))
    )
    disagreement_penalty = config.disagreement_penalty_weight * _clamp01(
        0.55 * disagreement_level
        + 0.25 * evidence.confidence_dispersion
        + 0.20 * max(0.0, 1.0 - evidence.family_diversity)
    )
    fragility_penalty = config.fragility_penalty_weight * _clamp01(
        0.35 * (1.0 if evidence.support_size <= 1 else 0.0)
        + 0.25 * (1.0 if evidence.support_share < config.minimum_support_share else 0.0)
        + 0.20 * evidence.entropy_penalty
        + 0.20 * max(0.0, 1.0 - evidence_quality)
    )

    invalid_answer_penalty = 0.0
    if submission_answer is None:
        invalid_answer_penalty += config.invalid_answer_penalty

    final_score = _clamp01(
        upstream_weighted_score
        + family_bonus
        + quality_bonus
        + evidence_bonus
        - caution_penalty
        - obligation_penalty
        - critique_penalty
        - disagreement_penalty
        - fragility_penalty
        - invalid_answer_penalty
    )
    return FinalScoreBreakdown(
        weighted_evidence_score=evidence_quality,
        upstream_weighted_score=upstream_weighted_score,
        evidence_bonus=_clamp01(evidence_bonus),
        family_bonus=_clamp01(family_bonus),
        quality_bonus=_clamp01(quality_bonus),
        disagreement_penalty=_clamp01(disagreement_penalty),
        fragility_penalty=_clamp01(fragility_penalty + obligation_penalty + critique_penalty),
        caution_penalty=_clamp01(caution_penalty),
        invalid_answer_penalty=_clamp01(invalid_answer_penalty),
        final_score=final_score,
    )


def _tie_break_key(record: _ClusterRecord, breakdown: FinalScoreBreakdown) -> tuple[float, float, float, float, float, float, float, str, str]:
    evidence = record.evidence
    signals = evidence.decomposed_signals
    return (
        breakdown.final_score,
        signals.get("answer_correctness_likelihood", evidence.mean_verifier_score),
        signals.get("symbolic_agreement", evidence.exact_symbolic_rate),
        evidence.support_share,
        evidence.family_diversity,
        evidence.best_member_score,
        1.0 - evidence.confidence_dispersion,
        record.cluster_id,
        record.display_answer,
    )


def _selection_warnings(
    winner: RankedCluster,
    *,
    runner_up: RankedCluster | None,
    config: FinalSelectorConfig,
) -> tuple[SelectionWarning, ...]:
    warnings: list[SelectionWarning] = []
    margin = winner.final_score - runner_up.final_score if runner_up is not None else winner.final_score
    evidence = winner.evidence

    if winner.submission_answer is None:
        raw = winner.candidate.answer_canonical or winner.candidate.answer
        if raw is None or not _INTEGER_RE.match(str(raw).strip()):
            warnings.append(SelectionWarning.NON_INTEGER_WINNER)
        else:
            warnings.append(SelectionWarning.OUT_OF_RANGE_WINNER)

    if winner.final_score < config.minimum_final_score:
        warnings.append(SelectionWarning.LOW_CONFIDENCE)
    if margin < config.minimum_margin:
        warnings.append(SelectionWarning.LOW_MARGIN)
    if runner_up is not None:
        disagreement_level = _clamp01(
            0.60 * max(0.0, 1.0 - margin)
            + 0.20 * abs(winner.evidence.support_share - runner_up.evidence.support_share)
            + 0.20 * max(
                winner.evidence.confidence_dispersion,
                runner_up.evidence.confidence_dispersion,
            )
        )
        if disagreement_level > 0.42:
            warnings.append(SelectionWarning.HIGH_DISAGREEMENT)

    if evidence.support_size <= 1 or evidence.support_share < config.minimum_support_share:
        warnings.append(SelectionWarning.FRAGILE_WINNER)
    if "low_support_outlier" in winner.cautions or "singleton" in winner.cautions:
        warnings.append(SelectionWarning.OUTLIER_WINNER)
    if "weak_evidence" in winner.cautions or evidence.mean_evidence_quality < 0.40:
        warnings.append(SelectionWarning.WEAK_EVIDENCE)
    if winner.cautions:
        warnings.append(SelectionWarning.CLUSTER_CAUTION)
    if evidence.mean_discharge_fraction < config.minimum_discharge_fraction:
        warnings.append(SelectionWarning.LOW_DISCHARGE)
    if evidence.mean_open_obligation_burden > config.maximum_obligation_burden:
        warnings.append(SelectionWarning.HIGH_OBLIGATION_BURDEN)
    if evidence.critique_ratio > 0.60 or "critique_heavy" in winner.cautions:
        warnings.append(SelectionWarning.CRITIQUE_HEAVY_FAMILY)
    if evidence.family_diversity < 0.34 and evidence.support_size > 1:
        warnings.append(SelectionWarning.FRAGMENTED_SUPPORT)
    if evidence.mean_evidence_quality >= 0.60 and evidence.family_diversity < 0.34:
        warnings.append(SelectionWarning.STRONG_EVIDENCE_FRAGMENTED_SUPPORT)

    abstention_pressure = _abstention_pressure(winner, runner_up=runner_up, config=config)
    if config.allow_abstention and abstention_pressure >= 0.55:
        warnings.append(SelectionWarning.ABSTENTION_RECOMMENDED)

    if not warnings:
        return (SelectionWarning.NONE,)
    return tuple(_dedupe_warnings(warnings))


def _prediction_signal_decomposition(
    winner: RankedCluster,
    *,
    runner_up: RankedCluster | None,
    margin: float,
    warnings: Sequence[SelectionWarning],
    config: FinalSelectorConfig,
) -> dict[str, float]:
    signals = dict(winner.evidence.decomposed_signals)
    disagreement_level = _disagreement_level(winner, runner_up=runner_up, margin=margin)
    abstention_pressure = _abstention_pressure(winner, runner_up=runner_up, config=config)
    warning_pressure = _warning_pressure(warnings)

    decomposition = {
        **_empty_signal_map(),
        **_normalize_signal_map(signals),
        "support_share": _clamp01(winner.evidence.support_share),
        "diversity_weighted_support": _clamp01(winner.evidence.diversity_weighted_support),
        "family_diversity": _clamp01(winner.evidence.family_diversity),
        "support_dispersion": _clamp01(winner.evidence.confidence_dispersion),
        "selector_upstream_weight": _clamp01(winner.score_breakdown.upstream_weighted_score),
        "selector_evidence_bonus": _clamp01(winner.score_breakdown.evidence_bonus),
        "selector_family_bonus": _clamp01(winner.score_breakdown.family_bonus),
        "selector_quality_bonus": _clamp01(winner.score_breakdown.quality_bonus),
        "selector_disagreement_penalty": _clamp01(winner.score_breakdown.disagreement_penalty),
        "selector_fragility_penalty": _clamp01(winner.score_breakdown.fragility_penalty),
        "selector_caution_penalty": _clamp01(winner.score_breakdown.caution_penalty),
        "selector_invalid_answer_penalty": _clamp01(winner.score_breakdown.invalid_answer_penalty),
        "selector_confidence": _clamp01(winner.final_score),
        "margin_to_runner_up": _clamp01(margin),
        "disagreement_level": disagreement_level,
        "warning_pressure": warning_pressure,
        "abstention_pressure": abstention_pressure,
        "winner_fragility": _winner_fragility(winner),
        "winner_warning_count": _clamp01(len([w for w in warnings if w is not SelectionWarning.NONE]) / 8.0),
    }
    return decomposition


def _prediction_confidence(
    winner: RankedCluster,
    *,
    runner_up: RankedCluster | None,
    margin: float,
    warnings: Sequence[SelectionWarning],
    config: FinalSelectorConfig,
) -> float:
    disagreement_level = _disagreement_level(winner, runner_up=runner_up, margin=margin)
    abstention_pressure = _abstention_pressure(winner, runner_up=runner_up, config=config)
    warning_pressure = _warning_pressure(warnings)
    confidence = _clamp01(
        winner.final_score
        - 0.20 * disagreement_level
        - 0.18 * abstention_pressure
        - 0.10 * warning_pressure
        + 0.06 * _clamp01(margin)
    )
    return confidence


def _build_rationale(
    winner: RankedCluster,
    *,
    runner_up: RankedCluster | None,
    margin: float,
    warnings: Sequence[SelectionWarning],
) -> SelectionRationale:
    evidence = winner.evidence
    factors: list[str] = []
    cautionary: list[str] = []

    if winner.score_breakdown.upstream_weighted_score >= 0.55:
        factors.append("strong_upstream_weighted_evidence")
    if evidence.exact_symbolic_rate >= 0.55:
        factors.append("strong_symbolic_agreement")
    if evidence.mean_verifier_score >= 0.55:
        factors.append("strong_verifier_support")
    if evidence.mean_discharge_fraction >= 0.55:
        factors.append("good_obligation_discharge")
    if evidence.support_share >= 0.25:
        factors.append("meaningful_family_support")
    if evidence.family_diversity >= 0.50:
        factors.append("diverse_branch_family_support")
    if evidence.mean_retrieval_relevance >= 0.35:
        factors.append("retrieval_consistent_support")
    if evidence.mean_provenance_strength >= 0.40:
        factors.append("operator_provenance_support")

    if SelectionWarning.LOW_MARGIN in warnings:
        cautionary.append("near_tie_with_runner_up")
    if SelectionWarning.HIGH_DISAGREEMENT in warnings:
        cautionary.append("high_family_disagreement")
    if SelectionWarning.FRAGILE_WINNER in warnings:
        cautionary.append("fragile_winner")
    if SelectionWarning.WEAK_EVIDENCE in warnings:
        cautionary.append("weak_evidence")
    if SelectionWarning.LOW_DISCHARGE in warnings:
        cautionary.append("low_discharge")
    if SelectionWarning.HIGH_OBLIGATION_BURDEN in warnings:
        cautionary.append("high_open_obligation_burden")
    if SelectionWarning.CRITIQUE_HEAVY_FAMILY in warnings:
        cautionary.append("critique_heavy_family")
    if SelectionWarning.FRAGMENTED_SUPPORT in warnings:
        cautionary.append("fragmented_answer_family_support")

    summary = (
        "Selected the answer family with the best bounded combination of upstream weighted evidence, "
        "family-level support structure, and final-stage safety checks."
    )
    tie_break_explanation = _tie_break_explanation(winner, runner_up=runner_up, margin=margin)

    return SelectionRationale(
        summary=summary,
        winning_factors=tuple(factors),
        cautionary_factors=tuple(cautionary),
        tie_break_explanation=tie_break_explanation,
        margin_to_runner_up=margin,
        evidence=evidence,
        score_breakdown=winner.score_breakdown,
    )


def _tie_break_explanation(
    winner: RankedCluster,
    *,
    runner_up: RankedCluster | None,
    margin: float,
) -> str:
    if runner_up is None:
        return "Only one valid answer family remained after weighted aggregation."
    if margin >= 0.10:
        return "Winner led by a clear final-selector margin after bounded arbitration."
    if winner.evidence.exact_symbolic_rate > runner_up.evidence.exact_symbolic_rate:
        return "Near-tie resolved in favor of stronger symbolic agreement and discharge evidence."
    if winner.evidence.mean_verifier_score > runner_up.evidence.mean_verifier_score:
        return "Near-tie resolved in favor of stronger verifier-backed correctness likelihood."
    if winner.evidence.support_share > runner_up.evidence.support_share:
        return "Near-tie resolved in favor of stronger family support share."
    if winner.evidence.family_diversity > runner_up.evidence.family_diversity:
        return "Near-tie resolved in favor of more diverse branch-family support."
    return "Near-tie resolved deterministically by bounded tie-break hierarchy."


def _global_disagreement(records: Sequence[_ClusterRecord]) -> float:
    if not records:
        return 0.0
    support_distribution = [record.evidence.support_share for record in records]
    entropy_like = _clamp01(sum(value for value in support_distribution if value > 0.0))
    _ = entropy_like
    if len(records) == 1:
        return 0.0
    sorted_support = sorted((record.evidence.support_share for record in records), reverse=True)
    top = sorted_support[0]
    runner = sorted_support[1] if len(sorted_support) > 1 else 0.0
    spread = max(0.0, 1.0 - (top - runner))
    mean_dispersion = _mean([record.evidence.confidence_dispersion for record in records])
    mean_fragmentation = _mean([1.0 - record.evidence.family_diversity for record in records])
    return _clamp01(0.45 * spread + 0.30 * mean_dispersion + 0.25 * mean_fragmentation)


def _disagreement_level(
    winner: RankedCluster,
    *,
    runner_up: RankedCluster | None,
    margin: float,
) -> float:
    if runner_up is None:
        return 0.0
    return _clamp01(
        0.55 * max(0.0, 1.0 - margin)
        + 0.20 * max(winner.evidence.confidence_dispersion, runner_up.evidence.confidence_dispersion)
        + 0.15 * abs(winner.evidence.support_share - runner_up.evidence.support_share)
        + 0.10 * max(1.0 - winner.evidence.family_diversity, 1.0 - runner_up.evidence.family_diversity)
    )


def _abstention_pressure(
    winner: RankedCluster,
    *,
    runner_up: RankedCluster | None,
    config: FinalSelectorConfig,
) -> float:
    margin = winner.final_score - runner_up.final_score if runner_up is not None else winner.final_score
    disagreement = _disagreement_level(winner, runner_up=runner_up, margin=margin)
    return _clamp01(
        0.30 * (1.0 if winner.final_score < config.minimum_final_score else 0.0)
        + 0.20 * (1.0 if margin < config.minimum_margin else 0.0)
        + 0.15 * (1.0 if winner.evidence.support_share < config.minimum_support_share else 0.0)
        + 0.15 * (1.0 if winner.evidence.mean_discharge_fraction < config.minimum_discharge_fraction else 0.0)
        + 0.10 * disagreement
        + 0.10 * _winner_fragility(winner)
    )


def _winner_fragility(winner: RankedCluster) -> float:
    evidence = winner.evidence
    return _clamp01(
        0.30 * (1.0 if evidence.support_size <= 1 else 0.0)
        + 0.20 * (1.0 if evidence.support_share < 0.18 else 0.0)
        + 0.20 * evidence.entropy_penalty
        + 0.15 * max(0.0, 1.0 - evidence.mean_discharge_fraction)
        + 0.15 * evidence.confidence_dispersion
    )


def _warning_pressure(warnings: Sequence[SelectionWarning]) -> float:
    if not warnings:
        return 0.0
    active = [warning for warning in warnings if warning is not SelectionWarning.NONE]
    return _clamp01(len(active) / 8.0)


def _parse_submission_answer(answer: str | None) -> int | None:
    if answer is None:
        return None
    text = str(answer).strip()
    if not text or not _INTEGER_RE.match(text):
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    if value < ANSWER_MIN or value > ANSWER_MAX:
        return None
    return value


def _infer_total_branch_count(ranked: Sequence[RankedCluster]) -> int:
    return sum(max(1, int(cluster.evidence.support_size)) for cluster in ranked)


def _infer_surviving_branch_count(ranked: Sequence[RankedCluster]) -> int:
    return sum(len(cluster.evidence.branch_ids) for cluster in ranked)


def _normalize_signal_map(values: Mapping[str, Any]) -> dict[str, float]:
    return {str(key): _clamp01(value) for key, value in values.items()}


def _empty_signal_map() -> dict[str, float]:
    return {
        "logical_consistency": 0.0,
        "symbolic_agreement": 0.0,
        "completeness": 0.0,
        "repairability": 0.0,
        "answer_correctness_likelihood": 0.0,
        "step_quality": 0.0,
        "prefix_quality": 0.0,
        "prm_prefix_quality": 0.0,
        "retrieval_compatibility": 0.0,
        "retrieval_support": 0.0,
        "operator_reliability": 0.0,
        "open_obligation_burden": 0.0,
        "discharge_fraction": 0.0,
        "critique_support": 0.0,
        "confidence_dispersion": 0.0,
        "family_diversity": 0.0,
    }


def _dedupe_warnings(warnings: Sequence[SelectionWarning]) -> list[SelectionWarning]:
    seen: set[str] = set()
    ordered: list[SelectionWarning] = []
    for warning in warnings:
        if warning.value not in seen:
            seen.add(warning.value)
            ordered.append(warning)
    return ordered


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


__all__ = [
    "FinalScoreBreakdown",
    "FinalSelectionResult",
    "FinalSelectorConfig",
    "RankedCluster",
    "SelectedClusterEvidence",
    "SelectionRationale",
    "SelectionWarning",
    "rank_weighted_clusters",
    "select_final_answer",
]
