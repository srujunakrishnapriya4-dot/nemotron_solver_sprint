"""Deterministic metric, scoring, calibration, and evaluation helpers.

This module is the shared numeric contract for the repo:
- routing confidence and uncertainty summaries
- branch / verifier / aggregation score composition
- retrieval ranking and diversity helpers
- offline calibration and evaluation metrics

It intentionally avoids runtime/model dependencies and keeps all helpers pure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from time import monotonic_ns
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

from src.common.constants import (
    ROUTING_HIGH_ENTROPY_THRESHOLD,
    W_ANSWER_AGREEMENT,
    W_BRANCH_NOVELTY,
    W_SYMBOLIC_CHECK,
    W_TOOL_CONSISTENCY,
    W_VERIFIER,
)
from src.common.utils import safe_normalize_text


EPSILON = 1e-12
DEFAULT_ECE_BUCKETS = 10
DEFAULT_HISTOGRAM_BOUNDARIES = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


@dataclass(frozen=True)
class CounterMetric:
    name: str
    value: float = 0.0
    unit: str = "count"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def increment(self, amount: Any = 1.0) -> "CounterMetric":
        return CounterMetric(
            name=self.name,
            value=self.value + _as_float(amount, default=0.0),
            unit=self.unit,
            metadata=dict(self.metadata),
        )

    def merge(self, other: "CounterMetric") -> "CounterMetric":
        _ensure_metric_compatible(self.name, other.name, metric_type="counter")
        return CounterMetric(
            name=self.name,
            value=self.value + other.value,
            unit=self.unit,
            metadata=dict(self.metadata or other.metadata),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "counter",
            "value": self.value,
            "unit": self.unit,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TimerMetric:
    name: str
    count: int = 0
    total_seconds: float = 0.0
    min_seconds: float | None = None
    max_seconds: float | None = None
    last_seconds: float | None = None
    unit: str = "seconds"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def mean_seconds(self) -> float:
        return safe_divide(self.total_seconds, self.count, default=0.0)

    def record(self, duration_seconds: Any) -> "TimerMetric":
        value = max(0.0, _as_float(duration_seconds, default=0.0))
        next_count = self.count + 1
        next_min = value if self.min_seconds is None else min(self.min_seconds, value)
        next_max = value if self.max_seconds is None else max(self.max_seconds, value)
        return TimerMetric(
            name=self.name,
            count=next_count,
            total_seconds=self.total_seconds + value,
            min_seconds=next_min,
            max_seconds=next_max,
            last_seconds=value,
            unit=self.unit,
            metadata=dict(self.metadata),
        )

    def merge(self, other: "TimerMetric") -> "TimerMetric":
        _ensure_metric_compatible(self.name, other.name, metric_type="timer")
        min_seconds = self.min_seconds
        if other.min_seconds is not None:
            min_seconds = other.min_seconds if min_seconds is None else min(min_seconds, other.min_seconds)
        max_seconds = self.max_seconds
        if other.max_seconds is not None:
            max_seconds = other.max_seconds if max_seconds is None else max(max_seconds, other.max_seconds)
        return TimerMetric(
            name=self.name,
            count=self.count + other.count,
            total_seconds=self.total_seconds + other.total_seconds,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            last_seconds=other.last_seconds if other.last_seconds is not None else self.last_seconds,
            unit=self.unit,
            metadata=dict(self.metadata or other.metadata),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "timer",
            "count": self.count,
            "total_seconds": self.total_seconds,
            "mean_seconds": self.mean_seconds,
            "min_seconds": self.min_seconds,
            "max_seconds": self.max_seconds,
            "last_seconds": self.last_seconds,
            "unit": self.unit,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class HistogramMetric:
    name: str
    boundaries: tuple[float, ...]
    bucket_counts: tuple[int, ...]
    count: int = 0
    total: float = 0.0
    min_value: float | None = None
    max_value: float | None = None
    unit: str = "value"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        name: str,
        *,
        boundaries: Sequence[Any] = DEFAULT_HISTOGRAM_BOUNDARIES,
        unit: str = "value",
        metadata: Mapping[str, Any] | None = None,
    ) -> "HistogramMetric":
        normalized = tuple(sorted({_as_float(boundary, default=0.0) for boundary in boundaries}))
        return cls(
            name=name,
            boundaries=normalized,
            bucket_counts=tuple(0 for _ in range(len(normalized) + 1)),
            unit=unit,
            metadata=dict(metadata or {}),
        )

    @property
    def mean(self) -> float:
        return safe_divide(self.total, self.count, default=0.0)

    def record(self, value: Any) -> "HistogramMetric":
        numeric = _as_float(value, default=0.0)
        index = _histogram_bucket_index(numeric, self.boundaries)
        counts = list(self.bucket_counts)
        counts[index] += 1
        next_min = numeric if self.min_value is None else min(self.min_value, numeric)
        next_max = numeric if self.max_value is None else max(self.max_value, numeric)
        return HistogramMetric(
            name=self.name,
            boundaries=self.boundaries,
            bucket_counts=tuple(counts),
            count=self.count + 1,
            total=self.total + numeric,
            min_value=next_min,
            max_value=next_max,
            unit=self.unit,
            metadata=dict(self.metadata),
        )

    def merge(self, other: "HistogramMetric") -> "HistogramMetric":
        _ensure_metric_compatible(self.name, other.name, metric_type="histogram")
        if self.boundaries != other.boundaries:
            raise ValueError("Histogram boundaries must match to merge metrics deterministically.")
        min_value = self.min_value
        if other.min_value is not None:
            min_value = other.min_value if min_value is None else min(min_value, other.min_value)
        max_value = self.max_value
        if other.max_value is not None:
            max_value = other.max_value if max_value is None else max(max_value, other.max_value)
        return HistogramMetric(
            name=self.name,
            boundaries=self.boundaries,
            bucket_counts=tuple(left + right for left, right in zip(self.bucket_counts, other.bucket_counts)),
            count=self.count + other.count,
            total=self.total + other.total,
            min_value=min_value,
            max_value=max_value,
            unit=self.unit,
            metadata=dict(self.metadata or other.metadata),
        )

    def as_dict(self) -> dict[str, Any]:
        buckets = []
        lower: float | None = None
        for index, count in enumerate(self.bucket_counts):
            upper = self.boundaries[index] if index < len(self.boundaries) else None
            buckets.append(
                {
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "count": count,
                }
            )
            lower = upper
        return {
            "name": self.name,
            "type": "histogram",
            "count": self.count,
            "total": self.total,
            "mean": self.mean,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "unit": self.unit,
            "boundaries": list(self.boundaries),
            "buckets": buckets,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MetricCollection:
    counters: tuple[CounterMetric, ...] = ()
    timers: tuple[TimerMetric, ...] = ()
    histograms: tuple[HistogramMetric, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "counters": [metric.as_dict() for metric in self.counters],
            "timers": [metric.as_dict() for metric in self.timers],
            "histograms": [metric.as_dict() for metric in self.histograms],
        }


@dataclass(frozen=True)
class DistributionSummary:
    top_label: str | None
    top_probability: float
    second_probability: float
    margin: float
    entropy: float
    normalized_entropy: float
    dominant_ratio: float
    consensus_strength: float
    uncertainty_penalty: float
    confidence: float
    total_mass: float
    support_size: int
    probabilities: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DiversityCoverageSummary:
    total_items: int
    unique_items: int
    diversity_ratio: float
    coverage_ratio: float
    dominant_ratio: float
    counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalQualitySummary:
    mean_score: float
    top_score: float
    support_ratio: float
    coverage_ratio: float
    diversity_ratio: float
    quality_score: float


@dataclass(frozen=True)
class BranchScoreComponents:
    verifier_probability: float = 0.0
    tool_consistency: float = 0.0
    answer_agreement: float = 0.0
    branch_novelty: float = 0.0
    exact_symbolic_check: float = 0.0
    retrieval_support: float = 0.0
    penalty: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "verifier_probability": self.verifier_probability,
            "tool_consistency": self.tool_consistency,
            "answer_agreement": self.answer_agreement,
            "branch_novelty": self.branch_novelty,
            "exact_symbolic_check": self.exact_symbolic_check,
            "retrieval_support": self.retrieval_support,
            "penalty": self.penalty,
        }


@dataclass(frozen=True)
class BranchScoreWeights:
    verifier_weight: float = W_VERIFIER
    tool_consistency_weight: float = W_TOOL_CONSISTENCY
    answer_agreement_weight: float = W_ANSWER_AGREEMENT
    branch_novelty_weight: float = W_BRANCH_NOVELTY
    exact_symbolic_weight: float = W_SYMBOLIC_CHECK
    retrieval_bonus_weight: float = 0.05


@dataclass(frozen=True)
class BranchScoreDecomposition:
    components: BranchScoreComponents
    weights: BranchScoreWeights
    weighted_core: float
    retrieval_bonus: float
    penalty: float
    composite_score: float


@dataclass(frozen=True)
class CalibrationBucketSummary:
    bucket_index: int
    lower_bound: float
    upper_bound: float
    count: int
    mean_confidence: float
    empirical_accuracy: float
    confidence_gap: float


@dataclass(frozen=True)
class BinaryCalibrationSummary:
    count: int
    accuracy: float
    mean_confidence: float
    positive_rate: float
    brier_score: float
    expected_calibration_error: float
    max_calibration_error: float
    buckets: tuple[CalibrationBucketSummary, ...] = ()


def monotonic_time_ns() -> int:
    return monotonic_ns()


def duration_seconds_from_ns(start_ns: int, end_ns: int | None = None) -> float:
    finish = monotonic_ns() if end_ns is None else int(end_ns)
    start = int(start_ns)
    if finish <= start:
        return 0.0
    return (finish - start) / 1_000_000_000.0


def record_timer_ns(metric: TimerMetric, *, start_ns: int, end_ns: int | None = None) -> TimerMetric:
    return metric.record(duration_seconds_from_ns(start_ns, end_ns))


def clamp(value: Any, lower: float, upper: float) -> float:
    low = _as_float(lower, default=0.0)
    high = _as_float(upper, default=1.0)
    if low > high:
        low, high = high, low
    numeric = _as_float(value, default=low)
    return max(low, min(high, numeric))


def clamp01(value: Any) -> float:
    return clamp(value, 0.0, 1.0)


def safe_divide(numerator: Any, denominator: Any, *, default: float = 0.0) -> float:
    num = _as_float(numerator, default=0.0)
    den = _as_float(denominator, default=0.0)
    if abs(den) <= EPSILON:
        return float(default)
    return num / den


def safe_ratio(numerator: Any, denominator: Any, *, default: float = 0.0) -> float:
    return safe_divide(numerator, denominator, default=default)


def normalize_score(
    value: Any,
    *,
    lower: float = 0.0,
    upper: float = 1.0,
    default: float = 0.0,
) -> float:
    low = _as_float(lower, default=0.0)
    high = _as_float(upper, default=1.0)
    if abs(high - low) <= EPSILON:
        return clamp01(default)
    return clamp01(safe_divide(_as_float(value, default=low) - low, high - low, default=default))


def stable_mean(values: Sequence[Any], *, default: float = 0.0) -> float:
    numeric = [_as_float(value, default=0.0) for value in values]
    if not numeric:
        return float(default)
    return sum(numeric) / len(numeric)


def weighted_mean(values: Sequence[Any], weights: Sequence[Any], *, default: float = 0.0) -> float:
    if not values or not weights:
        return float(default)
    usable = []
    for value, weight in zip(values, weights):
        w = max(0.0, _as_float(weight, default=0.0))
        if w <= EPSILON:
            continue
        usable.append((_as_float(value, default=0.0), w))
    if not usable:
        return float(default)
    total_weight = sum(weight for _, weight in usable)
    return safe_divide(sum(value * weight for value, weight in usable), total_weight, default=default)


def safe_log(value: Any, *, eps: float = EPSILON) -> float:
    return math.log(max(_as_float(value, default=0.0), max(eps, EPSILON)))


def stable_softmax(values: Sequence[Any], *, temperature: float = 1.0) -> tuple[float, ...]:
    if not values:
        return ()
    temp = max(_as_float(temperature, default=1.0), EPSILON)
    logits = [_as_float(value, default=0.0) / temp for value in values]
    max_logit = max(logits)
    exps = [math.exp(logit - max_logit) for logit in logits]
    total = sum(exps)
    if total <= EPSILON:
        uniform = 1.0 / len(values)
        return tuple(uniform for _ in values)
    return tuple(exp_value / total for exp_value in exps)


def normalize_probabilities(
    values: Mapping[Any, Any] | Sequence[Any],
    *,
    temperature: float = 1.0,
) -> dict[str, float] | tuple[float, ...]:
    labels, raw_values = _coerce_distribution_items(values)
    if not raw_values:
        return {} if isinstance(values, Mapping) else ()

    temp = max(_as_float(temperature, default=1.0), EPSILON)
    adjusted = [max(0.0, value) for value in raw_values]
    if temp != 1.0:
        adjusted = [value ** (1.0 / temp) if value > 0.0 else 0.0 for value in adjusted]
    total = sum(adjusted)
    if total <= EPSILON:
        probabilities = [1.0 / len(adjusted) for _ in adjusted]
    else:
        probabilities = [value / total for value in adjusted]
    if labels is None:
        return tuple(probabilities)
    return {label: probability for label, probability in zip(labels, probabilities)}


def entropy(values: Mapping[Any, Any] | Sequence[Any], *, normalize: bool = False) -> float:
    probabilities = _normalized_distribution_values(values)
    if len(probabilities) <= 1:
        return 0.0
    total = -sum(probability * safe_log(probability) for probability in probabilities if probability > 0.0)
    if not normalize:
        return total
    max_entropy = safe_log(len(probabilities))
    return clamp01(safe_divide(total, max_entropy, default=0.0))


def normalized_entropy(values: Mapping[Any, Any] | Sequence[Any]) -> float:
    return entropy(values, normalize=True)


def dominant_answer_ratio(values: Mapping[Any, Any] | Sequence[Any]) -> float:
    summary = distribution_summary(values)
    return summary.dominant_ratio


def consensus_strength(values: Mapping[Any, Any] | Sequence[Any]) -> float:
    summary = distribution_summary(values)
    return summary.consensus_strength


def uncertainty_penalty(values: Mapping[Any, Any] | Sequence[Any], *, weight: float = 1.0) -> float:
    summary = distribution_summary(values)
    return clamp01(_as_float(weight, default=1.0) * summary.uncertainty_penalty)


def distribution_summary(values: Mapping[Any, Any] | Sequence[Any]) -> DistributionSummary:
    labels, normalized = _coerce_distribution_items(values, normalize_values=True)
    if not normalized:
        return DistributionSummary(
            top_label=None,
            top_probability=0.0,
            second_probability=0.0,
            margin=0.0,
            entropy=0.0,
            normalized_entropy=0.0,
            dominant_ratio=0.0,
            consensus_strength=0.0,
            uncertainty_penalty=0.0,
            confidence=0.0,
            total_mass=0.0,
            support_size=0,
            probabilities={},
        )

    ordered = sorted(
        zip(labels or [str(index) for index in range(len(normalized))], normalized),
        key=lambda item: (-item[1], item[0]),
    )
    top_label, top_probability = ordered[0]
    second_probability = ordered[1][1] if len(ordered) > 1 else 0.0
    margin = clamp01(top_probability - second_probability)
    ent = entropy(normalized)
    ent_norm = entropy(normalized, normalize=True)
    dominant = clamp01(top_probability)
    consensus = clamp01(0.60 * dominant + 0.40 * margin)
    confidence = clamp01(0.55 * margin + 0.45 * (1.0 - ent_norm))
    penalty = clamp01(0.65 * ent_norm + 0.35 * (1.0 - dominant))

    return DistributionSummary(
        top_label=top_label if labels is not None else None,
        top_probability=dominant,
        second_probability=clamp01(second_probability),
        margin=margin,
        entropy=ent,
        normalized_entropy=ent_norm,
        dominant_ratio=dominant,
        consensus_strength=consensus,
        uncertainty_penalty=penalty,
        confidence=confidence,
        total_mass=1.0,
        support_size=len(normalized),
        probabilities={label: probability for label, probability in ordered},
    )


def should_early_stop(
    values: Mapping[Any, Any] | Sequence[Any],
    *,
    min_consensus: float = 0.82,
    max_normalized_entropy: float = 0.18,
    min_dominant_ratio: float = 0.70,
    min_samples: int = 4,
) -> bool:
    summary = distribution_summary(values)
    sample_count = _distribution_sample_count(values)
    return bool(
        sample_count >= max(1, int(min_samples))
        and summary.consensus_strength >= clamp01(min_consensus)
        and summary.normalized_entropy <= clamp01(max_normalized_entropy)
        and summary.dominant_ratio >= clamp01(min_dominant_ratio)
    )


def should_expand_search(
    values: Mapping[Any, Any] | Sequence[Any],
    *,
    route_uncertainty: float = 0.0,
    entropy_trigger: float = 0.26,
    low_consensus_trigger: float = 0.55,
    difficulty_pressure: float = 0.0,
) -> bool:
    summary = distribution_summary(values)
    return bool(
        summary.normalized_entropy >= clamp01(entropy_trigger)
        or summary.consensus_strength <= clamp01(low_consensus_trigger)
        or clamp01(route_uncertainty) >= ROUTING_HIGH_ENTROPY_THRESHOLD
        or clamp01(difficulty_pressure) >= 0.65
    )


def minmax_normalize(values: Sequence[Any], *, default: float = 0.0) -> tuple[float, ...]:
    numeric = [_as_float(value, default=0.0) for value in values]
    if not numeric:
        return ()
    low = min(numeric)
    high = max(numeric)
    if abs(high - low) <= EPSILON:
        return tuple(clamp01(default) for _ in numeric)
    return tuple(clamp01((value - low) / (high - low)) for value in numeric)


def normalize_ranking_scores(
    values: Mapping[Any, Any] | Sequence[Any],
    *,
    method: Literal["minmax", "softmax"] = "minmax",
    temperature: float = 1.0,
) -> dict[str, float] | tuple[float, ...]:
    labels, numeric = _coerce_distribution_items(values)
    if not numeric:
        return {} if isinstance(values, Mapping) else ()
    if method == "softmax":
        normalized = stable_softmax(numeric, temperature=temperature)
    else:
        normalized = minmax_normalize(numeric)
    if labels is None:
        return normalized
    return {label: score for label, score in zip(labels, normalized)}


def passes_threshold(value: Any, threshold: float, *, inclusive: bool = True) -> bool:
    numeric = _as_float(value, default=0.0)
    boundary = _as_float(threshold, default=0.0)
    return numeric >= boundary if inclusive else numeric > boundary


def retrieval_quality_score(
    *,
    similarity: float,
    lexical_overlap: float = 0.0,
    structural_alignment: float = 0.0,
    support: float = 0.0,
    penalty: float = 0.0,
) -> float:
    return clamp01(
        0.55 * clamp01(similarity)
        + 0.15 * clamp01(lexical_overlap)
        + 0.20 * clamp01(structural_alignment)
        + 0.10 * clamp01(support)
        - clamp01(penalty)
    )


def diversity_coverage_summary(
    items: Sequence[Any],
    *,
    key: Callable[[Any], Any] | None = None,
    universe: Iterable[Any] | None = None,
) -> DiversityCoverageSummary:
    if not items:
        return DiversityCoverageSummary(
            total_items=0,
            unique_items=0,
            diversity_ratio=0.0,
            coverage_ratio=0.0,
            dominant_ratio=0.0,
            counts={},
        )

    resolved_key = key or (lambda value: value)
    counts: dict[str, int] = {}
    for item in items:
        label = str(resolved_key(item)).strip()
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1

    total = sum(counts.values())
    if total <= 0:
        return DiversityCoverageSummary(
            total_items=len(items),
            unique_items=0,
            diversity_ratio=0.0,
            coverage_ratio=0.0,
            dominant_ratio=0.0,
            counts={},
        )

    unique = len(counts)
    diversity = clamp01(safe_ratio(unique, total, default=0.0))
    dominant = clamp01(max(counts.values()) / total)
    if universe is None:
        coverage = diversity
    else:
        universe_labels = {str(value).strip() for value in universe if str(value).strip()}
        coverage = clamp01(safe_ratio(unique, len(universe_labels), default=0.0))
    return DiversityCoverageSummary(
        total_items=total,
        unique_items=unique,
        diversity_ratio=diversity,
        coverage_ratio=coverage,
        dominant_ratio=dominant,
        counts=dict(sorted(counts.items())),
    )


def retrieval_quality_summary(
    scores: Sequence[Any],
    *,
    support_count: int | None = None,
    universe_size: int | None = None,
) -> RetrievalQualitySummary:
    normalized_scores = [clamp01(score) for score in scores]
    diversity = diversity_coverage_summary(tuple(round(score, 6) for score in normalized_scores))
    support_ratio = clamp01(
        safe_ratio(support_count if support_count is not None else len(normalized_scores), universe_size or len(normalized_scores) or 1)
    )
    mean_score = stable_mean(normalized_scores)
    top_score = max(normalized_scores) if normalized_scores else 0.0
    quality = clamp01(0.55 * mean_score + 0.25 * top_score + 0.10 * support_ratio + 0.10 * diversity.diversity_ratio)
    return RetrievalQualitySummary(
        mean_score=mean_score,
        top_score=top_score,
        support_ratio=support_ratio,
        coverage_ratio=diversity.coverage_ratio,
        diversity_ratio=diversity.diversity_ratio,
        quality_score=quality,
    )


def route_confidence_summary(probabilities: Mapping[Any, Any] | Sequence[Any]) -> DistributionSummary:
    return distribution_summary(probabilities)


def route_confidence(probabilities: Mapping[Any, Any] | Sequence[Any]) -> float:
    return route_confidence_summary(probabilities).confidence


def difficulty_pressure(
    difficulty_score: Any,
    *,
    route_uncertainty: float = 0.0,
    parse_confidence: float = 1.0,
) -> float:
    return clamp01(
        0.55 * clamp01(difficulty_score)
        + 0.30 * clamp01(route_uncertainty)
        + 0.15 * (1.0 - clamp01(parse_confidence))
    )


def calibrate_probability(
    probability: Any,
    *,
    scale: float = 1.0,
    offset: float = 0.0,
    floor: float = 0.0,
    ceiling: float = 1.0,
) -> float:
    value = _as_float(scale, default=1.0) * clamp01(probability) + _as_float(offset, default=0.0)
    return clamp(value, floor, ceiling)


def decompose_branch_score(
    components: BranchScoreComponents | Mapping[str, Any] | Any,
    *,
    weights: BranchScoreWeights | None = None,
    round_digits: int = 6,
) -> BranchScoreDecomposition:
    resolved_components = branch_score_components(components)
    resolved_weights = weights or BranchScoreWeights()
    weighted_core = (
        resolved_weights.verifier_weight * resolved_components.verifier_probability
        + resolved_weights.tool_consistency_weight * resolved_components.tool_consistency
        + resolved_weights.answer_agreement_weight * resolved_components.answer_agreement
        + resolved_weights.branch_novelty_weight * resolved_components.branch_novelty
        + resolved_weights.exact_symbolic_weight * resolved_components.exact_symbolic_check
    )
    retrieval_bonus = resolved_weights.retrieval_bonus_weight * resolved_components.retrieval_support
    composite = clamp01(weighted_core + retrieval_bonus - resolved_components.penalty)
    return BranchScoreDecomposition(
        components=resolved_components,
        weights=resolved_weights,
        weighted_core=round(weighted_core, round_digits),
        retrieval_bonus=round(retrieval_bonus, round_digits),
        penalty=round(clamp01(resolved_components.penalty), round_digits),
        composite_score=round(composite, round_digits),
    )


def compose_branch_score(
    components: BranchScoreComponents | Mapping[str, Any] | Any,
    *,
    weights: BranchScoreWeights | None = None,
    round_digits: int = 6,
) -> float:
    return decompose_branch_score(components, weights=weights, round_digits=round_digits).composite_score


def branch_score_components(source: BranchScoreComponents | Mapping[str, Any] | Any) -> BranchScoreComponents:
    if isinstance(source, BranchScoreComponents):
        return source
    extractor = source.get if isinstance(source, Mapping) else lambda key, default=0.0: getattr(source, key, default)
    return BranchScoreComponents(
        verifier_probability=clamp01(extractor("verifier_probability", 0.0)),
        tool_consistency=clamp01(extractor("tool_consistency", 0.0)),
        answer_agreement=clamp01(extractor("answer_agreement", 0.0)),
        branch_novelty=clamp01(extractor("branch_novelty", 0.0)),
        exact_symbolic_check=clamp01(extractor("exact_symbolic_check", extractor("symbolic_check", 0.0))),
        retrieval_support=clamp01(extractor("retrieval_support", 0.0)),
        penalty=clamp01(extractor("penalty", 0.0)),
    )


def candidate_cluster_weight(
    *,
    verifier_score: float,
    tool_consistency: float,
    answer_agreement: float,
    branch_novelty: float,
    symbolic_check: float,
    retrieval_support: float = 0.0,
    entropy_penalty: float = 0.0,
    outlier_penalty: float = 0.0,
    weights: BranchScoreWeights | None = None,
) -> float:
    base = compose_branch_score(
        BranchScoreComponents(
            verifier_probability=verifier_score,
            tool_consistency=tool_consistency,
            answer_agreement=answer_agreement,
            branch_novelty=branch_novelty,
            exact_symbolic_check=symbolic_check,
            retrieval_support=retrieval_support,
            penalty=clamp01(entropy_penalty) + clamp01(outlier_penalty),
        ),
        weights=weights,
    )
    return clamp01(base)


def exact_match(left: Any, right: Any) -> bool:
    return safe_normalize_text(left, lowercase=False) == safe_normalize_text(right, lowercase=False)


def canonical_match(left: Any, right: Any) -> bool:
    return safe_normalize_text(left, lowercase=True) == safe_normalize_text(right, lowercase=True)


def accuracy_score(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    pairs = list(zip(y_true, y_pred))
    if not pairs:
        return 0.0
    correct = sum(1 for truth, pred in pairs if canonical_match(truth, pred))
    return safe_ratio(correct, len(pairs), default=0.0)


def brier_score(y_true: Sequence[Any], y_prob: Sequence[Any]) -> float:
    pairs = list(zip(y_true, y_prob))
    if not pairs:
        return 0.0
    errors = []
    for truth, probability in pairs:
        target = 1.0 if _as_bool_label(truth) else 0.0
        prob = clamp01(probability)
        errors.append((prob - target) ** 2)
    return stable_mean(errors)


def calibration_buckets(
    y_true: Sequence[Any],
    y_prob: Sequence[Any],
    *,
    num_buckets: int = DEFAULT_ECE_BUCKETS,
) -> tuple[CalibrationBucketSummary, ...]:
    pairs = [(1.0 if _as_bool_label(truth) else 0.0, clamp01(prob)) for truth, prob in zip(y_true, y_prob)]
    if not pairs:
        return ()

    bucket_count = max(1, int(num_buckets))
    assignments: list[list[tuple[float, float]]] = [[] for _ in range(bucket_count)]
    for truth, prob in pairs:
        index = min(bucket_count - 1, int(prob * bucket_count))
        assignments[index].append((truth, prob))

    summaries: list[CalibrationBucketSummary] = []
    for index, bucket in enumerate(assignments):
        lower = index / bucket_count
        upper = (index + 1) / bucket_count
        if not bucket:
            summaries.append(
                CalibrationBucketSummary(
                    bucket_index=index,
                    lower_bound=lower,
                    upper_bound=upper,
                    count=0,
                    mean_confidence=0.0,
                    empirical_accuracy=0.0,
                    confidence_gap=0.0,
                )
            )
            continue
        truths = [item[0] for item in bucket]
        probs = [item[1] for item in bucket]
        mean_conf = stable_mean(probs)
        accuracy = stable_mean(truths)
        summaries.append(
            CalibrationBucketSummary(
                bucket_index=index,
                lower_bound=lower,
                upper_bound=upper,
                count=len(bucket),
                mean_confidence=mean_conf,
                empirical_accuracy=accuracy,
                confidence_gap=abs(mean_conf - accuracy),
            )
        )
    return tuple(summaries)


def expected_calibration_error(
    y_true: Sequence[Any],
    y_prob: Sequence[Any],
    *,
    num_buckets: int = DEFAULT_ECE_BUCKETS,
) -> float:
    buckets = calibration_buckets(y_true, y_prob, num_buckets=num_buckets)
    total = sum(bucket.count for bucket in buckets)
    if total <= 0:
        return 0.0
    return sum((bucket.count / total) * bucket.confidence_gap for bucket in buckets)


def evaluate_binary_calibration(
    y_true: Sequence[Any],
    y_prob: Sequence[Any],
    *,
    threshold: float = 0.5,
    num_buckets: int = DEFAULT_ECE_BUCKETS,
) -> BinaryCalibrationSummary:
    probs = [clamp01(probability) for probability in y_prob]
    truths = [1.0 if _as_bool_label(value) else 0.0 for value in y_true]
    preds = [1.0 if probability >= clamp01(threshold) else 0.0 for probability in probs]
    buckets = calibration_buckets(truths, probs, num_buckets=num_buckets)
    max_gap = max((bucket.confidence_gap for bucket in buckets), default=0.0)
    return BinaryCalibrationSummary(
        count=min(len(truths), len(probs)),
        accuracy=accuracy_score(truths, preds),
        mean_confidence=stable_mean(probs),
        positive_rate=stable_mean(truths),
        brier_score=brier_score(truths, probs),
        expected_calibration_error=expected_calibration_error(truths, probs, num_buckets=num_buckets),
        max_calibration_error=max_gap,
        buckets=buckets,
    )


def _as_float(value: Any, *, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(numeric):
        return float(default)
    if math.isinf(numeric):
        return float(default if numeric < 0 else 1.0e12)
    return numeric


def _as_bool_label(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "correct"}:
            return True
        if lowered in {"0", "false", "no", "incorrect"}:
            return False
    return bool(_as_float(value, default=1.0 if bool(value) else 0.0) >= 0.5)


def _coerce_distribution_items(
    values: Mapping[Any, Any] | Sequence[Any],
    *,
    normalize_values: bool = False,
) -> tuple[list[str] | None, list[float]]:
    if isinstance(values, Mapping):
        ordered = sorted(((str(key), max(0.0, _as_float(value, default=0.0))) for key, value in values.items()), key=lambda item: item[0])
        labels = [label for label, _ in ordered]
        numeric = [value for _, value in ordered]
    else:
        labels = None
        numeric = [max(0.0, _as_float(value, default=0.0)) for value in values]

    if normalize_values:
        total = sum(numeric)
        if total <= EPSILON and numeric:
            numeric = [1.0 / len(numeric) for _ in numeric]
        elif total > EPSILON:
            numeric = [value / total for value in numeric]
    return labels, numeric


def _normalized_distribution_values(values: Mapping[Any, Any] | Sequence[Any]) -> list[float]:
    _, numeric = _coerce_distribution_items(values, normalize_values=True)
    return numeric


def _distribution_sample_count(values: Mapping[Any, Any] | Sequence[Any]) -> int:
    if isinstance(values, Mapping):
        raw_total = sum(max(0.0, _as_float(value, default=0.0)) for value in values.values())
        if raw_total > 1.5:
            return int(round(raw_total))
        return len(values)
    return len(values)


def _ensure_metric_compatible(left_name: str, right_name: str, *, metric_type: str) -> None:
    if left_name != right_name:
        raise ValueError(f"Cannot merge {metric_type} metrics with different names: {left_name!r} vs {right_name!r}.")


def _histogram_bucket_index(value: float, boundaries: Sequence[float]) -> int:
    for index, boundary in enumerate(boundaries):
        if value <= boundary:
            return index
    return len(boundaries)


__all__ = [
    "BinaryCalibrationSummary",
    "CounterMetric",
    "BranchScoreComponents",
    "BranchScoreDecomposition",
    "BranchScoreWeights",
    "CalibrationBucketSummary",
    "DEFAULT_ECE_BUCKETS",
    "DEFAULT_HISTOGRAM_BOUNDARIES",
    "DiversityCoverageSummary",
    "DistributionSummary",
    "EPSILON",
    "HistogramMetric",
    "MetricCollection",
    "RetrievalQualitySummary",
    "TimerMetric",
    "accuracy_score",
    "brier_score",
    "branch_score_components",
    "calibrate_probability",
    "calibration_buckets",
    "candidate_cluster_weight",
    "canonical_match",
    "clamp",
    "clamp01",
    "compose_branch_score",
    "consensus_strength",
    "decompose_branch_score",
    "difficulty_pressure",
    "distribution_summary",
    "diversity_coverage_summary",
    "dominant_answer_ratio",
    "duration_seconds_from_ns",
    "entropy",
    "evaluate_binary_calibration",
    "exact_match",
    "expected_calibration_error",
    "minmax_normalize",
    "normalize_probabilities",
    "normalize_ranking_scores",
    "normalize_score",
    "normalized_entropy",
    "passes_threshold",
    "monotonic_time_ns",
    "record_timer_ns",
    "retrieval_quality_score",
    "retrieval_quality_summary",
    "route_confidence",
    "route_confidence_summary",
    "safe_divide",
    "safe_log",
    "safe_ratio",
    "should_early_stop",
    "should_expand_search",
    "stable_mean",
    "stable_softmax",
    "uncertainty_penalty",
    "weighted_mean",
]
