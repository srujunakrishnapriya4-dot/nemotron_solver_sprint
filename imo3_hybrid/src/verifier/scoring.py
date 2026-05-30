"""
Typed verifier score decomposition and verdict assembly.

This module owns runtime score fusion for verifier outputs. It deliberately keeps
symbolic agreement, logical consistency, completeness, answer correctness
likelihood, repairability, and uncertainty as separate typed signals before
assembling any branch score or verdict.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field

from src.branches.branch_state import BranchScoreBreakdown
from src.common.schemas import VerifierLabel as LegacyVerifierLabel

from .verifier_labels import (
    BranchSupervisionLabel,
    ReasoningQualityTag,
    VerifierLabelBundle,
    VerifierVerdict,
)


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))


def _normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def _mean(*values: float) -> float:
    if not values:
        return 0.0
    return _clamp01(sum(float(value) for value in values) / len(values))


def _extract_score(mapping: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return _clamp01(mapping[key])
    return None


def _extract_int(mapping: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            try:
                return max(0, int(mapping[key]))
            except (TypeError, ValueError):
                return 0
    return None


def _normalize_symbolic_status(value: Any) -> str:
    text = _normalize_text(value).lower().replace(" ", "_")
    if text in {
        "success",
        "contradiction",
        "unsupported",
        "timeout",
        "malformed_input",
        "execution_failure",
    }:
        return text
    return ""


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        str_strip_whitespace=True,
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )


class VerifierScoreMode(str, Enum):
    DISTILLED = "distilled"
    LLM_BACKED = "llm_backed"
    LABEL_DERIVED = "label_derived"
    LEGACY_LABEL = "legacy_label"
    HEURISTIC = "heuristic"


class ScoreComponentName(str, Enum):
    LOGICAL_CONSISTENCY = "logical_consistency"
    SYMBOLIC_AGREEMENT = "symbolic_agreement"
    COMPLETENESS = "completeness"
    ANSWER_CORRECTNESS_LIKELIHOOD = "answer_correctness_likelihood"
    REPAIRABILITY = "repairability"
    STEP_QUALITY = "step_quality"
    PREFIX_QUALITY = "prefix_quality"
    MODEL_CONFIDENCE = "model_confidence"
    UNCERTAINTY = "uncertainty"


class ScoreThresholdName(str, Enum):
    PRUNE = "prune"
    REPAIR = "repair"
    UNCERTAIN = "uncertain"
    STRONG_PASS = "strong_pass"


class ComponentCalibration(StrictModel):
    scale: float = 1.0
    offset: float = 0.0
    floor: float = 0.0
    ceiling: float = 1.0

    def apply(self, value: float) -> float:
        return max(self.floor, min(self.ceiling, self.scale * _clamp01(value) + self.offset))


class ScoreCalibrationProfile(StrictModel):
    logical_consistency: ComponentCalibration = Field(default_factory=ComponentCalibration)
    symbolic_agreement: ComponentCalibration = Field(default_factory=ComponentCalibration)
    completeness: ComponentCalibration = Field(default_factory=ComponentCalibration)
    answer_correctness_likelihood: ComponentCalibration = Field(default_factory=ComponentCalibration)
    repairability: ComponentCalibration = Field(default_factory=ComponentCalibration)
    model_confidence: ComponentCalibration = Field(default_factory=ComponentCalibration)


class ScoreThresholds(StrictModel):
    prune_threshold: float = Field(default=0.24, ge=0.0, le=1.0)
    repair_threshold: float = Field(default=0.56, ge=0.0, le=1.0)
    uncertain_threshold: float = Field(default=0.34, ge=0.0, le=1.0)
    strong_pass_threshold: float = Field(default=0.74, ge=0.0, le=1.0)
    logical_floor: float = Field(default=0.36, ge=0.0, le=1.0)
    symbolic_floor: float = Field(default=0.34, ge=0.0, le=1.0)
    completeness_floor: float = Field(default=0.30, ge=0.0, le=1.0)

    def lookup(self, name: ScoreThresholdName) -> float:
        return {
            ScoreThresholdName.PRUNE: self.prune_threshold,
            ScoreThresholdName.REPAIR: self.repair_threshold,
            ScoreThresholdName.UNCERTAIN: self.uncertain_threshold,
            ScoreThresholdName.STRONG_PASS: self.strong_pass_threshold,
        }[name]


class VerifierScoreInput(StrictModel):
    problem_id: str = ""
    branch_id: str = ""
    model_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    logical_consistency: float = Field(default=0.0, ge=0.0, le=1.0)
    symbolic_agreement: float = Field(default=0.0, ge=0.0, le=1.0)
    completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    answer_correctness_likelihood: float | None = Field(default=None)
    repairability: float = Field(default=0.0, ge=0.0, le=1.0)
    step_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    prefix_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    open_obligation_burden: float = Field(default=0.0, ge=0.0, le=1.0)
    route_uncertainty: float = Field(default=0.0, ge=0.0, le=1.0)
    symbolic_status: str = ""
    symbolic_exact: bool = False
    symbolic_partial_support: bool = False
    discharged_obligation_count: int = Field(default=0, ge=0)
    contradicted_obligation_count: int = Field(default=0, ge=0)
    step_count: int = Field(default=0, ge=0)
    artifact_state: str = ""
    mode: VerifierScoreMode = VerifierScoreMode.HEURISTIC
    quality_tag: ReasoningQualityTag = ReasoningQualityTag.UNKNOWN
    failure_type: str | None = None
    repair_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoreComponent(StrictModel):
    name: ScoreComponentName
    raw_score: float = Field(default=0.0, ge=0.0, le=1.0)
    calibrated_score: float = Field(default=0.0, ge=0.0, le=1.0)
    weight: float = Field(default=0.0, ge=0.0, le=1.0)
    contribution: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""


class UncertaintyBreakdown(StrictModel):
    calibration_gap: float = Field(default=0.0, ge=0.0, le=1.0)
    consistency_conflict: float = Field(default=0.0, ge=0.0, le=1.0)
    incompleteness_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    route_uncertainty: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_sparsity: float = Field(default=0.0, ge=0.0, le=1.0)
    shallow_reasoning_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    obligation_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    total: float = Field(default=0.0, ge=0.0, le=1.0)


class VerifierDecisionFlags(StrictModel):
    strong_pass: bool = False
    should_prune: bool = False
    should_repair: bool = False
    uncertain: bool = False
    aggregation_ready: bool = False


class VerifierScoreBundle(StrictModel):
    problem_id: str
    branch_id: str
    mode: VerifierScoreMode
    artifact_state: str = ""
    model_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    logical_consistency: float = Field(default=0.0, ge=0.0, le=1.0)
    symbolic_agreement: float = Field(default=0.0, ge=0.0, le=1.0)
    completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    answer_correctness_likelihood: float = Field(default=0.0, ge=0.0, le=1.0)
    repairability: float = Field(default=0.0, ge=0.0, le=1.0)
    step_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    prefix_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    open_obligation_burden: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty: UncertaintyBreakdown
    branch_score: float = Field(default=0.0, ge=0.0, le=1.0)
    verdict: VerifierVerdict
    flags: VerifierDecisionFlags
    components: tuple[ScoreComponent, ...] = Field(default_factory=tuple)
    quality_tag: ReasoningQualityTag = ReasoningQualityTag.UNKNOWN
    failure_type: str | None = None
    repair_type: str | None = None
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def probability(self) -> float:
        return self.answer_correctness_likelihood

    @property
    def score(self) -> float:
        return self.branch_score

    @property
    def composite_score(self) -> float:
        return self.branch_score

    @property
    def verifier_score(self) -> float:
        return self.branch_score

    def to_branch_score_breakdown(
        self,
        *,
        answer_agreement: float = 0.0,
        branch_novelty: float = 0.0,
        retrieval_support: float = 0.0,
    ) -> BranchScoreBreakdown:
        return BranchScoreBreakdown(
            verifier_probability=self.answer_correctness_likelihood,
            tool_consistency=self.symbolic_agreement,
            answer_agreement=_clamp01(answer_agreement),
            branch_novelty=_clamp01(branch_novelty),
            exact_symbolic_check=self.symbolic_agreement,
            retrieval_support=_clamp01(retrieval_support),
            logical_consistency=self.logical_consistency,
            completeness=self.completeness,
            repairability=self.repairability,
            step_quality=self.step_quality,
            prefix_quality=self.prefix_quality,
            open_obligation_burden=self.open_obligation_burden,
            penalty=_clamp01(self.uncertainty.total + 0.20 * self.open_obligation_burden),
        )

    def to_aggregation_features(self) -> dict[str, float]:
        return {
            "verifier_score": self.branch_score,
            "answer_correctness_likelihood": self.answer_correctness_likelihood,
            "logical_consistency": self.logical_consistency,
            "symbolic_agreement": self.symbolic_agreement,
            "completeness": self.completeness,
            "repairability": self.repairability,
            "step_quality": self.step_quality,
            "prefix_quality": self.prefix_quality,
            "open_obligation_burden": self.open_obligation_burden,
            "uncertainty": self.uncertainty.total,
        }


def normalize_score(value: Any) -> float:
    return _clamp01(value)


def calibrate_score(
    value: Any,
    calibration: ComponentCalibration | None = None,
) -> float:
    return (calibration or ComponentCalibration()).apply(_clamp01(value))


def _coerce_quality_tag(value: Any) -> ReasoningQualityTag:
    if isinstance(value, ReasoningQualityTag):
        return value
    try:
        return ReasoningQualityTag(str(value))
    except Exception:
        return ReasoningQualityTag.UNKNOWN


def _coerce_mode(value: Any) -> VerifierScoreMode:
    if isinstance(value, VerifierScoreMode):
        return value
    text = _normalize_text(value).lower()
    if text in {"model", "distilled"}:
        return VerifierScoreMode.DISTILLED
    if text in {"deepseek", "llm", "llm_backed"}:
        return VerifierScoreMode.LLM_BACKED
    if text in {"label", "label_derived"}:
        return VerifierScoreMode.LABEL_DERIVED
    if text in {"legacy", "legacy_label"}:
        return VerifierScoreMode.LEGACY_LABEL
    return VerifierScoreMode.HEURISTIC


def _input_from_legacy_label(label: LegacyVerifierLabel) -> VerifierScoreInput:
    return VerifierScoreInput(
        branch_id=label.branch_id,
        model_confidence=_clamp01(label.composite_score),
        logical_consistency=label.logical_consistency,
        symbolic_agreement=label.symbolic_agreement,
        completeness=label.completeness,
        answer_correctness_likelihood=label.final_answer_correct_probability,
        repairability=label.repairability,
        step_count=len(label.step_correctness),
        mode=VerifierScoreMode.LEGACY_LABEL,
        quality_tag=_coerce_quality_tag(
            "correct_answer_wrong_reasoning"
            if label.final_answer_correct and label.logical_consistency < 0.75
            else "wrong_answer_plausible_reasoning"
            if not label.final_answer_correct and label.logical_consistency >= 0.60
            else "unknown"
        ),
        failure_type=label.failure_type,
        repair_type=label.repair_type,
        metadata={"legacy_label": label.model_dump(mode="json")},
    )


def _input_from_branch_label(label: BranchSupervisionLabel) -> VerifierScoreInput:
    return VerifierScoreInput(
        problem_id=label.context.problem_id,
        branch_id=label.context.branch_id,
        model_confidence=label.confidence.confidence,
        logical_consistency=label.consistency.logical_consistency,
        symbolic_agreement=label.consistency.symbolic_agreement,
        completeness=label.completeness.completeness,
        answer_correctness_likelihood=label.correctness.answer_correct_likelihood,
        repairability=label.repairability.repairability,
        step_count=label.context.total_step_count,
        mode=VerifierScoreMode.LABEL_DERIVED,
        quality_tag=label.correctness.quality_tag,
        failure_type=label.repairability.failure_type,
        repair_type=label.repairability.repair_type,
        metadata={"branch_label_id": label.label_id},
    )


def _input_from_bundle(bundle: VerifierLabelBundle) -> VerifierScoreInput:
    payload = _input_from_branch_label(bundle.branch_label)
    return payload.model_copy(
        update={
            "metadata": {
                **payload.metadata,
                "bundle_id": bundle.bundle_id,
                "schema_version": bundle.schema_version,
            }
        }
    )


def _input_from_mapping(data: Mapping[str, Any], secondary: Any = None, **kwargs: Any) -> VerifierScoreInput:
    merged = dict(data)
    if kwargs:
        merged.update(kwargs)
    metadata_mapping = dict(merged.get("metadata", {})) if isinstance(merged.get("metadata"), Mapping) else {}

    embedded_label = merged.get("label")
    if isinstance(embedded_label, BranchSupervisionLabel):
        baseline = _input_from_branch_label(embedded_label)
    elif isinstance(embedded_label, VerifierLabelBundle):
        baseline = _input_from_bundle(embedded_label)
    elif isinstance(embedded_label, LegacyVerifierLabel):
        baseline = _input_from_legacy_label(embedded_label)
    elif isinstance(embedded_label, Mapping):
        try:
            baseline = _input_from_legacy_label(LegacyVerifierLabel.model_validate(embedded_label))
        except Exception:
            baseline = VerifierScoreInput()
    elif isinstance(secondary, BranchSupervisionLabel):
        baseline = _input_from_branch_label(secondary)
    elif isinstance(secondary, VerifierLabelBundle):
        baseline = _input_from_bundle(secondary)
    elif isinstance(secondary, LegacyVerifierLabel):
        baseline = _input_from_legacy_label(secondary)
    elif isinstance(secondary, Mapping):
        try:
            baseline = _input_from_legacy_label(LegacyVerifierLabel.model_validate(secondary))
        except Exception:
            baseline = VerifierScoreInput()
    else:
        baseline = VerifierScoreInput()

    explicit_answer = _extract_score(
        merged,
        "answer_correctness_likelihood",
        "final_answer_correct_probability",
        "probability",
        "verifier_probability",
    )
    logical_raw = _extract_score(merged, "logical_consistency")
    symbolic_raw = _extract_score(merged, "symbolic_agreement", "symbolic_consistency")
    completeness_raw = _extract_score(merged, "completeness")
    repairability_raw = _extract_score(merged, "repairability")
    step_quality_raw = _extract_score(merged, "step_quality")
    prefix_quality_raw = _extract_score(merged, "prefix_quality")
    open_obligation_burden = _extract_score(merged, "open_obligation_burden")
    contradicted_obligation_count = _extract_int(merged, "contradicted_obligation_count")
    discharged_obligation_count = _extract_int(merged, "discharged_obligation_count")
    symbolic_status = _normalize_symbolic_status(
        merged.get("symbolic_status", metadata_mapping.get("symbolic_status", metadata_mapping.get("status", "")))
    )
    symbolic_partial_support = bool(
        merged.get("symbolic_partial_support", metadata_mapping.get("partial_support", False))
    )
    symbolic_exact = bool(
        merged.get("symbolic_exact", metadata_mapping.get("symbolic_exact", metadata_mapping.get("exact", False)))
    )
    if contradicted_obligation_count is None:
        contradicted_obligation_count = len(
            list(
                merged.get(
                    "contradicted_obligation_ids",
                    metadata_mapping.get("contradicted_obligation_ids", []),
                )
                or []
            )
        )
    if discharged_obligation_count is None:
        discharged_obligation_count = len(
            list(
                merged.get(
                    "discharged_obligation_ids",
                    metadata_mapping.get("discharged_obligation_ids", []),
                )
                or []
            )
        )
    model_confidence = _extract_score(
        merged,
        "model_confidence",
        "confidence",
        "overall_score",
        "branch_score",
        "composite_score",
    )
    return VerifierScoreInput(
        problem_id=str(merged.get("problem_id", baseline.problem_id)),
        branch_id=str(merged.get("branch_id", baseline.branch_id)),
        model_confidence=_clamp01(model_confidence if model_confidence is not None else baseline.model_confidence),
        logical_consistency=_clamp01(logical_raw if logical_raw is not None else baseline.logical_consistency),
        symbolic_agreement=_clamp01(symbolic_raw if symbolic_raw is not None else baseline.symbolic_agreement),
        completeness=_clamp01(completeness_raw if completeness_raw is not None else baseline.completeness),
        answer_correctness_likelihood=explicit_answer if explicit_answer is not None else baseline.answer_correctness_likelihood,
        repairability=_clamp01(repairability_raw if repairability_raw is not None else baseline.repairability),
        step_quality=_clamp01(step_quality_raw if step_quality_raw is not None else baseline.step_quality),
        prefix_quality=_clamp01(prefix_quality_raw if prefix_quality_raw is not None else baseline.prefix_quality),
        open_obligation_burden=_clamp01(
            open_obligation_burden if open_obligation_burden is not None else baseline.open_obligation_burden
        ),
        route_uncertainty=_clamp01(merged.get("route_uncertainty", baseline.route_uncertainty)),
        symbolic_status=symbolic_status or _normalize_symbolic_status(baseline.metadata.get("symbolic_status", "")),
        symbolic_exact=symbolic_exact,
        symbolic_partial_support=symbolic_partial_support,
        discharged_obligation_count=max(
            0,
            discharged_obligation_count
            if discharged_obligation_count is not None
            else int(baseline.metadata.get("discharged_obligation_count", 0) or 0),
        ),
        contradicted_obligation_count=max(
            0,
            contradicted_obligation_count
            if contradicted_obligation_count is not None
            else int(baseline.metadata.get("contradicted_obligation_count", 0) or 0),
        ),
        step_count=max(0, int(merged.get("step_count", baseline.step_count))),
        artifact_state=_normalize_text(merged.get("artifact_state", baseline.artifact_state)),
        mode=_coerce_mode(merged.get("mode", baseline.mode.value)),
        quality_tag=_coerce_quality_tag(merged.get("quality_tag", baseline.quality_tag.value)),
        failure_type=_normalize_text(merged.get("failure_type", baseline.failure_type)) or None,
        repair_type=_normalize_text(merged.get("repair_type", baseline.repair_type)) or None,
        metadata={
            **baseline.metadata,
            **metadata_mapping,
        },
    )


def coerce_score_input(primary: Any = None, secondary: Any = None, **kwargs: Any) -> VerifierScoreInput:
    if isinstance(primary, VerifierScoreInput):
        return primary
    if isinstance(primary, VerifierLabelBundle):
        return _input_from_bundle(primary)
    if isinstance(primary, BranchSupervisionLabel):
        return _input_from_branch_label(primary)
    if isinstance(primary, LegacyVerifierLabel):
        return _input_from_legacy_label(primary)
    if isinstance(primary, Mapping):
        return _input_from_mapping(primary, secondary, **kwargs)
    if isinstance(primary, BaseModel):
        return _input_from_mapping(primary.model_dump(mode="json"), secondary, **kwargs)
    if primary is None:
        return _input_from_mapping({}, secondary, **kwargs)
    if hasattr(primary, "__dict__"):
        return _input_from_mapping(vars(primary), secondary, **kwargs)
    return _input_from_mapping({}, secondary, **kwargs)


def _derive_answer_likelihood(score_input: VerifierScoreInput) -> float:
    if score_input.answer_correctness_likelihood is not None:
        return _clamp01(score_input.answer_correctness_likelihood)
    return _clamp01(
        0.46 * score_input.logical_consistency
        + 0.28 * score_input.symbolic_agreement
        + 0.16 * score_input.completeness
        + 0.10 * score_input.model_confidence
    )


def _effective_obligation_burden(
    open_obligation_burden: float,
    *,
    discharged_obligation_count: int,
    contradicted_obligation_count: int,
) -> float:
    burden = _clamp01(open_obligation_burden)
    burden += min(0.24, 0.08 * max(0, contradicted_obligation_count))
    burden -= min(0.12, 0.03 * max(0, discharged_obligation_count))
    return _clamp01(burden)


def _shallow_reasoning_risk(
    *,
    step_quality: float,
    prefix_quality: float,
    open_obligation_burden: float,
) -> float:
    step_gap = max(0.0, 0.55 - _clamp01(step_quality)) / 0.55
    prefix_gap = max(0.0, 0.60 - _clamp01(prefix_quality)) / 0.60
    return _clamp01(
        0.35 * step_gap
        + 0.50 * prefix_gap
        + 0.15 * _clamp01(open_obligation_burden)
    )


def _compute_uncertainty(
    *,
    model_confidence: float,
    answer_correctness_likelihood: float,
    logical_consistency: float,
    symbolic_agreement: float,
    completeness: float,
    step_quality: float,
    prefix_quality: float,
    open_obligation_burden: float,
    route_uncertainty: float,
    symbolic_status: str = "",
    symbolic_partial_support: bool = False,
    discharged_obligation_count: int = 0,
    contradicted_obligation_count: int = 0,
) -> UncertaintyBreakdown:
    calibration_gap = abs(model_confidence - answer_correctness_likelihood)
    consistency_conflict = abs(logical_consistency - symbolic_agreement)
    incompleteness_risk = 1.0 - completeness
    evidence_sparsity = 1.0 - _mean(logical_consistency, symbolic_agreement, completeness, answer_correctness_likelihood)
    obligation_risk = _effective_obligation_burden(
        open_obligation_burden,
        discharged_obligation_count=discharged_obligation_count,
        contradicted_obligation_count=contradicted_obligation_count,
    )
    shallow_reasoning_risk = _shallow_reasoning_risk(
        step_quality=step_quality,
        prefix_quality=prefix_quality,
        open_obligation_burden=obligation_risk,
    )
    symbolic_support_gap = 0.0
    if symbolic_status in {"unsupported", "timeout"}:
        symbolic_support_gap = 0.32 if symbolic_partial_support else 0.52
    elif symbolic_status in {"malformed_input", "execution_failure"}:
        symbolic_support_gap = 0.60
    elif symbolic_status == "contradiction":
        symbolic_support_gap = 1.0
    total = _clamp01(
        0.24 * calibration_gap
        + 0.16 * consistency_conflict
        + 0.12 * incompleteness_risk
        + 0.12 * route_uncertainty
        + 0.08 * evidence_sparsity
        + 0.10 * shallow_reasoning_risk
        + 0.10 * symbolic_support_gap
        + 0.08 * obligation_risk
    )
    return UncertaintyBreakdown(
        calibration_gap=_clamp01(calibration_gap),
        consistency_conflict=_clamp01(consistency_conflict),
        incompleteness_risk=_clamp01(incompleteness_risk),
        route_uncertainty=_clamp01(route_uncertainty),
        evidence_sparsity=_clamp01(evidence_sparsity),
        shallow_reasoning_risk=_clamp01(shallow_reasoning_risk),
        obligation_risk=_clamp01(obligation_risk),
        total=total,
    )


def _apply_symbolic_trust_policy(
    score_input: VerifierScoreInput,
    *,
    logical_consistency: float,
    symbolic_agreement: float,
    completeness: float,
    answer_correctness_likelihood: float,
    repairability: float,
    step_quality: float,
    prefix_quality: float,
    open_obligation_burden: float,
    route_uncertainty: float,
) -> tuple[float, float, float, float, float, float, float, dict[str, Any]]:
    status = _normalize_symbolic_status(score_input.symbolic_status)
    effective_obligation_burden = _effective_obligation_burden(
        open_obligation_burden,
        discharged_obligation_count=score_input.discharged_obligation_count,
        contradicted_obligation_count=score_input.contradicted_obligation_count,
    )
    adjustments: dict[str, Any] = {
        "symbolic_status": status,
        "symbolic_partial_support": bool(score_input.symbolic_partial_support),
        "symbolic_exact": bool(score_input.symbolic_exact),
        "contradicted_obligation_count": int(score_input.contradicted_obligation_count),
        "discharged_obligation_count": int(score_input.discharged_obligation_count),
        "effective_open_obligation_burden": effective_obligation_burden,
    }

    if status == "contradiction":
        symbolic_agreement = min(symbolic_agreement, 0.02)
        logical_consistency = min(logical_consistency, 0.18)
        completeness = min(completeness, 0.35)
        answer_correctness_likelihood = min(answer_correctness_likelihood, 0.12)
        repairability = max(repairability, 0.72)
        effective_obligation_burden = max(effective_obligation_burden, 0.90)
        route_uncertainty = max(route_uncertainty, 0.85)
        adjustments["trust_penalty"] = "symbolic_contradiction"
    elif status in {"malformed_input", "execution_failure"}:
        symbolic_agreement = min(symbolic_agreement, 0.20)
        logical_consistency = min(logical_consistency, 0.30)
        completeness = min(completeness, 0.45)
        answer_correctness_likelihood = min(answer_correctness_likelihood, 0.35)
        route_uncertainty = max(route_uncertainty, 0.70)
        adjustments["trust_penalty"] = "symbolic_runtime_failure"
    elif status == "unsupported":
        if score_input.symbolic_partial_support:
            symbolic_agreement = min(symbolic_agreement, 0.55 if score_input.symbolic_exact else 0.45)
            logical_consistency = min(logical_consistency, 0.60 if score_input.symbolic_exact else 0.52)
            completeness = min(completeness, 0.68)
            answer_correctness_likelihood = min(
                answer_correctness_likelihood,
                0.52 if score_input.symbolic_exact else 0.45,
            )
            route_uncertainty = max(route_uncertainty, 0.30)
            adjustments["trust_penalty"] = "symbolic_partial_support"
        else:
            symbolic_agreement = min(symbolic_agreement, 0.25)
            logical_consistency = min(logical_consistency, 0.42)
            completeness = min(completeness, 0.55)
            answer_correctness_likelihood = min(answer_correctness_likelihood, 0.40)
            route_uncertainty = max(route_uncertainty, 0.50)
            adjustments["trust_penalty"] = "symbolic_unsupported"

    if score_input.contradicted_obligation_count > 0:
        effective_obligation_burden = max(
            effective_obligation_burden,
            min(1.0, 0.55 + 0.15 * score_input.contradicted_obligation_count),
        )
        logical_consistency = min(logical_consistency, 0.25)
        answer_correctness_likelihood = min(answer_correctness_likelihood, 0.20)
        adjustments["obligation_penalty"] = "contradicted_obligations"

    if effective_obligation_burden >= 0.45:
        completeness = min(completeness, 0.60)
        answer_correctness_likelihood = min(
            answer_correctness_likelihood,
            max(0.18, 0.64 - 0.30 * effective_obligation_burden),
        )
        adjustments["obligation_pressure"] = "unresolved_obligations"

    shallow_risk = _shallow_reasoning_risk(
        step_quality=step_quality,
        prefix_quality=prefix_quality,
        open_obligation_burden=effective_obligation_burden,
    )
    if shallow_risk >= 0.45:
        logical_consistency = min(logical_consistency, max(0.22, 0.72 - 0.35 * shallow_risk))
        answer_correctness_likelihood = min(
            answer_correctness_likelihood,
            max(0.16, 0.70 - 0.45 * shallow_risk),
        )
        adjustments["reasoning_penalty"] = "shallow_prefix_support"

    return (
        _clamp01(logical_consistency),
        _clamp01(symbolic_agreement),
        _clamp01(completeness),
        _clamp01(answer_correctness_likelihood),
        _clamp01(repairability),
        _clamp01(effective_obligation_burden),
        _clamp01(route_uncertainty),
        adjustments,
    )


def _build_components(
    *,
    model_confidence: float,
    logical_consistency: float,
    symbolic_agreement: float,
    completeness: float,
    answer_correctness_likelihood: float,
    repairability: float,
    step_quality: float,
    prefix_quality: float,
    uncertainty: UncertaintyBreakdown,
) -> tuple[ScoreComponent, ...]:
    weighted = (
        (ScoreComponentName.LOGICAL_CONSISTENCY, logical_consistency, 0.24),
        (ScoreComponentName.SYMBOLIC_AGREEMENT, symbolic_agreement, 0.20),
        (ScoreComponentName.COMPLETENESS, completeness, 0.12),
        (ScoreComponentName.ANSWER_CORRECTNESS_LIKELIHOOD, answer_correctness_likelihood, 0.24),
        (ScoreComponentName.REPAIRABILITY, repairability, 0.08),
        (ScoreComponentName.STEP_QUALITY, step_quality, 0.05),
        (ScoreComponentName.PREFIX_QUALITY, prefix_quality, 0.07),
        (ScoreComponentName.MODEL_CONFIDENCE, model_confidence, 0.0),
    )
    components = []
    for name, score, weight in weighted:
        components.append(
            ScoreComponent(
                name=name,
                raw_score=_clamp01(score),
                calibrated_score=_clamp01(score),
                weight=weight,
                contribution=_clamp01(score * weight),
                rationale=name.value,
            )
        )
    components.append(
        ScoreComponent(
            name=ScoreComponentName.UNCERTAINTY,
            raw_score=uncertainty.total,
            calibrated_score=uncertainty.total,
            weight=0.0,
            contribution=0.0,
            rationale="penalty_only_signal",
        )
    )
    return tuple(components)


def assemble_verdict(
    *,
    logical_consistency: float,
    symbolic_agreement: float,
    completeness: float,
    answer_correctness_likelihood: float,
    repairability: float,
    step_quality: float,
    prefix_quality: float,
    open_obligation_burden: float,
    symbolic_status: str,
    uncertainty: UncertaintyBreakdown,
    quality_tag: ReasoningQualityTag,
    thresholds: ScoreThresholds,
) -> tuple[VerifierVerdict, VerifierDecisionFlags]:
    strong_pass = (
        answer_correctness_likelihood >= thresholds.strong_pass_threshold
        and logical_consistency >= 0.72
        and symbolic_agreement >= 0.70
        and completeness >= 0.55
        and step_quality >= 0.55
        and prefix_quality >= 0.62
        and open_obligation_burden <= 0.35
        and symbolic_status in {"", "success"}
        and uncertainty.total < thresholds.uncertain_threshold
    )
    inconsistent = (
        logical_consistency < thresholds.logical_floor
        or symbolic_agreement < thresholds.symbolic_floor
        or prefix_quality < 0.32
    )
    incomplete = completeness < thresholds.completeness_floor
    plausible = (
        logical_consistency >= 0.60
        and symbolic_agreement >= 0.58
        and prefix_quality >= 0.45
        and open_obligation_burden <= 0.60
    )
    uncertain = (
        uncertainty.total >= thresholds.uncertain_threshold
        or abs(answer_correctness_likelihood - logical_consistency) >= 0.20
    )
    should_repair = (
        not strong_pass
        and repairability >= thresholds.repair_threshold
        and (plausible or incomplete or quality_tag == ReasoningQualityTag.INCOMPLETE_BUT_SALVAGEABLE)
    )
    should_prune = (
        not strong_pass
        and not should_repair
        and (
            inconsistent
            or (
                answer_correctness_likelihood < thresholds.prune_threshold
                and repairability < thresholds.repair_threshold
            )
        )
    )
    if strong_pass:
        verdict = VerifierVerdict.ACCEPT
    elif plausible and incomplete and not inconsistent:
        verdict = VerifierVerdict.ACCEPT_WITH_RESERVATIONS
    elif should_repair:
        verdict = VerifierVerdict.REPAIRABLE
    elif should_prune:
        verdict = VerifierVerdict.REJECT
    else:
        verdict = VerifierVerdict.ESCALATE

    flags = VerifierDecisionFlags(
        strong_pass=strong_pass,
        should_prune=should_prune,
        should_repair=should_repair,
        uncertain=uncertain,
        aggregation_ready=not should_prune and (answer_correctness_likelihood >= 0.25 or plausible),
    )
    return verdict, flags


def score_verifier_bundle(
    primary: Any = None,
    secondary: Any = None,
    *,
    calibration: ScoreCalibrationProfile | None = None,
    thresholds: ScoreThresholds | None = None,
    **kwargs: Any,
) -> VerifierScoreBundle:
    score_input = coerce_score_input(primary, secondary, **kwargs)
    calibration_profile = calibration or ScoreCalibrationProfile()
    threshold_profile = thresholds or ScoreThresholds()
    symbolic_status = _normalize_symbolic_status(score_input.symbolic_status)

    model_confidence = calibrate_score(score_input.model_confidence, calibration_profile.model_confidence)
    logical_consistency = calibrate_score(score_input.logical_consistency, calibration_profile.logical_consistency)
    symbolic_agreement = calibrate_score(score_input.symbolic_agreement, calibration_profile.symbolic_agreement)
    completeness = calibrate_score(score_input.completeness, calibration_profile.completeness)
    repairability = calibrate_score(score_input.repairability, calibration_profile.repairability)
    step_quality = _clamp01(score_input.step_quality)
    prefix_quality = _clamp01(score_input.prefix_quality)
    answer_is_explicit = score_input.answer_correctness_likelihood is not None
    answer_correctness_likelihood = calibrate_score(
        _derive_answer_likelihood(score_input),
        calibration_profile.answer_correctness_likelihood,
    )
    (
        logical_consistency,
        symbolic_agreement,
        completeness,
        answer_correctness_likelihood,
        repairability,
        open_obligation_burden,
        route_uncertainty,
        trust_adjustments,
    ) = _apply_symbolic_trust_policy(
        score_input,
        logical_consistency=logical_consistency,
        symbolic_agreement=symbolic_agreement,
        completeness=completeness,
        answer_correctness_likelihood=answer_correctness_likelihood,
        repairability=repairability,
        step_quality=step_quality,
        prefix_quality=prefix_quality,
        open_obligation_burden=score_input.open_obligation_burden,
        route_uncertainty=score_input.route_uncertainty,
    )

    uncertainty = _compute_uncertainty(
        model_confidence=model_confidence,
        answer_correctness_likelihood=answer_correctness_likelihood,
        logical_consistency=logical_consistency,
        symbolic_agreement=symbolic_agreement,
        completeness=completeness,
        step_quality=step_quality,
        prefix_quality=prefix_quality,
        open_obligation_burden=open_obligation_burden,
        route_uncertainty=route_uncertainty,
        symbolic_status=symbolic_status,
        symbolic_partial_support=score_input.symbolic_partial_support,
        discharged_obligation_count=score_input.discharged_obligation_count,
        contradicted_obligation_count=score_input.contradicted_obligation_count,
    )
    components = _build_components(
        model_confidence=model_confidence,
        logical_consistency=logical_consistency,
        symbolic_agreement=symbolic_agreement,
        completeness=completeness,
        answer_correctness_likelihood=answer_correctness_likelihood,
        repairability=repairability,
        step_quality=step_quality,
        prefix_quality=prefix_quality,
        uncertainty=uncertainty,
    )

    inconsistency_penalty = 0.18 if logical_consistency < 0.35 or symbolic_agreement < 0.30 else 0.0
    shallow_penalty = 0.12 * uncertainty.shallow_reasoning_risk
    trust_penalty = 0.0
    if symbolic_status == "contradiction":
        trust_penalty += 0.22
    elif symbolic_status in {"malformed_input", "execution_failure"}:
        trust_penalty += 0.12
    elif symbolic_status == "unsupported":
        trust_penalty += 0.06 if score_input.symbolic_partial_support else 0.12
    if score_input.contradicted_obligation_count > 0:
        trust_penalty += min(0.18, 0.06 * score_input.contradicted_obligation_count)
    smooth_unsupported_gap = _clamp01(
        answer_correctness_likelihood - max(symbolic_agreement, prefix_quality, step_quality)
    )
    smooth_unsupported_penalty = 0.0
    if symbolic_status == "unsupported":
        smooth_unsupported_penalty = (
            0.08 if score_input.symbolic_partial_support else 0.14
        ) + 0.12 * smooth_unsupported_gap
    effective_obligation_penalty = 0.12 * open_obligation_burden
    resolution_credit = min(0.05, 0.015 * max(0, score_input.discharged_obligation_count))
    quality_tag_penalty = 0.0
    if score_input.quality_tag is ReasoningQualityTag.WRONG_ANSWER_PLAUSIBLE_REASONING:
        quality_tag_penalty = 0.06
    elif score_input.quality_tag is ReasoningQualityTag.CORRECT_ANSWER_WRONG_REASONING:
        quality_tag_penalty = 0.03
    answer_weight = 0.28 if answer_is_explicit else 0.18
    branch_score = _clamp01(
        answer_weight * answer_correctness_likelihood
        + 0.24 * logical_consistency
        + 0.18 * symbolic_agreement
        + 0.10 * completeness
        + 0.06 * repairability
        + 0.05 * step_quality
        + 0.07 * prefix_quality
        - effective_obligation_penalty
        - 0.14 * uncertainty.total
        - inconsistency_penalty
        - shallow_penalty
        - trust_penalty
        - smooth_unsupported_penalty
        - quality_tag_penalty
        + resolution_credit
    )

    verdict, flags = assemble_verdict(
        logical_consistency=logical_consistency,
        symbolic_agreement=symbolic_agreement,
        completeness=completeness,
        answer_correctness_likelihood=answer_correctness_likelihood,
        repairability=repairability,
        step_quality=step_quality,
        prefix_quality=prefix_quality,
        open_obligation_burden=open_obligation_burden,
        symbolic_status=symbolic_status,
        uncertainty=uncertainty,
        quality_tag=score_input.quality_tag,
        thresholds=threshold_profile,
    )

    summary = (
        f"verdict={verdict.value} "
        f"answer={answer_correctness_likelihood:.3f} "
        f"logical={logical_consistency:.3f} "
        f"symbolic={symbolic_agreement:.3f} "
        f"complete={completeness:.3f} "
        f"step={step_quality:.3f} "
        f"prefix={prefix_quality:.3f} "
        f"repair={repairability:.3f} "
        f"uncertainty={uncertainty.total:.3f}"
    )

    return VerifierScoreBundle(
        problem_id=score_input.problem_id,
        branch_id=score_input.branch_id,
        mode=score_input.mode,
        artifact_state=score_input.artifact_state,
        model_confidence=model_confidence,
        logical_consistency=logical_consistency,
        symbolic_agreement=symbolic_agreement,
        completeness=completeness,
        answer_correctness_likelihood=answer_correctness_likelihood,
        repairability=repairability,
        step_quality=step_quality,
        prefix_quality=prefix_quality,
        open_obligation_burden=open_obligation_burden,
        uncertainty=uncertainty,
        branch_score=branch_score,
        verdict=verdict,
        flags=flags,
        components=components,
        quality_tag=score_input.quality_tag,
        failure_type=score_input.failure_type,
        repair_type=score_input.repair_type,
        summary=summary,
        metadata={
            **dict(score_input.metadata),
            **trust_adjustments,
            "step_quality": step_quality,
            "prefix_quality": prefix_quality,
            "open_obligation_burden": open_obligation_burden,
            "route_uncertainty": route_uncertainty,
            "trust_penalty": _clamp01(trust_penalty),
            "shallow_penalty": _clamp01(shallow_penalty),
            "smooth_unsupported_penalty": _clamp01(smooth_unsupported_penalty),
            "quality_tag_penalty": _clamp01(quality_tag_penalty),
            "resolution_credit": _clamp01(resolution_credit),
            "answer_weight": answer_weight,
        },
    )


def score_bundle(primary: Any = None, secondary: Any = None, **kwargs: Any) -> VerifierScoreBundle:
    return score_verifier_bundle(primary, secondary, **kwargs)


def score_branch(primary: Any = None, secondary: Any = None, **kwargs: Any) -> VerifierScoreBundle:
    return score_verifier_bundle(primary, secondary, **kwargs)


def compute_verifier_score(primary: Any = None, secondary: Any = None, **kwargs: Any) -> VerifierScoreBundle:
    return score_verifier_bundle(primary, secondary, **kwargs)


def is_prunable(bundle: VerifierScoreBundle, thresholds: ScoreThresholds | None = None) -> bool:
    if thresholds is None:
        return bundle.flags.should_prune
    return bundle.branch_score < thresholds.prune_threshold and bundle.repairability < thresholds.repair_threshold


def should_trigger_repair(bundle: VerifierScoreBundle, thresholds: ScoreThresholds | None = None) -> bool:
    if thresholds is None:
        return bundle.flags.should_repair
    return (
        bundle.repairability >= thresholds.repair_threshold
        and not is_prunable(bundle, thresholds)
        and bundle.verdict in {VerifierVerdict.REPAIRABLE, VerifierVerdict.ACCEPT_WITH_RESERVATIONS}
    )


def is_uncertain(bundle: VerifierScoreBundle, thresholds: ScoreThresholds | None = None) -> bool:
    if thresholds is None:
        return bundle.flags.uncertain
    return bundle.uncertainty.total >= thresholds.uncertain_threshold


def is_strong_pass(bundle: VerifierScoreBundle, thresholds: ScoreThresholds | None = None) -> bool:
    if thresholds is None:
        return bundle.flags.strong_pass
    return (
        bundle.answer_correctness_likelihood >= thresholds.strong_pass_threshold
        and bundle.logical_consistency >= 0.72
        and bundle.symbolic_agreement >= 0.70
    )


__all__ = [
    "VerifierScoreMode",
    "ScoreComponentName",
    "ScoreThresholdName",
    "ComponentCalibration",
    "ScoreCalibrationProfile",
    "ScoreThresholds",
    "VerifierScoreInput",
    "ScoreComponent",
    "UncertaintyBreakdown",
    "VerifierDecisionFlags",
    "VerifierScoreBundle",
    "normalize_score",
    "calibrate_score",
    "coerce_score_input",
    "assemble_verdict",
    "score_verifier_bundle",
    "score_bundle",
    "score_branch",
    "compute_verifier_score",
    "is_prunable",
    "should_trigger_repair",
    "is_uncertain",
    "is_strong_pass",
]
