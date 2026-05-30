from __future__ import annotations

from enum import Enum
from hashlib import sha1
import json
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.branches.branch_state import BranchState
from src.branches.failure_classifier import BranchFailureDiagnosis
from src.common.constants import VERIFIER_PASS_THRESHOLD
from src.common.schemas import (
    BranchTrace,
    FailureType,
    VerifierLabel as LegacyVerifierLabel,
)

SCHEMA_VERSION = "verifier_labels.v1"

# Intentionally non-recursive for pydantic/schema safety.
LabelMetadata = dict[str, Any]


def _normalize_text(text: str | None) -> str:
    return " ".join((text or "").strip().split())


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _stable_hash(prefix: str, payload: Any) -> str:
    raw = f"{prefix}::{_stable_json(payload)}"
    return f"{prefix}_{sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _sorted_distribution(
    distribution: Mapping[str, float] | None,
) -> tuple["WeightedLabel", ...]:
    items = distribution or {}
    return tuple(
        WeightedLabel(name=name, weight=_clamp01(weight))
        for name, weight in sorted(items.items(), key=lambda item: (-float(item[1]), item[0]))
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return _clamp01(sum(float(v) for v in values) / len(values))


def _step_binary(score: float, explicit: bool | None = None) -> bool | None:
    if explicit is not None:
        return bool(explicit)
    if score <= 0.0:
        return False
    if score >= 1.0:
        return True
    return None


def _quality_tag_from_flags(
    *,
    answer_correct: bool | None,
    reasoning_soundness: float,
    completeness: float,
    repairability: float,
) -> "ReasoningQualityTag":
    if answer_correct is True and reasoning_soundness >= 0.75 and completeness >= 0.75:
        return ReasoningQualityTag.FULLY_CORRECT
    if answer_correct is True and reasoning_soundness < 0.75:
        return ReasoningQualityTag.CORRECT_ANSWER_WRONG_REASONING
    if answer_correct is False and reasoning_soundness >= 0.60:
        return ReasoningQualityTag.WRONG_ANSWER_PLAUSIBLE_REASONING
    if completeness < 0.55 and repairability >= 0.50:
        return ReasoningQualityTag.INCOMPLETE_BUT_SALVAGEABLE
    if repairability >= 0.65:
        return ReasoningQualityTag.PARTIAL_CORRECT_PREFIX
    return ReasoningQualityTag.WRONG_ANSWER_UNSOUND_REASONING


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        str_strip_whitespace=True,
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )


class LabelScope(str, Enum):
    STEP = "step"
    PREFIX = "prefix"
    BRANCH = "branch"


class VerifierLabelAxis(str, Enum):
    STEP_CORRECTNESS = "step_correctness"
    LOGICAL_CONSISTENCY = "logical_consistency"
    SYMBOLIC_AGREEMENT = "symbolic_agreement"
    COMPLETENESS = "completeness"
    ANSWER_CORRECTNESS = "answer_correctness"
    REPAIRABILITY = "repairability"
    VERDICT_CONFIDENCE = "verdict_confidence"


class ReasoningQualityTag(str, Enum):
    FULLY_CORRECT = "fully_correct"
    CORRECT_ANSWER_WRONG_REASONING = "correct_answer_wrong_reasoning"
    WRONG_ANSWER_PLAUSIBLE_REASONING = "wrong_answer_plausible_reasoning"
    WRONG_ANSWER_UNSOUND_REASONING = "wrong_answer_unsound_reasoning"
    INCOMPLETE_BUT_SALVAGEABLE = "incomplete_but_salvageable"
    PARTIAL_CORRECT_PREFIX = "partial_correct_prefix"
    REPAIRED_CORRECT = "repaired_correct"
    UNKNOWN = "unknown"


class VerifierVerdict(str, Enum):
    ACCEPT = "accept"
    ACCEPT_WITH_RESERVATIONS = "accept_with_reservations"
    REPAIRABLE = "repairable"
    REJECT = "reject"
    ESCALATE = "escalate"


class CalibrationGroupName(str, Enum):
    CORRECTNESS = "correctness"
    CONSISTENCY = "consistency"
    COMPLETENESS = "completeness"
    ANSWER = "answer"
    REPAIRABILITY = "repairability"
    VERDICT = "verdict"


class WeightedLabel(StrictModel):
    name: str
    weight: float = Field(default=0.0, ge=0.0, le=1.0)


class BranchLabelContext(StrictModel):
    branch_id: str
    problem_id: str = ""
    root_branch_id: str | None = None
    parent_branch_id: str | None = None
    branch_phase: str | None = None
    branch_depth: int = Field(default=0, ge=0)
    prefix_step_count: int = Field(default=0, ge=0)
    total_step_count: int = Field(default=0, ge=0)
    operator_sequence: tuple[str, ...] = Field(default_factory=tuple)
    route_problem_type: tuple[WeightedLabel, ...] = Field(default_factory=tuple)
    route_archetypes: tuple[WeightedLabel, ...] = Field(default_factory=tuple)
    retrieval_used: bool = False
    repair_count: int = Field(default=0, ge=0)
    symbolic_evidence_count: int = Field(default=0, ge=0)
    verifier_evidence_count: int = Field(default=0, ge=0)

    @classmethod
    def from_branch_trace(
        cls,
        trace: BranchTrace,
        *,
        prefix_step_count: int | None = None,
        route_problem_type: Mapping[str, float] | None = None,
        route_archetypes: Mapping[str, float] | None = None,
    ) -> "BranchLabelContext":
        prefix = len(trace.steps) if prefix_step_count is None else max(0, min(prefix_step_count, len(trace.steps)))
        return cls(
            branch_id=trace.branch_id,
            problem_id=trace.problem_id,
            branch_phase="trace",
            branch_depth=max(0, len(trace.steps) - 1),
            prefix_step_count=prefix,
            total_step_count=len(trace.steps),
            operator_sequence=tuple(trace.operator_sequence),
            route_problem_type=_sorted_distribution(route_problem_type),
            route_archetypes=_sorted_distribution(route_archetypes),
            retrieval_used=bool(trace.retrieval_used),
            repair_count=max(0, int(trace.repair_count)),
            symbolic_evidence_count=sum(1 for step in trace.steps if bool(step.symbolic_valid)),
            verifier_evidence_count=1 if float(trace.verifier_score or 0.0) > 0.0 else 0,
        )

    @classmethod
    def from_branch_state(
        cls,
        branch: BranchState,
        *,
        prefix_step_count: int | None = None,
    ) -> "BranchLabelContext":
        prefix = branch.active_cursor.step_count if prefix_step_count is None else max(
            0,
            min(prefix_step_count, branch.active_cursor.step_count),
        )
        return cls(
            branch_id=branch.branch_id,
            problem_id=branch.problem_id,
            root_branch_id=branch.root_branch_id,
            parent_branch_id=branch.parent_branch_id,
            branch_phase=branch.phase.value,
            branch_depth=max(0, len(branch.lineage)),
            prefix_step_count=prefix,
            total_step_count=branch.active_cursor.step_count,
            operator_sequence=tuple(
                step.operator_name for step in branch.active_steps()[:prefix] if step.operator_name
            ),
            route_problem_type=_sorted_distribution(branch.route.problem_type),
            route_archetypes=_sorted_distribution(branch.route.archetypes),
            retrieval_used=branch.active_cursor.retrieval_count > 0,
            repair_count=branch.active_cursor.repair_count,
            symbolic_evidence_count=branch.active_cursor.symbolic_count,
            verifier_evidence_count=branch.active_cursor.verifier_count,
        )


class CorrectnessTarget(StrictModel):
    scope: LabelScope
    step_is_correct: bool | None = None
    prefix_is_correct: bool | None = None
    branch_is_correct: bool | None = None
    step_correctness: float = Field(default=0.0, ge=0.0, le=1.0)
    prefix_correctness: float = Field(default=0.0, ge=0.0, le=1.0)
    branch_correctness: float = Field(default=0.0, ge=0.0, le=1.0)
    answer_correct: bool | None = None
    answer_correct_likelihood: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning_soundness: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_tag: ReasoningQualityTag = ReasoningQualityTag.UNKNOWN


class ConsistencyTarget(StrictModel):
    logical_consistency: float = Field(default=0.0, ge=0.0, le=1.0)
    cross_step_consistency: float = Field(default=0.0, ge=0.0, le=1.0)
    symbolic_agreement: float = Field(default=0.0, ge=0.0, le=1.0)
    tool_agreement: float = Field(default=0.0, ge=0.0, le=1.0)
    contradiction_detected: bool = False


class CompletenessTarget(StrictModel):
    completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    case_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    goal_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_case_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    terminal_ready: bool = False


class RepairabilityTarget(StrictModel):
    repairability: float = Field(default=0.0, ge=0.0, le=1.0)
    repairable: bool = False
    repair_type: str | None = None
    failure_type: str | None = None
    failure_location: str | None = None
    affected_scope: LabelScope = LabelScope.BRANCH
    prefix_keep_steps: int | None = Field(default=None, ge=0)
    prefix_rewrite_start: int | None = Field(default=None, ge=0)


class ConfidenceTarget(StrictModel):
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    calibrated_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_strength: float = Field(default=0.0, ge=0.0, le=1.0)


class VerdictTarget(StrictModel):
    verdict: VerifierVerdict
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    requires_repair: bool = False
    should_resample: bool = False
    escalate_to_search: bool = False
    rationale: str = ""


class VerifierModelTargets(StrictModel):
    branch_id: str
    prefix_step_count: int = Field(default=0, ge=0)
    step_correctness: tuple[float, ...] = Field(default_factory=tuple)
    logical_consistency: float = Field(default=0.0, ge=0.0, le=1.0)
    symbolic_agreement: float = Field(default=0.0, ge=0.0, le=1.0)
    completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    answer_correct_likelihood: float = Field(default=0.0, ge=0.0, le=1.0)
    repairability: float = Field(default=0.0, ge=0.0, le=1.0)
    verdict: VerifierVerdict
    verdict_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_tag: ReasoningQualityTag = ReasoningQualityTag.UNKNOWN


class StepSupervisionLabel(StrictModel):
    label_id: str
    scope: LabelScope = LabelScope.PREFIX
    context: BranchLabelContext
    step_index: int = Field(default=0, ge=0)
    step_id: str | None = None
    operator_name: str | None = None
    description: str = ""
    correctness: CorrectnessTarget
    consistency: ConsistencyTarget
    completeness: CompletenessTarget
    repairability: RepairabilityTarget
    confidence: ConfidenceTarget
    verdict: VerdictTarget

    def to_model_targets(self) -> VerifierModelTargets:
        return VerifierModelTargets(
            branch_id=self.context.branch_id,
            prefix_step_count=self.context.prefix_step_count,
            step_correctness=(self.correctness.step_correctness,),
            logical_consistency=self.consistency.logical_consistency,
            symbolic_agreement=self.consistency.symbolic_agreement,
            completeness=self.completeness.completeness,
            answer_correct_likelihood=self.correctness.answer_correct_likelihood,
            repairability=self.repairability.repairability,
            verdict=self.verdict.verdict,
            verdict_confidence=self.verdict.confidence,
            quality_tag=self.correctness.quality_tag,
        )

    @classmethod
    def create(
        cls,
        *,
        context: BranchLabelContext,
        step_index: int,
        step_id: str | None,
        operator_name: str | None,
        description: str,
        correctness: CorrectnessTarget,
        consistency: ConsistencyTarget,
        completeness: CompletenessTarget,
        repairability: RepairabilityTarget,
        confidence: ConfidenceTarget,
        verdict: VerdictTarget,
    ) -> "StepSupervisionLabel":
        payload = {
            "branch_id": context.branch_id,
            "prefix": context.prefix_step_count,
            "step_index": step_index,
            "step_id": step_id,
            "operator_name": operator_name,
            "quality_tag": correctness.quality_tag.value,
            "verdict": verdict.verdict.value,
        }
        return cls(
            label_id=_stable_hash("step_label", payload),
            context=context,
            step_index=step_index,
            step_id=step_id,
            operator_name=_normalize_text(operator_name) or None,
            description=_normalize_text(description),
            correctness=correctness,
            consistency=consistency,
            completeness=completeness,
            repairability=repairability,
            confidence=confidence,
            verdict=verdict,
        )


class BranchSupervisionLabel(StrictModel):
    label_id: str
    scope: LabelScope = LabelScope.BRANCH
    context: BranchLabelContext
    step_label_ids: tuple[str, ...] = Field(default_factory=tuple)
    correctness: CorrectnessTarget
    consistency: ConsistencyTarget
    completeness: CompletenessTarget
    repairability: RepairabilityTarget
    confidence: ConfidenceTarget
    verdict: VerdictTarget
    notes: str = ""

    def composite_score(self) -> float:
        weighted = (
            0.28 * self.correctness.branch_correctness
            + 0.20 * self.consistency.logical_consistency
            + 0.17 * self.consistency.symbolic_agreement
            + 0.15 * self.completeness.completeness
            + 0.10 * self.correctness.answer_correct_likelihood
            + 0.10 * self.repairability.repairability
        )
        return round(_clamp01(weighted), 6)

    def to_model_targets(
        self,
        step_labels: Sequence[StepSupervisionLabel] = (),
    ) -> VerifierModelTargets:
        return VerifierModelTargets(
            branch_id=self.context.branch_id,
            prefix_step_count=self.context.prefix_step_count,
            step_correctness=tuple(label.correctness.step_correctness for label in step_labels),
            logical_consistency=self.consistency.logical_consistency,
            symbolic_agreement=self.consistency.symbolic_agreement,
            completeness=self.completeness.completeness,
            answer_correct_likelihood=self.correctness.answer_correct_likelihood,
            repairability=self.repairability.repairability,
            verdict=self.verdict.verdict,
            verdict_confidence=self.verdict.confidence,
            quality_tag=self.correctness.quality_tag,
        )

    def to_legacy_schema(
        self,
        step_labels: Sequence[StepSupervisionLabel] = (),
    ) -> LegacyVerifierLabel:
        return LegacyVerifierLabel(
            branch_id=self.context.branch_id,
            step_correctness=[
                int(round(label.correctness.step_correctness))
                for label in sorted(step_labels, key=lambda item: item.step_index)
            ],
            symbolic_agreement=self.consistency.symbolic_agreement,
            logical_consistency=self.consistency.logical_consistency,
            completeness=self.completeness.completeness,
            final_answer_correct=int(self.correctness.answer_correct is True),
            final_answer_correct_probability=self.correctness.answer_correct_likelihood,
            repairability=self.repairability.repairability,
            repair_type=self.repairability.repair_type,
            failure_type=self.repairability.failure_type,
            failure_location=self.repairability.failure_location,
            composite_score=self.composite_score(),
        )

    @classmethod
    def from_legacy_schema(
        cls,
        legacy: LegacyVerifierLabel,
        *,
        problem_id: str = "",
        total_step_count: int | None = None,
        quality_tag: ReasoningQualityTag | None = None,
    ) -> "BranchSupervisionLabel":
        score = _clamp01(legacy.composite_score)
        answer_correct = bool(legacy.final_answer_correct)
        tag = quality_tag or _quality_tag_from_flags(
            answer_correct=answer_correct,
            reasoning_soundness=legacy.logical_consistency,
            completeness=legacy.completeness,
            repairability=legacy.repairability,
        )
        context = BranchLabelContext(
            branch_id=legacy.branch_id,
            problem_id=problem_id,
            branch_phase="legacy",
            prefix_step_count=total_step_count or len(legacy.step_correctness),
            total_step_count=total_step_count or len(legacy.step_correctness),
            symbolic_evidence_count=int(legacy.symbolic_agreement > 0.0),
            verifier_evidence_count=1,
        )
        return cls.create(
            context=context,
            step_label_ids=tuple(),
            correctness=CorrectnessTarget(
                scope=LabelScope.BRANCH,
                branch_is_correct=answer_correct and legacy.logical_consistency >= 0.75,
                branch_correctness=_mean(tuple(float(v) for v in legacy.step_correctness))
                if legacy.step_correctness
                else score,
                answer_correct=answer_correct,
                answer_correct_likelihood=legacy.final_answer_correct_probability,
                reasoning_soundness=legacy.logical_consistency,
                quality_tag=tag,
            ),
            consistency=ConsistencyTarget(
                logical_consistency=legacy.logical_consistency,
                cross_step_consistency=_mean(tuple(float(v) for v in legacy.step_correctness)),
                symbolic_agreement=legacy.symbolic_agreement,
                tool_agreement=legacy.symbolic_agreement,
                contradiction_detected=bool(
                    legacy.failure_type == FailureType.SYMBOLIC_MISMATCH.value
                ),
            ),
            completeness=CompletenessTarget(
                completeness=legacy.completeness,
                case_coverage=legacy.completeness,
                goal_coverage=legacy.completeness,
                missing_case_risk=_clamp01(1.0 - legacy.completeness),
                terminal_ready=legacy.completeness >= 0.70,
            ),
            repairability=RepairabilityTarget(
                repairability=legacy.repairability,
                repairable=legacy.repairability >= 0.50,
                repair_type=legacy.repair_type,
                failure_type=legacy.failure_type,
                failure_location=legacy.failure_location,
                affected_scope=LabelScope.BRANCH,
            ),
            confidence=ConfidenceTarget(
                confidence=score,
                calibrated_confidence=score,
                uncertainty=_clamp01(1.0 - score),
                evidence_strength=_mean(
                    (legacy.symbolic_agreement, legacy.logical_consistency, legacy.completeness)
                ),
            ),
            verdict=_verdict_from_targets(
                answer_correct_likelihood=legacy.final_answer_correct_probability,
                logical_consistency=legacy.logical_consistency,
                completeness=legacy.completeness,
                repairability=legacy.repairability,
                quality_tag=tag,
            ),
            notes="imported_from_legacy_schema",
        )

    @classmethod
    def create(
        cls,
        *,
        context: BranchLabelContext,
        step_label_ids: Sequence[str],
        correctness: CorrectnessTarget,
        consistency: ConsistencyTarget,
        completeness: CompletenessTarget,
        repairability: RepairabilityTarget,
        confidence: ConfidenceTarget,
        verdict: VerdictTarget,
        notes: str = "",
    ) -> "BranchSupervisionLabel":
        payload = {
            "branch_id": context.branch_id,
            "prefix": context.prefix_step_count,
            "quality_tag": correctness.quality_tag.value,
            "verdict": verdict.verdict.value,
            "step_label_ids": tuple(step_label_ids),
        }
        return cls(
            label_id=_stable_hash("branch_label", payload),
            context=context,
            step_label_ids=tuple(step_label_ids),
            correctness=correctness,
            consistency=consistency,
            completeness=completeness,
            repairability=repairability,
            confidence=confidence,
            verdict=verdict,
            notes=_normalize_text(notes),
        )


class CalibrationLabelGroup(StrictModel):
    group_id: str
    group_name: CalibrationGroupName
    branch_id: str
    label_ids: tuple[str, ...] = Field(default_factory=tuple)
    target_value: float = Field(default=0.0, ge=0.0, le=1.0)
    predicted_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    verdict: VerifierVerdict
    quality_tag: ReasoningQualityTag = ReasoningQualityTag.UNKNOWN

    @classmethod
    def create(
        cls,
        *,
        group_name: CalibrationGroupName,
        branch_id: str,
        label_ids: Sequence[str],
        target_value: float,
        predicted_confidence: float,
        verdict: VerifierVerdict,
        quality_tag: ReasoningQualityTag,
    ) -> "CalibrationLabelGroup":
        payload = {
            "group_name": group_name.value,
            "branch_id": branch_id,
            "label_ids": tuple(label_ids),
            "quality_tag": quality_tag.value,
        }
        return cls(
            group_id=_stable_hash("calibration_group", payload),
            group_name=group_name,
            branch_id=branch_id,
            label_ids=tuple(label_ids),
            target_value=_clamp01(target_value),
            predicted_confidence=_clamp01(predicted_confidence),
            verdict=verdict,
            quality_tag=quality_tag,
        )


class VerifierLabelBundle(StrictModel):
    schema_version: str = SCHEMA_VERSION
    bundle_id: str
    context: BranchLabelContext
    step_labels: tuple[StepSupervisionLabel, ...] = Field(default_factory=tuple)
    branch_label: BranchSupervisionLabel
    calibration_groups: tuple[CalibrationLabelGroup, ...] = Field(default_factory=tuple)
    metadata: LabelMetadata = Field(default_factory=dict)

    def to_legacy_schema(self) -> LegacyVerifierLabel:
        return self.branch_label.to_legacy_schema(self.step_labels)

    def to_model_targets(self) -> VerifierModelTargets:
        return self.branch_label.to_model_targets(self.step_labels)

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        return _stable_json(self.to_payload())

    @classmethod
    def create(
        cls,
        *,
        context: BranchLabelContext,
        step_labels: Sequence[StepSupervisionLabel],
        branch_label: BranchSupervisionLabel,
        calibration_groups: Sequence[CalibrationLabelGroup] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "VerifierLabelBundle":
        payload = {
            "branch_id": context.branch_id,
            "branch_label_id": branch_label.label_id,
            "step_label_ids": tuple(label.label_id for label in step_labels),
        }
        groups = tuple(calibration_groups or build_calibration_groups(branch_label, step_labels))
        return cls(
            bundle_id=_stable_hash("verifier_bundle", payload),
            context=context,
            step_labels=tuple(step_labels),
            branch_label=branch_label,
            calibration_groups=groups,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_json(cls, raw: str) -> "VerifierLabelBundle":
        return cls.model_validate_json(raw)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "VerifierLabelBundle":
        return cls.model_validate(payload)

    @classmethod
    def from_legacy_schema(
        cls,
        legacy: LegacyVerifierLabel,
        *,
        problem_id: str = "",
        quality_tag: ReasoningQualityTag | None = None,
    ) -> "VerifierLabelBundle":
        branch_label = BranchSupervisionLabel.from_legacy_schema(
            legacy,
            problem_id=problem_id,
            quality_tag=quality_tag,
        )
        step_labels = tuple(
            StepSupervisionLabel.create(
                context=branch_label.context.model_copy(update={"prefix_step_count": index + 1}),
                step_index=index,
                step_id=None,
                operator_name=None,
                description=f"legacy_step_{index}",
                correctness=CorrectnessTarget(
                    scope=LabelScope.PREFIX,
                    step_is_correct=bool(value),
                    prefix_is_correct=bool(value),
                    step_correctness=float(value),
                    prefix_correctness=_mean(
                        tuple(float(v) for v in legacy.step_correctness[: index + 1])
                    ),
                    answer_correct=bool(legacy.final_answer_correct),
                    answer_correct_likelihood=legacy.final_answer_correct_probability,
                    reasoning_soundness=legacy.logical_consistency,
                    quality_tag=branch_label.correctness.quality_tag,
                ),
                consistency=ConsistencyTarget(
                    logical_consistency=legacy.logical_consistency,
                    cross_step_consistency=_mean(
                        tuple(float(v) for v in legacy.step_correctness[: index + 1])
                    ),
                    symbolic_agreement=legacy.symbolic_agreement,
                    tool_agreement=legacy.symbolic_agreement,
                    contradiction_detected=bool(
                        legacy.failure_type == FailureType.SYMBOLIC_MISMATCH.value
                    ),
                ),
                completeness=CompletenessTarget(
                    completeness=_mean(
                        tuple(float(v) for v in legacy.step_correctness[: index + 1])
                    ),
                    case_coverage=_mean(
                        tuple(float(v) for v in legacy.step_correctness[: index + 1])
                    ),
                    goal_coverage=_mean(
                        tuple(float(v) for v in legacy.step_correctness[: index + 1])
                    ),
                    missing_case_risk=_clamp01(
                        1.0
                        - _mean(tuple(float(v) for v in legacy.step_correctness[: index + 1]))
                    ),
                    terminal_ready=index + 1 >= len(legacy.step_correctness),
                ),
                repairability=branch_label.repairability,
                confidence=ConfidenceTarget(
                    confidence=_mean((legacy.logical_consistency, legacy.symbolic_agreement)),
                    calibrated_confidence=_mean(
                        (legacy.logical_consistency, legacy.symbolic_agreement)
                    ),
                    uncertainty=_clamp01(
                        1.0 - _mean((legacy.logical_consistency, legacy.symbolic_agreement))
                    ),
                    evidence_strength=_mean(
                        (legacy.logical_consistency, legacy.symbolic_agreement)
                    ),
                ),
                verdict=branch_label.verdict,
            )
            for index, value in enumerate(legacy.step_correctness)
        )
        context = branch_label.context.model_copy(
            update={
                "prefix_step_count": len(step_labels),
                "total_step_count": len(step_labels),
            }
        )
        return cls.create(
            context=context,
            step_labels=step_labels,
            branch_label=branch_label.model_copy(
                update={
                    "context": context,
                    "step_label_ids": tuple(label.label_id for label in step_labels),
                }
            ),
            metadata={"source": "legacy_schema"},
        )

    @classmethod
    def from_branch_trace(
        cls,
        trace: BranchTrace,
        *,
        answer_correct: bool | None = None,
        answer_correct_likelihood: float | None = None,
        quality_tag: ReasoningQualityTag | None = None,
        route_problem_type: Mapping[str, float] | None = None,
        route_archetypes: Mapping[str, float] | None = None,
        failure_diagnosis: BranchFailureDiagnosis | None = None,
    ) -> "VerifierLabelBundle":
        total_steps = len(trace.steps)
        symbolic_scores = tuple(1.0 if bool(step.symbolic_valid) else 0.0 for step in trace.steps)
        branch_symbolic = (
            _mean(symbolic_scores) if symbolic_scores else _clamp01(float(trace.tool_consistency or 0.0))
        )
        branch_consistency = _clamp01(float(trace.verifier_score or trace.branch_score or 0.0))
        completeness = 1.0 if trace.answer else _clamp01(total_steps / max(total_steps + 1, 1))
        answer_likelihood = (
            _clamp01(answer_correct_likelihood)
            if answer_correct_likelihood is not None
            else _clamp01(float(trace.verifier_score or trace.branch_score or 0.0))
        )
        repairability_score = _clamp01(
            0.75 if trace.repaired else 0.40 if trace.failure_type is not None else 0.55
        )
        tag = quality_tag or _quality_tag_from_flags(
            answer_correct=answer_correct,
            reasoning_soundness=branch_consistency,
            completeness=completeness,
            repairability=repairability_score,
        )
        branch_context = BranchLabelContext.from_branch_trace(
            trace,
            route_problem_type=route_problem_type,
            route_archetypes=route_archetypes,
        )
        step_labels: list[StepSupervisionLabel] = []
        for index, step in enumerate(trace.steps):
            prefix_scores = symbolic_scores[: index + 1]
            prefix_correctness = _mean(prefix_scores) if prefix_scores else branch_consistency
            step_score = 1.0 if bool(step.symbolic_valid) else branch_consistency
            step_context = branch_context.model_copy(update={"prefix_step_count": index + 1})
            verdict = _verdict_from_targets(
                answer_correct_likelihood=answer_likelihood,
                logical_consistency=prefix_correctness,
                completeness=_clamp01((index + 1) / max(total_steps, 1)) if total_steps else completeness,
                repairability=repairability_score,
                quality_tag=tag,
            )
            step_labels.append(
                StepSupervisionLabel.create(
                    context=step_context,
                    step_index=index,
                    step_id=None,
                    operator_name=step.operator_used,
                    description=step.description,
                    correctness=CorrectnessTarget(
                        scope=LabelScope.PREFIX,
                        step_is_correct=_step_binary(step_score, bool(step.symbolic_valid)),
                        prefix_is_correct=_step_binary(prefix_correctness),
                        step_correctness=step_score,
                        prefix_correctness=prefix_correctness,
                        answer_correct=answer_correct,
                        answer_correct_likelihood=answer_likelihood,
                        reasoning_soundness=prefix_correctness,
                        quality_tag=tag if index + 1 == total_steps else ReasoningQualityTag.PARTIAL_CORRECT_PREFIX,
                    ),
                    consistency=ConsistencyTarget(
                        logical_consistency=prefix_correctness,
                        cross_step_consistency=prefix_correctness,
                        symbolic_agreement=_mean(prefix_scores) if prefix_scores else branch_symbolic,
                        tool_agreement=_mean(prefix_scores) if prefix_scores else branch_symbolic,
                        contradiction_detected=bool(not step.symbolic_valid),
                    ),
                    completeness=CompletenessTarget(
                        completeness=_clamp01((index + 1) / max(total_steps, 1)) if total_steps else completeness,
                        case_coverage=_clamp01((index + 1) / max(total_steps, 1)) if total_steps else completeness,
                        goal_coverage=_clamp01((index + 1) / max(total_steps, 1)) if total_steps else completeness,
                        missing_case_risk=_clamp01(1.0 - ((index + 1) / max(total_steps, 1))) if total_steps else 0.0,
                        terminal_ready=index + 1 == total_steps and bool(trace.answer),
                    ),
                    repairability=_repairability_target_from_failure(
                        repairability=repairability_score,
                        failure_type=trace.failure_type.value
                        if isinstance(trace.failure_type, FailureType)
                        else (trace.failure_type or None),
                        failure_location=trace.failure_location,
                        failure_diagnosis=failure_diagnosis,
                        prefix_step_count=index + 1,
                    ),
                    confidence=ConfidenceTarget(
                        confidence=_mean((step_score, prefix_correctness)),
                        calibrated_confidence=_mean((step_score, prefix_correctness)),
                        uncertainty=_clamp01(1.0 - _mean((step_score, prefix_correctness))),
                        evidence_strength=_mean((branch_symbolic, branch_consistency)),
                    ),
                    verdict=verdict,
                )
            )
        branch_label = BranchSupervisionLabel.create(
            context=branch_context.model_copy(update={"prefix_step_count": total_steps}),
            step_label_ids=tuple(label.label_id for label in step_labels),
            correctness=CorrectnessTarget(
                scope=LabelScope.BRANCH,
                branch_is_correct=answer_correct is True
                and branch_consistency >= 0.75
                and branch_symbolic >= 0.75,
                branch_correctness=_mean(
                    tuple(label.correctness.step_correctness for label in step_labels)
                )
                if step_labels
                else branch_consistency,
                answer_correct=answer_correct,
                answer_correct_likelihood=answer_likelihood,
                reasoning_soundness=branch_consistency,
                quality_tag=tag,
            ),
            consistency=ConsistencyTarget(
                logical_consistency=branch_consistency,
                cross_step_consistency=_mean(
                    tuple(label.consistency.cross_step_consistency for label in step_labels)
                )
                if step_labels
                else branch_consistency,
                symbolic_agreement=branch_symbolic,
                tool_agreement=_clamp01(float(trace.tool_consistency or branch_symbolic)),
                contradiction_detected=bool(trace.failure_type == FailureType.SYMBOLIC_MISMATCH),
            ),
            completeness=CompletenessTarget(
                completeness=completeness,
                case_coverage=completeness,
                goal_coverage=completeness,
                missing_case_risk=_clamp01(1.0 - completeness),
                terminal_ready=bool(trace.answer),
            ),
            repairability=_repairability_target_from_failure(
                repairability=repairability_score,
                failure_type=trace.failure_type.value
                if isinstance(trace.failure_type, FailureType)
                else (trace.failure_type or None),
                failure_location=trace.failure_location,
                failure_diagnosis=failure_diagnosis,
                prefix_step_count=total_steps,
            ),
            confidence=ConfidenceTarget(
                confidence=_mean((branch_symbolic, branch_consistency, answer_likelihood)),
                calibrated_confidence=_mean(
                    (branch_symbolic, branch_consistency, answer_likelihood)
                ),
                uncertainty=_clamp01(
                    1.0 - _mean((branch_symbolic, branch_consistency, answer_likelihood))
                ),
                evidence_strength=_mean(
                    (
                        float(trace.branch_score or 0.0),
                        float(trace.verifier_score or 0.0),
                        float(trace.tool_consistency or 0.0),
                    )
                ),
            ),
            verdict=_verdict_from_targets(
                answer_correct_likelihood=answer_likelihood,
                logical_consistency=branch_consistency,
                completeness=completeness,
                repairability=repairability_score,
                quality_tag=tag,
            ),
            notes="from_branch_trace",
        )
        return cls.create(
            context=branch_label.context,
            step_labels=step_labels,
            branch_label=branch_label,
            metadata={"source": "branch_trace"},
        )

    @classmethod
    def from_branch_state(
        cls,
        branch: BranchState,
        *,
        answer_correct: bool | None = None,
        answer_correct_likelihood: float | None = None,
        quality_tag: ReasoningQualityTag | None = None,
        failure_diagnosis: BranchFailureDiagnosis | None = None,
    ) -> "VerifierLabelBundle":
        active_steps = branch.active_steps()
        symbolic_evidence = branch.active_symbolic_evidence()
        verifier_evidence = branch.active_verifier_evidence()
        candidate = branch.current_candidate()

        symbolic_scores = tuple(
            (e.score if e.passed else 0.0) for e in symbolic_evidence
        )
        symbolic_score = (
            _mean(symbolic_scores)
            if symbolic_scores
            else _clamp01(branch.score_breakdown.exact_symbolic_check)
        )
        logical_consistency = (
            verifier_evidence[-1].logical_consistency
            if verifier_evidence
            else _clamp01(branch.score_breakdown.verifier_probability)
        )
        completeness = (
            verifier_evidence[-1].completeness
            if verifier_evidence
            else _clamp01(branch.score_breakdown.answer_agreement + 0.25)
        )
        repairability_score = (
            verifier_evidence[-1].repairability
            if verifier_evidence
            else _clamp01(0.40 + 0.15 * min(branch.active_cursor.repair_count, 2))
        )
        answer_likelihood = (
            _clamp01(answer_correct_likelihood)
            if answer_correct_likelihood is not None
            else _clamp01(
                max(
                    float(candidate.confidence if candidate is not None else 0.0),
                    float(branch.score_breakdown.answer_agreement),
                    float(branch.score_breakdown.verifier_probability),
                )
            )
        )
        answer_is_correct = (
            answer_correct
            if answer_correct is not None
            else (
                bool(candidate is not None and candidate.is_valid)
                if candidate is not None
                else None
            )
        )
        tag = quality_tag or _quality_tag_from_flags(
            answer_correct=answer_is_correct,
            reasoning_soundness=logical_consistency,
            completeness=completeness,
            repairability=repairability_score,
        )
        context = BranchLabelContext.from_branch_state(branch)
        step_labels: list[StepSupervisionLabel] = []

        for index, step in enumerate(active_steps):
            prefix = active_steps[: index + 1]
            prefix_symbolic = symbolic_scores[: index + 1]
            prefix_correctness = (
                _mean(prefix_symbolic)
                if prefix_symbolic
                else _clamp01(logical_consistency * ((index + 1) / max(len(active_steps), 1)))
            )
            step_score = (
                _clamp01(1.0 if getattr(step, "symbolic_valid", False) else prefix_correctness)
            )
            step_context = context.model_copy(update={"prefix_step_count": index + 1})
            step_verdict = _verdict_from_targets(
                answer_correct_likelihood=answer_likelihood,
                logical_consistency=prefix_correctness,
                completeness=_clamp01((index + 1) / max(len(active_steps), 1)) if active_steps else completeness,
                repairability=repairability_score,
                quality_tag=tag,
            )
            step_labels.append(
                StepSupervisionLabel.create(
                    context=step_context,
                    step_index=index,
                    step_id=getattr(step, "step_id", None),
                    operator_name=getattr(step, "operator_name", None),
                    description=getattr(step, "description", ""),
                    correctness=CorrectnessTarget(
                        scope=LabelScope.PREFIX,
                        step_is_correct=_step_binary(step_score, getattr(step, "symbolic_valid", None)),
                        prefix_is_correct=_step_binary(prefix_correctness),
                        step_correctness=step_score,
                        prefix_correctness=prefix_correctness,
                        answer_correct=answer_is_correct,
                        answer_correct_likelihood=answer_likelihood,
                        reasoning_soundness=prefix_correctness,
                        quality_tag=tag if index + 1 == len(active_steps) else ReasoningQualityTag.PARTIAL_CORRECT_PREFIX,
                    ),
                    consistency=ConsistencyTarget(
                        logical_consistency=prefix_correctness,
                        cross_step_consistency=_mean(
                            tuple(_clamp01(1.0 if getattr(s, "symbolic_valid", False) else 0.0) for s in prefix)
                        )
                        if prefix
                        else prefix_correctness,
                        symbolic_agreement=_mean(prefix_symbolic) if prefix_symbolic else symbolic_score,
                        tool_agreement=_mean(prefix_symbolic) if prefix_symbolic else symbolic_score,
                        contradiction_detected=bool(getattr(step, "symbolic_valid", True) is False),
                    ),
                    completeness=CompletenessTarget(
                        completeness=_clamp01((index + 1) / max(len(active_steps), 1)) if active_steps else completeness,
                        case_coverage=_clamp01((index + 1) / max(len(active_steps), 1)) if active_steps else completeness,
                        goal_coverage=_clamp01((index + 1) / max(len(active_steps), 1)) if active_steps else completeness,
                        missing_case_risk=_clamp01(1.0 - ((index + 1) / max(len(active_steps), 1))) if active_steps else 0.0,
                        terminal_ready=index + 1 == len(active_steps) and candidate is not None,
                    ),
                    repairability=_repairability_target_from_failure(
                        repairability=repairability_score,
                        failure_type=_branch_failure_type_string(branch),
                        failure_location=_branch_failure_location(branch),
                        failure_diagnosis=failure_diagnosis,
                        prefix_step_count=index + 1,
                    ),
                    confidence=ConfidenceTarget(
                        confidence=_mean((step_score, prefix_correctness)),
                        calibrated_confidence=_mean((step_score, prefix_correctness)),
                        uncertainty=_clamp01(1.0 - _mean((step_score, prefix_correctness))),
                        evidence_strength=_mean((symbolic_score, logical_consistency)),
                    ),
                    verdict=step_verdict,
                )
            )

        branch_label = BranchSupervisionLabel.create(
            context=context.model_copy(update={"prefix_step_count": len(active_steps)}),
            step_label_ids=tuple(label.label_id for label in step_labels),
            correctness=CorrectnessTarget(
                scope=LabelScope.BRANCH,
                branch_is_correct=answer_is_correct is True
                and logical_consistency >= VERIFIER_PASS_THRESHOLD
                and symbolic_score >= VERIFIER_PASS_THRESHOLD,
                branch_correctness=_mean(
                    tuple(label.correctness.step_correctness for label in step_labels)
                )
                if step_labels
                else logical_consistency,
                answer_correct=answer_is_correct,
                answer_correct_likelihood=answer_likelihood,
                reasoning_soundness=logical_consistency,
                quality_tag=tag,
            ),
            consistency=ConsistencyTarget(
                logical_consistency=logical_consistency,
                cross_step_consistency=_mean(
                    tuple(label.consistency.cross_step_consistency for label in step_labels)
                )
                if step_labels
                else logical_consistency,
                symbolic_agreement=symbolic_score,
                tool_agreement=symbolic_score,
                contradiction_detected=bool(
                    _branch_failure_type_string(branch) == FailureType.SYMBOLIC_MISMATCH.value
                ),
            ),
            completeness=CompletenessTarget(
                completeness=completeness,
                case_coverage=completeness,
                goal_coverage=completeness,
                missing_case_risk=_clamp01(1.0 - completeness),
                terminal_ready=bool(candidate is not None),
            ),
            repairability=_repairability_target_from_failure(
                repairability=repairability_score,
                failure_type=_branch_failure_type_string(branch),
                failure_location=_branch_failure_location(branch),
                failure_diagnosis=failure_diagnosis,
                prefix_step_count=len(active_steps),
            ),
            confidence=ConfidenceTarget(
                confidence=_mean((symbolic_score, logical_consistency, answer_likelihood)),
                calibrated_confidence=_mean((symbolic_score, logical_consistency, answer_likelihood)),
                uncertainty=_clamp01(
                    1.0 - _mean((symbolic_score, logical_consistency, answer_likelihood))
                ),
                evidence_strength=_mean(
                    (
                        float(branch.score_breakdown.verifier_probability),
                        float(branch.score_breakdown.exact_symbolic_check),
                        float(branch.score_breakdown.answer_agreement),
                    )
                ),
            ),
            verdict=_verdict_from_targets(
                answer_correct_likelihood=answer_likelihood,
                logical_consistency=logical_consistency,
                completeness=completeness,
                repairability=repairability_score,
                quality_tag=tag,
            ),
            notes="from_branch_state",
        )

        return cls.create(
            context=branch_label.context,
            step_labels=step_labels,
            branch_label=branch_label,
            metadata={"source": "branch_state"},
        )


def _repairability_target_from_failure(
    *,
    repairability: float,
    failure_type: str | None,
    failure_location: str | None,
    failure_diagnosis: BranchFailureDiagnosis | None,
    prefix_step_count: int,
) -> RepairabilityTarget:
    diagnosis_failure_type = getattr(failure_diagnosis, "failure_type", None)
    if isinstance(diagnosis_failure_type, FailureType):
        diagnosis_failure_type = diagnosis_failure_type.value
    elif diagnosis_failure_type is not None:
        diagnosis_failure_type = str(diagnosis_failure_type)

    diagnosis_location = _normalize_text(getattr(failure_diagnosis, "failure_location", None)) or None
    diagnosis_reason = _normalize_text(getattr(failure_diagnosis, "repair_type", None)) or None

    normalized_failure_type = diagnosis_failure_type or failure_type
    normalized_location = diagnosis_location or failure_location
    repairable = repairability >= 0.50

    return RepairabilityTarget(
        repairability=_clamp01(repairability),
        repairable=repairable,
        repair_type=diagnosis_reason,
        failure_type=normalized_failure_type,
        failure_location=normalized_location,
        affected_scope=LabelScope.PREFIX if normalized_location else LabelScope.BRANCH,
        prefix_keep_steps=max(0, prefix_step_count - 1) if repairable and prefix_step_count > 0 else None,
        prefix_rewrite_start=max(0, prefix_step_count - 1) if repairable and prefix_step_count > 0 else None,
    )


def _verdict_from_targets(
    *,
    answer_correct_likelihood: float,
    logical_consistency: float,
    completeness: float,
    repairability: float,
    quality_tag: ReasoningQualityTag,
) -> VerdictTarget:
    answer_correct_likelihood = _clamp01(answer_correct_likelihood)
    logical_consistency = _clamp01(logical_consistency)
    completeness = _clamp01(completeness)
    repairability = _clamp01(repairability)

    if (
        answer_correct_likelihood >= VERIFIER_PASS_THRESHOLD
        and logical_consistency >= VERIFIER_PASS_THRESHOLD
        and completeness >= 0.70
    ):
        verdict = VerifierVerdict.ACCEPT
        confidence = _mean((answer_correct_likelihood, logical_consistency, completeness))
        rationale = "high answer plausibility, consistency, and completeness"
        requires_repair = False
        should_resample = False
        escalate = False
    elif repairability >= 0.60:
        verdict = VerifierVerdict.REPAIRABLE
        confidence = _mean((repairability, logical_consistency))
        rationale = "branch appears locally repairable"
        requires_repair = True
        should_resample = False
        escalate = False
    elif logical_consistency >= 0.55 and completeness >= 0.45:
        verdict = VerifierVerdict.ACCEPT_WITH_RESERVATIONS
        confidence = _mean((logical_consistency, completeness))
        rationale = "partially consistent but incomplete or weakly justified"
        requires_repair = False
        should_resample = quality_tag in {
            ReasoningQualityTag.INCOMPLETE_BUT_SALVAGEABLE,
            ReasoningQualityTag.WRONG_ANSWER_PLAUSIBLE_REASONING,
        }
        escalate = False
    elif quality_tag in {
        ReasoningQualityTag.WRONG_ANSWER_PLAUSIBLE_REASONING,
        ReasoningQualityTag.INCOMPLETE_BUT_SALVAGEABLE,
    }:
        verdict = VerifierVerdict.ESCALATE
        confidence = _mean((logical_consistency, repairability))
        rationale = "plausible structure but not safe to accept"
        requires_repair = False
        should_resample = True
        escalate = True
    else:
        verdict = VerifierVerdict.REJECT
        confidence = _mean((1.0 - logical_consistency, 1.0 - answer_correct_likelihood))
        rationale = "insufficient consistency or answer plausibility"
        requires_repair = False
        should_resample = repairability >= 0.40
        escalate = should_resample

    return VerdictTarget(
        verdict=verdict,
        confidence=_clamp01(confidence),
        requires_repair=requires_repair,
        should_resample=should_resample,
        escalate_to_search=escalate,
        rationale=rationale,
    )


def build_calibration_groups(
    branch_label: BranchSupervisionLabel,
    step_labels: Sequence[StepSupervisionLabel],
) -> tuple[CalibrationLabelGroup, ...]:
    label_ids = tuple(label.label_id for label in step_labels)
    targets = branch_label.to_model_targets(step_labels)

    groups = (
        CalibrationLabelGroup.create(
            group_name=CalibrationGroupName.CORRECTNESS,
            branch_id=branch_label.context.branch_id,
            label_ids=label_ids,
            target_value=targets.answer_correct_likelihood,
            predicted_confidence=branch_label.confidence.calibrated_confidence,
            verdict=branch_label.verdict.verdict,
            quality_tag=branch_label.correctness.quality_tag,
        ),
        CalibrationLabelGroup.create(
            group_name=CalibrationGroupName.CONSISTENCY,
            branch_id=branch_label.context.branch_id,
            label_ids=label_ids,
            target_value=branch_label.consistency.logical_consistency,
            predicted_confidence=branch_label.confidence.calibrated_confidence,
            verdict=branch_label.verdict.verdict,
            quality_tag=branch_label.correctness.quality_tag,
        ),
        CalibrationLabelGroup.create(
            group_name=CalibrationGroupName.COMPLETENESS,
            branch_id=branch_label.context.branch_id,
            label_ids=label_ids,
            target_value=branch_label.completeness.completeness,
            predicted_confidence=branch_label.confidence.calibrated_confidence,
            verdict=branch_label.verdict.verdict,
            quality_tag=branch_label.correctness.quality_tag,
        ),
        CalibrationLabelGroup.create(
            group_name=CalibrationGroupName.ANSWER,
            branch_id=branch_label.context.branch_id,
            label_ids=label_ids,
            target_value=branch_label.correctness.answer_correct_likelihood,
            predicted_confidence=branch_label.verdict.confidence,
            verdict=branch_label.verdict.verdict,
            quality_tag=branch_label.correctness.quality_tag,
        ),
        CalibrationLabelGroup.create(
            group_name=CalibrationGroupName.REPAIRABILITY,
            branch_id=branch_label.context.branch_id,
            label_ids=label_ids,
            target_value=branch_label.repairability.repairability,
            predicted_confidence=branch_label.verdict.confidence,
            verdict=branch_label.verdict.verdict,
            quality_tag=branch_label.correctness.quality_tag,
        ),
        CalibrationLabelGroup.create(
            group_name=CalibrationGroupName.VERDICT,
            branch_id=branch_label.context.branch_id,
            label_ids=label_ids,
            target_value=1.0 if branch_label.verdict.verdict in {
                VerifierVerdict.ACCEPT,
                VerifierVerdict.ACCEPT_WITH_RESERVATIONS,
            } else 0.0,
            predicted_confidence=branch_label.verdict.confidence,
            verdict=branch_label.verdict.verdict,
            quality_tag=branch_label.correctness.quality_tag,
        ),
    )
    return groups


def _branch_failure_type_string(branch: BranchState) -> str | None:
    value = getattr(branch, "failure_type", None)
    if isinstance(value, FailureType):
        return value.value
    if value is not None:
        return str(value)
    candidate = branch.current_candidate()
    if candidate is not None:
        candidate_value = getattr(candidate, "failure_type", None)
        if isinstance(candidate_value, FailureType):
            return candidate_value.value
        if candidate_value is not None:
            return str(candidate_value)
    return None


def _branch_failure_location(branch: BranchState) -> str | None:
    value = getattr(branch, "failure_location", None)
    if value:
        return str(value)
    candidate = branch.current_candidate()
    if candidate is not None:
        candidate_value = getattr(candidate, "failure_location", None)
        if candidate_value:
            return str(candidate_value)
    return None


__all__ = [
    "SCHEMA_VERSION",
    "LabelScope",
    "VerifierLabelAxis",
    "ReasoningQualityTag",
    "VerifierVerdict",
    "CalibrationGroupName",
    "WeightedLabel",
    "BranchLabelContext",
    "CorrectnessTarget",
    "ConsistencyTarget",
    "CompletenessTarget",
    "RepairabilityTarget",
    "ConfidenceTarget",
    "VerdictTarget",
    "VerifierModelTargets",
    "StepSupervisionLabel",
    "BranchSupervisionLabel",
    "CalibrationLabelGroup",
    "VerifierLabelBundle",
    "build_calibration_groups",
]