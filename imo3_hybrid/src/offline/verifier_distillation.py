from __future__ import annotations

"""Typed verifier distillation dataset construction.

This module converts branch traces / branch states / distilled trace records into
stable verifier-training examples with preserved label decomposition, provenance,
and deterministic dataset emission.

It is intentionally more than a thin export wrapper:
- aligns examples with verifier_labels.py bundle + target structures
- generates branch-level and prefix-level supervision examples
- preserves plausible-wrong / unsound / contradicted / repaired distinctions
- emits runtime-verifier and stronger-verifier compatible payloads
- preserves richer decomposed supervision/control-plane signals
- provides deterministic artifact serialization helpers
"""

from enum import Enum
from hashlib import sha1
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.common.constants import (
    OFFLINE_ARTIFACT_VERSION,
    OFFLINE_SPLIT_RATIOS,
    OFFLINE_SPLIT_SALT,
)
from src.common.schemas import BranchTrace, FailureType, VerifierLabel as LegacyVerifierLabel
from src.verifier.verifier_labels import (
    BranchSupervisionLabel,
    CalibrationGroupName,
    LabelScope,
    ReasoningQualityTag,
    StepSupervisionLabel,
    VerifierLabelBundle,
    VerifierModelTargets,
    VerifierVerdict,
)

try:
    from src.branches.branch_state import BranchState
except Exception:  # pragma: no cover
    BranchState = Any  # type: ignore[assignment]

try:
    from src.offline.trace_distillation import (
        DistilledTraceArtifact,
        DistilledTraceClass,
        DistilledTraceRecord,
        TraceQualityTier,
    )
except Exception:  # pragma: no cover
    DistilledTraceArtifact = Any  # type: ignore[assignment]
    DistilledTraceClass = Any  # type: ignore[assignment]
    DistilledTraceRecord = Any  # type: ignore[assignment]
    TraceQualityTier = Any  # type: ignore[assignment]


SCHEMA_VERSION = "verifier_distillation.v2"
ARTIFACT_KIND = "verifier_training_dataset"
DEFAULT_ARTIFACT_VERSION = OFFLINE_ARTIFACT_VERSION


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        str_strip_whitespace=True,
        populate_by_name=True,
    )


class ExampleScope(str, Enum):
    BRANCH = "branch"
    PREFIX = "prefix"


class ExampleRetentionTag(str, Enum):
    POSITIVE = "positive"
    PLAUSIBLE_WRONG = "plausible_wrong"
    UNSOUND_WRONG = "unsound_wrong"
    SYMBOLIC_CONTRADICTED = "symbolic_contradicted"
    REPAIRED = "repaired"
    VERIFIER_REJECTED = "verifier_rejected"
    LEGACY = "legacy"
    UNKNOWN = "unknown"


class DistillationSourceType(str, Enum):
    BRANCH_TRACE = "branch_trace"
    BRANCH_STATE = "branch_state"
    TRACE_DISTILLATION_RECORD = "trace_distillation_record"
    TRACE_DISTILLATION_ARTIFACT = "trace_distillation_artifact"
    VERIFIER_BUNDLE = "verifier_bundle"
    LEGACY_LABEL = "legacy_label"


class SplitName(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    CALIBRATION = "calibration"


class VerifierDistillationConfig(StrictModel):
    artifact_version: str = DEFAULT_ARTIFACT_VERSION
    schema_version: str = SCHEMA_VERSION
    include_branch_examples: bool = True
    include_prefix_examples: bool = True
    min_prefix_steps: int = Field(default=1, ge=1)
    max_prefix_examples_per_branch: int | None = Field(default=None, ge=1)
    prefix_stride: int = Field(default=1, ge=1)
    include_terminal_prefix: bool = True
    retain_plausible_wrong: bool = True
    retain_unsound_wrong: bool = True
    retain_symbolic_contradicted: bool = True
    retain_repaired: bool = True
    retain_verifier_rejected: bool = True
    split_ratios: tuple[float, float, float] = OFFLINE_SPLIT_RATIOS
    split_salt: str = OFFLINE_SPLIT_SALT

    def normalized_split_ratios(self) -> tuple[float, float, float]:
        total = sum(self.split_ratios)
        if total <= 0:
            return (0.80, 0.10, 0.10)
        return tuple(float(x) / total for x in self.split_ratios)  # type: ignore[return-value]


class SourceProvenance(StrictModel):
    source_type: DistillationSourceType
    source_record_id: str
    problem_id: str = ""
    branch_id: str
    bundle_id: str | None = None
    label_id: str | None = None
    source_digest: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExampleTextFields(StrictModel):
    problem_text: str = ""
    prefix_text: str = ""
    reasoning_text: str = ""
    answer_text: str = ""
    canonical_answer: str = ""
    operator_sequence: tuple[str, ...] = Field(default_factory=tuple)
    route_problem_type: tuple[str, ...] = Field(default_factory=tuple)
    route_archetypes: tuple[str, ...] = Field(default_factory=tuple)


class VerifierLabelDecomposition(StrictModel):
    bundle: dict[str, Any]
    branch_label: dict[str, Any]
    step_label: dict[str, Any] | None = None
    legacy_label: dict[str, Any]
    model_targets: dict[str, Any]
    decomposed_signals: dict[str, float] = Field(default_factory=dict)
    proof_obligation_summary: dict[str, Any] = Field(default_factory=dict)
    retrieval_summary: dict[str, Any] = Field(default_factory=dict)
    operator_summary: dict[str, Any] = Field(default_factory=dict)
    route_compute_signals: dict[str, float] = Field(default_factory=dict)


class VerifierTrainingExample(StrictModel):
    example_id: str
    split: SplitName
    scope: ExampleScope
    retention_tag: ExampleRetentionTag
    problem_id: str = ""
    branch_id: str
    prefix_step_count: int = Field(default=0, ge=0)
    total_step_count: int = Field(default=0, ge=0)
    quality_tag: ReasoningQualityTag = ReasoningQualityTag.UNKNOWN
    verdict: VerifierVerdict
    texts: ExampleTextFields
    labels: VerifierLabelDecomposition
    provenance: SourceProvenance
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerifierTrainingArtifactMetadata(StrictModel):
    artifact_kind: str = ARTIFACT_KIND
    artifact_version: str = DEFAULT_ARTIFACT_VERSION
    schema_version: str = SCHEMA_VERSION
    artifact_id: str
    example_count: int
    source_count: int
    branch_count: int
    config_digest: str
    split_counts: dict[str, int] = Field(default_factory=dict)
    scope_counts: dict[str, int] = Field(default_factory=dict)
    retention_counts: dict[str, int] = Field(default_factory=dict)
    source_digest: str


class VerifierTrainingArtifact(StrictModel):
    metadata: VerifierTrainingArtifactMetadata
    examples: tuple[VerifierTrainingExample, ...] = Field(default_factory=tuple)


class VerifierDistillationReport(StrictModel):
    kept_count: int
    dropped_count: int
    split_counts: dict[str, int] = Field(default_factory=dict)
    scope_counts: dict[str, int] = Field(default_factory=dict)
    retention_counts: dict[str, int] = Field(default_factory=dict)
    dropped_reasons: dict[str, int] = Field(default_factory=dict)


class VerifierArtifactWriteResult(StrictModel):
    artifact_dir: str
    manifest_path: str
    examples_path: str
    split_paths: dict[str, str] = Field(default_factory=dict)


class _NormalizedBundle(StrictModel):
    source_type: DistillationSourceType
    source_record_id: str
    problem_id: str = ""
    branch_id: str
    bundle: dict[str, Any]
    source_split: SplitName | None = None
    problem_text: str = ""
    reasoning_steps: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    operator_sequence: tuple[str, ...] = Field(default_factory=tuple)
    answer_text: str = ""
    canonical_answer: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class _PrefixSelection(StrictModel):
    step_index: int = Field(ge=1)
    label_id: str
    operator_name: str = ""


# --------------------------
# utility helpers
# --------------------------


def _normalize_text(text: str | None) -> str:
    return " ".join((text or "").strip().split())


def _clamp01(value: float | int | None) -> float:
    try:
        return max(0.0, min(1.0, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _stable_hash(prefix: str, payload: Any) -> str:
    raw = f"{prefix}::{_stable_json(payload)}"
    return f"{prefix}_{sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _enum_value(value: Any) -> Any:
    raw = getattr(value, "value", value)
    return raw


def _as_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return _as_json_value(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(k): _as_json_value(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_as_json_value(v) for v in value]
    return _normalize_text(str(value))


def _reasoning_text_from_steps(steps: Sequence[Mapping[str, Any]], prefix_count: int | None = None) -> str:
    selected = list(steps[:prefix_count] if prefix_count is not None else steps)
    lines: list[str] = []
    for index, step in enumerate(selected, start=1):
        text = _normalize_text(str(step.get("description") or step.get("text") or ""))
        operator_name = _normalize_text(str(step.get("operator_name") or step.get("operator_used") or ""))
        if not text:
            continue
        if operator_name:
            lines.append(f"Step {index} [{operator_name}]: {text}")
        else:
            lines.append(f"Step {index}: {text}")
    return "\n".join(lines)


def _collect_operator_sequence_from_steps(steps: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    operators: list[str] = []
    for step in steps:
        name = _normalize_text(str(step.get("operator_name") or step.get("operator_used") or ""))
        if name:
            operators.append(name)
    return tuple(dict.fromkeys(operators))


def _split_for_problem(problem_id: str, config: VerifierDistillationConfig) -> SplitName:
    ratios = config.normalized_split_ratios()
    bucket = int(sha1(f"{config.split_salt}::{problem_id}".encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < ratios[0]:
        return SplitName.TRAIN
    if bucket < ratios[0] + ratios[1]:
        return SplitName.VALID
    return SplitName.CALIBRATION


def _resolve_split(
    problem_id: str,
    *,
    source_split: SplitName | None,
    config: VerifierDistillationConfig,
) -> SplitName:
    if source_split is not None:
        return source_split
    return _split_for_problem(problem_id, config)


def _coerce_split(value: Any) -> SplitName | None:
    if value is None:
        return None
    if isinstance(value, SplitName):
        return value
    raw = getattr(value, "value", value)
    try:
        return SplitName(str(raw))
    except Exception:
        return None


def _extract_float(mapping: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            try:
                return _clamp01(float(mapping[key]))
            except Exception:
                continue
    return None


def _extract_text(mapping: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            value = _normalize_text(str(mapping[key]))
            if value:
                return value
    return ""


def _safe_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return {}


def _default_quality_tag(name: str) -> ReasoningQualityTag:
    if hasattr(ReasoningQualityTag, name):
        return getattr(ReasoningQualityTag, name)
    return ReasoningQualityTag.UNKNOWN


def _default_verdict(name: str) -> VerifierVerdict:
    if hasattr(VerifierVerdict, name):
        return getattr(VerifierVerdict, name)
    members = list(VerifierVerdict)
    return members[0]


def _coerce_quality_tag(value: Any) -> ReasoningQualityTag:
    raw = str(getattr(value, "value", value) or "").strip()
    for member in ReasoningQualityTag:
        if member.value == raw:
            return member
    return ReasoningQualityTag.UNKNOWN


def _coerce_verdict(value: Any) -> VerifierVerdict:
    raw = str(getattr(value, "value", value) or "").strip()
    for member in VerifierVerdict:
        if member.value == raw:
            return member
    return _default_verdict("UNCERTAIN")


def _quality_tag_from_signals(signals: Mapping[str, Any], metadata: Mapping[str, Any] | None = None) -> ReasoningQualityTag:
    logical = _extract_float(signals, "logical_consistency") or 0.0
    symbolic = _extract_float(signals, "symbolic_agreement") or _extract_float(signals, "symbolic_score") or 0.0
    completeness = _extract_float(signals, "completeness") or 0.0
    answer_like = _extract_float(signals, "answer_correctness_likelihood", "verifier_score") or 0.0
    repaired = bool((metadata or {}).get("repaired")) or bool((metadata or {}).get("repair_count"))
    symbolic_valid = bool((metadata or {}).get("symbolic_valid"))
    failure_type = _normalize_text(str((metadata or {}).get("failure_type") or ""))

    if repaired and answer_like >= 0.70 and logical >= 0.55:
        return _default_quality_tag("REPAIRED_CORRECT")
    if answer_like >= 0.80 and logical >= 0.75 and symbolic >= 0.75 and completeness >= 0.65:
        return _default_quality_tag("FULLY_CORRECT")
    if answer_like >= 0.65 and logical >= 0.55 and completeness >= 0.40:
        return _default_quality_tag("PARTIAL_CORRECT_PREFIX")
    if answer_like >= 0.75 and (logical < 0.50 or symbolic < 0.50):
        return _default_quality_tag("CORRECT_ANSWER_WRONG_REASONING")
    if answer_like < 0.55 and logical >= 0.50:
        return _default_quality_tag("WRONG_ANSWER_PLAUSIBLE_REASONING")
    if not symbolic_valid or "symbolic" in failure_type:
        return _default_quality_tag("WRONG_ANSWER_UNSOUND_REASONING")
    if logical < 0.40:
        return _default_quality_tag("WRONG_ANSWER_UNSOUND_REASONING")
    return ReasoningQualityTag.UNKNOWN


def _verdict_from_signals(signals: Mapping[str, Any], metadata: Mapping[str, Any] | None = None) -> VerifierVerdict:
    logical = _extract_float(signals, "logical_consistency") or 0.0
    symbolic = _extract_float(signals, "symbolic_agreement") or _extract_float(signals, "symbolic_score") or 0.0
    answer_like = _extract_float(signals, "answer_correctness_likelihood", "verifier_score") or 0.0
    repairability = _extract_float(signals, "repairability") or 0.0
    repaired = bool((metadata or {}).get("repaired"))
    symbolic_valid = bool((metadata or {}).get("symbolic_valid"))

    if answer_like >= 0.75 and logical >= 0.70 and symbolic >= 0.70:
        return _default_verdict("ACCEPT")
    if repaired and answer_like >= 0.65:
        return _default_verdict("ACCEPT")
    if repairability >= 0.55 or (logical >= 0.45 and answer_like >= 0.45):
        return _default_verdict("REPAIRABLE")
    if not symbolic_valid and symbolic < 0.40:
        return _default_verdict("REJECT")
    if logical < 0.35 and answer_like < 0.35:
        return _default_verdict("REJECT")
    return _default_verdict("UNCERTAIN")


def _retention_tag_for_quality(
    quality: ReasoningQualityTag,
    verdict: VerifierVerdict,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ExampleRetentionTag:
    source_cls = _normalize_text(str((metadata or {}).get("trace_class") or ""))
    if "symbolic_contradicted" in source_cls:
        return ExampleRetentionTag.SYMBOLIC_CONTRADICTED
    if "repaired" in source_cls or quality == _default_quality_tag("REPAIRED_CORRECT"):
        return ExampleRetentionTag.REPAIRED
    if source_cls == "verifier_rejected":
        return ExampleRetentionTag.VERIFIER_REJECTED
    if quality in {_default_quality_tag("FULLY_CORRECT"), _default_quality_tag("PARTIAL_CORRECT_PREFIX")}:
        return ExampleRetentionTag.POSITIVE
    if quality == _default_quality_tag("CORRECT_ANSWER_WRONG_REASONING"):
        return ExampleRetentionTag.PLAUSIBLE_WRONG
    if quality == _default_quality_tag("WRONG_ANSWER_PLAUSIBLE_REASONING"):
        return ExampleRetentionTag.PLAUSIBLE_WRONG
    if quality == _default_quality_tag("WRONG_ANSWER_UNSOUND_REASONING"):
        return ExampleRetentionTag.UNSOUND_WRONG
    if verdict == _default_verdict("REPAIRABLE"):
        return ExampleRetentionTag.REPAIRED
    if verdict == _default_verdict("REJECT"):
        return ExampleRetentionTag.UNSOUND_WRONG
    return ExampleRetentionTag.UNKNOWN


def _should_keep_tag(tag: ExampleRetentionTag, config: VerifierDistillationConfig) -> bool:
    if tag == ExampleRetentionTag.POSITIVE:
        return True
    if tag == ExampleRetentionTag.PLAUSIBLE_WRONG:
        return config.retain_plausible_wrong
    if tag == ExampleRetentionTag.UNSOUND_WRONG:
        return config.retain_unsound_wrong
    if tag == ExampleRetentionTag.SYMBOLIC_CONTRADICTED:
        return config.retain_symbolic_contradicted
    if tag == ExampleRetentionTag.REPAIRED:
        return config.retain_repaired
    if tag == ExampleRetentionTag.VERIFIER_REJECTED:
        return config.retain_verifier_rejected
    return True


def _bundle_context_payload(
    *,
    branch_id: str,
    prefix_step_count: int,
    total_step_count: int,
    operator_sequence: Sequence[str],
    route_problem_type: Sequence[str],
    route_archetypes: Sequence[str],
) -> dict[str, Any]:
    return {
        "branch_id": branch_id,
        "prefix_step_count": int(prefix_step_count),
        "total_step_count": int(total_step_count),
        "operator_sequence": list(operator_sequence),
        "route_problem_type": list(route_problem_type),
        "route_archetypes": list(route_archetypes),
        "scope": LabelScope.BRANCH.value if hasattr(LabelScope, "BRANCH") else "branch",
    }


def _build_targets_from_signals(
    signals: Mapping[str, Any],
    *,
    quality_tag: ReasoningQualityTag,
    verdict: VerifierVerdict,
) -> dict[str, Any]:
    logical = _extract_float(signals, "logical_consistency") or 0.0
    symbolic = _extract_float(signals, "symbolic_agreement", "symbolic_score") or 0.0
    completeness = _extract_float(signals, "completeness") or 0.0
    repairability = _extract_float(signals, "repairability") or 0.0
    answer_like = _extract_float(signals, "answer_correctness_likelihood", "verifier_score") or 0.0
    step_quality = _extract_float(signals, "step_quality", "prm_step_quality") or 0.0
    prefix_quality = _extract_float(signals, "prefix_quality", "prm_prefix_quality") or 0.0
    retrieval_compatibility = _extract_float(signals, "retrieval_compatibility") or 0.0
    retrieval_support = _extract_float(signals, "retrieval_support") or 0.0
    operator_reliability = _extract_float(signals, "operator_reliability") or 0.0
    open_obligation_burden = _extract_float(signals, "open_obligation_burden") or 0.0
    discharge_fraction = _extract_float(signals, "discharge_fraction") or 0.0
    difficulty_intensity = _extract_float(signals, "difficulty_intensity") or 0.0
    proof_burden = _extract_float(signals, "proof_burden") or 0.0
    retrieval_need = _extract_float(signals, "retrieval_need") or 0.0
    repair_need = _extract_float(signals, "repair_need") or 0.0
    process_quality = _clamp01(
        0.28 * step_quality
        + 0.28 * prefix_quality
        + 0.18 * logical
        + 0.12 * symbolic
        + 0.14 * answer_like
    )

    return {
        "verdict": verdict.value,
        "quality_tag": quality_tag.value,
        "logical_consistency": logical,
        "symbolic_agreement": symbolic,
        "completeness": completeness,
        "repairability": repairability,
        "answer_correctness_likelihood": answer_like,
        "process_quality": process_quality,
        "step_quality": step_quality,
        "prefix_quality": prefix_quality,
        "retrieval_compatibility": retrieval_compatibility,
        "retrieval_support": retrieval_support,
        "operator_reliability": operator_reliability,
        "open_obligation_burden": open_obligation_burden,
        "discharge_fraction": discharge_fraction,
        "difficulty_intensity": difficulty_intensity,
        "proof_burden": proof_burden,
        "retrieval_need": retrieval_need,
        "repair_need": repair_need,
    }


def _build_legacy_label_payload(quality_tag: ReasoningQualityTag, verdict: VerifierVerdict, signals: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "quality_tag": quality_tag.value,
        "verdict": verdict.value,
        "verifier_score": _extract_float(signals, "verifier_score", "answer_correctness_likelihood") or 0.0,
    }


def _build_branch_label_payload(quality_tag: ReasoningQualityTag, verdict: VerifierVerdict, signals: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "correctness": {
            "quality_tag": quality_tag.value,
            "answer_correctness_likelihood": _extract_float(signals, "answer_correctness_likelihood", "verifier_score") or 0.0,
        },
        "verdict": {
            "verdict": verdict.value,
            "repairability": _extract_float(signals, "repairability") or 0.0,
        },
        "components": {
            "logical_consistency": _extract_float(signals, "logical_consistency") or 0.0,
            "symbolic_agreement": _extract_float(signals, "symbolic_agreement", "symbolic_score") or 0.0,
            "completeness": _extract_float(signals, "completeness") or 0.0,
            "step_quality": _extract_float(signals, "step_quality", "prm_step_quality") or 0.0,
            "prefix_quality": _extract_float(signals, "prefix_quality", "prm_prefix_quality") or 0.0,
        },
    }


def _build_step_label_payload(
    step_index: int,
    label_id: str,
    operator_name: str,
    quality_tag: ReasoningQualityTag,
    verdict: VerifierVerdict,
    signals: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "label_id": label_id,
        "step_index": int(step_index),
        "step_id": f"step::{step_index}",
        "operator_name": operator_name,
        "correctness": {
            "quality_tag": quality_tag.value,
            "answer_correctness_likelihood": _extract_float(signals, "answer_correctness_likelihood", "verifier_score") or 0.0,
        },
        "verdict": {
            "verdict": verdict.value,
            "repairability": _extract_float(signals, "repairability") or 0.0,
        },
        "components": {
            "logical_consistency": _extract_float(signals, "logical_consistency") or 0.0,
            "symbolic_agreement": _extract_float(signals, "symbolic_agreement", "symbolic_score") or 0.0,
            "step_quality": _extract_float(signals, "step_quality", "prm_step_quality") or 0.0,
            "prefix_quality": _extract_float(signals, "prefix_quality", "prm_prefix_quality") or 0.0,
        },
    }


def _proof_obligation_summary_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    obligations = metadata.get("proof_obligations")
    if not isinstance(obligations, Sequence) or isinstance(obligations, (str, bytes)):
        obligations = []
    total = 0
    discharged = 0
    statuses: dict[str, int] = {}
    for item in obligations:
        if not isinstance(item, Mapping):
            continue
        total += 1
        status = _normalize_text(str(item.get("status") or "unknown")) or "unknown"
        statuses[status] = statuses.get(status, 0) + 1
        if status in {"discharged", "resolved", "closed"}:
            discharged += 1
    open_burden = _extract_float(metadata, "open_obligation_burden") or _clamp01(1.0 - (discharged / total if total else 1.0))
    discharge_fraction = _extract_float(metadata, "discharge_fraction") or (discharged / total if total else 0.0)
    return {
        "count": total,
        "discharged_count": discharged,
        "status_counts": statuses,
        "open_obligation_burden": _clamp01(open_burden),
        "discharge_fraction": _clamp01(discharge_fraction),
    }


def _retrieval_summary_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "retrieval_used": bool(metadata.get("retrieval_used")),
        "retrieval_support": _extract_float(metadata, "retrieval_support", "retrieval_score") or 0.0,
        "retrieval_compatibility": _extract_float(metadata, "retrieval_compatibility") or 0.0,
    }


def _operator_summary_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "operator_reliability": _extract_float(metadata, "operator_reliability") or 0.0,
        "operator_provenance": _safe_mapping(metadata.get("operator_provenance")),
        "operator_sequence": list(metadata.get("operator_sequence") or []),
    }


def _route_compute_signals_from_metadata(metadata: Mapping[str, Any]) -> dict[str, float]:
    raw = metadata.get("route_compute_signals") or metadata.get("compute_signals") or {}
    if not isinstance(raw, Mapping):
        raw = {}
    out: dict[str, float] = {}
    for key in (
        "difficulty_intensity",
        "route_uncertainty",
        "proof_burden",
        "retrieval_need",
        "repair_need",
        "critique_aggressiveness",
        "repair_aggressiveness",
        "resample_aggressiveness",
    ):
        out[key] = _extract_float(raw, key) or 0.0
    return out


def _maybe_bundle_to_dict(bundle: Any) -> dict[str, Any]:
    if isinstance(bundle, Mapping):
        return _as_json_value(bundle)
    if isinstance(bundle, BaseModel):
        return bundle.model_dump(mode="json")
    if hasattr(bundle, "model_dump"):
        return bundle.model_dump(mode="json")
    return {}


def _build_label_decomposition(
    *,
    bundle_payload: dict[str, Any],
    branch_label_payload: dict[str, Any],
    step_label_payload: dict[str, Any] | None,
    legacy_label_payload: dict[str, Any],
    model_targets: dict[str, Any],
    metadata: Mapping[str, Any],
) -> VerifierLabelDecomposition:
    return VerifierLabelDecomposition(
        bundle=bundle_payload,
        branch_label=branch_label_payload,
        step_label=step_label_payload,
        legacy_label=legacy_label_payload,
        model_targets=model_targets,
        decomposed_signals={
            "logical_consistency": _extract_float(model_targets, "logical_consistency") or 0.0,
            "symbolic_agreement": _extract_float(model_targets, "symbolic_agreement") or 0.0,
            "completeness": _extract_float(model_targets, "completeness") or 0.0,
            "repairability": _extract_float(model_targets, "repairability") or 0.0,
            "answer_correctness_likelihood": _extract_float(model_targets, "answer_correctness_likelihood") or 0.0,
            "process_quality": _extract_float(model_targets, "process_quality") or 0.0,
            "step_quality": _extract_float(model_targets, "step_quality") or 0.0,
            "prefix_quality": _extract_float(model_targets, "prefix_quality") or 0.0,
        },
        proof_obligation_summary=_proof_obligation_summary_from_metadata(metadata),
        retrieval_summary=_retrieval_summary_from_metadata(metadata),
        operator_summary=_operator_summary_from_metadata(metadata),
        route_compute_signals=_route_compute_signals_from_metadata(metadata),
    )


# --------------------------
# source normalization
# --------------------------


def _signals_from_branch_trace(source: BranchTrace) -> dict[str, Any]:
    verifier_decomposition = _safe_mapping(getattr(source, "verifier_decomposition", {}))
    metadata = _safe_mapping(getattr(source, "metadata", {}))
    route_compute_signals = _safe_mapping(metadata.get("route_compute_signals") or metadata.get("compute_signals"))
    proof_obligations = _as_json_value(getattr(source, "proof_obligations", []))
    return {
        "verifier_score": _clamp01(getattr(source, "verifier_score", 0.0)),
        "logical_consistency": _clamp01(verifier_decomposition.get("logical_consistency", getattr(source, "logical_consistency", 0.0))),
        "symbolic_agreement": _clamp01(verifier_decomposition.get("symbolic_agreement", getattr(source, "symbolic_agreement", 0.0))),
        "completeness": _clamp01(verifier_decomposition.get("completeness", getattr(source, "completeness", 0.0))),
        "repairability": _clamp01(verifier_decomposition.get("repairability", getattr(source, "repairability", 0.0))),
        "answer_correctness_likelihood": _clamp01(
            verifier_decomposition.get(
                "answer_correctness_likelihood",
                getattr(source, "answer_correctness_likelihood", getattr(source, "verifier_score", 0.0)),
            )
        ),
        "step_quality": _clamp01(getattr(source, "step_quality", 0.0)),
        "prefix_quality": _clamp01(getattr(source, "prefix_quality", 0.0)),
        "prm_step_quality": _clamp01(getattr(source, "prm_step_quality", 0.0)),
        "prm_prefix_quality": _clamp01(getattr(source, "prm_prefix_quality", 0.0)),
        "retrieval_compatibility": _clamp01(getattr(source, "retrieval_compatibility", 0.0)),
        "retrieval_support": _clamp01(getattr(source, "retrieval_support", 0.0)),
        "operator_reliability": _clamp01(getattr(source, "operator_reliability", 0.0)),
        "open_obligation_burden": _clamp01(getattr(source, "open_obligation_burden", 0.0)),
        "discharge_fraction": _clamp01(getattr(source, "discharge_fraction", 0.0)),
        "route_compute_signals": route_compute_signals,
        "symbolic_valid": bool(getattr(source, "symbolic_valid", False)),
        "retrieval_used": bool(getattr(source, "retrieval_used", False)),
        "repaired": bool(getattr(source, "repaired", False)),
        "repair_count": int(getattr(source, "repair_count", 0) or 0),
        "failure_type": str(getattr(source, "failure_type", "") or ""),
        "proof_obligations": proof_obligations,
        "operator_provenance": _safe_mapping(metadata.get("operator_provenance")),
    }


def _normalize_from_branch_trace(source: BranchTrace) -> _NormalizedBundle:
    bundle_payload = _maybe_bundle_to_dict(VerifierLabelBundle.from_branch_trace(source))
    steps = tuple(
        {
            "description": step.description,
            "operator_used": step.operator_used,
            "symbolic_valid": bool(step.symbolic_valid),
            "symbolic_expression": step.symbolic_expression,
            "python_code": step.python_code,
            "python_result": step.python_result,
        }
        for step in source.steps
    )
    signals = _signals_from_branch_trace(source)
    metadata = {
        "source_split": _normalize_text(str(getattr(source, "source_split", "") or "")),
        "trace_class": _normalize_text(str(getattr(source, "trace_class", "") or "")),
        "symbolic_valid": bool(getattr(source, "symbolic_valid", False)),
        "retrieval_used": bool(getattr(source, "retrieval_used", False)),
        "repaired": bool(getattr(source, "repaired", False)),
        "repair_count": int(getattr(source, "repair_count", 0) or 0),
        "failure_type": str(getattr(source, "failure_type", "") or ""),
        "proof_obligations": _as_json_value(getattr(source, "proof_obligations", [])),
        "operator_sequence": list(getattr(source, "operator_sequence", [])),
        "operator_provenance": signals.get("operator_provenance", {}),
        "route_compute_signals": signals.get("route_compute_signals", {}),
        **{k: v for k, v in signals.items() if k != "route_compute_signals"},
        **_safe_mapping(getattr(source, "metadata", {})),
    }
    return _NormalizedBundle(
        source_type=DistillationSourceType.BRANCH_TRACE,
        source_record_id=source.branch_id,
        problem_id=source.problem_id,
        branch_id=source.branch_id,
        bundle=bundle_payload,
        source_split=_coerce_split(getattr(source, "source_split", None)),
        problem_text=_extract_text(metadata, "problem_text", "raw_problem_text"),
        reasoning_steps=steps,
        operator_sequence=tuple(getattr(source, "operator_sequence", []) or _collect_operator_sequence_from_steps(steps)),
        answer_text=_normalize_text(getattr(source, "answer", "") or ""),
        canonical_answer=_normalize_text(getattr(source, "answer_canonical", "") or getattr(source, "answer", "") or ""),
        metadata=metadata,
    )


def _normalize_from_branch_state(source: BranchState) -> _NormalizedBundle:
    steps = tuple(
        {
            "description": _normalize_text(getattr(step, "text", None) or getattr(step, "description", None) or ""),
            "operator_used": _normalize_text(getattr(step, "operator_name", None) or getattr(step, "operator_used", None) or ""),
            "symbolic_valid": bool(getattr(step, "symbolic_valid", False)),
            "symbolic_expression": getattr(step, "symbolic_expression", None),
            "python_code": getattr(step, "python_code", None),
            "python_result": getattr(step, "python_result", None),
        }
        for step in tuple(getattr(source, "steps", ()) or ())
    )
    metadata = _safe_mapping(getattr(source, "metadata", {}))
    route = getattr(source, "route", None)
    route_compute_signals = _safe_mapping(getattr(route, "compute_signals", {}))
    signals = {
        "verifier_score": _extract_float(metadata, "verifier_score", "answer_correctness_likelihood") or 0.0,
        "logical_consistency": _extract_float(metadata, "logical_consistency") or 0.0,
        "symbolic_agreement": _extract_float(metadata, "symbolic_agreement", "symbolic_score") or 0.0,
        "completeness": _extract_float(metadata, "completeness") or 0.0,
        "repairability": _extract_float(metadata, "repairability") or 0.0,
        "answer_correctness_likelihood": _extract_float(metadata, "answer_correctness_likelihood", "verifier_score") or 0.0,
        "step_quality": _extract_float(metadata, "step_quality", "prm_step_quality") or 0.0,
        "prefix_quality": _extract_float(metadata, "prefix_quality", "prm_prefix_quality") or 0.0,
        "retrieval_compatibility": _extract_float(metadata, "retrieval_compatibility") or 0.0,
        "retrieval_support": _extract_float(metadata, "retrieval_support", "retrieval_score") or 0.0,
        "operator_reliability": _extract_float(metadata, "operator_reliability") or 0.0,
        "open_obligation_burden": _extract_float(metadata, "open_obligation_burden") or 0.0,
        "discharge_fraction": _extract_float(metadata, "discharge_fraction") or 0.0,
        "route_compute_signals": route_compute_signals,
        "symbolic_valid": bool(metadata.get("symbolic_valid")),
        "retrieval_used": bool(metadata.get("retrieval_used")),
        "repaired": bool(metadata.get("repaired")),
        "repair_count": int(metadata.get("repair_count") or 0),
        "failure_type": str(metadata.get("failure_type") or ""),
        "proof_obligations": _as_json_value(metadata.get("proof_obligations") or []),
        "operator_provenance": _safe_mapping(metadata.get("operator_provenance")),
    }
    quality_tag = _quality_tag_from_signals(signals, signals)
    verdict = _verdict_from_signals(signals, signals)
    bundle_payload = {
        "bundle_id": _stable_hash("verifier_bundle", {"branch_id": getattr(source, "branch_id", ""), "metadata": metadata}),
        "context": _bundle_context_payload(
            branch_id=str(getattr(source, "branch_id", "")),
            prefix_step_count=len(steps),
            total_step_count=len(steps),
            operator_sequence=_collect_operator_sequence_from_steps(steps),
            route_problem_type=tuple(_safe_mapping(getattr(route, "problem_type_probs", {})).keys()),
            route_archetypes=tuple(_safe_mapping(getattr(route, "archetype_probs", {})).keys()),
        ),
        "branch_label": _build_branch_label_payload(quality_tag, verdict, signals),
        "step_labels": [],
        "calibration_groups": [],
    }
    merged_metadata = {
        **metadata,
        **signals,
    }
    return _NormalizedBundle(
        source_type=DistillationSourceType.BRANCH_STATE,
        source_record_id=str(getattr(source, "branch_id", "")),
        problem_id=str(getattr(getattr(source, "problem", None), "problem_id", "") or metadata.get("problem_id") or ""),
        branch_id=str(getattr(source, "branch_id", "")),
        bundle=bundle_payload,
        source_split=_coerce_split(metadata.get("source_split")),
        problem_text=_extract_text(metadata, "problem_text", "raw_problem_text"),
        reasoning_steps=steps,
        operator_sequence=_collect_operator_sequence_from_steps(steps),
        answer_text=_extract_text(metadata, "answer_text", "answer"),
        canonical_answer=_extract_text(metadata, "canonical_answer", "answer_canonical", "answer_text", "answer"),
        metadata=merged_metadata,
    )


def _normalize_from_distilled_trace_record(source: DistilledTraceRecord) -> _NormalizedBundle:
    outcome = _safe_mapping(getattr(source, "outcome", {}))
    provenance = _safe_mapping(getattr(source, "provenance", {}))
    metadata = _safe_mapping(getattr(source, "metadata", {}))
    steps = tuple(
        {
            "description": _normalize_text(getattr(step, "text", None) or getattr(step, "description", None) or ""),
            "operator_used": _normalize_text(getattr(step, "operator_name", None) or getattr(step, "operator_used", None) or ""),
            "symbolic_valid": bool(getattr(step, "symbolic_valid", False)),
            "symbolic_expression": getattr(step, "symbolic_expression", None),
            "python_code": getattr(step, "python_code", None),
            "python_result": getattr(step, "python_result", None),
        }
        for step in tuple(getattr(source, "steps", ()) or ())
    )
    signals = {
        "verifier_score": _extract_float(outcome, "verifier_score") or 0.0,
        "logical_consistency": _extract_float(outcome, "logical_consistency") or 0.0,
        "symbolic_agreement": _extract_float(outcome, "symbolic_score", "symbolic_agreement") or 0.0,
        "completeness": _extract_float(outcome, "completeness") or 0.0,
        "repairability": _extract_float(outcome, "repairability") or 0.0,
        "answer_correctness_likelihood": _extract_float(outcome, "answer_correctness_likelihood", "verifier_score") or 0.0,
        "step_quality": _extract_float(metadata, "step_quality", "prm_step_quality") or 0.0,
        "prefix_quality": _extract_float(metadata, "prefix_quality", "prm_prefix_quality") or 0.0,
        "retrieval_compatibility": _extract_float(metadata, "retrieval_compatibility") or 0.0,
        "retrieval_support": _extract_float(metadata, "retrieval_support") or 0.0,
        "operator_reliability": _extract_float(metadata, "operator_reliability") or 0.0,
        "open_obligation_burden": _extract_float(metadata, "open_obligation_burden") or 0.0,
        "discharge_fraction": _extract_float(metadata, "discharge_fraction") or 0.0,
        "route_compute_signals": _safe_mapping(provenance.get("route_snapshot", {})).get("compute_signals", {}),
        "symbolic_valid": bool(outcome.get("symbolic_valid")),
        "retrieval_used": bool(outcome.get("retrieval_used")),
        "repaired": bool(outcome.get("repaired")),
        "repair_count": int(outcome.get("repair_count") or 0),
        "failure_type": str(outcome.get("failure_type") or ""),
        "proof_obligations": metadata.get("proof_obligations") or [],
        "operator_provenance": _safe_mapping(metadata.get("operator_provenance")),
    }
    quality_tag = _quality_tag_from_signals(signals, {**metadata, **outcome, "trace_class": getattr(source, "classification", "")})
    verdict = _verdict_from_signals(signals, {**metadata, **outcome})
    bundle_payload = {
        "bundle_id": _stable_hash("verifier_bundle", {"record_id": getattr(source, "record_id", ""), "branch_id": getattr(source, "branch_id", "")}),
        "context": _bundle_context_payload(
            branch_id=str(getattr(source, "branch_id", "")),
            prefix_step_count=len(steps),
            total_step_count=len(steps),
            operator_sequence=tuple(getattr(source, "operator_sequence", ()) or _collect_operator_sequence_from_steps(steps)),
            route_problem_type=tuple(_safe_mapping(provenance.get("route_snapshot", {})).get("problem_type_probs", {}).keys()),
            route_archetypes=tuple(_safe_mapping(provenance.get("route_snapshot", {})).get("archetype_probs", {}).keys()),
        ),
        "branch_label": _build_branch_label_payload(quality_tag, verdict, signals),
        "step_labels": [],
        "calibration_groups": [],
    }
    merged_metadata = {
        **metadata,
        **outcome,
        **signals,
        "trace_class": _normalize_text(str(getattr(source, "classification", "") or "")),
        "quality_tier": _normalize_text(str(getattr(source, "quality_tier", "") or "")),
        "route_snapshot": provenance.get("route_snapshot", {}),
    }
    return _NormalizedBundle(
        source_type=DistillationSourceType.TRACE_DISTILLATION_RECORD,
        source_record_id=str(getattr(source, "record_id", "")),
        problem_id=str(getattr(source, "problem_id", "")),
        branch_id=str(getattr(source, "branch_id", "")),
        bundle=bundle_payload,
        source_split=_coerce_split(getattr(source, "split", None)),
        problem_text=_extract_text(merged_metadata, "problem_text", "raw_problem_text"),
        reasoning_steps=steps,
        operator_sequence=tuple(getattr(source, "operator_sequence", ()) or _collect_operator_sequence_from_steps(steps)),
        answer_text=_extract_text(outcome, "answer_raw"),
        canonical_answer=_extract_text(outcome, "answer_canonical", "answer_raw"),
        metadata=merged_metadata,
    )


def _normalize_from_distilled_trace_artifact(source: DistilledTraceArtifact) -> list[_NormalizedBundle]:
    records = tuple(getattr(source, "records", ()) or ())
    return [_normalize_from_distilled_trace_record(record) for record in records]


def _normalize_from_verifier_bundle(source: VerifierLabelBundle) -> _NormalizedBundle:
    bundle_payload = _maybe_bundle_to_dict(source)
    context = _safe_mapping(getattr(source, "context", {}))
    branch_label = _safe_mapping(getattr(source, "branch_label", {}))
    component_map = _safe_mapping(branch_label.get("components") or {})
    correctness = _safe_mapping(branch_label.get("correctness") or {})
    verdict_payload = _safe_mapping(branch_label.get("verdict") or {})
    signals = {
        "logical_consistency": _extract_float(component_map, "logical_consistency") or 0.0,
        "symbolic_agreement": _extract_float(component_map, "symbolic_agreement") or 0.0,
        "completeness": _extract_float(component_map, "completeness") or 0.0,
        "repairability": _extract_float(verdict_payload, "repairability") or 0.0,
        "answer_correctness_likelihood": _extract_float(correctness, "answer_correctness_likelihood") or 0.0,
        "step_quality": _extract_float(component_map, "step_quality") or 0.0,
        "prefix_quality": _extract_float(component_map, "prefix_quality") or 0.0,
    }
    return _NormalizedBundle(
        source_type=DistillationSourceType.VERIFIER_BUNDLE,
        source_record_id=_normalize_text(str(context.get("branch_id") or bundle_payload.get("bundle_id") or "")),
        problem_id=_normalize_text(str(context.get("problem_id") or "")),
        branch_id=_normalize_text(str(context.get("branch_id") or "")),
        bundle=bundle_payload,
        source_split=_coerce_split(context.get("source_split")),
        problem_text=_normalize_text(str(context.get("problem_text") or "")),
        reasoning_steps=tuple(_as_json_value(context.get("reasoning_steps") or [])),
        operator_sequence=tuple(context.get("operator_sequence") or ()),
        answer_text=_normalize_text(str(context.get("answer_text") or "")),
        canonical_answer=_normalize_text(str(context.get("canonical_answer") or "")),
        metadata={**signals, **context},
    )


def _normalize_from_legacy_label(source: LegacyVerifierLabel) -> _NormalizedBundle:
    payload = _maybe_bundle_to_dict(source)
    score = _extract_float(payload, "score", "verifier_score") or 0.0
    verdict = _default_verdict("ACCEPT") if score >= 0.70 else _default_verdict("REJECT")
    quality = _default_quality_tag("FULLY_CORRECT") if score >= 0.70 else _default_quality_tag("WRONG_ANSWER_UNSOUND_REASONING")
    bundle_payload = {
        "bundle_id": _stable_hash("verifier_bundle", payload),
        "context": _bundle_context_payload(
            branch_id=_extract_text(payload, "branch_id", "id") or "legacy",
            prefix_step_count=0,
            total_step_count=0,
            operator_sequence=(),
            route_problem_type=(),
            route_archetypes=(),
        ),
        "branch_label": _build_branch_label_payload(quality, verdict, {"verifier_score": score}),
        "step_labels": [],
        "calibration_groups": [],
    }
    return _NormalizedBundle(
        source_type=DistillationSourceType.LEGACY_LABEL,
        source_record_id=_stable_hash("legacy_verifier_label", payload),
        problem_id=_extract_text(payload, "problem_id"),
        branch_id=_extract_text(payload, "branch_id", "id") or "legacy",
        bundle=bundle_payload,
        source_split=None,
        problem_text="",
        reasoning_steps=tuple(),
        operator_sequence=tuple(),
        answer_text=_extract_text(payload, "answer"),
        canonical_answer=_extract_text(payload, "answer_canonical", "answer"),
        metadata={"verifier_score": score},
    )


def normalize_verifier_source(
    source: BranchTrace | BranchState | DistilledTraceRecord | DistilledTraceArtifact | VerifierLabelBundle | LegacyVerifierLabel,
) -> list[_NormalizedBundle]:
    if isinstance(source, BranchTrace):
        return [_normalize_from_branch_trace(source)]
    if type(source).__name__ == "BranchState":
        return [_normalize_from_branch_state(source)]
    if type(source).__name__ == "DistilledTraceRecord":
        return [_normalize_from_distilled_trace_record(source)]
    if type(source).__name__ == "DistilledTraceArtifact":
        return _normalize_from_distilled_trace_artifact(source)
    if isinstance(source, VerifierLabelBundle):
        return [_normalize_from_verifier_bundle(source)]
    if isinstance(source, LegacyVerifierLabel):
        return [_normalize_from_legacy_label(source)]
    raise TypeError(f"Unsupported verifier distillation source: {type(source)!r}")


# --------------------------
# example construction
# --------------------------


def _extract_bundle_quality(bundle_payload: Mapping[str, Any], metadata: Mapping[str, Any]) -> ReasoningQualityTag:
    branch_label = _safe_mapping(bundle_payload.get("branch_label"))
    correctness = _safe_mapping(branch_label.get("correctness"))
    maybe = correctness.get("quality_tag")
    if maybe is not None:
        coerced = _coerce_quality_tag(maybe)
        if coerced is not ReasoningQualityTag.UNKNOWN:
            return coerced
    return _quality_tag_from_signals(metadata, metadata)


def _extract_bundle_verdict(bundle_payload: Mapping[str, Any], metadata: Mapping[str, Any]) -> VerifierVerdict:
    branch_label = _safe_mapping(bundle_payload.get("branch_label"))
    verdict_payload = _safe_mapping(branch_label.get("verdict"))
    maybe = verdict_payload.get("verdict")
    if maybe is not None:
        return _coerce_verdict(maybe)
    return _verdict_from_signals(metadata, metadata)


def _prefix_selections(normalized: _NormalizedBundle, config: VerifierDistillationConfig) -> tuple[_PrefixSelection, ...]:
    total = len(normalized.reasoning_steps)
    if total < config.min_prefix_steps:
        return ()
    positions: list[int] = list(range(config.min_prefix_steps, total + 1, config.prefix_stride))
    if config.include_terminal_prefix and total not in positions:
        positions.append(total)
    positions = sorted(set(pos for pos in positions if 1 <= pos <= total))
    if config.max_prefix_examples_per_branch is not None:
        positions = positions[: config.max_prefix_examples_per_branch]
    selections: list[_PrefixSelection] = []
    for pos in positions:
        operator_name = ""
        if pos - 1 < len(normalized.reasoning_steps):
            operator_name = _normalize_text(str(normalized.reasoning_steps[pos - 1].get("operator_used") or ""))
        selections.append(
            _PrefixSelection(
                step_index=pos,
                label_id=_stable_hash(
                    "prefix_label",
                    {"source_record_id": normalized.source_record_id, "branch_id": normalized.branch_id, "step_index": pos},
                ),
                operator_name=operator_name,
            )
        )
    return tuple(selections)


def _example_texts(normalized: _NormalizedBundle, prefix_count: int | None = None) -> ExampleTextFields:
    reasoning_text = _reasoning_text_from_steps(normalized.reasoning_steps)
    prefix_text = _reasoning_text_from_steps(normalized.reasoning_steps, prefix_count=prefix_count)
    bundle_context = _safe_mapping(normalized.bundle.get("context"))
    route_problem_type = tuple(str(x) for x in bundle_context.get("route_problem_type") or ())
    route_archetypes = tuple(str(x) for x in bundle_context.get("route_archetypes") or ())
    return ExampleTextFields(
        problem_text=_normalize_text(normalized.problem_text),
        prefix_text=prefix_text,
        reasoning_text=reasoning_text,
        answer_text=_normalize_text(normalized.answer_text),
        canonical_answer=_normalize_text(normalized.canonical_answer),
        operator_sequence=tuple(normalized.operator_sequence),
        route_problem_type=route_problem_type,
        route_archetypes=route_archetypes,
    )


def _provenance(normalized: _NormalizedBundle, *, label_id: str, bundle_id: str) -> SourceProvenance:
    digest_payload = {
        "source_type": normalized.source_type.value,
        "source_record_id": normalized.source_record_id,
        "branch_id": normalized.branch_id,
        "bundle_id": bundle_id,
        "label_id": label_id,
        "metadata": normalized.metadata,
    }
    return SourceProvenance(
        source_type=normalized.source_type,
        source_record_id=normalized.source_record_id,
        problem_id=normalized.problem_id,
        branch_id=normalized.branch_id,
        bundle_id=bundle_id,
        label_id=label_id,
        source_digest=_stable_hash("verifier_source", digest_payload),
        metadata=dict(normalized.metadata),
    )


def _branch_example(normalized: _NormalizedBundle, config: VerifierDistillationConfig) -> VerifierTrainingExample:
    quality_tag = _extract_bundle_quality(normalized.bundle, normalized.metadata)
    verdict = _extract_bundle_verdict(normalized.bundle, normalized.metadata)
    tag = _retention_tag_for_quality(quality_tag, verdict, metadata=normalized.metadata)
    texts = _example_texts(normalized)
    bundle_id = _normalize_text(str(normalized.bundle.get("bundle_id") or _stable_hash("bundle", normalized.bundle)))
    split = _resolve_split(
        normalized.problem_id or normalized.branch_id,
        source_split=normalized.source_split,
        config=config,
    )
    provenance = _provenance(normalized, label_id=f"{bundle_id}::branch", bundle_id=bundle_id)
    model_targets = _build_targets_from_signals(normalized.metadata, quality_tag=quality_tag, verdict=verdict)
    label_decomposition = _build_label_decomposition(
        bundle_payload=normalized.bundle,
        branch_label_payload=_safe_mapping(normalized.bundle.get("branch_label")),
        step_label_payload=None,
        legacy_label_payload=_build_legacy_label_payload(quality_tag, verdict, normalized.metadata),
        model_targets=model_targets,
        metadata=normalized.metadata,
    )
    payload = {
        "source_record_id": normalized.source_record_id,
        "bundle_id": bundle_id,
        "scope": ExampleScope.BRANCH.value,
        "split": split.value,
    }
    return VerifierTrainingExample(
        example_id=_stable_hash("verifier_example", payload),
        split=split,
        scope=ExampleScope.BRANCH,
        retention_tag=tag,
        problem_id=normalized.problem_id,
        branch_id=normalized.branch_id,
        prefix_step_count=len(normalized.reasoning_steps),
        total_step_count=len(normalized.reasoning_steps),
        quality_tag=quality_tag,
        verdict=verdict,
        texts=texts,
        labels=label_decomposition,
        provenance=provenance,
        metadata={
            "calibration_group_ids": [
                group.get("group_id")
                for group in (_safe_mapping(g) for g in normalized.bundle.get("calibration_groups") or [])
                if group.get("group_id")
            ],
            "step_label_count": len(_prefix_selections(normalized, config)),
            **dict(normalized.metadata),
        },
    )


def _prefix_quality_adjustment(base: dict[str, Any], prefix_ratio: float) -> dict[str, Any]:
    adjusted = dict(base)
    adjusted["completeness"] = _clamp01((adjusted.get("completeness") or 0.0) * prefix_ratio)
    adjusted["prefix_quality"] = _clamp01(adjusted.get("prefix_quality") or 0.0)
    adjusted["step_quality"] = _clamp01(adjusted.get("step_quality") or 0.0)
    adjusted["process_quality"] = _clamp01(
        0.40 * adjusted.get("prefix_quality", 0.0)
        + 0.30 * adjusted.get("step_quality", 0.0)
        + 0.20 * adjusted.get("logical_consistency", 0.0)
        + 0.10 * adjusted.get("symbolic_agreement", 0.0)
    )
    return adjusted


def _prefix_example(normalized: _NormalizedBundle, selection: _PrefixSelection, config: VerifierDistillationConfig) -> VerifierTrainingExample:
    prefix_ratio = selection.step_index / max(1, len(normalized.reasoning_steps))
    quality_tag = _extract_bundle_quality(normalized.bundle, normalized.metadata)
    verdict = _extract_bundle_verdict(normalized.bundle, normalized.metadata)
    tag = _retention_tag_for_quality(quality_tag, verdict, metadata=normalized.metadata)
    texts = _example_texts(normalized, prefix_count=selection.step_index)
    bundle_id = _normalize_text(str(normalized.bundle.get("bundle_id") or _stable_hash("bundle", normalized.bundle)))
    split = _resolve_split(
        normalized.problem_id or normalized.branch_id,
        source_split=normalized.source_split,
        config=config,
    )
    provenance = _provenance(normalized, label_id=selection.label_id, bundle_id=bundle_id)
    targets = _prefix_quality_adjustment(
        _build_targets_from_signals(normalized.metadata, quality_tag=quality_tag, verdict=verdict),
        prefix_ratio,
    )
    step_payload = _build_step_label_payload(
        step_index=selection.step_index,
        label_id=selection.label_id,
        operator_name=selection.operator_name,
        quality_tag=quality_tag,
        verdict=verdict,
        signals=targets,
    )
    label_decomposition = _build_label_decomposition(
        bundle_payload=normalized.bundle,
        branch_label_payload=_safe_mapping(normalized.bundle.get("branch_label")),
        step_label_payload=step_payload,
        legacy_label_payload=_build_legacy_label_payload(quality_tag, verdict, targets),
        model_targets=targets,
        metadata=normalized.metadata,
    )
    payload = {
        "source_record_id": normalized.source_record_id,
        "bundle_id": bundle_id,
        "label_id": selection.label_id,
        "scope": ExampleScope.PREFIX.value,
        "split": split.value,
    }
    return VerifierTrainingExample(
        example_id=_stable_hash("verifier_example", payload),
        split=split,
        scope=ExampleScope.PREFIX,
        retention_tag=tag,
        problem_id=normalized.problem_id,
        branch_id=normalized.branch_id,
        prefix_step_count=selection.step_index,
        total_step_count=len(normalized.reasoning_steps),
        quality_tag=quality_tag,
        verdict=verdict,
        texts=texts,
        labels=label_decomposition,
        provenance=provenance,
        metadata={
            "step_index": selection.step_index,
            "step_id": f"step::{selection.step_index}",
            "operator_name": selection.operator_name,
            **dict(normalized.metadata),
        },
    )


# --------------------------
# public API
# --------------------------


def distill_verifier_dataset(
    sources: Sequence[BranchTrace | BranchState | DistilledTraceRecord | DistilledTraceArtifact | VerifierLabelBundle | LegacyVerifierLabel],
    *,
    config: VerifierDistillationConfig | None = None,
) -> tuple[VerifierTrainingArtifact, VerifierDistillationReport]:
    """Build a deterministic verifier-training dataset artifact."""

    resolved_config = config or VerifierDistillationConfig()
    normalized: list[_NormalizedBundle] = []
    for source in sources:
        normalized.extend(normalize_verifier_source(source))

    kept: list[VerifierTrainingExample] = []
    dropped_reasons: dict[str, int] = {}
    for item in normalized:
        if resolved_config.include_branch_examples:
            branch_ex = _branch_example(item, resolved_config)
            if _should_keep_tag(branch_ex.retention_tag, resolved_config):
                kept.append(branch_ex)
            else:
                dropped_reasons[f"drop_{branch_ex.retention_tag.value}"] = dropped_reasons.get(
                    f"drop_{branch_ex.retention_tag.value}",
                    0,
                ) + 1

        if resolved_config.include_prefix_examples:
            for selection in _prefix_selections(item, resolved_config):
                prefix_ex = _prefix_example(item, selection, resolved_config)
                if _should_keep_tag(prefix_ex.retention_tag, resolved_config):
                    kept.append(prefix_ex)
                else:
                    dropped_reasons[f"drop_{prefix_ex.retention_tag.value}"] = dropped_reasons.get(
                        f"drop_{prefix_ex.retention_tag.value}",
                        0,
                    ) + 1

    kept.sort(key=lambda ex: (ex.split.value, ex.scope.value, ex.problem_id, ex.branch_id, ex.example_id))
    examples = tuple(kept)

    split_counts: dict[str, int] = {}
    scope_counts: dict[str, int] = {}
    retention_counts: dict[str, int] = {}
    branch_ids: set[str] = set()
    source_ids: set[str] = set()
    for example in examples:
        split_counts[example.split.value] = split_counts.get(example.split.value, 0) + 1
        scope_counts[example.scope.value] = scope_counts.get(example.scope.value, 0) + 1
        retention_counts[example.retention_tag.value] = retention_counts.get(example.retention_tag.value, 0) + 1
        branch_ids.add(example.branch_id)
        source_ids.add(example.provenance.source_record_id)

    config_digest = _stable_hash("verifier_config", resolved_config.model_dump(mode="json"))
    source_digest = _stable_hash(
        "verifier_sources",
        [
            {
                "example_id": ex.example_id,
                "source_digest": ex.provenance.source_digest,
            }
            for ex in examples
        ],
    )
    artifact_id = _stable_hash(
        "verifier_training_artifact",
        {
            "config_digest": config_digest,
            "source_digest": source_digest,
            "example_count": len(examples),
        },
    )
    metadata = VerifierTrainingArtifactMetadata(
        artifact_id=artifact_id,
        example_count=len(examples),
        source_count=len(source_ids),
        branch_count=len(branch_ids),
        config_digest=config_digest,
        split_counts=split_counts,
        scope_counts=scope_counts,
        retention_counts=retention_counts,
        source_digest=source_digest,
    )
    artifact = VerifierTrainingArtifact(metadata=metadata, examples=examples)
    report = VerifierDistillationReport(
        kept_count=len(examples),
        dropped_count=sum(dropped_reasons.values()),
        split_counts=split_counts,
        scope_counts=scope_counts,
        retention_counts=retention_counts,
        dropped_reasons=dropped_reasons,
    )
    return artifact, report


def artifact_examples_by_split(artifact: VerifierTrainingArtifact) -> dict[SplitName, tuple[VerifierTrainingExample, ...]]:
    grouped: dict[SplitName, list[VerifierTrainingExample]] = {
        SplitName.TRAIN: [],
        SplitName.VALID: [],
        SplitName.CALIBRATION: [],
    }
    for example in artifact.examples:
        grouped[example.split].append(example)
    return {split: tuple(items) for split, items in grouped.items()}


def write_verifier_training_artifact(
    artifact: VerifierTrainingArtifact,
    output_dir: str | Path,
    *,
    include_split_files: bool = True,
) -> VerifierArtifactWriteResult:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = out_dir / artifact.metadata.artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = artifact_dir / "manifest.json"
    examples_path = artifact_dir / "examples.jsonl"
    manifest_path.write_text(_stable_json(artifact.metadata.model_dump(mode="json")) + "\n", encoding="utf-8")

    with examples_path.open("w", encoding="utf-8") as handle:
        for example in artifact.examples:
            handle.write(_stable_json(example.model_dump(mode="json")) + "\n")

    split_paths: dict[str, str] = {}
    if include_split_files:
        for split, examples in artifact_examples_by_split(artifact).items():
            path = artifact_dir / f"{split.value}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for example in examples:
                    handle.write(_stable_json(example.model_dump(mode="json")) + "\n")
            split_paths[split.value] = str(path)

    return VerifierArtifactWriteResult(
        artifact_dir=str(artifact_dir),
        manifest_path=str(manifest_path),
        examples_path=str(examples_path),
        split_paths=split_paths,
    )


def load_verifier_training_artifact(path: str | Path) -> VerifierTrainingArtifact:
    input_path = Path(path)
    if input_path.is_dir():
        manifest_path = input_path / "manifest.json"
        examples_path = input_path / "examples.jsonl"
    elif input_path.name == "manifest.json":
        manifest_path = input_path
        examples_path = input_path.with_name("examples.jsonl")
    else:
        raise ValueError(f"Unsupported artifact path: {path}")

    metadata = VerifierTrainingArtifactMetadata.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    examples: list[VerifierTrainingExample] = []
    with examples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            examples.append(VerifierTrainingExample.model_validate_json(stripped))
    return VerifierTrainingArtifact(metadata=metadata, examples=tuple(examples))


def runtime_verifier_rows(artifact: VerifierTrainingArtifact) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for example in artifact.examples:
        rows.append(
            {
                "example_id": example.example_id,
                "split": example.split.value,
                "scope": example.scope.value,
                "problem_id": example.problem_id,
                "branch_id": example.branch_id,
                "prefix_step_count": example.prefix_step_count,
                "input_text": example.texts.prefix_text or example.texts.reasoning_text,
                "answer_text": example.texts.answer_text,
                "canonical_answer": example.texts.canonical_answer,
                "targets": dict(example.labels.model_targets),
                "retention_tag": example.retention_tag.value,
                "quality_tag": example.quality_tag.value,
                "verdict": example.verdict.value,
                "decomposed_signals": dict(example.labels.decomposed_signals),
                "proof_obligation_summary": dict(example.labels.proof_obligation_summary),
                "retrieval_summary": dict(example.labels.retrieval_summary),
                "operator_summary": dict(example.labels.operator_summary),
                "route_compute_signals": dict(example.labels.route_compute_signals),
                "provenance": example.provenance.model_dump(mode="json"),
                "metadata": dict(example.metadata),
            }
        )
    return tuple(rows)


def stronger_verifier_rows(artifact: VerifierTrainingArtifact) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for example in artifact.examples:
        rows.append(
            {
                "example_id": example.example_id,
                "split": example.split.value,
                "scope": example.scope.value,
                "problem_id": example.problem_id,
                "branch_id": example.branch_id,
                "problem_text": example.texts.problem_text,
                "reasoning_text": example.texts.reasoning_text,
                "prefix_text": example.texts.prefix_text,
                "answer_text": example.texts.answer_text,
                "canonical_answer": example.texts.canonical_answer,
                "operator_sequence": list(example.texts.operator_sequence),
                "route_problem_type": list(example.texts.route_problem_type),
                "route_archetypes": list(example.texts.route_archetypes),
                "label_bundle": dict(example.labels.bundle),
                "branch_label": dict(example.labels.branch_label),
                "step_label": dict(example.labels.step_label or {}),
                "legacy_label": dict(example.labels.legacy_label),
                "model_targets": dict(example.labels.model_targets),
                "decomposed_signals": dict(example.labels.decomposed_signals),
                "proof_obligation_summary": dict(example.labels.proof_obligation_summary),
                "retrieval_summary": dict(example.labels.retrieval_summary),
                "operator_summary": dict(example.labels.operator_summary),
                "route_compute_signals": dict(example.labels.route_compute_signals),
                "retention_tag": example.retention_tag.value,
                "quality_tag": example.quality_tag.value,
                "verdict": example.verdict.value,
                "provenance": example.provenance.model_dump(mode="json"),
                "metadata": dict(example.metadata),
            }
        )
    return tuple(rows)