from __future__ import annotations

"""Unified offline training dataset construction.

Merged implementation:
- keeps the older artifact/provenance/split discipline and typed source normalization
- ports the newer repair-pass signal surfacing so richer runtime evidence becomes
  active offline supervision instead of passive archived metadata
"""

from enum import Enum
from hashlib import sha1
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.common.constants import (
    OFFLINE_ARTIFACT_VERSION,
    OFFLINE_SPLIT_RATIOS,
    OFFLINE_SPLIT_SALT,
)
from src.common.schemas import BranchTrace

from src.offline.trace_distillation import DistilledTraceArtifact, DistilledTraceRecord
try:
    from src.offline.verifier_distillation import (
        VerifierTrainingArtifact,
        VerifierTrainingExample,
    )
except Exception:  # pragma: no cover
    VerifierTrainingArtifact = Any  # type: ignore
    VerifierTrainingExample = Any  # type: ignore


SCHEMA_VERSION = "training_data_builder.v3"
ARTIFACT_KIND = "offline_training_corpus"
DEFAULT_ARTIFACT_VERSION = OFFLINE_ARTIFACT_VERSION


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        str_strip_whitespace=True,
        populate_by_name=True,
    )


class UnifiedTaskFamily(str, Enum):
    ROUTER = "router"
    VERIFIER = "verifier"
    OPERATOR_MINING = "operator_mining"
    CALIBRATION = "calibration"


class UnifiedSourceType(str, Enum):
    TRACE_RECORD = "trace_record"
    TRACE_ARTIFACT = "trace_artifact"
    VERIFIER_EXAMPLE = "verifier_example"
    VERIFIER_ARTIFACT = "verifier_artifact"
    BRANCH_TRACE = "branch_trace"
    GENERIC_MAPPING = "generic_mapping"


class SplitName(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    CALIBRATION = "calibration"


class SplitConflictPolicy(str, Enum):
    HASH = "hash"
    PREFER_SOURCE = "prefer_source"
    ERROR = "error"


class TrainingDataBuilderConfig(StrictModel):
    artifact_version: str = DEFAULT_ARTIFACT_VERSION
    schema_version: str = SCHEMA_VERSION
    include_router_examples: bool = True
    include_verifier_examples: bool = True
    include_operator_examples: bool = True
    include_calibration_examples: bool = True
    prefer_source_split: bool = True
    split_ratios: tuple[float, float, float] = OFFLINE_SPLIT_RATIOS
    split_salt: str = OFFLINE_SPLIT_SALT
    drop_router_examples_without_problem_text: bool = False
    drop_examples_without_targets: bool = False
    split_conflict_policy: SplitConflictPolicy = SplitConflictPolicy.HASH

    def normalized_split_ratios(self) -> tuple[float, float, float]:
        total = sum(self.split_ratios)
        if total <= 0:
            return (0.80, 0.10, 0.10)
        return tuple(float(x) / total for x in self.split_ratios)  # type: ignore[return-value]


class TrainingRecordProvenance(StrictModel):
    source_type: UnifiedSourceType
    source_record_id: str
    upstream_artifact_id: str | None = None
    problem_id: str = ""
    branch_id: str = ""
    source_digest: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class UnifiedTrainingExample(StrictModel):
    record_id: str
    task_family: UnifiedTaskFamily
    split: SplitName
    problem_id: str = ""
    branch_id: str = ""
    input_fields: dict[str, Any] = Field(default_factory=dict)
    target_fields: dict[str, Any] = Field(default_factory=dict)
    feature_fields: dict[str, Any] = Field(default_factory=dict)
    provenance: TrainingRecordProvenance
    metadata: dict[str, Any] = Field(default_factory=dict)


class OfflineTrainingArtifactMetadata(StrictModel):
    artifact_kind: str = ARTIFACT_KIND
    artifact_version: str = DEFAULT_ARTIFACT_VERSION
    schema_version: str = SCHEMA_VERSION
    artifact_id: str
    record_count: int
    source_count: int
    problem_count: int
    task_counts: dict[str, int] = Field(default_factory=dict)
    split_counts: dict[str, int] = Field(default_factory=dict)
    problem_split_counts: dict[str, int] = Field(default_factory=dict)
    split_conflicts: dict[str, list[str]] = Field(default_factory=dict)
    upstream_artifact_ids: tuple[str, ...] = Field(default_factory=tuple)
    config_digest: str
    source_digest: str


class OfflineTrainingArtifact(StrictModel):
    metadata: OfflineTrainingArtifactMetadata
    records: tuple[UnifiedTrainingExample, ...] = Field(default_factory=tuple)


class TrainingDataBuildReport(StrictModel):
    kept_count: int
    dropped_count: int
    task_counts: dict[str, int] = Field(default_factory=dict)
    split_counts: dict[str, int] = Field(default_factory=dict)
    dropped_reasons: dict[str, int] = Field(default_factory=dict)


class TrainingArtifactWriteResult(StrictModel):
    artifact_dir: str
    manifest_path: str
    records_path: str
    task_paths: dict[str, str] = Field(default_factory=dict)
    split_paths: dict[str, str] = Field(default_factory=dict)


class _NormalizedTrace(StrictModel):
    source_type: UnifiedSourceType
    source_record_id: str
    upstream_artifact_id: str | None = None
    problem_id: str = ""
    branch_id: str
    split: SplitName
    classification: str
    quality_tier: str
    steps: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    operator_sequence: tuple[str, ...] = Field(default_factory=tuple)
    archetypes: tuple[str, ...] = Field(default_factory=tuple)
    answer_raw: str = ""
    answer_canonical: str = ""
    route_snapshot: dict[str, Any] = Field(default_factory=dict)
    outcome: dict[str, Any] = Field(default_factory=dict)
    evidence: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)
    decomposed_signals: dict[str, Any] = Field(default_factory=dict)
    verifier_decomposition: dict[str, Any] = Field(default_factory=dict)
    proof_obligation_summary: dict[str, Any] = Field(default_factory=dict)
    route_compute_signals: dict[str, Any] = Field(default_factory=dict)
    operator_context: dict[str, Any] = Field(default_factory=dict)


class _NormalizedVerifier(StrictModel):
    source_type: UnifiedSourceType
    source_record_id: str
    upstream_artifact_id: str | None = None
    problem_id: str = ""
    branch_id: str
    split: SplitName
    texts: dict[str, Any] = Field(default_factory=dict)
    labels: dict[str, Any] = Field(default_factory=dict)
    retention_tag: str = ""
    quality_tag: str = ""
    verdict: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    decomposed_signals: dict[str, Any] = Field(default_factory=dict)
    verifier_decomposition: dict[str, Any] = Field(default_factory=dict)
    route_compute_signals: dict[str, Any] = Field(default_factory=dict)


def _normalize_text(text: str | None) -> str:
    return " ".join((text or "").strip().split())


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _stable_hash(prefix: str, payload: Any) -> str:
    raw = f"{prefix}::{_stable_json(payload)}"
    return f"{prefix}_{sha1(raw.encode('utf-8')).hexdigest()[:16]}"


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


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _split_from_value(value: str | SplitName | Any) -> SplitName:
    raw = value.value if isinstance(value, Enum) else str(value)
    try:
        return SplitName(raw)
    except Exception:
        return SplitName.TRAIN


def _hash_split(problem_id: str, config: TrainingDataBuilderConfig) -> SplitName:
    ratios = config.normalized_split_ratios()
    token = f"{config.split_salt}::{problem_id}"
    bucket = int(sha1(token.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < ratios[0]:
        return SplitName.TRAIN
    if bucket < ratios[0] + ratios[1]:
        return SplitName.VALID
    return SplitName.CALIBRATION


def _resolve_split(problem_id: str, source_split: SplitName | None, config: TrainingDataBuilderConfig) -> SplitName:
    if config.prefer_source_split and source_split is not None:
        return source_split
    return _hash_split(problem_id or "unknown_problem", config)


def _resolve_problem_splits(
    items: Sequence[_NormalizedTrace | _NormalizedVerifier],
    config: TrainingDataBuilderConfig,
) -> tuple[dict[str, SplitName], dict[str, list[str]]]:
    problem_to_splits: dict[str, list[SplitName]] = {}
    for item in items:
        if item.problem_id:
            problem_to_splits.setdefault(item.problem_id, []).append(item.split)

    resolved: dict[str, SplitName] = {}
    conflicts: dict[str, list[str]] = {}
    for problem_id, splits in sorted(problem_to_splits.items()):
        unique = sorted({split.value for split in splits})
        if len(unique) <= 1:
            resolved[problem_id] = _split_from_value(unique[0]) if unique else _hash_split(problem_id, config)
            continue
        conflicts[problem_id] = unique
        if config.split_conflict_policy is SplitConflictPolicy.ERROR:
            raise ValueError(f"Split conflict for problem_id={problem_id}: {unique}")
        if config.split_conflict_policy is SplitConflictPolicy.PREFER_SOURCE:
            resolved[problem_id] = splits[0]
        else:
            resolved[problem_id] = _hash_split(problem_id, config)
    return resolved, conflicts


def _record_provenance(
    *,
    source_type: UnifiedSourceType,
    source_record_id: str,
    upstream_artifact_id: str | None,
    problem_id: str,
    branch_id: str,
    metadata: Mapping[str, Any] | None,
) -> TrainingRecordProvenance:
    payload = {
        "source_type": source_type.value,
        "source_record_id": source_record_id,
        "upstream_artifact_id": upstream_artifact_id,
        "problem_id": problem_id,
        "branch_id": branch_id,
        "metadata": metadata or {},
    }
    return TrainingRecordProvenance(
        source_type=source_type,
        source_record_id=source_record_id,
        upstream_artifact_id=upstream_artifact_id,
        problem_id=problem_id,
        branch_id=branch_id,
        source_digest=_stable_hash("training_source", payload),
        metadata=dict(metadata or {}),
    )


def _proof_summary_from_obligations(proof_obligations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(proof_obligations)
    open_count = 0
    discharged_count = 0
    status_histogram: dict[str, int] = {}
    for item in proof_obligations:
        status = _normalize_text(str(item.get("status") or item.get("state") or "unknown")).lower() or "unknown"
        status_histogram[status] = status_histogram.get(status, 0) + 1
        if status in {"open", "pending", "unresolved"}:
            open_count += 1
        if status in {"discharged", "resolved", "closed", "proved"}:
            discharged_count += 1
    return {
        "obligation_count": total,
        "open_count": open_count,
        "discharged_count": discharged_count,
        "open_fraction": _clamp01(open_count / float(total)) if total else 0.0,
        "discharge_fraction": _clamp01(discharged_count / float(total)) if total else 0.0,
        "status_histogram": status_histogram,
    }


def _extract_decomposed_signals(payload: Mapping[str, Any]) -> dict[str, Any]:
    verifier_decomposition = payload.get("verifier_decomposition") or {}
    if not isinstance(verifier_decomposition, Mapping):
        verifier_decomposition = {}
    route_compute = payload.get("route_compute_signals") or payload.get("compute_signals") or {}
    if not isinstance(route_compute, Mapping):
        route_compute = {}
    signal_map = {
        "verifier_score": payload.get("verifier_score"),
        "logical_consistency": payload.get("logical_consistency", verifier_decomposition.get("logical_consistency")),
        "symbolic_agreement": payload.get("symbolic_agreement", verifier_decomposition.get("symbolic_agreement")),
        "completeness": payload.get("completeness", verifier_decomposition.get("completeness")),
        "repairability": payload.get("repairability", verifier_decomposition.get("repairability")),
        "answer_correctness_likelihood": payload.get("answer_correctness_likelihood", verifier_decomposition.get("answer_correctness_likelihood")),
        "step_quality": payload.get("step_quality", verifier_decomposition.get("step_quality")),
        "prefix_quality": payload.get("prefix_quality", verifier_decomposition.get("prefix_quality")),
        "prm_prefix_quality": payload.get("prm_prefix_quality", verifier_decomposition.get("prm_prefix_quality")),
        "prm_step_quality": payload.get("prm_step_quality", verifier_decomposition.get("prm_step_quality")),
        "retrieval_compatibility": payload.get("retrieval_compatibility"),
        "retrieval_support": payload.get("retrieval_support"),
        "operator_reliability": payload.get("operator_reliability"),
        "discharge_fraction": payload.get("discharge_fraction"),
        "open_obligation_burden": payload.get("open_obligation_burden"),
        "difficulty_intensity": route_compute.get("difficulty_intensity"),
        "route_uncertainty": route_compute.get("route_uncertainty"),
        "proof_burden": route_compute.get("proof_burden", payload.get("open_obligation_burden")),
        "retrieval_need": route_compute.get("retrieval_need"),
        "repair_need": route_compute.get("repair_need"),
        "critique_aggressiveness": route_compute.get("critique_aggressiveness"),
        "repair_aggressiveness": route_compute.get("repair_aggressiveness"),
        "resample_aggressiveness": route_compute.get("resample_aggressiveness"),
    }
    return {str(k): round(_clamp01(v), 6) for k, v in signal_map.items() if v is not None}


def _extract_operator_context_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    operator_sequence = payload.get("operator_sequence") or []
    if not isinstance(operator_sequence, Sequence) or isinstance(operator_sequence, (str, bytes)):
        operator_sequence = []
    return {
        "operator_sequence": [str(x) for x in operator_sequence],
        "operator_reliability": round(_clamp01(payload.get("operator_reliability", 0.0)), 6),
        "operator_provenance": dict(payload.get("operator_provenance") or {}),
        "archetype_used": _normalize_text(str(payload.get("archetype_used") or "")),
    }


def _normalize_trace_record(record: DistilledTraceRecord, upstream_artifact_id: str | None = None) -> _NormalizedTrace:
    outcome = record.outcome.model_dump(mode="json")
    trace_meta = _as_json_value(record.metadata)
    prov = record.provenance.model_dump(mode="json")
    evidence = tuple(_as_json_value(ev.model_dump(mode="json")) for ev in record.evidence)
    prov_meta = dict(prov.get("metadata") or {})
    trace_meta_dict = dict(trace_meta) if isinstance(trace_meta, dict) else {}
    proof_summary = dict(
        prov_meta.get("proof_obligation_summary")
        or trace_meta_dict.get("proof_obligation_summary")
        or _proof_summary_from_obligations(prov_meta.get("proof_obligations", []))
    )
    route_compute = dict(
        prov_meta.get("route_compute_signals")
        or trace_meta_dict.get("route_compute_signals")
        or {}
    )
    payload = {
        **outcome,
        **trace_meta_dict,
        "verifier_decomposition": trace_meta_dict.get("verifier_decomposition") or outcome.get("metadata", {}).get("verifier_decomposition", {}),
        "route_compute_signals": route_compute,
        "proof_obligations": prov_meta.get("proof_obligations", []),
        "discharge_fraction": trace_meta_dict.get("discharge_fraction", proof_summary.get("discharge_fraction")),
        "open_obligation_burden": trace_meta_dict.get("open_obligation_burden", proof_summary.get("open_fraction")),
        "operator_sequence": list(record.operator_sequence),
        "operator_reliability": trace_meta_dict.get("operator_reliability", 0.0),
        "operator_provenance": trace_meta_dict.get("operator_provenance", {}),
        "archetype_used": record.archetypes[0] if record.archetypes else None,
        "retrieval_compatibility": trace_meta_dict.get("retrieval_compatibility", 0.0),
        "retrieval_support": trace_meta_dict.get("retrieval_support", 0.0),
    }
    return _NormalizedTrace(
        source_type=UnifiedSourceType.TRACE_RECORD,
        source_record_id=record.record_id,
        upstream_artifact_id=upstream_artifact_id,
        problem_id=record.problem_id,
        branch_id=record.branch_id,
        split=_split_from_value(record.split),
        classification=record.classification.value if isinstance(record.classification, Enum) else str(record.classification),
        quality_tier=record.quality_tier.value if isinstance(record.quality_tier, Enum) else str(record.quality_tier),
        steps=tuple(_as_json_value(step.model_dump(mode="json")) for step in record.steps),
        operator_sequence=tuple(record.operator_sequence),
        archetypes=tuple(record.archetypes),
        answer_raw=_normalize_text(record.outcome.answer_raw),
        answer_canonical=_normalize_text(record.outcome.answer_canonical),
        route_snapshot=_as_json_value(record.provenance.route_snapshot),
        outcome=_as_json_value(outcome),
        evidence=evidence,
        metadata={
            "trace_metadata": trace_meta,
            "trace_provenance": prov,
            "problem_text": trace_meta_dict.get("problem_text", ""),
        },
        decomposed_signals=_extract_decomposed_signals(payload),
        verifier_decomposition=dict(payload.get("verifier_decomposition") or {}),
        proof_obligation_summary=proof_summary,
        route_compute_signals=dict(route_compute),
        operator_context=_extract_operator_context_from_payload(payload),
    )


def _normalize_verifier_example(example: VerifierTrainingExample, upstream_artifact_id: str | None = None) -> _NormalizedVerifier:
    example_meta = _as_json_value(example.metadata)
    example_meta_dict = dict(example_meta) if isinstance(example_meta, dict) else {}
    payload = {
        **example.labels.model_dump(mode="json"),
        **example_meta_dict,
        "verifier_decomposition": example_meta_dict.get("verifier_decomposition", {}),
        "route_compute_signals": example_meta_dict.get("route_compute_signals", {}),
    }
    return _NormalizedVerifier(
        source_type=UnifiedSourceType.VERIFIER_EXAMPLE,
        source_record_id=example.example_id,
        upstream_artifact_id=upstream_artifact_id,
        problem_id=example.problem_id,
        branch_id=example.branch_id,
        split=_split_from_value(example.split),
        texts=_as_json_value(example.texts.model_dump(mode="json")),
        labels=_as_json_value(example.labels.model_dump(mode="json")),
        retention_tag=example.retention_tag.value,
        quality_tag=example.quality_tag.value,
        verdict=example.verdict.value,
        metadata={
            "scope": example.scope.value,
            "prefix_step_count": example.prefix_step_count,
            "total_step_count": example.total_step_count,
            "example_metadata": example_meta,
            "example_provenance": _as_json_value(example.provenance.model_dump(mode="json")),
        },
        decomposed_signals=_extract_decomposed_signals(payload),
        verifier_decomposition=dict(payload.get("verifier_decomposition") or {}),
        route_compute_signals=dict(payload.get("route_compute_signals") or {}),
    )


def _normalize_branch_trace(trace: BranchTrace) -> _NormalizedTrace:
    operator_sequence = tuple(
        trace.operator_sequence
        or [str(step.operator_used) for step in trace.steps if step.operator_used]
    )
    payload = {
        "verifier_score": trace.verifier_score,
        "logical_consistency": getattr(trace, "logical_consistency", 0.0),
        "symbolic_agreement": getattr(trace, "symbolic_agreement", 0.0),
        "completeness": getattr(trace, "completeness", 0.0),
        "repairability": getattr(trace, "repairability", 0.0),
        "answer_correctness_likelihood": getattr(trace, "answer_correctness_likelihood", 0.0),
        "step_quality": getattr(trace, "step_quality", 0.0),
        "prefix_quality": getattr(trace, "prefix_quality", 0.0),
        "prm_prefix_quality": getattr(trace, "prm_prefix_quality", 0.0),
        "prm_step_quality": getattr(trace, "prm_step_quality", 0.0),
        "retrieval_compatibility": getattr(trace, "retrieval_compatibility", 0.0),
        "retrieval_support": getattr(trace, "retrieval_support", 0.0),
        "operator_reliability": getattr(trace, "operator_reliability", 0.0),
        "discharge_fraction": getattr(trace, "discharge_fraction", 0.0),
        "open_obligation_burden": getattr(trace, "open_obligation_burden", 0.0),
        "verifier_decomposition": getattr(trace, "verifier_decomposition", {}) or {},
        "route_compute_signals": dict((trace.metadata or {}).get("route_compute_signals", {})) if isinstance(trace.metadata, Mapping) else {},
        "operator_sequence": list(operator_sequence),
        "operator_provenance": dict((trace.metadata or {}).get("operator_provenance", {})) if isinstance(trace.metadata, Mapping) else {},
        "archetype_used": trace.archetype_used,
        "proof_obligations": getattr(trace, "proof_obligations", []) or [],
    }
    proof_summary = _proof_summary_from_obligations(payload["proof_obligations"])
    if not payload["discharge_fraction"]:
        payload["discharge_fraction"] = proof_summary.get("discharge_fraction", 0.0)
    if not payload["open_obligation_burden"]:
        payload["open_obligation_burden"] = proof_summary.get("open_fraction", 0.0)
    return _NormalizedTrace(
        source_type=UnifiedSourceType.BRANCH_TRACE,
        source_record_id=trace.branch_id,
        upstream_artifact_id=None,
        problem_id=trace.problem_id,
        branch_id=trace.branch_id,
        split=SplitName.TRAIN,
        classification="raw_branch_trace",
        quality_tier="unknown",
        steps=tuple(_as_json_value(step.model_dump(mode="json")) for step in trace.steps),
        operator_sequence=operator_sequence,
        archetypes=tuple([trace.archetype_used] if trace.archetype_used else ()),
        answer_raw=_normalize_text(trace.answer),
        answer_canonical=_normalize_text(trace.answer_canonical or trace.answer),
        route_snapshot=_as_json_value((trace.provenance or {}).get("route_snapshot", {})),
        outcome={
            "symbolic_valid": bool(trace.symbolic_valid),
            "branch_score": trace.branch_score,
            "verifier_score": trace.verifier_score,
            "failure_type": _as_json_value(trace.failure_type),
            "failure_location": trace.failure_location,
            "repaired": bool(trace.repaired),
            "repair_count": int(trace.repair_count),
        },
        metadata={
            "full_reasoning": _normalize_text(trace.full_reasoning),
            "tool_consistency": trace.tool_consistency,
            "retrieval_used": bool(trace.retrieval_used),
            "trace_metadata": _as_json_value(trace.metadata or {}),
        },
        decomposed_signals=_extract_decomposed_signals(payload),
        verifier_decomposition=dict(payload.get("verifier_decomposition") or {}),
        proof_obligation_summary=proof_summary,
        route_compute_signals=dict(payload.get("route_compute_signals") or {}),
        operator_context=_extract_operator_context_from_payload(payload),
    )


def normalize_training_source(
    source: DistilledTraceArtifact | DistilledTraceRecord | VerifierTrainingArtifact | VerifierTrainingExample | BranchTrace,
) -> tuple[_NormalizedTrace | _NormalizedVerifier, ...]:
    if isinstance(source, DistilledTraceArtifact):
        return tuple(_normalize_trace_record(record, source.metadata.artifact_id) for record in source.records)
    if isinstance(source, DistilledTraceRecord):
        return (_normalize_trace_record(source, None),)
    if VerifierTrainingArtifact is not Any and isinstance(source, VerifierTrainingArtifact):
        return tuple(_normalize_verifier_example(example, source.metadata.artifact_id) for example in source.examples)
    if VerifierTrainingExample is not Any and isinstance(source, VerifierTrainingExample):
        return (_normalize_verifier_example(source, None),)
    if isinstance(source, BranchTrace):
        return (_normalize_branch_trace(source),)
    raise TypeError(f"Unsupported training data source type: {type(source)!r}")


def _reasoning_text_from_steps(steps: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for index, step in enumerate(steps, start=1):
        text = _normalize_text(str(step.get("text") or step.get("description") or ""))
        operator_name = _normalize_text(str(step.get("operator_name") or step.get("operator_used") or ""))
        if not text:
            continue
        prefix = f"Step {index}"
        if operator_name:
            prefix += f" [{operator_name}]"
        lines.append(f"{prefix}: {text}")
    return "\n".join(lines)


def _base_trace_feature_fields(trace: _NormalizedTrace) -> dict[str, Any]:
    return {
        "decomposed_signals": dict(trace.decomposed_signals),
        "verifier_decomposition": dict(trace.verifier_decomposition),
        "proof_obligation_summary": dict(trace.proof_obligation_summary),
        "route_compute_signals": dict(trace.route_compute_signals),
        "operator_context": dict(trace.operator_context),
        "retrieval_compatibility": float(trace.decomposed_signals.get("retrieval_compatibility", 0.0) or 0.0),
        "retrieval_support": float(trace.decomposed_signals.get("retrieval_support", 0.0) or 0.0),
        "quality_tier": trace.quality_tier,
        "classification": trace.classification,
        "answer_canonical": trace.answer_canonical,
        "outcome": dict(trace.outcome),
    }


def _build_router_example(trace: _NormalizedTrace, config: TrainingDataBuilderConfig) -> UnifiedTrainingExample | None:
    route_snapshot = dict(trace.route_snapshot)
    problem_text = _normalize_text(str(route_snapshot.get("problem_text") or trace.metadata.get("problem_text") or ""))
    if config.drop_router_examples_without_problem_text and not problem_text:
        return None
    targets = {
        "problem_type": _as_json_value(route_snapshot.get("problem_type", {})),
        "archetypes": list(trace.archetypes) or _as_json_value(route_snapshot.get("archetypes", {})),
        "difficulty": _as_json_value(route_snapshot.get("difficulty", None)),
        "route_compute_signals": dict(trace.route_compute_signals),
        "retrieval_need": trace.decomposed_signals.get("retrieval_need", 0.0),
        "repair_need": trace.decomposed_signals.get("repair_need", 0.0),
        "proof_obligation_burden": trace.proof_obligation_summary.get("open_fraction", trace.decomposed_signals.get("open_obligation_burden", 0.0)),
    }
    if config.drop_examples_without_targets and not any(targets.values()):
        return None
    split = _resolve_split(trace.problem_id, trace.split, config)
    provenance = _record_provenance(
        source_type=trace.source_type,
        source_record_id=trace.source_record_id,
        upstream_artifact_id=trace.upstream_artifact_id,
        problem_id=trace.problem_id,
        branch_id=trace.branch_id,
        metadata=trace.metadata,
    )
    return UnifiedTrainingExample(
        record_id=_stable_hash("training_record", {"task_family": "router", "source_record_id": trace.source_record_id, "split": split.value}),
        task_family=UnifiedTaskFamily.ROUTER,
        split=split,
        problem_id=trace.problem_id,
        branch_id=trace.branch_id,
        input_fields={
            "problem_text": problem_text,
            "reasoning_text": _reasoning_text_from_steps(trace.steps),
            "state_summary": route_snapshot,
        },
        target_fields=targets,
        feature_fields={**_base_trace_feature_fields(trace), "operator_sequence": list(trace.operator_sequence)},
        provenance=provenance,
        metadata={"builder_task": "router"},
    )


def _build_operator_example(trace: _NormalizedTrace, config: TrainingDataBuilderConfig) -> UnifiedTrainingExample | None:
    if config.drop_examples_without_targets and not trace.operator_sequence:
        return None
    split = _resolve_split(trace.problem_id, trace.split, config)
    provenance = _record_provenance(
        source_type=trace.source_type,
        source_record_id=trace.source_record_id,
        upstream_artifact_id=trace.upstream_artifact_id,
        problem_id=trace.problem_id,
        branch_id=trace.branch_id,
        metadata=trace.metadata,
    )
    target_fields = {
        "operator_sequence": list(trace.operator_sequence),
        "archetypes": list(trace.archetypes),
        "classification": trace.classification,
        "operator_reliability": trace.operator_context.get("operator_reliability", 0.0),
        "proof_discharge_fraction": trace.proof_obligation_summary.get("discharge_fraction", 0.0),
    }
    return UnifiedTrainingExample(
        record_id=_stable_hash("training_record", {"task_family": "operator_mining", "source_record_id": trace.source_record_id, "split": split.value}),
        task_family=UnifiedTaskFamily.OPERATOR_MINING,
        split=split,
        problem_id=trace.problem_id,
        branch_id=trace.branch_id,
        input_fields={
            "steps": list(trace.steps),
            "reasoning_text": _reasoning_text_from_steps(trace.steps),
            "evidence": list(trace.evidence),
        },
        target_fields=target_fields,
        feature_fields=_base_trace_feature_fields(trace),
        provenance=provenance,
        metadata={"builder_task": "operator_mining"},
    )


def _build_trace_calibration_example(trace: _NormalizedTrace, config: TrainingDataBuilderConfig) -> UnifiedTrainingExample | None:
    split = _resolve_split(trace.problem_id, trace.split, config)
    provenance = _record_provenance(
        source_type=trace.source_type,
        source_record_id=trace.source_record_id,
        upstream_artifact_id=trace.upstream_artifact_id,
        problem_id=trace.problem_id,
        branch_id=trace.branch_id,
        metadata=trace.metadata,
    )
    return UnifiedTrainingExample(
        record_id=_stable_hash("training_record", {"task_family": "calibration", "source_record_id": trace.source_record_id, "source_type": trace.source_type.value, "split": split.value}),
        task_family=UnifiedTaskFamily.CALIBRATION,
        split=split,
        problem_id=trace.problem_id,
        branch_id=trace.branch_id,
        input_fields={
            "reasoning_text": _reasoning_text_from_steps(trace.steps),
            "answer_canonical": trace.answer_canonical,
        },
        target_fields={
            "classification": trace.classification,
            "quality_tier": trace.quality_tier,
            "outcome": trace.outcome,
            "final_answer_correct": trace.outcome.get("final_answer_correct"),
            "symbolic_valid": trace.outcome.get("symbolic_valid"),
        },
        feature_fields=_base_trace_feature_fields(trace),
        provenance=provenance,
        metadata={"builder_task": "calibration_from_trace"},
    )


def _build_verifier_example(verifier: _NormalizedVerifier, config: TrainingDataBuilderConfig) -> UnifiedTrainingExample | None:
    if config.drop_examples_without_targets and not verifier.labels:
        return None
    split = _resolve_split(verifier.problem_id, verifier.split, config)
    provenance = _record_provenance(
        source_type=verifier.source_type,
        source_record_id=verifier.source_record_id,
        upstream_artifact_id=verifier.upstream_artifact_id,
        problem_id=verifier.problem_id,
        branch_id=verifier.branch_id,
        metadata=verifier.metadata,
    )
    return UnifiedTrainingExample(
        record_id=_stable_hash("training_record", {"task_family": "verifier", "source_record_id": verifier.source_record_id, "split": split.value}),
        task_family=UnifiedTaskFamily.VERIFIER,
        split=split,
        problem_id=verifier.problem_id,
        branch_id=verifier.branch_id,
        input_fields={
            "problem_text": verifier.texts.get("problem_text", ""),
            "prefix_text": verifier.texts.get("prefix_text", ""),
            "reasoning_text": verifier.texts.get("reasoning_text", ""),
            "answer_text": verifier.texts.get("answer_text", ""),
        },
        target_fields={
            "labels": verifier.labels,
            "retention_tag": verifier.retention_tag,
            "quality_tag": verifier.quality_tag,
            "verdict": verifier.verdict,
        },
        feature_fields={
            "texts": verifier.texts,
            "decomposed_signals": dict(verifier.decomposed_signals),
            "verifier_decomposition": dict(verifier.verifier_decomposition),
            "route_compute_signals": dict(verifier.route_compute_signals),
        },
        provenance=provenance,
        metadata={"builder_task": "verifier"},
    )


def _build_verifier_calibration_example(verifier: _NormalizedVerifier, config: TrainingDataBuilderConfig) -> UnifiedTrainingExample | None:
    split = _resolve_split(verifier.problem_id, verifier.split, config)
    provenance = _record_provenance(
        source_type=verifier.source_type,
        source_record_id=verifier.source_record_id,
        upstream_artifact_id=verifier.upstream_artifact_id,
        problem_id=verifier.problem_id,
        branch_id=verifier.branch_id,
        metadata=verifier.metadata,
    )
    return UnifiedTrainingExample(
        record_id=_stable_hash("training_record", {"task_family": "calibration", "source_record_id": verifier.source_record_id, "source_type": verifier.source_type.value, "split": split.value}),
        task_family=UnifiedTaskFamily.CALIBRATION,
        split=split,
        problem_id=verifier.problem_id,
        branch_id=verifier.branch_id,
        input_fields={
            "prefix_text": verifier.texts.get("prefix_text", ""),
            "reasoning_text": verifier.texts.get("reasoning_text", ""),
        },
        target_fields={
            "retention_tag": verifier.retention_tag,
            "quality_tag": verifier.quality_tag,
            "verdict": verifier.verdict,
            "labels": verifier.labels,
        },
        feature_fields={
            "problem_text": verifier.texts.get("problem_text", ""),
            "canonical_answer": verifier.texts.get("canonical_answer", ""),
            "decomposed_signals": dict(verifier.decomposed_signals),
            "verifier_decomposition": dict(verifier.verifier_decomposition),
            "route_compute_signals": dict(verifier.route_compute_signals),
        },
        provenance=provenance,
        metadata={"builder_task": "calibration_from_verifier"},
    )


def build_training_artifact(
    sources: Sequence[DistilledTraceArtifact | DistilledTraceRecord | VerifierTrainingArtifact | VerifierTrainingExample | BranchTrace],
    *,
    config: TrainingDataBuilderConfig | None = None,
) -> tuple[OfflineTrainingArtifact, TrainingDataBuildReport]:
    resolved_config = config or TrainingDataBuilderConfig()
    normalized_items: list[_NormalizedTrace | _NormalizedVerifier] = []
    for source in sources:
        normalized_items.extend(normalize_training_source(source))

    resolved_problem_splits, split_conflicts = _resolve_problem_splits(normalized_items, resolved_config)

    records: list[UnifiedTrainingExample] = []
    dropped_reasons: dict[str, int] = {}
    for item in normalized_items:
        source_split_override = resolved_problem_splits.get(item.problem_id) if item.problem_id else None
        if source_split_override is not None:
            item = item.model_copy(update={"split": source_split_override})
        candidates: list[UnifiedTrainingExample | None] = []
        if isinstance(item, _NormalizedTrace):
            if resolved_config.include_router_examples:
                candidates.append(_build_router_example(item, resolved_config))
            if resolved_config.include_operator_examples:
                candidates.append(_build_operator_example(item, resolved_config))
            if resolved_config.include_calibration_examples:
                candidates.append(_build_trace_calibration_example(item, resolved_config))
        else:
            if resolved_config.include_verifier_examples:
                candidates.append(_build_verifier_example(item, resolved_config))
            if resolved_config.include_calibration_examples:
                candidates.append(_build_verifier_calibration_example(item, resolved_config))

        for candidate in candidates:
            if candidate is None:
                dropped_reasons["dropped_empty_candidate"] = dropped_reasons.get("dropped_empty_candidate", 0) + 1
                continue
            records.append(candidate)

    records.sort(key=lambda r: (r.task_family.value, r.split.value, r.problem_id, r.branch_id, r.record_id))

    task_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    problem_split_counts: dict[str, int] = {}
    for record in records:
        task_counts[record.task_family.value] = task_counts.get(record.task_family.value, 0) + 1
        split_counts[record.split.value] = split_counts.get(record.split.value, 0) + 1
    for split in resolved_problem_splits.values():
        problem_split_counts[split.value] = problem_split_counts.get(split.value, 0) + 1
    if split_conflicts:
        dropped_reasons["problem_split_conflicts"] = len(split_conflicts)

    config_digest = _stable_hash("training_builder_config", resolved_config.model_dump(mode="json"))
    source_digest = _stable_hash(
        "training_builder_sources",
        [{
            "source_type": item.source_type.value,
            "source_record_id": item.source_record_id,
            "upstream_artifact_id": item.upstream_artifact_id,
            "problem_id": item.problem_id,
            "branch_id": item.branch_id,
        } for item in normalized_items],
    )
    artifact_id = _stable_hash(
        "offline_training_artifact",
        {"artifact_version": resolved_config.artifact_version, "config_digest": config_digest, "record_ids": [r.record_id for r in records]},
    )
    artifact = OfflineTrainingArtifact(
        metadata=OfflineTrainingArtifactMetadata(
            artifact_version=resolved_config.artifact_version,
            schema_version=resolved_config.schema_version,
            artifact_id=artifact_id,
            record_count=len(records),
            source_count=len(normalized_items),
            problem_count=len({record.problem_id for record in records if record.problem_id}),
            task_counts=task_counts,
            split_counts=split_counts,
            problem_split_counts=problem_split_counts,
            split_conflicts=split_conflicts,
            upstream_artifact_ids=tuple(sorted({item.upstream_artifact_id for item in normalized_items if item.upstream_artifact_id})),
            config_digest=config_digest,
            source_digest=source_digest,
        ),
        records=tuple(records),
    )
    report = TrainingDataBuildReport(
        kept_count=len(records),
        dropped_count=sum(dropped_reasons.values()),
        task_counts=task_counts,
        split_counts=split_counts,
        dropped_reasons=dropped_reasons,
    )
    return artifact, report


def artifact_records_by_task(artifact: OfflineTrainingArtifact) -> dict[UnifiedTaskFamily, tuple[UnifiedTrainingExample, ...]]:
    grouped: dict[UnifiedTaskFamily, list[UnifiedTrainingExample]] = {task: [] for task in UnifiedTaskFamily}
    for record in artifact.records:
        grouped[record.task_family].append(record)
    return {task: tuple(items) for task, items in grouped.items()}


def artifact_records_by_split(artifact: OfflineTrainingArtifact) -> dict[SplitName, tuple[UnifiedTrainingExample, ...]]:
    grouped: dict[SplitName, list[UnifiedTrainingExample]] = {split: [] for split in SplitName}
    for record in artifact.records:
        grouped[record.split].append(record)
    return {split: tuple(items) for split, items in grouped.items()}


def router_training_rows(artifact: OfflineTrainingArtifact) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "record_id": record.record_id,
            "split": record.split.value,
            "problem_id": record.problem_id,
            "branch_id": record.branch_id,
            "problem_text": record.input_fields.get("problem_text", ""),
            "reasoning_text": record.input_fields.get("reasoning_text", ""),
            "targets": record.target_fields,
            "features": record.feature_fields,
            "provenance": record.provenance.model_dump(mode="json"),
        }
        for record in artifact.records
        if record.task_family == UnifiedTaskFamily.ROUTER
    )


def verifier_training_rows(artifact: OfflineTrainingArtifact) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "record_id": record.record_id,
            "split": record.split.value,
            "problem_id": record.problem_id,
            "branch_id": record.branch_id,
            "input_fields": record.input_fields,
            "targets": record.target_fields,
            "features": record.feature_fields,
            "provenance": record.provenance.model_dump(mode="json"),
        }
        for record in artifact.records
        if record.task_family == UnifiedTaskFamily.VERIFIER
    )


def operator_mining_rows(artifact: OfflineTrainingArtifact) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "record_id": record.record_id,
            "split": record.split.value,
            "problem_id": record.problem_id,
            "branch_id": record.branch_id,
            "steps": record.input_fields.get("steps", []),
            "reasoning_text": record.input_fields.get("reasoning_text", ""),
            "targets": record.target_fields,
            "features": record.feature_fields,
            "provenance": record.provenance.model_dump(mode="json"),
        }
        for record in artifact.records
        if record.task_family == UnifiedTaskFamily.OPERATOR_MINING
    )


def calibration_rows(artifact: OfflineTrainingArtifact) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "record_id": record.record_id,
            "split": record.split.value,
            "problem_id": record.problem_id,
            "branch_id": record.branch_id,
            "inputs": record.input_fields,
            "targets": record.target_fields,
            "features": record.feature_fields,
            "provenance": record.provenance.model_dump(mode="json"),
        }
        for record in artifact.records
        if record.task_family == UnifiedTaskFamily.CALIBRATION
    )


def write_training_artifact(
    artifact: OfflineTrainingArtifact,
    output_dir: str | Path,
    *,
    include_task_files: bool = True,
    include_split_files: bool = True,
) -> TrainingArtifactWriteResult:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = out_dir / artifact.metadata.artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = artifact_dir / "manifest.json"
    records_path = artifact_dir / "records.jsonl"
    manifest_path.write_text(_stable_json(artifact.metadata.model_dump(mode="json")) + "\n", encoding="utf-8")
    with records_path.open("w", encoding="utf-8") as handle:
        for record in artifact.records:
            handle.write(_stable_json(record.model_dump(mode="json")) + "\n")

    task_paths: dict[str, str] = {}
    if include_task_files:
        for task, records in artifact_records_by_task(artifact).items():
            path = artifact_dir / f"{task.value}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(_stable_json(record.model_dump(mode="json")) + "\n")
            task_paths[task.value] = str(path)

    split_paths: dict[str, str] = {}
    if include_split_files:
        for split, records in artifact_records_by_split(artifact).items():
            path = artifact_dir / f"{split.value}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(_stable_json(record.model_dump(mode="json")) + "\n")
            split_paths[split.value] = str(path)

    return TrainingArtifactWriteResult(
        artifact_dir=str(artifact_dir),
        manifest_path=str(manifest_path),
        records_path=str(records_path),
        task_paths=task_paths,
        split_paths=split_paths,
    )


def load_training_artifact(path: str | Path) -> OfflineTrainingArtifact:
    input_path = Path(path)
    if input_path.is_dir():
        manifest_path = input_path / "manifest.json"
        records_path = input_path / "records.jsonl"
    elif input_path.name == "manifest.json":
        manifest_path = input_path
        records_path = input_path.with_name("records.jsonl")
    else:
        raise ValueError(f"Unsupported artifact path: {path}")

    metadata = OfflineTrainingArtifactMetadata.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    records: list[UnifiedTrainingExample] = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(UnifiedTrainingExample.model_validate_json(stripped))
    return OfflineTrainingArtifact(metadata=metadata, records=tuple(records))


__all__ = [
    "ARTIFACT_KIND",
    "DEFAULT_ARTIFACT_VERSION",
    "SCHEMA_VERSION",
    "OfflineTrainingArtifact",
    "OfflineTrainingArtifactMetadata",
    "SplitName",
    "TrainingArtifactWriteResult",
    "TrainingDataBuildReport",
    "TrainingDataBuilderConfig",
    "TrainingRecordProvenance",
    "UnifiedSourceType",
    "UnifiedTaskFamily",
    "UnifiedTrainingExample",
    "artifact_records_by_split",
    "artifact_records_by_task",
    "build_training_artifact",
    "calibration_rows",
    "load_training_artifact",
    "normalize_training_source",
    "operator_mining_rows",
    "router_training_rows",
    "verifier_training_rows",
    "write_training_artifact",
]
