from __future__ import annotations

"""Manifest-backed offline archetype/routing training preparation.

This module turns router-facing offline artifacts into a deterministic,
problem-level archetype training dataset with explicit splits, provenance, and
readiness metadata. It is intentionally a real pipeline component rather than a
loose JSON helper.

Repair emphasis for this pass:
- consume richer offline/runtime supervision beyond shallow domain labels
- preserve archetype-conditioned evidence useful for routing/runtime control
- keep outputs manifest-backed, deterministic, and backward-compatible
"""

import hashlib
import json
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.common.constants import (
    OFFLINE_ARTIFACT_VERSION,
    OFFLINE_SPLIT_RATIOS,
    OFFLINE_SPLIT_SALT,
)
from src.common.schemas import ParsedProblem
from src.offline.training_data_builder import (
    OfflineTrainingArtifact,
    SplitName as BuilderSplitName,
    UnifiedTaskFamily,
    UnifiedTrainingExample,
)
from src.routing.archetype_predictor import get_supported_archetypes


SCHEMA_VERSION = "archetype_training.v2"
ARTIFACT_KIND = "archetype_training_manifest"
DEFAULT_ARTIFACT_VERSION = OFFLINE_ARTIFACT_VERSION


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        str_strip_whitespace=True,
        populate_by_name=True,
    )


class ArchetypeTrainingStatus(str, Enum):
    READY = "ready"
    MISSING_INPUT = "missing_input"
    INVALID_DATASET = "invalid_dataset"
    UNAVAILABLE_RUNTIME = "unavailable_runtime"
    COMPLETED = "completed"


class SplitStrategy(str, Enum):
    SOURCE = "source"
    MANIFEST = "manifest"
    HASH = "hash"


class RuntimeBoundary(str, Enum):
    MANIFEST_ONLY = "manifest_only"
    EXTERNAL_TRAINER_REQUIRED = "external_trainer_required"
    TRAINER_EXECUTED = "trainer_executed"


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    CALIBRATION = "calibration"


class ArchetypeTrainingConfig(StrictModel):
    run_name: str = "archetype_training"
    output_dir: str = "artifacts/router/archetype_training"
    dataset_path: str | None = None
    split_manifest_path: str | None = None
    artifact_version: str = DEFAULT_ARTIFACT_VERSION
    schema_version: str = SCHEMA_VERSION
    split_strategy: SplitStrategy = SplitStrategy.SOURCE
    split_ratios: tuple[float, float, float] = OFFLINE_SPLIT_RATIOS
    split_salt: str = OFFLINE_SPLIT_SALT
    min_examples: int = Field(default=4, ge=1)
    min_valid_examples: int = Field(default=1, ge=0)
    min_calibration_examples: int = Field(default=1, ge=0)
    preserve_problem_metadata: bool = True
    emit_jsonl: bool = True
    write_summary_json: bool = True
    allow_unlabeled_examples: bool = False
    require_signal_rich_examples: bool = False

    @model_validator(mode="after")
    def _validate_manifest_strategy(self) -> "ArchetypeTrainingConfig":
        if self.split_strategy == SplitStrategy.MANIFEST and not self.split_manifest_path:
            raise ValueError("split_manifest_path is required when split_strategy='manifest'")
        return self

    def normalized_split_ratios(self) -> tuple[float, float, float]:
        total = sum(self.split_ratios)
        if total <= 0:
            return (0.80, 0.10, 0.10)
        return tuple(float(value) / total for value in self.split_ratios)  # type: ignore[return-value]


class ArchetypeTrainingExample(StrictModel):
    problem_id: str
    raw_text: str
    domain: str
    target: str = ""
    answer_type: str = ""
    archetype_labels: list[str] = Field(default_factory=list)
    label_scores: dict[str, float] = Field(default_factory=dict)
    source_split: DatasetSplit | None = None
    source_artifact_id: str | None = None
    source_record_ids: list[str] = Field(default_factory=list)
    source_metadata: dict[str, Any] = Field(default_factory=dict)

    # Additive richer supervision fields.
    feature_fields: dict[str, Any] = Field(default_factory=dict)
    supervision_summary: dict[str, Any] = Field(default_factory=dict)


class SplitManifestEntry(StrictModel):
    problem_id: str
    split: DatasetSplit


class DatasetLabelStats(StrictModel):
    total_examples: int = 0
    labeled_examples: int = 0
    unlabeled_examples: int = 0
    label_frequencies: dict[str, int] = Field(default_factory=dict)


class DatasetValidationReport(StrictModel):
    valid: bool
    num_examples: int = 0
    num_unique_problem_ids: int = 0
    duplicate_problem_ids: list[str] = Field(default_factory=list)
    split_conflict_problem_ids: list[str] = Field(default_factory=list)
    unsupported_labels: dict[str, list[str]] = Field(default_factory=dict)
    empty_label_problem_ids: list[str] = Field(default_factory=list)
    stats: DatasetLabelStats = Field(default_factory=DatasetLabelStats)
    notes: list[str] = Field(default_factory=list)


class SplitSummary(StrictModel):
    split_name: str
    count: int = 0
    label_frequencies: dict[str, int] = Field(default_factory=dict)
    problem_ids: list[str] = Field(default_factory=list)


class ArchetypeSplitBundle(StrictModel):
    train: list[ArchetypeTrainingExample] = Field(default_factory=list)
    valid: list[ArchetypeTrainingExample] = Field(default_factory=list)
    calibration: list[ArchetypeTrainingExample] = Field(default_factory=list)
    split_strategy: SplitStrategy
    manifest_source: str | None = None
    train_summary: SplitSummary = Field(default_factory=lambda: SplitSummary(split_name="train"))
    valid_summary: SplitSummary = Field(default_factory=lambda: SplitSummary(split_name="valid"))
    calibration_summary: SplitSummary = Field(default_factory=lambda: SplitSummary(split_name="calibration"))


class ReadinessReport(StrictModel):
    ready_for_training: bool = False
    min_examples_ok: bool = False
    valid_examples_ok: bool = False
    calibration_examples_ok: bool = False
    unlabeled_policy_ok: bool = False


class ArchetypeTrainingArtifactMetadata(StrictModel):
    artifact_kind: str = ARTIFACT_KIND
    artifact_version: str = DEFAULT_ARTIFACT_VERSION
    schema_version: str = SCHEMA_VERSION
    artifact_id: str
    run_name: str
    problem_count: int
    split_counts: dict[str, int] = Field(default_factory=dict)
    label_frequencies: dict[str, int] = Field(default_factory=dict)
    supported_labels: tuple[str, ...] = Field(default_factory=tuple)
    source_artifact_ids: tuple[str, ...] = Field(default_factory=tuple)
    config_digest: str
    source_digest: str
    readiness: ReadinessReport
    checkpoint_references: dict[str, str] = Field(default_factory=dict)


class ArchetypeTrainingManifest(StrictModel):
    metadata: ArchetypeTrainingArtifactMetadata
    config: ArchetypeTrainingConfig
    validation: DatasetValidationReport
    split_bundle: ArchetypeSplitBundle
    trainer_report: dict[str, Any] = Field(default_factory=dict)


class EmittedArtifactPaths(StrictModel):
    run_dir: str
    manifest_path: str | None = None
    train_manifest_path: str | None = None
    valid_manifest_path: str | None = None
    calibration_manifest_path: str | None = None
    split_summary_path: str | None = None
    dataset_validation_path: str | None = None
    trained_model_path: str | None = None
    trainer_report_path: str | None = None


class ArchetypeTrainingResult(StrictModel):
    status: ArchetypeTrainingStatus
    runtime_boundary: RuntimeBoundary
    config: ArchetypeTrainingConfig
    validation: DatasetValidationReport
    split_bundle: ArchetypeSplitBundle | None = None
    artifacts: EmittedArtifactPaths | None = None
    trainer_report: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ExternalTrainerProtocol:
    def __call__(
        self,
        *,
        train_examples: list[ArchetypeTrainingExample],
        eval_examples: list[ArchetypeTrainingExample],
        config: ArchetypeTrainingConfig,
        run_dir: Path,
    ) -> Mapping[str, Any]:
        raise NotImplementedError


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _stable_hash(prefix: str, payload: Any) -> str:
    raw = f"{prefix}::{_stable_json(payload)}"
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _normalize_text(text: str | None) -> str:
    return " ".join((text or "").strip().split())


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _coerce_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return _coerce_json_value(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _coerce_json_value(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_coerce_json_value(item) for item in value]
    return _normalize_text(str(value))


def _coerce_split(value: DatasetSplit | BuilderSplitName | str | None) -> DatasetSplit | None:
    if value is None:
        return None
    raw = value.value if isinstance(value, Enum) else str(value)
    try:
        return DatasetSplit(raw)
    except Exception:
        return None


def _hash_split(problem_id: str, config: ArchetypeTrainingConfig) -> DatasetSplit:
    ratios = config.normalized_split_ratios()
    token = f"{config.split_salt}::{problem_id or 'unknown_problem'}"
    bucket = int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < ratios[0]:
        return DatasetSplit.TRAIN
    if bucket < ratios[0] + ratios[1]:
        return DatasetSplit.VALID
    return DatasetSplit.CALIBRATION


def _top_key(distribution: Any) -> str:
    if isinstance(distribution, Mapping) and distribution:
        ranked = sorted(distribution.items(), key=lambda item: (-float(item[1]), str(item[0])))
        return str(ranked[0][0])
    return "unknown"


def _normalize_score_distribution(payload: Any) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, value in payload.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        out[str(key)] = max(0.0, numeric)
    total = sum(out.values())
    if total > 0:
        out = {key: value / total for key, value in out.items()}
    return {key: round(out[key], 6) for key in sorted(out)}


def _normalize_label_payload(
    payload: Any,
    *,
    supported: set[str],
) -> tuple[list[str], dict[str, float], list[str]]:
    labels: list[str] = []
    scores: dict[str, float] = {}
    unsupported: list[str] = []
    if isinstance(payload, Mapping):
        for label, score in payload.items():
            label_name = str(label)
            try:
                numeric_score = float(score)
            except (TypeError, ValueError):
                numeric_score = 0.0
            if label_name in supported:
                scores[label_name] = max(0.0, numeric_score)
            else:
                unsupported.append(label_name)
        labels = sorted(name for name, score in scores.items() if score > 0.0)
        return labels, scores, sorted(set(unsupported))

    values: list[str]
    if payload is None:
        values = []
    elif isinstance(payload, str):
        values = [_normalize_text(payload)] if _normalize_text(payload) else []
    else:
        values = [_normalize_text(str(item)) for item in list(payload)]

    for label in values:
        if not label:
            continue
        if label in supported:
            labels.append(label)
            scores[label] = 1.0
        else:
            unsupported.append(label)
    return sorted(set(labels)), {key: scores[key] for key in sorted(scores)}, sorted(set(unsupported))


def _extract_float(mapping: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return _clamp01(mapping[key])
    return None


def _extract_mapping(mapping: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _operator_pattern_summary(feature_fields: Mapping[str, Any], provenance_meta: Mapping[str, Any]) -> dict[str, Any]:
    operator_seq = provenance_meta.get("operator_sequence") or feature_fields.get("operator_sequence") or []
    if not isinstance(operator_seq, Sequence) or isinstance(operator_seq, (str, bytes)):
        operator_seq = []
    ops = [_normalize_text(str(item)) for item in operator_seq if _normalize_text(str(item))]
    counts = Counter(ops)
    return {
        "operator_sequence": ops,
        "top_operators": [name for name, _ in counts.most_common(5)],
        "operator_count": len(ops),
        "unique_operator_count": len(counts),
    }


def _route_compute_signal_summary(feature_fields: Mapping[str, Any], provenance_meta: Mapping[str, Any]) -> dict[str, float]:
    payload = (
        _extract_mapping(provenance_meta, "route_compute_signals", "compute_signals")
        or _extract_mapping(feature_fields, "route_compute_signals", "compute_signals")
        or {}
    )
    return {
        key: round(_clamp01(payload.get(key, 0.0)), 6)
        for key in (
            "difficulty_intensity",
            "route_uncertainty",
            "proof_burden",
            "retrieval_need",
            "repair_need",
            "critique_aggressiveness",
            "repair_aggressiveness",
            "resample_aggressiveness",
        )
    }


def _supervision_summary(
    *,
    feature_fields: Mapping[str, Any],
    target_fields: Mapping[str, Any],
    provenance_meta: Mapping[str, Any],
) -> dict[str, Any]:
    verifier = _extract_mapping(feature_fields, "decomposed_signals", "verifier_decomposition")
    proof = _extract_mapping(feature_fields, "proof_obligation_summary")
    retrieval = _extract_mapping(feature_fields, "retrieval_summary")
    operator = _extract_mapping(feature_fields, "operator_summary")
    route_compute = _route_compute_signal_summary(feature_fields, provenance_meta)
    process = {
        "step_quality": round(
            _extract_float(feature_fields, "step_quality", "prm_step_quality")
            or verifier.get("step_quality", 0.0),
            6,
        ),
        "prefix_quality": round(
            _extract_float(feature_fields, "prefix_quality", "prm_prefix_quality")
            or verifier.get("prefix_quality", 0.0),
            6,
        ),
    }

    return {
        "logical_consistency": round(
            _extract_float(feature_fields, "logical_consistency")
            or verifier.get("logical_consistency", 0.0),
            6,
        ),
        "symbolic_agreement": round(
            _extract_float(feature_fields, "symbolic_agreement")
            or verifier.get("symbolic_agreement", 0.0),
            6,
        ),
        "completeness": round(
            _extract_float(feature_fields, "completeness")
            or verifier.get("completeness", 0.0),
            6,
        ),
        "repairability": round(
            _extract_float(feature_fields, "repairability")
            or verifier.get("repairability", 0.0),
            6,
        ),
        "answer_correctness_likelihood": round(
            _extract_float(feature_fields, "answer_correctness_likelihood", "verifier_score")
            or verifier.get("answer_correctness_likelihood", 0.0),
            6,
        ),
        "process_quality": process,
        "proof_obligation_burden": round(
            _extract_float(feature_fields, "open_obligation_burden")
            or proof.get("open_obligation_burden", 0.0),
            6,
        ),
        "discharge_fraction": round(
            _extract_float(feature_fields, "discharge_fraction")
            or proof.get("discharge_fraction", 0.0),
            6,
        ),
        "retrieval_compatibility": round(
            _extract_float(feature_fields, "retrieval_compatibility")
            or retrieval.get("retrieval_compatibility", 0.0),
            6,
        ),
        "retrieval_support": round(
            _extract_float(feature_fields, "retrieval_support")
            or retrieval.get("retrieval_support", 0.0),
            6,
        ),
        "operator_reliability": round(
            _extract_float(feature_fields, "operator_reliability")
            or operator.get("operator_reliability", 0.0),
            6,
        ),
        "route_compute_signals": route_compute,
        "problem_type_distribution": _normalize_score_distribution(target_fields.get("problem_type")),
        "archetype_distribution": _normalize_score_distribution(target_fields.get("archetypes")),
    }


def _router_feature_fields(
    *,
    raw_text: str,
    target_fields: Mapping[str, Any],
    feature_fields: Mapping[str, Any],
    provenance_meta: Mapping[str, Any],
) -> dict[str, Any]:
    route_compute = _route_compute_signal_summary(feature_fields, provenance_meta)
    operator_summary = _operator_pattern_summary(feature_fields, provenance_meta)
    proof_summary = _extract_mapping(feature_fields, "proof_obligation_summary")
    retrieval_summary = _extract_mapping(feature_fields, "retrieval_summary")
    verifier_summary = _extract_mapping(feature_fields, "decomposed_signals", "verifier_decomposition")
    return {
        "problem_length": len(raw_text),
        "problem_type_distribution": _normalize_score_distribution(target_fields.get("problem_type")),
        "archetype_distribution": _normalize_score_distribution(target_fields.get("archetypes")),
        "route_compute_signals": route_compute,
        "operator_patterns": operator_summary,
        "proof_obligation_summary": _coerce_json_value(proof_summary),
        "retrieval_summary": _coerce_json_value(retrieval_summary),
        "verifier_summary": _coerce_json_value(verifier_summary),
        "difficulty_seed": _extract_float(feature_fields, "difficulty_intensity"),
        "retrieval_need": route_compute.get("retrieval_need", 0.0),
        "repair_need": route_compute.get("repair_need", 0.0),
        "proof_burden": route_compute.get("proof_burden", 0.0),
    }


def load_archetype_training_dataset(
    dataset_path: str | Path,
) -> list[ParsedProblem | UnifiedTrainingExample | Mapping[str, Any]]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    if path.is_dir():
        records_path = path / "records.jsonl"
        manifest_path = path / "manifest.json"
        if records_path.exists():
            return [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if manifest_path.exists():
            return load_archetype_training_dataset(manifest_path)
        raise ValueError(f"Directory does not contain records.jsonl or manifest.json: {path}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix != ".json":
        raise ValueError(f"Unsupported dataset file type: {path.suffix}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if payload.get("artifact_kind") == "offline_training_corpus":
            sibling_records = path.with_name("records.jsonl")
            if sibling_records.exists():
                return load_archetype_training_dataset(sibling_records)
            raise ValueError("Offline training manifest found without sibling records.jsonl")
        if "split_bundle" in payload and isinstance(payload["split_bundle"], Mapping):
            split_bundle = payload["split_bundle"]
            rows: list[Mapping[str, Any]] = []
            for key in ("train", "valid", "calibration"):
                value = split_bundle.get(key, [])
                if isinstance(value, list):
                    rows.extend(value)
            return rows
        if "records" in payload and isinstance(payload["records"], list):
            return list(payload["records"])
        if "examples" in payload and isinstance(payload["examples"], list):
            return list(payload["examples"])
        if "problems" in payload and isinstance(payload["problems"], list):
            return list(payload["problems"])
        raise ValueError("JSON dataset must contain a top-level 'records', 'examples', or 'problems' list")
    if isinstance(payload, list):
        return list(payload)
    raise ValueError("Unsupported JSON dataset payload")


def convert_problem_to_archetype_example(
    problem: ParsedProblem,
    *,
    supported_archetypes: Sequence[str] | None = None,
    preserve_problem_metadata: bool = True,
) -> ArchetypeTrainingExample:
    supported = set(supported_archetypes or get_supported_archetypes())
    labels, scores, _ = _normalize_label_payload(problem.likely_archetypes, supported=supported)
    metadata: dict[str, Any] = {}
    feature_fields: dict[str, Any] = {}
    if preserve_problem_metadata:
        metadata = {
            "difficulty_seed": problem.difficulty_seed,
            "parse_quality": dict(problem.parse_quality),
            "symmetries": list(problem.symmetries),
            "parity_cues": list(problem.parity_cues),
            "integrality_constraints": list(problem.integrality_constraints),
            "proof_targets": list(getattr(problem, "proof_targets", []) or []),
            "proof_obligation_hints": list(getattr(problem, "proof_obligation_hints", []) or []),
            "original_metadata": dict(problem.metadata),
        }
        feature_fields = {
            "symmetry_count": len(problem.symmetries),
            "parity_cue_count": len(problem.parity_cues),
            "integrality_constraint_count": len(problem.integrality_constraints),
            "proof_target_count": len(getattr(problem, "proof_targets", []) or []),
            "proof_obligation_hint_count": len(getattr(problem, "proof_obligation_hints", []) or []),
            "likely_archetype_count": len(problem.likely_archetypes),
        }
    return ArchetypeTrainingExample(
        problem_id=problem.problem_id,
        raw_text=_normalize_text(problem.raw_text),
        domain=problem.domain.value if hasattr(problem.domain, "value") else str(problem.domain),
        target=problem.target,
        answer_type=problem.answer_type,
        archetype_labels=labels,
        label_scores=scores,
        source_metadata=metadata,
        feature_fields=feature_fields,
        supervision_summary={},
    )


def _unified_to_archetype_example(
    record: UnifiedTrainingExample,
    *,
    supported_archetypes: Sequence[str],
    preserve_problem_metadata: bool,
) -> tuple[ArchetypeTrainingExample | None, list[str]]:
    if record.task_family != UnifiedTaskFamily.ROUTER:
        return None, []

    provenance_meta = dict(record.provenance.metadata)
    trace_meta = provenance_meta.get("trace_metadata", {}) if isinstance(provenance_meta.get("trace_metadata"), Mapping) else {}
    raw_text = _normalize_text(
        str(
            record.input_fields.get("problem_text")
            or trace_meta.get("problem_text")
            or ""
        )
    )
    labels, scores, unsupported = _normalize_label_payload(record.target_fields.get("archetypes"), supported=set(supported_archetypes))

    record_features = _coerce_json_value(record.feature_fields)
    provenance_payload = _coerce_json_value(provenance_meta)
    source_meta = {
        "record_targets": _coerce_json_value(record.target_fields),
        "record_features": record_features,
        "provenance_metadata": provenance_payload,
    }
    if preserve_problem_metadata:
        source_meta["builder_metadata"] = _coerce_json_value(record.metadata)

    router_features = _router_feature_fields(
        raw_text=raw_text,
        target_fields=dict(record.target_fields),
        feature_fields=dict(record.feature_fields),
        provenance_meta=provenance_meta,
    )
    supervision_summary = _supervision_summary(
        feature_fields=dict(record.feature_fields),
        target_fields=dict(record.target_fields),
        provenance_meta=provenance_meta,
    )

    example = ArchetypeTrainingExample(
        problem_id=record.problem_id,
        raw_text=raw_text,
        domain=_normalize_text(str(trace_meta.get("domain") or _top_key(record.target_fields.get("problem_type")) or "unknown")) or "unknown",
        target=_normalize_text(str(trace_meta.get("target") or "")),
        answer_type=_normalize_text(str(trace_meta.get("answer_type") or "")),
        archetype_labels=labels,
        label_scores=scores,
        source_split=_coerce_split(record.split),
        source_artifact_id=record.provenance.upstream_artifact_id,
        source_record_ids=sorted({record.record_id, record.provenance.source_record_id}),
        source_metadata=source_meta,
        feature_fields=router_features,
        supervision_summary=supervision_summary,
    )
    return example, unsupported


def _coerce_source_item(
    item: ParsedProblem | UnifiedTrainingExample | OfflineTrainingArtifact | ArchetypeTrainingExample | Mapping[str, Any],
    *,
    supported_archetypes: Sequence[str],
    preserve_problem_metadata: bool,
) -> tuple[list[ArchetypeTrainingExample], dict[str, list[str]]]:
    unsupported: dict[str, list[str]] = {}
    if isinstance(item, OfflineTrainingArtifact):
        examples: list[ArchetypeTrainingExample] = []
        for record in item.records:
            converted, unsupported_labels = _unified_to_archetype_example(
                record,
                supported_archetypes=supported_archetypes,
                preserve_problem_metadata=preserve_problem_metadata,
            )
            if converted is not None:
                examples.append(converted)
            if unsupported_labels:
                unsupported[record.problem_id] = unsupported_labels
        return examples, unsupported

    if isinstance(item, UnifiedTrainingExample):
        converted, unsupported_labels = _unified_to_archetype_example(
            item,
            supported_archetypes=supported_archetypes,
            preserve_problem_metadata=preserve_problem_metadata,
        )
        if converted is None:
            return [], {}
        if unsupported_labels:
            unsupported[item.problem_id] = unsupported_labels
        return [converted], unsupported

    if isinstance(item, ParsedProblem):
        return [
            convert_problem_to_archetype_example(
                item,
                supported_archetypes=supported_archetypes,
                preserve_problem_metadata=preserve_problem_metadata,
            )
        ], {}

    if isinstance(item, ArchetypeTrainingExample):
        return [item], {}

    if isinstance(item, Mapping):
        payload = dict(item)
        if "task_family" in payload and "provenance" in payload:
            record = UnifiedTrainingExample.model_validate(payload)
            return _coerce_source_item(
                record,
                supported_archetypes=supported_archetypes,
                preserve_problem_metadata=preserve_problem_metadata,
            )
        if "raw_text" in payload and "problem_id" in payload and "domain" in payload:
            return [ArchetypeTrainingExample.model_validate(payload)], {}
        problem = ParsedProblem.model_validate(payload)
        return _coerce_source_item(
            problem,
            supported_archetypes=supported_archetypes,
            preserve_problem_metadata=preserve_problem_metadata,
        )

    raise TypeError(f"Unsupported archetype training source type: {type(item)!r}")


def _merge_examples(
    examples: Sequence[ArchetypeTrainingExample],
) -> tuple[list[ArchetypeTrainingExample], list[str], list[str]]:
    grouped: dict[str, list[ArchetypeTrainingExample]] = {}
    for example in examples:
        grouped.setdefault(example.problem_id, []).append(example)

    merged: list[ArchetypeTrainingExample] = []
    duplicate_problem_ids: list[str] = []
    split_conflict_problem_ids: list[str] = []

    for problem_id, items in sorted(grouped.items()):
        if len(items) > 1:
            duplicate_problem_ids.append(problem_id)
        raw_text = max((item.raw_text for item in items), key=len, default="")
        domain = next((item.domain for item in items if item.domain and item.domain != "unknown"), "unknown")
        target = next((item.target for item in items if item.target), "")
        answer_type = next((item.answer_type for item in items if item.answer_type), "")
        label_set = sorted({label for item in items for label in item.archetype_labels})
        label_scores: dict[str, float] = {}
        for item in items:
            for label, score in item.label_scores.items():
                label_scores[label] = max(label_scores.get(label, 0.0), float(score))
        source_splits = sorted({item.source_split.value for item in items if item.source_split is not None})
        if len(source_splits) > 1:
            split_conflict_problem_ids.append(problem_id)

        merged_feature_fields: dict[str, Any] = {}
        for item in items:
            for key, value in item.feature_fields.items():
                if isinstance(value, (int, float)):
                    merged_feature_fields[key] = max(float(merged_feature_fields.get(key, 0.0)), float(value))
                elif isinstance(value, Mapping):
                    current = merged_feature_fields.get(key, {})
                    if not isinstance(current, Mapping):
                        current = {}
                    current_dict = dict(current)
                    for sub_key, sub_val in value.items():
                        if isinstance(sub_val, (int, float)):
                            current_dict[str(sub_key)] = max(float(current_dict.get(str(sub_key), 0.0)), float(sub_val))
                        else:
                            current_dict[str(sub_key)] = _coerce_json_value(sub_val)
                    merged_feature_fields[key] = current_dict
                else:
                    merged_feature_fields[key] = _coerce_json_value(value)

        merged_supervision: dict[str, Any] = {}
        for item in items:
            for key, value in item.supervision_summary.items():
                if isinstance(value, (int, float)):
                    merged_supervision[key] = max(float(merged_supervision.get(key, 0.0)), float(value))
                elif isinstance(value, Mapping):
                    current = merged_supervision.get(key, {})
                    if not isinstance(current, Mapping):
                        current = {}
                    current_dict = dict(current)
                    for sub_key, sub_val in value.items():
                        if isinstance(sub_val, (int, float)):
                            current_dict[str(sub_key)] = max(float(current_dict.get(str(sub_key), 0.0)), float(sub_val))
                        else:
                            current_dict[str(sub_key)] = _coerce_json_value(sub_val)
                    merged_supervision[key] = current_dict
                else:
                    merged_supervision[key] = _coerce_json_value(value)

        merged_metadata = {
            "merged_example_count": len(items),
            "source_artifact_ids": sorted({item.source_artifact_id for item in items if item.source_artifact_id}),
            "source_splits": source_splits,
            "source_metadata": [_coerce_json_value(item.source_metadata) for item in items],
        }
        merged.append(
            ArchetypeTrainingExample(
                problem_id=problem_id,
                raw_text=raw_text,
                domain=domain,
                target=target,
                answer_type=answer_type,
                archetype_labels=label_set,
                label_scores={key: label_scores[key] for key in sorted(label_scores)},
                source_split=items[0].source_split if len(source_splits) == 1 else None,
                source_artifact_id=next((item.source_artifact_id for item in items if item.source_artifact_id), None),
                source_record_ids=sorted({record_id for item in items for record_id in item.source_record_ids}),
                source_metadata=merged_metadata,
                feature_fields=_coerce_json_value(merged_feature_fields),
                supervision_summary=_coerce_json_value(merged_supervision),
            )
        )

    return merged, duplicate_problem_ids, split_conflict_problem_ids


def _load_split_manifest(path: str | Path) -> dict[str, DatasetSplit]:
    payload = load_archetype_training_dataset(path)
    entries: dict[str, DatasetSplit] = {}
    for item in payload:
        entry = SplitManifestEntry.model_validate(item)
        entries[entry.problem_id] = entry.split
    return entries


def _resolve_example_split(
    example: ArchetypeTrainingExample,
    *,
    config: ArchetypeTrainingConfig,
    manifest_splits: Mapping[str, DatasetSplit] | None,
) -> DatasetSplit:
    if config.split_strategy == SplitStrategy.MANIFEST:
        if manifest_splits and example.problem_id in manifest_splits:
            return manifest_splits[example.problem_id]
        return _hash_split(example.problem_id, config)
    if config.split_strategy == SplitStrategy.SOURCE and example.source_split is not None:
        return example.source_split
    return _hash_split(example.problem_id, config)


def _split_summary(name: str, examples: Sequence[ArchetypeTrainingExample]) -> SplitSummary:
    label_counter = Counter(label for example in examples for label in example.archetype_labels)
    return SplitSummary(
        split_name=name,
        count=len(examples),
        label_frequencies=dict(sorted(label_counter.items())),
        problem_ids=[example.problem_id for example in examples],
    )


def _build_split_bundle(
    examples: Sequence[ArchetypeTrainingExample],
    *,
    config: ArchetypeTrainingConfig,
) -> ArchetypeSplitBundle:
    manifest_splits = _load_split_manifest(config.split_manifest_path) if config.split_manifest_path else None
    assigned: list[tuple[DatasetSplit, ArchetypeTrainingExample]] = []
    for example in examples:
        split = _resolve_example_split(example, config=config, manifest_splits=manifest_splits)
        assigned.append((split, example.model_copy(update={"source_split": split})))

    train = [example for split, example in assigned if split == DatasetSplit.TRAIN]
    valid = [example for split, example in assigned if split == DatasetSplit.VALID]
    calibration = [example for split, example in assigned if split == DatasetSplit.CALIBRATION]

    train.sort(key=lambda item: item.problem_id)
    valid.sort(key=lambda item: item.problem_id)
    calibration.sort(key=lambda item: item.problem_id)

    return ArchetypeSplitBundle(
        train=train,
        valid=valid,
        calibration=calibration,
        split_strategy=config.split_strategy,
        manifest_source=str(config.split_manifest_path) if config.split_manifest_path else None,
        train_summary=_split_summary("train", train),
        valid_summary=_split_summary("valid", valid),
        calibration_summary=_split_summary("calibration", calibration),
    )


def _validate_examples(
    examples: Sequence[ArchetypeTrainingExample],
    *,
    config: ArchetypeTrainingConfig,
    supported_archetypes: Sequence[str],
    duplicate_problem_ids: Sequence[str],
    split_conflict_problem_ids: Sequence[str],
    unsupported_labels: Mapping[str, list[str]],
    split_bundle: ArchetypeSplitBundle | None = None,
) -> tuple[DatasetValidationReport, ReadinessReport]:
    supported = set(supported_archetypes)
    label_counter = Counter(label for example in examples for label in example.archetype_labels)
    empty_label_problem_ids = sorted(example.problem_id for example in examples if not example.archetype_labels)
    unsupported_filtered = {
        problem_id: sorted({label for label in labels if label not in supported})
        for problem_id, labels in unsupported_labels.items()
        if any(label not in supported for label in labels)
    }
    notes: list[str] = []
    if duplicate_problem_ids:
        notes.append("Problem-level duplicates were merged deterministically before splitting.")
    if split_conflict_problem_ids:
        notes.append("Conflicting source splits were neutralized at the problem level before final split assignment.")
    if any(example.feature_fields for example in examples):
        notes.append("Richer routing supervision fields were preserved for offline archetype training.")
    if config.require_signal_rich_examples and not any(example.supervision_summary for example in examples):
        notes.append("Signal-rich example requirement was enabled but no richer supervision summaries were present.")

    readiness = ReadinessReport(
        min_examples_ok=len(examples) >= config.min_examples,
        valid_examples_ok=(split_bundle is None or len(split_bundle.valid) >= config.min_valid_examples),
        calibration_examples_ok=(split_bundle is None or len(split_bundle.calibration) >= config.min_calibration_examples),
        unlabeled_policy_ok=(config.allow_unlabeled_examples or not empty_label_problem_ids),
    )
    if config.require_signal_rich_examples and not any(example.supervision_summary for example in examples):
        readiness = readiness.model_copy(update={"ready_for_training": False})
    readiness = readiness.model_copy(
        update={
            "ready_for_training": all(
                (
                    readiness.min_examples_ok,
                    readiness.valid_examples_ok,
                    readiness.calibration_examples_ok,
                    readiness.unlabeled_policy_ok,
                    not unsupported_filtered,
                    (not config.require_signal_rich_examples or any(example.supervision_summary for example in examples)),
                )
            )
        }
    )

    validation = DatasetValidationReport(
        valid=readiness.ready_for_training,
        num_examples=len(examples),
        num_unique_problem_ids=len({example.problem_id for example in examples}),
        duplicate_problem_ids=sorted(set(duplicate_problem_ids)),
        split_conflict_problem_ids=sorted(set(split_conflict_problem_ids)),
        unsupported_labels=unsupported_filtered,
        empty_label_problem_ids=empty_label_problem_ids,
        stats=DatasetLabelStats(
            total_examples=len(examples),
            labeled_examples=sum(1 for example in examples if example.archetype_labels),
            unlabeled_examples=sum(1 for example in examples if not example.archetype_labels),
            label_frequencies=dict(sorted(label_counter.items())),
        ),
        notes=notes,
    )
    return validation, readiness


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_stable_json(row) + "\n")


def _save_training_manifest(
    *,
    manifest: ArchetypeTrainingManifest,
    output_dir: str | Path,
    trained_model_path: str | None = None,
) -> EmittedArtifactPaths:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    run_dir = base / manifest.metadata.artifact_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "manifest.json"
    train_path = run_dir / "train.jsonl"
    valid_path = run_dir / "valid.jsonl"
    calibration_path = run_dir / "calibration.jsonl"
    validation_path = run_dir / "dataset_validation.json"
    summary_path = run_dir / "split_summary.json"

    manifest_path.write_text(_stable_json(manifest.model_dump(mode="json")) + "\n", encoding="utf-8")
    if manifest.config.emit_jsonl:
        _write_jsonl(train_path, (example.model_dump(mode="json") for example in manifest.split_bundle.train))
        _write_jsonl(valid_path, (example.model_dump(mode="json") for example in manifest.split_bundle.valid))
        _write_jsonl(calibration_path, (example.model_dump(mode="json") for example in manifest.split_bundle.calibration))
    validation_path.write_text(_stable_json(manifest.validation.model_dump(mode="json")) + "\n", encoding="utf-8")
    if manifest.config.write_summary_json:
        summary_path.write_text(
            _stable_json(
                {
                    "train": manifest.split_bundle.train_summary.model_dump(mode="json"),
                    "valid": manifest.split_bundle.valid_summary.model_dump(mode="json"),
                    "calibration": manifest.split_bundle.calibration_summary.model_dump(mode="json"),
                }
            )
            + "\n",
            encoding="utf-8",
        )

    return EmittedArtifactPaths(
        run_dir=str(run_dir),
        manifest_path=str(manifest_path),
        train_manifest_path=str(train_path) if manifest.config.emit_jsonl else None,
        valid_manifest_path=str(valid_path) if manifest.config.emit_jsonl else None,
        calibration_manifest_path=str(calibration_path) if manifest.config.emit_jsonl else None,
        split_summary_path=str(summary_path) if manifest.config.write_summary_json else None,
        dataset_validation_path=str(validation_path),
        trained_model_path=trained_model_path,
    )


def build_archetype_training_manifest(
    sources: Sequence[ParsedProblem | UnifiedTrainingExample | OfflineTrainingArtifact | ArchetypeTrainingExample | Mapping[str, Any]],
    *,
    config: ArchetypeTrainingConfig | None = None,
    supported_archetypes: Sequence[str] | None = None,
) -> tuple[ArchetypeTrainingManifest | None, DatasetValidationReport, list[str]]:
    resolved_config = config or ArchetypeTrainingConfig()
    supported = tuple(supported_archetypes or get_supported_archetypes())

    normalized_examples: list[ArchetypeTrainingExample] = []
    unsupported_labels: dict[str, list[str]] = {}
    for source in sources:
        examples, source_unsupported = _coerce_source_item(
            source,
            supported_archetypes=supported,
            preserve_problem_metadata=resolved_config.preserve_problem_metadata,
        )
        normalized_examples.extend(examples)
        unsupported_labels.update(source_unsupported)

    if not normalized_examples:
        validation = DatasetValidationReport(valid=False, notes=["No archetype-training examples were available."])
        return None, validation, ["No router-compatible offline records or parsed problems were found."]

    merged_examples, duplicate_problem_ids, split_conflict_problem_ids = _merge_examples(normalized_examples)
    split_bundle = _build_split_bundle(merged_examples, config=resolved_config)
    validation, readiness = _validate_examples(
        merged_examples,
        config=resolved_config,
        supported_archetypes=supported,
        duplicate_problem_ids=duplicate_problem_ids,
        split_conflict_problem_ids=split_conflict_problem_ids,
        unsupported_labels=unsupported_labels,
        split_bundle=split_bundle,
    )

    config_digest = _stable_hash("archetype_training_config", resolved_config.model_dump(mode="json"))
    source_digest = _stable_hash(
        "archetype_training_sources",
        [
            {
                "problem_id": example.problem_id,
                "source_record_ids": example.source_record_ids,
                "source_split": example.source_split.value if example.source_split else None,
                "labels": example.archetype_labels,
                "feature_fields": example.feature_fields,
                "supervision_summary": example.supervision_summary,
            }
            for example in merged_examples
        ],
    )
    artifact_id = _stable_hash(
        "archetype_training_manifest",
        {
            "run_name": resolved_config.run_name,
            "artifact_version": resolved_config.artifact_version,
            "config_digest": config_digest,
            "problem_ids": [example.problem_id for example in merged_examples],
        },
    )

    manifest = ArchetypeTrainingManifest(
        metadata=ArchetypeTrainingArtifactMetadata(
            artifact_version=resolved_config.artifact_version,
            schema_version=resolved_config.schema_version,
            artifact_id=artifact_id,
            run_name=resolved_config.run_name,
            problem_count=len(merged_examples),
            split_counts={
                DatasetSplit.TRAIN.value: len(split_bundle.train),
                DatasetSplit.VALID.value: len(split_bundle.valid),
                DatasetSplit.CALIBRATION.value: len(split_bundle.calibration),
            },
            label_frequencies=dict(sorted(validation.stats.label_frequencies.items())),
            supported_labels=tuple(sorted(supported)),
            source_artifact_ids=tuple(sorted({example.source_artifact_id for example in merged_examples if example.source_artifact_id})),
            config_digest=config_digest,
            source_digest=source_digest,
            readiness=readiness,
        ),
        config=resolved_config,
        validation=validation,
        split_bundle=split_bundle,
    )
    return manifest, validation, list(validation.notes)


def load_archetype_training_manifest(path: str | Path) -> ArchetypeTrainingManifest:
    input_path = Path(path)
    if input_path.is_dir():
        manifest_path = input_path / "manifest.json"
    elif input_path.name == "manifest.json":
        manifest_path = input_path
    else:
        raise ValueError(f"Unsupported artifact path: {path}")
    return ArchetypeTrainingManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))


def archetype_examples_by_split(
    manifest: ArchetypeTrainingManifest,
) -> dict[DatasetSplit, tuple[ArchetypeTrainingExample, ...]]:
    return {
        DatasetSplit.TRAIN: tuple(manifest.split_bundle.train),
        DatasetSplit.VALID: tuple(manifest.split_bundle.valid),
        DatasetSplit.CALIBRATION: tuple(manifest.split_bundle.calibration),
    }


def router_training_rows(manifest: ArchetypeTrainingManifest) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for split, examples in archetype_examples_by_split(manifest).items():
        for example in examples:
            rows.append(
                {
                    "split": split.value,
                    "problem_id": example.problem_id,
                    "problem_text": example.raw_text,
                    "domain": example.domain,
                    "target": example.target,
                    "answer_type": example.answer_type,
                    "archetype_labels": list(example.archetype_labels),
                    "label_scores": dict(example.label_scores),
                    "source_artifact_id": example.source_artifact_id,
                    "source_record_ids": list(example.source_record_ids),
                    "source_metadata": _coerce_json_value(example.source_metadata),
                    "feature_fields": _coerce_json_value(example.feature_fields),
                    "supervision_summary": _coerce_json_value(example.supervision_summary),
                }
            )
    rows.sort(key=lambda row: (row["split"], row["problem_id"]))
    return tuple(rows)


def run_archetype_training(
    sources: Sequence[ParsedProblem | UnifiedTrainingExample | OfflineTrainingArtifact | ArchetypeTrainingExample | Mapping[str, Any]] | None = None,
    *,
    config: ArchetypeTrainingConfig | None = None,
    supported_archetypes: Sequence[str] | None = None,
    external_trainer: ExternalTrainerProtocol | None = None,
) -> ArchetypeTrainingResult:
    resolved_config = config or ArchetypeTrainingConfig()
    resolved_sources = list(sources or [])
    if resolved_config.dataset_path:
        resolved_sources.extend(load_archetype_training_dataset(resolved_config.dataset_path))

    if not resolved_sources:
        validation = DatasetValidationReport(valid=False, notes=["No dataset sources were provided."])
        return ArchetypeTrainingResult(
            status=ArchetypeTrainingStatus.MISSING_INPUT,
            runtime_boundary=RuntimeBoundary.MANIFEST_ONLY,
            config=resolved_config,
            validation=validation,
            notes=["Provide parsed problems, router examples, or an offline training artifact."],
        )

    manifest, validation, notes = build_archetype_training_manifest(
        resolved_sources,
        config=resolved_config,
        supported_archetypes=supported_archetypes,
    )
    if manifest is None:
        return ArchetypeTrainingResult(
            status=ArchetypeTrainingStatus.MISSING_INPUT,
            runtime_boundary=RuntimeBoundary.MANIFEST_ONLY,
            config=resolved_config,
            validation=validation,
            notes=notes,
        )

    artifacts = _save_training_manifest(manifest=manifest, output_dir=resolved_config.output_dir)
    if not validation.valid:
        return ArchetypeTrainingResult(
            status=ArchetypeTrainingStatus.INVALID_DATASET,
            runtime_boundary=RuntimeBoundary.MANIFEST_ONLY,
            config=resolved_config,
            validation=validation,
            split_bundle=manifest.split_bundle,
            artifacts=artifacts,
            notes=notes,
        )

    if external_trainer is None:
        return ArchetypeTrainingResult(
            status=ArchetypeTrainingStatus.READY,
            runtime_boundary=RuntimeBoundary.EXTERNAL_TRAINER_REQUIRED,
            config=resolved_config,
            validation=validation,
            split_bundle=manifest.split_bundle,
            artifacts=artifacts,
            notes=notes + ["Dataset manifests were emitted; an external trainer can now consume train/valid safely."],
        )

    run_dir = Path(artifacts.run_dir)
    try:
        trainer_report = dict(
            external_trainer(
                train_examples=list(manifest.split_bundle.train),
                eval_examples=list(manifest.split_bundle.valid),
                config=resolved_config,
                run_dir=run_dir,
            )
        )
    except Exception as exc:
        return ArchetypeTrainingResult(
            status=ArchetypeTrainingStatus.UNAVAILABLE_RUNTIME,
            runtime_boundary=RuntimeBoundary.EXTERNAL_TRAINER_REQUIRED,
            config=resolved_config,
            validation=validation,
            split_bundle=manifest.split_bundle,
            artifacts=artifacts,
            notes=notes + [f"External trainer failed: {exc}"],
        )

    trained_model_path = None
    for candidate_key in ("trained_model_path", "model_path", "checkpoint_path"):
        candidate = trainer_report.get(candidate_key)
        if candidate:
            trained_model_path = str(candidate)
            break
    trainer_report_path = run_dir / "trainer_report.json"
    trainer_report_path.write_text(_stable_json(_coerce_json_value(trainer_report)) + "\n", encoding="utf-8")
    manifest_with_trainer = manifest.model_copy(
        update={
            "metadata": manifest.metadata.model_copy(
                update={
                    "checkpoint_references": (
                        {"trained_model_path": trained_model_path} if trained_model_path else {}
                    )
                }
            ),
            "trainer_report": _coerce_json_value(trainer_report),
        }
    )
    if artifacts.manifest_path:
        Path(artifacts.manifest_path).write_text(
            _stable_json(manifest_with_trainer.model_dump(mode="json")) + "\n",
            encoding="utf-8",
        )
    updated_artifacts = artifacts.model_copy(
        update={
            "trained_model_path": trained_model_path,
            "trainer_report_path": str(trainer_report_path),
        }
    )

    return ArchetypeTrainingResult(
        status=ArchetypeTrainingStatus.COMPLETED,
        runtime_boundary=RuntimeBoundary.TRAINER_EXECUTED,
        config=resolved_config,
        validation=validation,
        split_bundle=manifest.split_bundle,
        artifacts=updated_artifacts,
        trainer_report=trainer_report,
        notes=notes,
    )


__all__ = [
    "ARTIFACT_KIND",
    "DEFAULT_ARTIFACT_VERSION",
    "SCHEMA_VERSION",
    "ArchetypeSplitBundle",
    "ArchetypeTrainingArtifactMetadata",
    "ArchetypeTrainingConfig",
    "ArchetypeTrainingExample",
    "ArchetypeTrainingManifest",
    "ArchetypeTrainingResult",
    "ArchetypeTrainingStatus",
    "DatasetLabelStats",
    "DatasetSplit",
    "DatasetValidationReport",
    "EmittedArtifactPaths",
    "ExternalTrainerProtocol",
    "ReadinessReport",
    "RuntimeBoundary",
    "SplitManifestEntry",
    "SplitStrategy",
    "SplitSummary",
    "archetype_examples_by_split",
    "build_archetype_training_manifest",
    "convert_problem_to_archetype_example",
    "load_archetype_training_manifest",
    "load_archetype_training_dataset",
    "router_training_rows",
    "run_archetype_training",
]