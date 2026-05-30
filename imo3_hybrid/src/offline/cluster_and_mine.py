from __future__ import annotations

"""Deterministic clustered operator mining artifact construction.

Merged implementation:
- keeps the older richer cluster/promotion structure, representative-trace
  selection, and miner integration
- keeps the newer repair-pass intent that richer runtime evidence must stay
  active in mining and promotion safety rather than passive archived metadata
"""

import hashlib
import json
from collections import Counter, defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.common.constants import (
    OFFLINE_ARTIFACT_VERSION,
    OFFLINE_SPLIT_RATIOS,
    OFFLINE_SPLIT_SALT,
)
from src.common.schemas import BranchTrace, FailureType
from src.operators.operator_miner import OperatorMiner
from src.operators.operator_types import (
    OperatorMiningCandidate,
    OperatorMiningReport,
    operator_family_for_name,
)
from src.offline.trace_distillation import DistilledTraceArtifact, DistilledTraceRecord


SCHEMA_VERSION = "cluster_and_mine.v4"
ARTIFACT_KIND = "clustered_operator_mining_artifact"
DEFAULT_ARTIFACT_VERSION = OFFLINE_ARTIFACT_VERSION


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        str_strip_whitespace=True,
        populate_by_name=True,
    )


class SplitName(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    CALIBRATION = "calibration"


class MiningSourceType(str, Enum):
    RAW_BRANCH_TRACE = "raw_branch_trace"
    DISTILLED_TRACE_RECORD = "distilled_trace_record"


class MiningConfig(StrictModel):
    artifact_version: str = DEFAULT_ARTIFACT_VERSION
    schema_version: str = SCHEMA_VERSION
    split_ratios: tuple[float, float, float] = OFFLINE_SPLIT_RATIOS
    split_salt: str = OFFLINE_SPLIT_SALT

    min_cluster_support: int = Field(default=2, ge=1)
    min_distinct_problem_support: int = Field(default=1, ge=1)
    min_symbolic_success_rate: float = Field(default=0.55, ge=0.0, le=1.0)
    min_mean_verifier_score: float = Field(default=0.55, ge=0.0, le=1.0)
    min_operator_purity: float = Field(default=0.60, ge=0.0, le=1.0)
    max_failure_rate: float = Field(default=0.45, ge=0.0, le=1.0)

    min_mean_logical_consistency: float = Field(default=0.45, ge=0.0, le=1.0)
    min_mean_prm_prefix_quality: float = Field(default=0.35, ge=0.0, le=1.0)
    min_mean_operator_reliability: float = Field(default=0.25, ge=0.0, le=1.0)
    min_mean_retrieval_compatibility: float = Field(default=0.15, ge=0.0, le=1.0)
    max_mean_proof_obligation_burden: float = Field(default=0.70, ge=0.0, le=1.0)
    min_mean_discharge_fraction: float = Field(default=0.20, ge=0.0, le=1.0)

    max_step_count_bucket: int = Field(default=12, ge=1)
    sequence_prefix: int = Field(default=4, ge=1)

    require_holdout_for_promotion: bool = False
    min_holdout_support: int = Field(default=0, ge=0)

    miner_min_support: int = Field(default=3, ge=1)
    miner_min_symbolic_success_rate: float = Field(default=0.55, ge=0.0, le=1.0)
    miner_min_mean_verifier_gain: float = Field(default=0.02)

    representative_symbolic_bonus: float = Field(default=0.10, ge=0.0, le=1.0)
    representative_verifier_weight: float = Field(default=0.45, ge=0.0, le=1.0)
    representative_signal_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    representative_reliability_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    representative_proof_weight: float = Field(default=0.15, ge=0.0, le=1.0)


class RichSignalBundle(StrictModel):
    verifier_score: float = 0.0
    logical_consistency: float = 0.0
    symbolic_agreement: float = 0.0
    completeness: float = 0.0
    repairability: float = 0.0
    answer_correctness_likelihood: float = 0.0
    prm_prefix_quality: float = 0.0
    prm_step_quality_mean: float = 0.0
    retrieval_support: float = 0.0
    retrieval_compatibility: float = 0.0
    operator_reliability: float = 0.0
    tool_consistency: float = 0.0
    discharge_fraction: float = 0.0
    open_obligation_burden: float = 0.0
    operator_purity: float = 0.0


class TraceArtifactRecord(StrictModel):
    record_id: str
    source_type: MiningSourceType
    source_record_id: str
    source_artifact_id: str | None = None
    problem_id: str
    branch_id: str
    split: SplitName
    classification: str = ""
    quality_tier: str = ""
    answer: str | None = None
    answer_canonical: str | None = None
    archetype_used: str | None = None
    domain_hint: str | None = None
    operator_sequence: tuple[str, ...] = Field(default_factory=tuple)
    operator_families: tuple[str, ...] = Field(default_factory=tuple)
    step_count: int = Field(default=0, ge=0)
    step_count_bucket: int = Field(default=0, ge=0)
    symbolic_valid: bool = False
    verifier_score: float = Field(default=0.0, ge=0.0, le=1.0)
    branch_score: float = Field(default=0.0, ge=0.0, le=1.0)
    tool_consistency: float = Field(default=0.0, ge=0.0, le=1.0)
    repaired: bool = False
    repair_count: int = Field(default=0, ge=0)
    retrieval_used: bool = False
    failure_type: str | None = None
    signature: str
    cluster_key: str
    route_snapshot: dict[str, Any] = Field(default_factory=dict)
    steps: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    signals: RichSignalBundle = Field(default_factory=RichSignalBundle)
    verifier_decomposition: dict[str, float] = Field(default_factory=dict)
    proof_obligation_summary: dict[str, Any] = Field(default_factory=dict)
    route_compute_signals: dict[str, float] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ---- compatibility surface for richer direct typed access ----
    @property
    def retrieval_compatibility(self) -> float:
        return float(self.signals.retrieval_compatibility)

    @property
    def retrieval_support(self) -> float:
        return float(self.signals.retrieval_support)

    @property
    def logical_consistency(self) -> float:
        return float(self.signals.logical_consistency)

    @property
    def completeness(self) -> float:
        return float(self.signals.completeness)

    @property
    def repairability(self) -> float:
        return float(self.signals.repairability)

    @property
    def answer_correctness_likelihood(self) -> float:
        return float(self.signals.answer_correctness_likelihood)

    @property
    def prm_prefix_quality(self) -> float:
        return float(self.signals.prm_prefix_quality)

    @property
    def prm_step_quality_mean(self) -> float:
        return float(self.signals.prm_step_quality_mean)

    @property
    def operator_reliability(self) -> float:
        return float(self.signals.operator_reliability)

    @property
    def symbolic_agreement(self) -> float:
        return float(self.signals.symbolic_agreement)

    @property
    def discharge_fraction(self) -> float:
        return float(self.signals.discharge_fraction)

    @property
    def open_obligation_burden(self) -> float:
        return float(self.signals.open_obligation_burden)

class ClusterPromotionSafety(StrictModel):
    min_cluster_support_ok: bool = False
    min_problem_support_ok: bool = False
    symbolic_success_ok: bool = False
    verifier_score_ok: bool = False
    operator_purity_ok: bool = False
    failure_rate_ok: bool = False
    logical_consistency_ok: bool = False
    prm_prefix_quality_ok: bool = False
    operator_reliability_ok: bool = False
    retrieval_compatibility_ok: bool = True
    proof_burden_ok: bool = True
    discharge_fraction_ok: bool = True
    holdout_support_ok: bool = True

    @property
    def promotion_safe(self) -> bool:
        return all((
            self.min_cluster_support_ok,
            self.min_problem_support_ok,
            self.symbolic_success_ok,
            self.verifier_score_ok,
            self.operator_purity_ok,
            self.failure_rate_ok,
            self.logical_consistency_ok,
            self.prm_prefix_quality_ok,
            self.operator_reliability_ok,
            self.retrieval_compatibility_ok,
            self.proof_burden_ok,
            self.discharge_fraction_ok,
            self.holdout_support_ok,
        ))


class ClusterSummary(StrictModel):
    cluster_id: str
    cluster_key: str
    member_count: int = 0
    member_count_by_split: dict[str, int] = Field(default_factory=dict)
    distinct_problem_count: int = 0
    distinct_problem_count_by_split: dict[str, int] = Field(default_factory=dict)
    dominant_archetype: str | None = None
    dominant_operator: str | None = None
    dominant_family: str | None = None
    operator_purity: float = 0.0
    symbolic_success_rate: float = 0.0
    mean_verifier_score: float = 0.0
    failure_rate: float = 0.0
    repaired_fraction: float = 0.0
    retrieval_fraction: float = 0.0
    answer_diversity: int = 0
    problem_ids: tuple[str, ...] = Field(default_factory=tuple)
    trace_ids: tuple[str, ...] = Field(default_factory=tuple)
    operator_histogram: dict[str, int] = Field(default_factory=dict)
    family_histogram: dict[str, int] = Field(default_factory=dict)
    answer_histogram: dict[str, int] = Field(default_factory=dict)
    failure_histogram: dict[str, int] = Field(default_factory=dict)
    class_histogram: dict[str, int] = Field(default_factory=dict)
    quality_histogram: dict[str, int] = Field(default_factory=dict)
    representative_trace_id: str | None = None
    representative_signature: str | None = None
    representative_problem_id: str | None = None
    representative_score: float = 0.0
    promotion: ClusterPromotionSafety = Field(default_factory=ClusterPromotionSafety)
    mean_retrieval_compatibility: float = 0.0
    mean_retrieval_support: float = 0.0
    mean_operator_reliability: float = 0.0
    mean_prm_prefix_quality: float = 0.0
    mean_prm_step_quality: float = 0.0
    mean_proof_obligation_burden: float = 0.0
    mean_discharge_fraction: float = 0.0
    mean_logical_consistency: float = 0.0
    mean_symbolic_agreement: float = 0.0
    mean_completeness: float = 0.0
    mean_repairability: float = 0.0
    mean_answer_correctness_likelihood: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClusterAndMineArtifactMetadata(StrictModel):
    artifact_kind: str = ARTIFACT_KIND
    artifact_version: str = DEFAULT_ARTIFACT_VERSION
    schema_version: str = SCHEMA_VERSION
    artifact_id: str
    run_fingerprint: str
    record_count: int
    cluster_count: int
    promoted_cluster_count: int
    config_digest: str
    source_digest: str
    split_counts: dict[str, int] = Field(default_factory=dict)
    source_artifact_ids: tuple[str, ...] = Field(default_factory=tuple)


class ClusterAndMineArtifact(StrictModel):
    metadata: ClusterAndMineArtifactMetadata
    config: MiningConfig
    records: tuple[TraceArtifactRecord, ...] = Field(default_factory=tuple)
    clusters: tuple[ClusterSummary, ...] = Field(default_factory=tuple)
    operator_candidates: tuple[OperatorMiningCandidate, ...] = Field(default_factory=tuple)
    operator_report: OperatorMiningReport = Field(default_factory=OperatorMiningReport)


class ClusterAndMineWriteResult(StrictModel):
    artifact_dir: str
    manifest_path: str
    records_path: str
    clusters_path: str
    operator_candidates_path: str
    split_paths: dict[str, str] = Field(default_factory=dict)


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _stable_hash(prefix: str, payload: Any) -> str:
    raw = f"{prefix}::{_stable_json(payload)}"
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value or 0.0)))
    except Exception:
        return 0.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = _normalize_text(str(value)).lower()
    return lowered in {"1", "true", "yes", "y", "pass", "passed"}


def _coerce_split(value: Any, *, config: MiningConfig, problem_id: str) -> SplitName:
    raw = getattr(value, "value", value)
    try:
        return SplitName(str(raw))
    except Exception:
        ratios = config.split_ratios
        token = f"{config.split_salt}::{problem_id}"
        bucket = int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        if bucket < ratios[0]:
            return SplitName.TRAIN
        if bucket < ratios[0] + ratios[1]:
            return SplitName.VALID
        return SplitName.CALIBRATION


def _proof_summary(proof_obligations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(proof_obligations)
    open_count = 0
    discharged_count = 0
    hist: dict[str, int] = {}
    for item in proof_obligations:
        status = _normalize_text(item.get("status") or item.get("state") or "unknown").lower() or "unknown"
        hist[status] = hist.get(status, 0) + 1
        if status in {"open", "pending", "unresolved"}:
            open_count += 1
        if status in {"discharged", "resolved", "closed", "proved"}:
            discharged_count += 1
    return {
        "count": total,
        "open_count": open_count,
        "discharged_count": discharged_count,
        "open_fraction": _clamp01(open_count / float(total)) if total else 0.0,
        "discharge_fraction": _clamp01(discharged_count / float(total)) if total else 0.0,
        "status_histogram": hist,
    }


class ClusterAndMineOrchestrator:
    def __init__(self, config: MiningConfig | None = None) -> None:
        self.config = config or MiningConfig()
        self._miner = OperatorMiner(
            min_support=self.config.miner_min_support,
            min_symbolic_success_rate=self.config.miner_min_symbolic_success_rate,
            min_mean_verifier_gain=self.config.miner_min_mean_verifier_gain,
        )

    def run(self, traces: Sequence[BranchTrace | DistilledTraceRecord | Mapping[str, Any] | DistilledTraceArtifact]) -> ClusterAndMineArtifact:
        records = self._coerce_records(traces)
        for record in records:
            self._miner.ingest_trace(self._miner_payload(record))
        operator_candidates = tuple(self._miner.emit_candidates())
        operator_report = self._miner.report()
        clusters = tuple(self._cluster_records(records))

        run_fingerprint = self._fingerprint(records)
        config_digest = _stable_hash("cluster_and_mine_config", self.config.model_dump(mode="json"))
        source_digest = _stable_hash("cluster_and_mine_sources", [{"record_id": r.record_id, "signature": r.signature} for r in records])
        artifact_id = _stable_hash("cluster_and_mine_artifact", {"config": config_digest, "source": source_digest})
        split_counts: dict[str, int] = {}
        for record in records:
            split_counts[record.split.value] = split_counts.get(record.split.value, 0) + 1

        return ClusterAndMineArtifact(
            metadata=ClusterAndMineArtifactMetadata(
                artifact_version=self.config.artifact_version,
                schema_version=self.config.schema_version,
                artifact_id=artifact_id,
                run_fingerprint=run_fingerprint,
                record_count=len(records),
                cluster_count=len(clusters),
                promoted_cluster_count=sum(1 for c in clusters if c.promotion.promotion_safe),
                config_digest=config_digest,
                source_digest=source_digest,
                split_counts=split_counts,
                source_artifact_ids=tuple(sorted({r.source_artifact_id for r in records if r.source_artifact_id})),
            ),
            config=self.config,
            records=tuple(records),
            clusters=clusters,
            operator_candidates=operator_candidates,
            operator_report=operator_report,
        )

    def save(self, artifact: ClusterAndMineArtifact, output_dir: str | Path) -> ClusterAndMineWriteResult:
        base = Path(output_dir)
        artifact_dir = base / artifact.metadata.artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = artifact_dir / "manifest.json"
        records_path = artifact_dir / "records.jsonl"
        clusters_path = artifact_dir / "clusters.jsonl"
        operator_candidates_path = artifact_dir / "operator_candidates.jsonl"
        manifest_path.write_text(
            _stable_json({
                **artifact.metadata.model_dump(mode="json"),
                "config": artifact.config.model_dump(mode="json"),
                "operator_report": artifact.operator_report.model_dump(mode="json"),
            }) + "\n",
            encoding="utf-8",
        )
        self._write_jsonl(records_path, (r.model_dump(mode="json") for r in artifact.records))
        self._write_jsonl(clusters_path, (c.model_dump(mode="json") for c in artifact.clusters))
        self._write_jsonl(operator_candidates_path, (c.model_dump(mode="json") for c in artifact.operator_candidates))
        split_paths: dict[str, str] = {}
        for split in SplitName:
            path = artifact_dir / f"{split.value}.jsonl"
            self._write_jsonl(path, (r.model_dump(mode="json") for r in artifact.records if r.split is split))
            split_paths[split.value] = str(path)
        return ClusterAndMineWriteResult(
            artifact_dir=str(artifact_dir),
            manifest_path=str(manifest_path),
            records_path=str(records_path),
            clusters_path=str(clusters_path),
            operator_candidates_path=str(operator_candidates_path),
            split_paths=split_paths,
        )

    def _coerce_records(self, sources: Sequence[BranchTrace | DistilledTraceRecord | Mapping[str, Any] | DistilledTraceArtifact]) -> list[TraceArtifactRecord]:
        records: list[TraceArtifactRecord] = []
        for source in sources:
            if isinstance(source, DistilledTraceArtifact):
                for record in source.records:
                    records.append(self._record_from_distilled(record, source.metadata.artifact_id))
            elif isinstance(source, DistilledTraceRecord) or (hasattr(source, "classification") and hasattr(source, "provenance")):
                records.append(self._record_from_distilled(source, None))  # type: ignore[arg-type]
            elif isinstance(source, BranchTrace):
                records.append(self._record_from_branch_trace(source))
            elif isinstance(source, Mapping):
                if "classification" in source and "provenance" in source:
                    records.append(self._record_from_distilled(DistilledTraceRecord.model_validate(source), None))
                else:
                    records.append(self._record_from_generated_raw_trace(source))
            else:
                raise TypeError(f"Unsupported mining source type: {type(source)!r}")
        return sorted(records, key=lambda item: (item.problem_id, item.branch_id, item.record_id))

    def _record_from_distilled(self, record: DistilledTraceRecord, upstream_artifact_id: str | None) -> TraceArtifactRecord:
        operator_sequence = tuple(record.operator_sequence)
        operator_families = tuple(operator_family_for_name(name).value for name in operator_sequence)
        route_snapshot = dict(record.provenance.route_snapshot or {})
        provenance_meta = dict(record.provenance.metadata or {})
        outcome_meta = dict(getattr(record.outcome, "metadata", {}) or {})
        trace_meta = dict(record.metadata or {})
        proof_summary = dict(provenance_meta.get("proof_obligation_summary") or trace_meta.get("proof_obligation_summary") or _proof_summary(provenance_meta.get("proof_obligations", [])))
        route_compute = {str(k): _clamp01(v) for k, v in dict(provenance_meta.get("route_compute_signals", {}) or trace_meta.get("route_compute_signals", {})).items()}
        signals = RichSignalBundle(
            verifier_score=_clamp01(record.outcome.verifier_score),
            logical_consistency=_clamp01(record.outcome.logical_consistency),
            symbolic_agreement=_clamp01(outcome_meta.get("symbolic_agreement", trace_meta.get("symbolic_agreement", 0.0))),
            completeness=_clamp01(record.outcome.completeness),
            repairability=_clamp01(record.outcome.repairability),
            answer_correctness_likelihood=_clamp01(outcome_meta.get("answer_correctness_likelihood", trace_meta.get("answer_correctness_likelihood", 0.0))),
            prm_prefix_quality=_clamp01(outcome_meta.get("prm_prefix_quality", trace_meta.get("prm_prefix_quality", 0.0))),
            prm_step_quality_mean=_clamp01(outcome_meta.get("prm_step_quality", trace_meta.get("prm_step_quality", 0.0))),
            retrieval_support=_clamp01(outcome_meta.get("retrieval_support", trace_meta.get("retrieval_support", 0.0))),
            retrieval_compatibility=_clamp01(outcome_meta.get("retrieval_compatibility", trace_meta.get("retrieval_compatibility", 0.0))),
            operator_reliability=_clamp01(outcome_meta.get("operator_reliability", trace_meta.get("operator_reliability", 0.0))),
            tool_consistency=_clamp01(record.outcome.symbolic_score),
            discharge_fraction=_clamp01(outcome_meta.get("discharge_fraction", trace_meta.get("discharge_fraction", proof_summary.get("discharge_fraction", 0.0)))),
            open_obligation_burden=_clamp01(outcome_meta.get("open_obligation_burden", trace_meta.get("open_obligation_burden", proof_summary.get("open_fraction", 0.0)))),
            operator_purity=_clamp01(trace_meta.get("operator_purity", 0.0)),
        )
        step_payload = tuple({
            "step_index": step.step_index,
            "description": step.text,
            "operator_used": step.operator_name,
            "symbolic_valid": step.symbolic_valid,
            "kind": step.kind,
            "phase": step.phase,
            "metadata": dict(step.metadata),
        } for step in record.steps)
        signature = _stable_hash("mining_signature", {
            "problem_id": record.problem_id,
            "branch_id": record.branch_id,
            "operators": operator_sequence,
            "classification": getattr(record.classification, "value", record.classification),
            "answer": record.outcome.answer_canonical,
        })
        cluster_key = self._cluster_key(
            archetype=(record.archetypes[0] if record.archetypes else None),
            domain=self._infer_domain_hint(record.archetypes, route_snapshot),
            operator_sequence=operator_sequence,
            step_count=len(record.steps),
            symbolic_valid=record.outcome.symbolic_valid,
            repaired=record.outcome.repaired,
        )
        return TraceArtifactRecord(
            record_id=record.record_id,
            source_type=MiningSourceType.DISTILLED_TRACE_RECORD,
            source_record_id=record.record_id,
            source_artifact_id=upstream_artifact_id,
            problem_id=record.problem_id,
            branch_id=record.branch_id,
            split=_coerce_split(record.split, config=self.config, problem_id=record.problem_id),
            classification=getattr(record.classification, "value", record.classification),
            quality_tier=getattr(record.quality_tier, "value", record.quality_tier),
            answer=record.outcome.answer_raw,
            answer_canonical=record.outcome.answer_canonical,
            archetype_used=(record.archetypes[0] if record.archetypes else None),
            domain_hint=self._infer_domain_hint(record.archetypes, route_snapshot),
            operator_sequence=operator_sequence,
            operator_families=operator_families,
            step_count=len(record.steps),
            step_count_bucket=min(len(record.steps), self.config.max_step_count_bucket),
            symbolic_valid=bool(record.outcome.symbolic_valid),
            verifier_score=signals.verifier_score,
            branch_score=_clamp01(record.outcome.branch_score),
            tool_consistency=signals.tool_consistency,
            repaired=bool(record.outcome.repaired),
            repair_count=int(record.outcome.repair_count),
            retrieval_used=bool(record.outcome.retrieval_used),
            failure_type=_normalize_text(record.outcome.failure_type) or None,
            signature=signature,
            cluster_key=cluster_key,
            route_snapshot=route_snapshot,
            steps=step_payload,
            signals=signals,
            verifier_decomposition={str(k): _clamp01(v) for k, v in dict(outcome_meta.get("verifier_decomposition", trace_meta.get("verifier_decomposition", {}))).items()},
            proof_obligation_summary=proof_summary,
            route_compute_signals=route_compute,
            provenance={
                "source_type": getattr(record.provenance.source_type, "value", record.provenance.source_type),
                "source_record_id": record.provenance.source_record_id,
                "source_digest": record.provenance.source_digest,
                "lineage": list(record.provenance.lineage),
                "route_snapshot": route_snapshot,
                **provenance_meta,
            },
            metadata={
                "quality_tier": getattr(record.quality_tier, "value", record.quality_tier),
                "classification_basis": trace_meta.get("classification_basis", {}),
                "problem_text": trace_meta.get("problem_text", ""),
                "operator_provenance": trace_meta.get("operator_provenance", {}),
            },
        )

    def _record_from_branch_trace(self, trace: BranchTrace) -> TraceArtifactRecord:
        operator_sequence = tuple(trace.operator_sequence or tuple(str(step.operator_used) for step in trace.steps if step.operator_used))
        operator_families = tuple(operator_family_for_name(name).value for name in operator_sequence)
        failure_type = trace.failure_type.value if isinstance(trace.failure_type, FailureType) else trace.failure_type
        proof_summary = _proof_summary(getattr(trace, "proof_obligations", []) or [])
        trace_metadata = dict(trace.metadata or {}) if isinstance(trace.metadata, Mapping) else {}
        route_compute = {str(k): _clamp01(v) for k, v in dict(trace_metadata.get("route_compute_signals", {})).items()}
        signals = RichSignalBundle(
            verifier_score=_clamp01(getattr(trace, "verifier_score", 0.0)),
            logical_consistency=_clamp01(getattr(trace, "logical_consistency", 0.0)),
            symbolic_agreement=_clamp01(getattr(trace, "symbolic_agreement", 0.0)),
            completeness=_clamp01(getattr(trace, "completeness", 0.0)),
            repairability=_clamp01(getattr(trace, "repairability", 0.0)),
            answer_correctness_likelihood=_clamp01(getattr(trace, "answer_correctness_likelihood", 0.0)),
            prm_prefix_quality=_clamp01(getattr(trace, "prm_prefix_quality", 0.0)),
            prm_step_quality_mean=_clamp01(getattr(trace, "prm_step_quality", 0.0)),
            retrieval_support=_clamp01(getattr(trace, "retrieval_support", 0.0)),
            retrieval_compatibility=_clamp01(getattr(trace, "retrieval_compatibility", 0.0)),
            operator_reliability=_clamp01(getattr(trace, "operator_reliability", 0.0)),
            tool_consistency=_clamp01(getattr(trace, "tool_consistency", 0.0)),
            discharge_fraction=_clamp01(getattr(trace, "discharge_fraction", proof_summary.get("discharge_fraction", 0.0))),
            open_obligation_burden=_clamp01(getattr(trace, "open_obligation_burden", proof_summary.get("open_fraction", 0.0))),
            operator_purity=_clamp01(trace_metadata.get("operator_purity", 0.0)),
        )
        signature = _stable_hash("mining_signature", {
            "problem_id": trace.problem_id,
            "branch_id": trace.branch_id,
            "operators": operator_sequence,
            "answer": trace.answer_canonical or trace.answer,
        })
        cluster_key = self._cluster_key(
            archetype=_normalize_text(trace.archetype_used) or None,
            domain=self._infer_domain_hint([trace.archetype_used] if trace.archetype_used else (), route_compute),
            operator_sequence=operator_sequence,
            step_count=len(trace.steps),
            symbolic_valid=bool(trace.symbolic_valid),
            repaired=bool(trace.repaired),
        )
        return TraceArtifactRecord(
        record_id=_stable_hash("raw_trace_record", {"problem_id": trace.problem_id, "branch_id": trace.branch_id}),
        source_type=MiningSourceType.RAW_BRANCH_TRACE,
        source_record_id=trace.branch_id,
        problem_id=trace.problem_id,
        branch_id=trace.branch_id,
        split=_coerce_split(getattr(trace, "source_split", None), config=self.config, problem_id=trace.problem_id),
        classification="raw_branch_trace",
        quality_tier="unknown",
        answer=trace.answer,
        answer_canonical=trace.answer_canonical or trace.answer,
        archetype_used=_normalize_text(trace.archetype_used) or None,
        domain_hint=self._infer_domain_hint([trace.archetype_used] if trace.archetype_used else (), {}),
        operator_sequence=operator_sequence,
        operator_families=operator_families,
        step_count=len(trace.steps),
        step_count_bucket=min(len(trace.steps), self.config.max_step_count_bucket),
        symbolic_valid=bool(trace.symbolic_valid),
        verifier_score=signals.verifier_score,
        branch_score=_clamp01(trace.branch_score),
        tool_consistency=signals.tool_consistency,
        repaired=bool(trace.repaired),
        repair_count=int(trace.repair_count),
        retrieval_used=bool(trace.retrieval_used),
        failure_type=_normalize_text(str(failure_type)) or None,
        signature=signature,
        cluster_key=cluster_key,
        route_snapshot=dict((trace.provenance or {}).get("route_snapshot", {})),
        steps=tuple({
            "step_index": i,
            "description": step.description,
            "operator_used": step.operator_used,
            "symbolic_valid": step.symbolic_valid,
        } for i, step in enumerate(trace.steps, start=1)),
        signals=signals,
        verifier_decomposition={
            str(k): _clamp01(v)
            for k, v in dict(getattr(trace, "verifier_decomposition", {}) or {}).items()
        },
        proof_obligation_summary=proof_summary,
        route_compute_signals=route_compute,
        provenance={"source_type": "branch_trace", "source_record_id": trace.branch_id, **dict(trace.provenance or {})},
        metadata={
            "full_reasoning": _normalize_text(trace.full_reasoning),
            "operator_provenance": dict(trace_metadata.get("operator_provenance", {}) or {}),
            "verifier_decomposition": {
                str(k): _clamp01(v)
                for k, v in dict(getattr(trace, "verifier_decomposition", {}) or {}).items()
            },
            "proof_obligation_summary": dict(proof_summary),
            "route_compute_signals": dict(route_compute),
        },
    )
    def _record_from_generated_raw_trace(self, source: Mapping[str, Any]) -> TraceArtifactRecord:
        problem_id = _normalize_text(str(source.get("problem_id") or ""))
        branch_id = _normalize_text(str(source.get("branch_id") or source.get("trace_id") or ""))
        if not problem_id or not branch_id:
            raise ValueError("Raw trace mapping is missing problem_id or branch_id")
        steps = source.get("steps", []) or []
        step_payload = tuple({
            "step_index": int(_safe_float(step.get("index", step.get("step_index", position)), float(position))),
            "description": _normalize_text(str(step.get("description") or step.get("text") or "")),
            "operator_used": _normalize_text(str(step.get("operator_name") or step.get("operator_used") or "")) or None,
            "symbolic_valid": _safe_bool(step.get("symbolic_valid", False)),
            "kind": _normalize_text(str(step.get("kind") or "")),
            "phase": _normalize_text(str(step.get("phase") or "")),
        } for position, step in enumerate(steps, start=1) if isinstance(step, Mapping))
        sequence_source = source.get("operator_sequence")
        if not isinstance(sequence_source, Sequence) or isinstance(sequence_source, (str, bytes)):
            sequence_source = [step.get("operator_name") or step.get("operator_used") for step in steps if isinstance(step, Mapping)]
        operator_sequence = tuple(item for item in (_normalize_text(str(op)) for op in sequence_source) if item)
        operator_families = tuple(operator_family_for_name(name).value for name in operator_sequence)
        route_snapshot = dict(source.get("route_snapshot", {}) if isinstance(source.get("route_snapshot"), Mapping) else {})
        raw_archetypes = source.get("archetypes")
        if isinstance(raw_archetypes, Sequence) and not isinstance(raw_archetypes, (str, bytes)):
            archetypes = [_normalize_text(str(item)) for item in raw_archetypes if _normalize_text(str(item))]
        elif isinstance(route_snapshot.get("archetypes"), Mapping):
            archetypes = [_normalize_text(str(k)) for k, v in route_snapshot["archetypes"].items() if _normalize_text(str(k)) and _safe_float(v) > 0.0]
        else:
            archetypes = []
        archetype_used = self._primary_archetype(archetypes)
        proof_summary = dict(source.get("proof_obligation_summary") if isinstance(source.get("proof_obligation_summary"), Mapping) else _proof_summary(source.get("proof_obligations", []) or []))
        route_compute = {str(k): _clamp01(v) for k, v in dict(source.get("route_compute_signals", {}) if isinstance(source.get("route_compute_signals"), Mapping) else {}).items()}
        signals = RichSignalBundle(
            verifier_score=_clamp01(_safe_float(source.get("verifier_score"), 0.0)),
            logical_consistency=_clamp01(_safe_float(source.get("logical_consistency"), 0.0)),
            symbolic_agreement=_clamp01(_safe_float(source.get("symbolic_agreement"), 0.0)),
            completeness=_clamp01(_safe_float(source.get("completeness"), 0.0)),
            repairability=_clamp01(_safe_float(source.get("repairability"), 0.0)),
            answer_correctness_likelihood=_clamp01(_safe_float(source.get("answer_correctness_likelihood"), 0.0)),
            prm_prefix_quality=_clamp01(_safe_float(source.get("prm_prefix_quality"), 0.0)),
            prm_step_quality_mean=_clamp01(_safe_float(source.get("prm_step_quality"), 0.0)),
            retrieval_support=_clamp01(_safe_float(source.get("retrieval_support"), 0.0)),
            retrieval_compatibility=_clamp01(_safe_float(source.get("retrieval_compatibility"), 0.0)),
            operator_reliability=_clamp01(_safe_float(source.get("operator_reliability"), 0.0)),
            tool_consistency=_clamp01(_safe_float(source.get("tool_consistency"), _safe_float(source.get("symbolic_score"), 0.0))),
            discharge_fraction=_clamp01(_safe_float(source.get("discharge_fraction"), proof_summary.get("discharge_fraction", 0.0))),
            open_obligation_burden=_clamp01(_safe_float(source.get("open_obligation_burden"), proof_summary.get("open_fraction", 0.0))),
            operator_purity=_clamp01(_safe_float(source.get("operator_purity"), 0.0)),
        )
        answer = _normalize_text(str(source.get("answer_raw") or source.get("answer") or ""))
        answer_canonical = _normalize_text(str(source.get("answer_canonical") or answer))
        signature = _stable_hash("mining_signature", {"problem_id": problem_id, "branch_id": branch_id, "operators": operator_sequence, "answer": answer_canonical})
        cluster_key = self._cluster_key(
            archetype=archetype_used,
            domain=self._infer_domain_hint(archetypes, route_snapshot),
            operator_sequence=operator_sequence,
            step_count=len(step_payload),
            symbolic_valid=_safe_bool(source.get("symbolic_valid", False)),
            repaired=_safe_bool(source.get("repaired", False)),
        )
        return TraceArtifactRecord(
            record_id=_stable_hash("raw_trace_record", {"problem_id": problem_id, "branch_id": branch_id}),
            source_type=MiningSourceType.RAW_BRANCH_TRACE,
            source_record_id=branch_id,
            source_artifact_id=None,
            problem_id=problem_id,
            branch_id=branch_id,
            split=_coerce_split(source.get("split"), config=self.config, problem_id=problem_id),
            classification=_normalize_text(str(source.get("record_type") or "raw_branch_trace")) or "raw_branch_trace",
            quality_tier="unknown",
            answer=answer or None,
            answer_canonical=answer_canonical or None,
            archetype_used=archetype_used,
            domain_hint=self._infer_domain_hint(archetypes, route_snapshot),
            operator_sequence=operator_sequence,
            operator_families=operator_families,
            step_count=len(step_payload),
            step_count_bucket=min(len(step_payload), self.config.max_step_count_bucket),
            symbolic_valid=_safe_bool(source.get("symbolic_valid", False)),
            verifier_score=signals.verifier_score,
            branch_score=_clamp01(_safe_float(source.get("branch_score"), 0.0)),
            tool_consistency=signals.tool_consistency,
            repaired=_safe_bool(source.get("repaired", False)),
            repair_count=int(_safe_float(source.get("repair_count"), 0.0)),
            retrieval_used=_safe_bool(source.get("retrieval_used", False)),
            failure_type=_normalize_text(str(source.get("failure_type") or "")) or None,
            signature=signature,
            cluster_key=cluster_key,
            route_snapshot=route_snapshot,
            steps=step_payload,
            signals=signals,
            verifier_decomposition={str(k): _clamp01(v) for k, v in dict(source.get("verifier_decomposition", {}) if isinstance(source.get("verifier_decomposition"), Mapping) else {}).items()},
            proof_obligation_summary=proof_summary,
            route_compute_signals=route_compute,
            provenance={"source_type": "generated_raw_trace", "source_record_id": branch_id, "run_id": _normalize_text(str(source.get("run_id") or ""))},
            metadata={"problem_text": _normalize_text(str(source.get("problem_text") or "")), "operator_provenance": dict(source.get("operator_provenance", {}) if isinstance(source.get("operator_provenance"), Mapping) else {})},
        )

    def _cluster_records(self, records: Sequence[TraceArtifactRecord]) -> list[ClusterSummary]:
        grouped: dict[str, list[TraceArtifactRecord]] = defaultdict(list)
        for record in records:
            grouped[record.cluster_key].append(record)

        clusters: list[ClusterSummary] = []
        for cluster_key, members in sorted(grouped.items()):
            operator_hist = Counter(op for m in members for op in m.operator_sequence)
            family_hist = Counter(f for m in members for f in m.operator_families)
            answer_hist = Counter(_normalize_text(m.answer_canonical or m.answer) or "" for m in members)
            failure_hist = Counter(_normalize_text(m.failure_type) or "none" for m in members)
            class_hist = Counter(m.classification for m in members)
            quality_hist = Counter(m.quality_tier for m in members)
            member_count_by_split = Counter(m.split.value for m in members)

            problem_ids = sorted({m.problem_id for m in members})
            distinct_problem_count_by_split = {split.value: len({m.problem_id for m in members if m.split is split}) for split in SplitName}
            dominant_operator, dominant_operator_count = self._dominant_item(operator_hist, "")
            dominant_family, _ = self._dominant_item(family_hist, "")
            dominant_archetype, _ = self._dominant_item(Counter(_normalize_text(m.archetype_used) or "" for m in members), "")

            operator_purity = dominant_operator_count / float(len(members)) if members else 0.0
            symbolic_success_rate = sum(1 for m in members if m.symbolic_valid) / float(len(members)) if members else 0.0
            mean_verifier_score = self._mean([m.signals.verifier_score for m in members])
            failure_rate = sum(1 for m in members if m.failure_type) / float(len(members)) if members else 0.0
            representative = self._select_representative_trace(members)
            holdout_support = distinct_problem_count_by_split.get(SplitName.VALID.value, 0) + distinct_problem_count_by_split.get(SplitName.CALIBRATION.value, 0)
            promotion = ClusterPromotionSafety(
                min_cluster_support_ok=len(members) >= self.config.min_cluster_support,
                min_problem_support_ok=len(problem_ids) >= self.config.min_distinct_problem_support,
                symbolic_success_ok=symbolic_success_rate >= self.config.min_symbolic_success_rate,
                verifier_score_ok=mean_verifier_score >= self.config.min_mean_verifier_score,
                operator_purity_ok=operator_purity >= self.config.min_operator_purity,
                failure_rate_ok=failure_rate <= self.config.max_failure_rate,
                logical_consistency_ok=self._mean([m.signals.logical_consistency for m in members]) >= self.config.min_mean_logical_consistency,
                prm_prefix_quality_ok=self._mean([m.signals.prm_prefix_quality for m in members]) >= self.config.min_mean_prm_prefix_quality,
                operator_reliability_ok=self._mean([m.signals.operator_reliability for m in members]) >= self.config.min_mean_operator_reliability,
                retrieval_compatibility_ok=self._mean([m.signals.retrieval_compatibility for m in members]) >= self.config.min_mean_retrieval_compatibility,
                proof_burden_ok=self._mean([m.signals.open_obligation_burden for m in members]) <= self.config.max_mean_proof_obligation_burden,
                discharge_fraction_ok=self._mean([m.signals.discharge_fraction for m in members]) >= self.config.min_mean_discharge_fraction,
                holdout_support_ok=(not self.config.require_holdout_for_promotion) or (holdout_support >= self.config.min_holdout_support),
            )
            cluster_id = _stable_hash("operator_cluster", {"cluster_key": cluster_key, "member_ids": [m.record_id for m in members]})
            clusters.append(
                ClusterSummary(
                    cluster_id=cluster_id,
                    cluster_key=cluster_key,
                    member_count=len(members),
                    member_count_by_split=dict(member_count_by_split),
                    distinct_problem_count=len(problem_ids),
                    distinct_problem_count_by_split=distinct_problem_count_by_split,
                    dominant_archetype=dominant_archetype or None,
                    dominant_operator=dominant_operator or None,
                    dominant_family=dominant_family or None,
                    operator_purity=round(_clamp01(operator_purity), 6),
                    symbolic_success_rate=round(_clamp01(symbolic_success_rate), 6),
                    mean_verifier_score=round(_clamp01(mean_verifier_score), 6),
                    failure_rate=round(_clamp01(failure_rate), 6),
                    repaired_fraction=round(_clamp01(sum(1 for m in members if m.repaired) / float(len(members)) if members else 0.0), 6),
                    retrieval_fraction=round(_clamp01(sum(1 for m in members if m.retrieval_used) / float(len(members)) if members else 0.0), 6),
                    answer_diversity=len([k for k in answer_hist if k]),
                    problem_ids=tuple(problem_ids),
                    trace_ids=tuple(m.record_id for m in members),
                    operator_histogram=dict(operator_hist),
                    family_histogram=dict(family_hist),
                    answer_histogram=dict(answer_hist),
                    failure_histogram=dict(failure_hist),
                    class_histogram=dict(class_hist),
                    quality_histogram=dict(quality_hist),
                    representative_trace_id=representative.record_id if representative else None,
                    representative_signature=representative.signature if representative else None,
                    representative_problem_id=representative.problem_id if representative else None,
                    representative_score=round(self._representative_score(representative), 6) if representative else 0.0,
                    promotion=promotion,
                    mean_retrieval_compatibility=round(_clamp01(self._mean([m.signals.retrieval_compatibility for m in members])), 6),
                    mean_retrieval_support=round(_clamp01(self._mean([m.signals.retrieval_support for m in members])), 6),
                    mean_operator_reliability=round(_clamp01(self._mean([m.signals.operator_reliability for m in members])), 6),
                    mean_prm_prefix_quality=round(_clamp01(self._mean([m.signals.prm_prefix_quality for m in members])), 6),
                    mean_prm_step_quality=round(_clamp01(self._mean([m.signals.prm_step_quality_mean for m in members])), 6),
                    mean_proof_obligation_burden=round(_clamp01(self._mean([m.signals.open_obligation_burden for m in members])), 6),
                    mean_discharge_fraction=round(_clamp01(self._mean([m.signals.discharge_fraction for m in members])), 6),
                    mean_logical_consistency=round(_clamp01(self._mean([m.signals.logical_consistency for m in members])), 6),
                    mean_symbolic_agreement=round(_clamp01(self._mean([m.signals.symbolic_agreement for m in members])), 6),
                    mean_completeness=round(_clamp01(self._mean([m.signals.completeness for m in members])), 6),
                    mean_repairability=round(_clamp01(self._mean([m.signals.repairability for m in members])), 6),
                    mean_answer_correctness_likelihood=round(_clamp01(self._mean([m.signals.answer_correctness_likelihood for m in members])), 6),
                    metadata={
                        "verifier_decomposition": [m.verifier_decomposition for m in members],
                        "route_compute_signals": [m.route_compute_signals for m in members],
                        "proof_obligation_context": [m.proof_obligation_summary for m in members],
                        "operator_provenance": [m.metadata.get("operator_provenance", {}) for m in members],
                    },
                )
            )
        return sorted(clusters, key=lambda c: (-int(c.promotion.promotion_safe), -c.member_count, c.cluster_key))

    def _representative_score(self, record: TraceArtifactRecord | None) -> float:
        if record is None:
            return 0.0
        score = (
            self.config.representative_verifier_weight * record.signals.verifier_score
            + self.config.representative_signal_weight * (
                0.25 * record.signals.logical_consistency
                + 0.25 * record.signals.completeness
                + 0.20 * record.signals.prm_prefix_quality
                + 0.15 * record.signals.retrieval_compatibility
                + 0.15 * record.signals.symbolic_agreement
            )
            + self.config.representative_reliability_weight * record.signals.operator_reliability
            + self.config.representative_proof_weight * record.signals.discharge_fraction
        )
        if record.symbolic_valid:
            score += self.config.representative_symbolic_bonus
        score -= 0.10 * record.signals.open_obligation_burden
        return score

    def _select_representative_trace(self, members: Sequence[TraceArtifactRecord]) -> TraceArtifactRecord | None:
        if not members:
            return None
        return max(
            members,
            key=lambda m: (
                self._representative_score(m),
                m.signals.verifier_score,
                -m.signals.open_obligation_burden,
                -m.repair_count,
                m.record_id,
            ),
        )

    def _miner_payload(self, record: TraceArtifactRecord) -> dict[str, Any]:
        return {
            "problem_id": record.problem_id,
            "branch_id": record.branch_id,
            "answer_canonical": record.answer_canonical,
            "operator_sequence": list(record.operator_sequence),
            "symbolic_valid": record.symbolic_valid,
            "verifier_score": record.verifier_score,
            "branch_score": record.branch_score,
            "logical_consistency": record.signals.logical_consistency,
            "prm_prefix_quality": record.signals.prm_prefix_quality,
            "retrieval_compatibility": record.signals.retrieval_compatibility,
            "operator_reliability": record.signals.operator_reliability,
            "discharge_fraction": record.signals.discharge_fraction,
            "open_obligation_burden": record.signals.open_obligation_burden,
            "split": record.split.value,
        }

    def _fingerprint(self, records: Sequence[TraceArtifactRecord]) -> str:
        return _stable_hash("cluster_and_mine_run", [{"record_id": r.record_id, "cluster_key": r.cluster_key} for r in records])

    def _dominant_item(self, histogram: Counter[str], default: str) -> tuple[str, int]:
        if not histogram:
            return default, 0
        item, count = max(histogram.items(), key=lambda kv: (kv[1], kv[0]))
        return item, count

    def _cluster_key(self, *, archetype: str | None, domain: str | None, operator_sequence: Sequence[str], step_count: int, symbolic_valid: bool, repaired: bool) -> str:
        prefix = tuple(operator_sequence[: self.config.sequence_prefix])
        return _stable_hash("cluster_key", {
            "archetype": _normalize_text(archetype),
            "domain": _normalize_text(domain),
            "operator_prefix": prefix,
            "step_bucket": min(step_count, self.config.max_step_count_bucket),
            "symbolic_valid": bool(symbolic_valid),
            "repaired": bool(repaired),
        })

    def _infer_domain_hint(self, archetypes: Sequence[str], route_snapshot: Mapping[str, Any]) -> str | None:
        domain = _normalize_text(str(route_snapshot.get("domain") or route_snapshot.get("problem_type") or ""))
        if domain:
            return domain
        lowered = {(_normalize_text(a)).lower() for a in archetypes if _normalize_text(a)}
        if {"modular", "parity", "divisibility"} & lowered:
            return "number_theory"
        if {"construction"} & lowered:
            return "geometry"
        if {"extremal", "pigeonhole"} & lowered:
            return "combinatorics"
        if {"symbolic_manipulation", "bounding"} & lowered:
            return "algebra"
        return None

    def _primary_archetype(self, archetypes: Sequence[str]) -> str | None:
        cleaned = [_normalize_text(a) for a in archetypes if _normalize_text(a)]
        return cleaned[0] if cleaned else None

    def _mean(self, values: Sequence[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    def _write_jsonl(self, path: Path, rows: Sequence[dict[str, Any]] | Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(_stable_json(row) + "\n")

