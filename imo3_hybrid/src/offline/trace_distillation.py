from __future__ import annotations

"""Deterministic offline trace distillation.

This module converts structured online reasoning trajectories into versioned,
provenance-preserving offline artifacts that later training / mining stages can
consume directly.

It intentionally does more than cleaning JSON:
- ingests branch / trace outputs from the online loop
- preserves step-level and evidence-level structure
- classifies trajectories into stable supervision classes
- emits deterministic artifacts with lineage + split metadata
- exposes helpers for downstream verifier / router / operator consumers
"""

from dataclasses import dataclass
from enum import Enum
from hashlib import sha1
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, Field

from src.aggregation.canonicalize import canonicalize_competition_answer
from src.common.constants import (
    OFFLINE_ARTIFACT_VERSION,
    OFFLINE_SPLIT_RATIOS,
    OFFLINE_SPLIT_SALT,
    VERIFIER_PASS_THRESHOLD,
)
from src.common.schemas import BranchTrace, FailureType, ReasoningStep, VerifierLabel

try:  # Optional at import time during partial builds.
    from src.branches.branch_state import (
        BranchCandidateState,
        BranchPhase,
        BranchState,
        BranchStep,
        CritiqueEvidence,
        RetrievalEvidence,
        SymbolicEvidence,
        VerifierEvidence,
    )
except Exception:  # pragma: no cover - allows isolated file use during early bootstrap.
    BranchState = Any  # type: ignore[assignment]
    BranchStep = Any  # type: ignore[assignment]
    RetrievalEvidence = Any  # type: ignore[assignment]
    SymbolicEvidence = Any  # type: ignore[assignment]
    VerifierEvidence = Any  # type: ignore[assignment]
    CritiqueEvidence = Any  # type: ignore[assignment]
    BranchCandidateState = Any  # type: ignore[assignment]
    BranchPhase = Any  # type: ignore[assignment]


SCHEMA_VERSION = "trace_distillation.v1"
ARTIFACT_KIND = "distilled_trace_corpus"
DEFAULT_ARTIFACT_VERSION = OFFLINE_ARTIFACT_VERSION


JsonValue = Any


class DistilledTraceClass(str, Enum):
    CORRECT_STRONG = "correct_strong"
    CORRECT_WEAK = "correct_weak"
    REPAIRED_SUCCESS = "repaired_success"
    REPAIRED_FAIL = "repaired_fail"
    VERIFIER_REJECTED = "verifier_rejected"
    SYMBOLIC_CONTRADICTED = "symbolic_contradicted"


class TraceQualityTier(str, Enum):
    STRONG = "strong"
    WEAK = "weak"
    REPAIRED = "repaired"
    FAILED = "failed"


class SplitName(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    CALIBRATION = "calibration"


class InputSourceType(str, Enum):
    BRANCH_TRACE = "branch_trace"
    BRANCH_STATE = "branch_state"
    GENERIC_MAPPING = "generic_mapping"


class DistillationConfig(BaseModel):
    artifact_version: str = DEFAULT_ARTIFACT_VERSION
    schema_version: str = SCHEMA_VERSION
    min_step_count: int = 1
    min_reasoning_chars: int = 12
    verifier_pass_threshold: float = Field(VERIFIER_PASS_THRESHOLD, ge=0.0, le=1.0)
    strong_verifier_threshold: float = Field(0.80, ge=0.0, le=1.0)
    weak_verifier_threshold: float = Field(0.55, ge=0.0, le=1.0)
    strong_symbolic_threshold: float = Field(0.90, ge=0.0, le=1.0)
    weak_symbolic_threshold: float = Field(0.55, ge=0.0, le=1.0)
    strong_completeness_threshold: float = Field(0.70, ge=0.0, le=1.0)
    weak_completeness_threshold: float = Field(0.40, ge=0.0, le=1.0)
    preserve_rejected_without_answer: bool = True
    keep_duplicate_problem_answer_pairs: bool = True
    split_ratios: tuple[float, float, float] = OFFLINE_SPLIT_RATIOS
    split_salt: str = OFFLINE_SPLIT_SALT

    def normalized_split_ratios(self) -> tuple[float, float, float]:
        total = sum(self.split_ratios)
        if total <= 0:
            return (0.80, 0.10, 0.10)
        return tuple(float(x) / total for x in self.split_ratios)  # type: ignore[return-value]


class DistilledTraceStep(BaseModel):
    step_id: str
    step_index: int
    kind: str = "reasoning"
    phase: str = "reasoning"
    text: str
    operator_name: str | None = None
    summary: str | None = None
    symbolic_expression: str | None = None
    symbolic_valid: bool | None = None
    python_code: str | None = None
    python_result: str | None = None
    node_id: str | None = None
    state_fingerprint: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DistilledEvidenceRecord(BaseModel):
    evidence_id: str
    kind: str
    score: float | None = None
    passed: bool | None = None
    summary: str = ""
    source_id: str | None = None
    supporting_step_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceOutcomeRecord(BaseModel):
    answer_raw: str | None = None
    answer_canonical: str | None = None
    verifier_score: float = Field(0.0, ge=0.0, le=1.0)
    symbolic_score: float = Field(0.0, ge=0.0, le=1.0)
    logical_consistency: float = Field(0.0, ge=0.0, le=1.0)
    completeness: float = Field(0.0, ge=0.0, le=1.0)
    repairability: float = Field(0.0, ge=0.0, le=1.0)
    branch_score: float = Field(0.0, ge=0.0, le=1.0)
    symbolic_valid: bool = False
    retrieval_used: bool = False
    self_critiqued: bool = False
    repaired: bool = False
    repair_count: int = 0
    failure_type: str | None = None
    failure_location: str | None = None
    final_answer_correct: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceProvenanceRecord(BaseModel):
    source_type: InputSourceType
    source_record_id: str
    problem_id: str
    branch_id: str
    root_branch_id: str | None = None
    parent_branch_id: str | None = None
    lineage: list[str] = Field(default_factory=list)
    route_snapshot: dict[str, Any] = Field(default_factory=dict)
    source_digest: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DistilledTraceRecord(BaseModel):
    record_id: str
    trace_id: str
    problem_id: str
    branch_id: str
    classification: DistilledTraceClass
    quality_tier: TraceQualityTier
    split: SplitName
    operator_sequence: list[str] = Field(default_factory=list)
    archetypes: list[str] = Field(default_factory=list)
    steps: list[DistilledTraceStep] = Field(default_factory=list)
    evidence: list[DistilledEvidenceRecord] = Field(default_factory=list)
    outcome: TraceOutcomeRecord
    provenance: TraceProvenanceRecord
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_positive(self) -> bool:
        return self.classification in {
            DistilledTraceClass.CORRECT_STRONG,
            DistilledTraceClass.CORRECT_WEAK,
            DistilledTraceClass.REPAIRED_SUCCESS,
        }

    @property
    def is_failure(self) -> bool:
        return not self.is_positive


class DistilledTraceArtifactMetadata(BaseModel):
    artifact_kind: str = ARTIFACT_KIND
    artifact_version: str = DEFAULT_ARTIFACT_VERSION
    schema_version: str = SCHEMA_VERSION
    artifact_id: str
    record_count: int
    problem_count: int
    source_count: int
    config_digest: str
    class_counts: dict[str, int] = Field(default_factory=dict)
    split_counts: dict[str, int] = Field(default_factory=dict)
    source_digest: str


class DistilledTraceArtifact(BaseModel):
    metadata: DistilledTraceArtifactMetadata
    records: list[DistilledTraceRecord] = Field(default_factory=list)


class TraceDistillationReport(BaseModel):
    kept_count: int
    dropped_count: int
    class_counts: dict[str, int] = Field(default_factory=dict)
    split_counts: dict[str, int] = Field(default_factory=dict)
    dropped_reasons: dict[str, int] = Field(default_factory=dict)


class DistilledArtifactWriteResult(BaseModel):
    artifact_dir: str
    manifest_path: str
    records_path: str
    split_paths: dict[str, str] = Field(default_factory=dict)


class _NormalizedInput(BaseModel):
    source_type: InputSourceType
    problem_id: str
    branch_id: str
    trace_id: str
    root_branch_id: str | None = None
    parent_branch_id: str | None = None
    lineage: list[str] = Field(default_factory=list)
    operator_sequence: list[str] = Field(default_factory=list)
    archetypes: list[str] = Field(default_factory=list)
    steps: list[DistilledTraceStep] = Field(default_factory=list)
    evidence: list[DistilledEvidenceRecord] = Field(default_factory=list)
    outcome: TraceOutcomeRecord
    route_snapshot: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class _ClassificationDecision:
    classification: DistilledTraceClass
    quality_tier: TraceQualityTier


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _stable_hash(prefix: str, payload: Any) -> str:
    raw = f"{prefix}::{_stable_json(payload)}"
    return f"{prefix}_{sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _normalize_text(text: str | None) -> str:
    return " ".join((text or "").strip().split())


def _clamp01(value: float | int | None) -> float:
    try:
        return max(0.0, min(1.0, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _as_json_value(value: Any) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _as_json_value(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, BaseModel):
        return _as_json_value(value.model_dump(mode="json"))
    if isinstance(value, (list, tuple, set)):
        return [_as_json_value(v) for v in value]
    return _normalize_text(str(value))


def _extract_bool(mapping: Mapping[str, Any], *keys: str) -> bool | None:
    for key in keys:
        if key in mapping:
            value = mapping[key]
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"1", "true", "yes", "y", "pass", "passed", "correct"}:
                    return True
                if lowered in {"0", "false", "no", "n", "fail", "failed", "incorrect"}:
                    return False
    return None


def _extract_float(mapping: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in mapping:
            try:
                return float(mapping[key])
            except (TypeError, ValueError):
                continue
    return None


def _extract_str(mapping: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            text = _normalize_text(str(mapping[key]))
            if text:
                return text
    return None


def _coerce_reasoning_step(step: ReasoningStep) -> DistilledTraceStep:
    payload = {
        "index": int(step.step_num),
        "description": _normalize_text(step.description),
        "operator": _normalize_text(step.operator_used),
        "symbolic_expression": _normalize_text(step.symbolic_expression),
        "python_code": _normalize_text(step.python_code),
        "python_result": _normalize_text(step.python_result),
    }
    return DistilledTraceStep(
        step_id=_stable_hash("distilled_step", payload),
        step_index=int(step.step_num),
        kind="reasoning",
        phase="reasoning",
        text=_normalize_text(step.description),
        operator_name=_normalize_text(step.operator_used) or None,
        summary=None,
        symbolic_expression=_normalize_text(step.symbolic_expression) or None,
        symbolic_valid=bool(step.symbolic_valid),
        python_code=_normalize_text(step.python_code) or None,
        python_result=_normalize_text(step.python_result) or None,
        metadata={},
    )


def _coerce_branch_step(step: BranchStep) -> DistilledTraceStep:
    payload = {
        "index": int(getattr(step, "index", 0)),
        "kind": getattr(getattr(step, "kind", None), "value", getattr(step, "kind", "reasoning")),
        "phase": getattr(getattr(step, "phase", None), "value", getattr(step, "phase", "reasoning")),
        "description": _normalize_text(getattr(step, "description", "")),
        "operator": _normalize_text(getattr(step, "operator_name", None)),
        "node_id": getattr(step, "node_id", None),
        "state_fingerprint": getattr(step, "state_fingerprint", None),
    }
    return DistilledTraceStep(
        step_id=_stable_hash("distilled_step", payload),
        step_index=int(getattr(step, "index", 0)),
        kind=str(getattr(getattr(step, "kind", None), "value", getattr(step, "kind", "reasoning"))),
        phase=str(getattr(getattr(step, "phase", None), "value", getattr(step, "phase", "reasoning"))),
        text=_normalize_text(getattr(step, "description", "")),
        operator_name=_normalize_text(getattr(step, "operator_name", None)) or None,
        summary=_normalize_text(getattr(step, "summary_text", None)) or None,
        node_id=getattr(step, "node_id", None),
        state_fingerprint=getattr(step, "state_fingerprint", None),
        metadata={},
    )


def _coerce_mapping_step(step: Mapping[str, Any], index: int) -> DistilledTraceStep:
    text = _extract_str(step, "text", "description", "content", "reasoning") or ""
    payload = {
        "index": index,
        "kind": _extract_str(step, "kind") or "reasoning",
        "phase": _extract_str(step, "phase") or "reasoning",
        "text": text,
        "operator": _extract_str(step, "operator_name", "operator_used", "operator"),
    }
    return DistilledTraceStep(
        step_id=_stable_hash("distilled_step", payload),
        step_index=index,
        kind=payload["kind"],
        phase=payload["phase"],
        text=text,
        operator_name=payload["operator"],
        summary=_extract_str(step, "summary"),
        symbolic_expression=_extract_str(step, "symbolic_expression"),
        symbolic_valid=_extract_bool(step, "symbolic_valid"),
        python_code=_extract_str(step, "python_code"),
        python_result=_extract_str(step, "python_result"),
        node_id=_extract_str(step, "node_id"),
        state_fingerprint=_extract_str(step, "state_fingerprint"),
        metadata={str(k): _as_json_value(v) for k, v in sorted(step.items()) if str(k) not in {
            "text", "description", "content", "reasoning", "kind", "phase", "operator_name", "operator_used",
            "operator", "summary", "symbolic_expression", "symbolic_valid", "python_code", "python_result",
            "node_id", "state_fingerprint",
        }},
    )


def _coerce_retrieval_evidence(evidence: RetrievalEvidence) -> DistilledEvidenceRecord:
    summary = _normalize_text(getattr(evidence, "problem", "")) or _normalize_text(getattr(evidence, "solution", ""))
    return DistilledEvidenceRecord(
        evidence_id=str(getattr(evidence, "evidence_id", _stable_hash("retrieval_evidence", getattr(evidence, "trace_id", "")))),
        kind="retrieval",
        score=_clamp01(getattr(evidence, "score", 0.0)),
        passed=None,
        summary=summary,
        source_id=getattr(evidence, "trace_id", None),
        metadata={
            "answer": _as_json_value(getattr(evidence, "answer", None)),
            "domain": _as_json_value(getattr(evidence, "domain", None)),
            "archetypes": _as_json_value(getattr(evidence, "archetypes", ())),
            "operators_used": _as_json_value(getattr(evidence, "operators_used", ())),
            "operator_support": _as_json_value(getattr(evidence, "operator_support", {})),
            "source": _as_json_value(getattr(evidence, "source", None)),
        },
    )


def _coerce_symbolic_evidence(evidence: SymbolicEvidence) -> DistilledEvidenceRecord:
    return DistilledEvidenceRecord(
        evidence_id=str(getattr(evidence, "evidence_id", _stable_hash("symbolic_evidence", getattr(evidence, "summary", "")))),
        kind="symbolic",
        score=_clamp01(getattr(evidence, "score", 0.0)),
        passed=bool(getattr(evidence, "passed", False)),
        summary=_normalize_text(getattr(evidence, "summary", "")),
        source_id=_extract_str({"check_name": getattr(evidence, "check_name", None)}, "check_name"),
        supporting_step_ids=list(getattr(evidence, "supporting_step_ids", ()) or ()),
        metadata={"check_name": _as_json_value(getattr(evidence, "check_name", None))},
    )


def _coerce_verifier_evidence(evidence: VerifierEvidence) -> DistilledEvidenceRecord:
    score = _clamp01(getattr(evidence, "probability", 0.0))
    return DistilledEvidenceRecord(
        evidence_id=str(getattr(evidence, "evidence_id", _stable_hash("verifier_evidence", getattr(evidence, "summary", "")))),
        kind="verifier",
        score=score,
        passed=score >= VERIFIER_PASS_THRESHOLD,
        summary=_normalize_text(getattr(evidence, "summary", "")),
        metadata={
            "logical_consistency": _as_json_value(getattr(evidence, "logical_consistency", 0.0)),
            "completeness": _as_json_value(getattr(evidence, "completeness", 0.0)),
            "repairability": _as_json_value(getattr(evidence, "repairability", 0.0)),
        },
    )


def _coerce_critique_evidence(evidence: CritiqueEvidence) -> DistilledEvidenceRecord:
    return DistilledEvidenceRecord(
        evidence_id=str(getattr(evidence, "evidence_id", _stable_hash("critique_evidence", getattr(evidence, "summary", "")))),
        kind="critique",
        score=_clamp01(getattr(evidence, "critique_score", 0.0)),
        passed=not bool(getattr(evidence, "changed_answer", False)),
        summary=_normalize_text(getattr(evidence, "summary", "")),
        metadata={
            "changed_answer": _as_json_value(getattr(evidence, "changed_answer", False)),
            "recommended_answer": _as_json_value(getattr(evidence, "recommended_answer", None)),
        },
    )


def _coerce_branch_candidate(candidate: BranchCandidateState | None) -> tuple[str | None, str | None]:
    if candidate is None:
        return None, None
    raw = _normalize_text(getattr(candidate, "raw_answer", None)) or None
    canonical = _normalize_text(getattr(candidate, "canonical_answer", None)) or None
    return raw, canonical



def _normalize_branch_trace(source: BranchTrace) -> _NormalizedInput:
    steps = [_coerce_reasoning_step(step) for step in source.steps]
    answer_raw = _normalize_text(source.answer) or None
    answer_canonical = _normalize_text(source.answer_canonical) or None
    if answer_canonical is None:
        answer_canonical = canonicalize_competition_answer(answer_raw, require_strong=False) or answer_raw

    symbolic_score = _clamp01(source.symbolic_agreement or (1.0 if source.symbolic_valid else 0.0))
    verifier_score = _clamp01(source.verifier_score)
    logical = _clamp01(source.logical_consistency or verifier_score)
    completeness = _clamp01(source.completeness or verifier_score)
    repairability = _clamp01(source.repairability or (0.75 if source.repaired else 0.0))

    evidence: list[DistilledEvidenceRecord] = []
    if steps:
        evidence.append(
            DistilledEvidenceRecord(
                evidence_id=_stable_hash("trace_evidence", {"branch_id": source.branch_id, "reasoning": source.full_reasoning}),
                kind="trace",
                score=verifier_score,
                passed=source.symbolic_valid,
                summary=_normalize_text(source.full_reasoning[:240]),
                metadata={
                    "generation_time_sec": _as_json_value(source.generation_time_sec),
                    "verifier_decomposition": _as_json_value(source.verifier_decomposition),
                    "retrieval_compatibility": _as_json_value(source.retrieval_compatibility),
                    "operator_reliability": _as_json_value(source.operator_reliability),
                },
            )
        )

    proof_obligation_summary = {
        "count": len(source.proof_obligations or []),
        "discharge_fraction": _clamp01(source.discharge_fraction),
        "open_obligation_burden": _clamp01(source.open_obligation_burden),
    }
    metadata = {
        "archetype_used": _as_json_value(source.archetype_used),
        "generation_time_sec": _as_json_value(source.generation_time_sec),
        "full_reasoning": _as_json_value(_normalize_text(source.full_reasoning)),
        "verifier_decomposition": _as_json_value(source.verifier_decomposition),
        "prm_step_quality": _as_json_value(source.prm_step_quality),
        "prm_prefix_quality": _as_json_value(source.prm_prefix_quality),
        "retrieval_support": _as_json_value(source.retrieval_support),
        "retrieval_compatibility": _as_json_value(source.retrieval_compatibility),
        "operator_reliability": _as_json_value(source.operator_reliability),
        "repair_locality": _as_json_value(source.repair_locality),
        "proof_obligation_summary": _as_json_value(proof_obligation_summary),
        "proof_obligations": _as_json_value(source.proof_obligations),
        "provenance": _as_json_value(source.provenance),
        "metadata": _as_json_value(source.metadata),
    }
    return _NormalizedInput(
        source_type=InputSourceType.BRANCH_TRACE,
        problem_id=source.problem_id,
        branch_id=source.branch_id,
        trace_id=source.branch_id,
        operator_sequence=list(source.operator_sequence),
        archetypes=[source.archetype_used] if source.archetype_used else [],
        steps=steps,
        evidence=evidence,
        outcome=TraceOutcomeRecord(
            answer_raw=answer_raw,
            answer_canonical=answer_canonical,
            verifier_score=verifier_score,
            symbolic_score=symbolic_score,
            logical_consistency=logical,
            completeness=completeness,
            repairability=repairability,
            branch_score=_clamp01(source.branch_score),
            symbolic_valid=bool(source.symbolic_valid),
            retrieval_used=bool(source.retrieval_used),
            self_critiqued=bool(source.self_critiqued),
            repaired=bool(source.repaired),
            repair_count=int(source.repair_count),
            failure_type=source.failure_type.value if isinstance(source.failure_type, FailureType) else _normalize_text(str(source.failure_type)) or None,
            failure_location=_normalize_text(source.failure_location) or None,
            final_answer_correct=None,
            metadata={
                "answer_correctness_likelihood": _clamp01(source.answer_correctness_likelihood),
                "symbolic_agreement": _clamp01(source.symbolic_agreement),
                "step_quality": _clamp01(source.step_quality),
                "prefix_quality": _clamp01(source.prefix_quality),
                "prm_step_quality": _clamp01(source.prm_step_quality),
                "prm_prefix_quality": _clamp01(source.prm_prefix_quality),
                "retrieval_support": _clamp01(source.retrieval_support),
                "retrieval_compatibility": _clamp01(source.retrieval_compatibility),
                "operator_reliability": _clamp01(source.operator_reliability),
                "repair_locality": _clamp01(source.repair_locality),
                "discharge_fraction": _clamp01(source.discharge_fraction),
                "open_obligation_burden": _clamp01(source.open_obligation_burden),
                "verifier_decomposition": _as_json_value(source.verifier_decomposition),
            },
        ),
        route_snapshot={
            "compute_signals": _as_json_value(source.metadata.get("route_compute_signals", {})) if isinstance(source.metadata, Mapping) else {},
            "source_split": _as_json_value(source.source_split),
        },
        metadata=metadata,
    )


def _normalize_branch_state(source: BranchState) -> _NormalizedInput:
    active_steps = list(source.active_steps()) if hasattr(source, "active_steps") else list(getattr(source, "steps", []))
    steps = [_coerce_branch_step(step) for step in active_steps]
    evidence: list[DistilledEvidenceRecord] = []
    evidence.extend(_coerce_retrieval_evidence(ev) for ev in getattr(source, "active_retrieval_evidence", lambda: tuple())())
    evidence.extend(_coerce_symbolic_evidence(ev) for ev in getattr(source, "active_symbolic_evidence", lambda: tuple())())
    evidence.extend(_coerce_verifier_evidence(ev) for ev in getattr(source, "active_verifier_evidence", lambda: tuple())())
    evidence.extend(_coerce_critique_evidence(ev) for ev in getattr(source, "active_critique_evidence", lambda: tuple())())

    candidate = source.current_candidate() if hasattr(source, "current_candidate") else None
    answer_raw, answer_canonical = _coerce_branch_candidate(candidate)
    if answer_canonical is None:
        answer_canonical = canonicalize_competition_answer(answer_raw, require_strong=False) or answer_raw

    symbolic_items = [ev for ev in evidence if ev.kind == "symbolic"]
    verifier_items = [ev for ev in evidence if ev.kind == "verifier"]
    critique_items = [ev for ev in evidence if ev.kind == "critique"]
    retrieval_items = [ev for ev in evidence if ev.kind == "retrieval"]

    symbolic_score = max((_clamp01(ev.score) for ev in symbolic_items), default=0.0)
    verifier_score = max((_clamp01(ev.score) for ev in verifier_items), default=_clamp01(source.score_breakdown.verifier_probability if getattr(source, "score_breakdown", None) else 0.0))
    logical_consistency = max((_clamp01(ev.metadata.get("logical_consistency", 0.0)) for ev in verifier_items), default=verifier_score)
    completeness = max((_clamp01(ev.metadata.get("completeness", 0.0)) for ev in verifier_items), default=0.0)
    repairability = max((_clamp01(ev.metadata.get("repairability", 0.0)) for ev in verifier_items), default=0.75 if getattr(source.active_cursor, "repair_count", 0) > 0 else 0.0)

    symbolic_valid = any(ev.passed for ev in symbolic_items) if symbolic_items else False
    route = getattr(source, "route", None)
    parsed_problem = getattr(source, "parsed_problem", None)
    problem_text = _normalize_text(getattr(parsed_problem, "raw_text", None))
    route_snapshot = {
        "problem_type": _as_json_value(getattr(route, "problem_type", {})),
        "archetypes": _as_json_value(getattr(route, "archetypes", {})),
        "difficulty": _as_json_value(getattr(getattr(route, "difficulty", None), "value", getattr(route, "difficulty", None))),
        "branch_budget": _as_json_value(getattr(route, "branch_budget", None)),
        "retrieval_depth": _as_json_value(getattr(route, "retrieval_depth", None)),
        "compute_signals": _as_json_value(getattr(route, "compute_signals", {})),
        "problem_text": problem_text,
        "domain": _as_json_value(getattr(getattr(parsed_problem, "domain", None), "value", getattr(parsed_problem, "domain", None))),
        "target": _as_json_value(getattr(parsed_problem, "target", None)),
        "answer_type": _as_json_value(getattr(parsed_problem, "answer_type", None)),
    }

    archetypes = sorted(str(k) for k, v in getattr(route, "archetypes", {}).items() if float(v) > 0.0)
    operator_sequence = [step.operator_name for step in steps if step.operator_name]
    operator_sequence.extend([str(item) for item in getattr(getattr(source, "metadata", {}), "get", lambda *args, **kwargs: [])("operator_sequence", []) or [] if item])
    operator_sequence = list(dict.fromkeys(operator_sequence))

    metadata = {
        "phase": _as_json_value(getattr(getattr(source, "phase", None), "value", getattr(source, "phase", None))),
        "root_node_id": _as_json_value(getattr(source, "root_node_id", None)),
        "latest_node_id": _as_json_value(getattr(source, "latest_node_id", None)),
        "latest_state_fingerprint": _as_json_value(getattr(source, "latest_state_fingerprint", None)),
        "score_breakdown": _as_json_value(getattr(getattr(source, "score_breakdown", None), "__dict__", {})),
        "repair_history": _as_json_value([getattr(item, "__dict__", item) for item in getattr(source, "repair_history", [])]),
        "patch_history": _as_json_value([getattr(item, "__dict__", item) for item in getattr(source, "patch_history", [])]),
        "problem_text": problem_text,
        "target": _as_json_value(getattr(parsed_problem, "target", None)),
        "answer_type": _as_json_value(getattr(parsed_problem, "answer_type", None)),
        "domain": _as_json_value(getattr(getattr(parsed_problem, "domain", None), "value", getattr(parsed_problem, "domain", None))),
        "calibration_metadata": _as_json_value(getattr(route, "calibration_metadata", {})),
    }

    return _NormalizedInput(
        source_type=InputSourceType.BRANCH_STATE,
        problem_id=source.problem_id,
        branch_id=source.branch_id,
        trace_id=source.branch_id,
        root_branch_id=getattr(source, "root_branch_id", None),
        parent_branch_id=getattr(source, "parent_branch_id", None),
        lineage=list(getattr(source, "lineage", ()) or ()),
        operator_sequence=operator_sequence,
        archetypes=archetypes,
        steps=steps,
        evidence=evidence,
        outcome=TraceOutcomeRecord(
            answer_raw=answer_raw,
            answer_canonical=answer_canonical,
            verifier_score=verifier_score,
            symbolic_score=symbolic_score,
            logical_consistency=logical_consistency,
            completeness=completeness,
            repairability=repairability,
            branch_score=_clamp01(source.composite_score() if hasattr(source, "composite_score") else 0.0),
            symbolic_valid=symbolic_valid,
            retrieval_used=bool(retrieval_items),
            self_critiqued=bool(critique_items),
            repaired=bool(getattr(getattr(source, "active_cursor", None), "repair_count", 0)),
            repair_count=int(getattr(getattr(source, "active_cursor", None), "repair_count", 0)),
            failure_type=_extract_str(getattr(source, "metadata", {}), "failure_type") or None,
            failure_location=_extract_str(getattr(source, "metadata", {}), "failure_location") or None,
            final_answer_correct=None,
            metadata={
                "retrieval_compatibility": max((_clamp01(ev.metadata.get("compatibility", ev.metadata.get("retrieval_compatibility", 0.0))) for ev in retrieval_items), default=0.0),
                "retrieval_support": max((_clamp01(ev.score) for ev in retrieval_items), default=0.0),
                "operator_reliability": _clamp01(_extract_float(getattr(source, "metadata", {}), "operator_reliability") or 0.0),
                "verifier_decomposition": {
                    "logical_consistency": logical_consistency,
                    "completeness": completeness,
                    "repairability": repairability,
                },
                "prm_prefix_quality": _clamp01(_extract_float(getattr(source, "metadata", {}), "prm_prefix_quality", "prefix_quality") or 0.0),
                "proof_obligation_burden": _clamp01(_extract_float(getattr(source, "metadata", {}), "open_obligation_burden") or 0.0),
            },
        ),
        route_snapshot=route_snapshot,
        metadata=metadata,
    )


def _normalize_mapping(source: Mapping[str, Any]) -> _NormalizedInput:
    problem_id = _extract_str(source, "problem_id") or "unknown_problem"
    branch_id = _extract_str(source, "branch_id", "trace_id", "record_id") or _stable_hash("branch", source)
    steps_raw = source.get("steps", ()) or ()
    steps = [_coerce_mapping_step(step, index + 1) for index, step in enumerate(steps_raw) if isinstance(step, Mapping)]

    evidence: list[DistilledEvidenceRecord] = []
    for index, item in enumerate(source.get("evidence", ()) or ()):
        if not isinstance(item, Mapping):
            continue
        evidence.append(
            DistilledEvidenceRecord(
                evidence_id=_extract_str(item, "evidence_id") or _stable_hash("evidence", {"idx": index, **dict(item)}),
                kind=_extract_str(item, "kind", "type") or "other",
                score=_extract_float(item, "score", "probability"),
                passed=_extract_bool(item, "passed"),
                summary=_extract_str(item, "summary", "text", "description") or "",
                source_id=_extract_str(item, "source_id", "trace_id"),
                supporting_step_ids=list(item.get("supporting_step_ids", ()) or ()),
                metadata={str(k): _as_json_value(v) for k, v in sorted(item.items()) if str(k) not in {
                    "evidence_id", "kind", "type", "score", "probability", "passed", "summary", "text", "description", "source_id", "trace_id", "supporting_step_ids"
                }},
            )
        )

    answer_raw = _extract_str(source, "answer", "answer_raw")
    answer_canonical = _extract_str(source, "answer_canonical") or canonicalize_competition_answer(answer_raw, require_strong=False) or answer_raw
    outcome = TraceOutcomeRecord(
        answer_raw=answer_raw,
        answer_canonical=answer_canonical,
        verifier_score=_clamp01(_extract_float(source, "verifier_score", "probability", "final_score") or 0.0),
        symbolic_score=_clamp01(_extract_float(source, "symbolic_score") or (1.0 if _extract_bool(source, "symbolic_valid") else 0.0)),
        logical_consistency=_clamp01(_extract_float(source, "logical_consistency") or 0.0),
        completeness=_clamp01(_extract_float(source, "completeness") or 0.0),
        repairability=_clamp01(_extract_float(source, "repairability") or 0.0),
        branch_score=_clamp01(_extract_float(source, "branch_score", "composite_score") or 0.0),
        symbolic_valid=bool(_extract_bool(source, "symbolic_valid") or False),
        retrieval_used=bool(_extract_bool(source, "retrieval_used") or False),
        self_critiqued=bool(_extract_bool(source, "self_critiqued") or False),
        repaired=bool(_extract_bool(source, "repaired") or False),
        repair_count=int(source.get("repair_count", 0) or 0),
        failure_type=_extract_str(source, "failure_type"),
        failure_location=_extract_str(source, "failure_location"),
        final_answer_correct=_extract_bool(source, "final_answer_correct", "is_correct", "answer_correct"),
        metadata={
            "answer_correctness_likelihood": _clamp01(_extract_float(source, "answer_correctness_likelihood") or 0.0),
            "symbolic_agreement": _clamp01(_extract_float(source, "symbolic_agreement") or 0.0),
            "step_quality": _clamp01(_extract_float(source, "step_quality", "prm_step_quality") or 0.0),
            "prefix_quality": _clamp01(_extract_float(source, "prefix_quality", "prm_prefix_quality") or 0.0),
            "prm_step_quality": _clamp01(_extract_float(source, "prm_step_quality") or 0.0),
            "prm_prefix_quality": _clamp01(_extract_float(source, "prm_prefix_quality") or 0.0),
            "retrieval_support": _clamp01(_extract_float(source, "retrieval_support") or 0.0),
            "retrieval_compatibility": _clamp01(_extract_float(source, "retrieval_compatibility") or 0.0),
            "operator_reliability": _clamp01(_extract_float(source, "operator_reliability") or 0.0),
            "discharge_fraction": _clamp01(_extract_float(source, "discharge_fraction") or 0.0),
            "open_obligation_burden": _clamp01(_extract_float(source, "open_obligation_burden") or 0.0),
            "verifier_decomposition": _as_json_value(source.get("verifier_decomposition", {})),
        },
    )

    operator_sequence = [str(item) for item in source.get("operator_sequence", ()) or () if str(item).strip()]
    archetypes = [str(item) for item in source.get("archetypes", ()) or () if str(item).strip()]
    metadata = {str(k): _as_json_value(v) for k, v in sorted(source.items()) if str(k) not in {
        "problem_id", "branch_id", "trace_id", "record_id", "steps", "evidence", "answer", "answer_raw", "answer_canonical",
        "verifier_score", "probability", "final_score", "symbolic_score", "symbolic_valid", "logical_consistency", "completeness",
        "repairability", "branch_score", "composite_score", "retrieval_used", "self_critiqued", "repaired", "repair_count",
        "failure_type", "failure_location", "final_answer_correct", "is_correct", "answer_correct", "operator_sequence", "archetypes",
    }}

    return _NormalizedInput(
        source_type=InputSourceType.GENERIC_MAPPING,
        problem_id=problem_id,
        branch_id=branch_id,
        trace_id=branch_id,
        root_branch_id=_extract_str(source, "root_branch_id"),
        parent_branch_id=_extract_str(source, "parent_branch_id"),
        lineage=list(source.get("lineage", ()) or ()),
        operator_sequence=operator_sequence,
        archetypes=archetypes,
        steps=steps,
        evidence=evidence,
        outcome=outcome,
        route_snapshot={str(k): _as_json_value(v) for k, v in sorted((source.get("route_snapshot") or {}).items())} if isinstance(source.get("route_snapshot"), Mapping) else {},
        metadata=metadata,
    )


def normalize_trace_source(source: BranchTrace | BranchState | Mapping[str, Any]) -> _NormalizedInput:
    if isinstance(source, BranchTrace):
        return _normalize_branch_trace(source)
    if hasattr(source, "active_steps") and hasattr(source, "current_candidate") and hasattr(source, "route"):
        return _normalize_branch_state(source)  # type: ignore[arg-type]
    if isinstance(source, Mapping):
        return _normalize_mapping(source)
    raise TypeError(f"Unsupported trace source type: {type(source)!r}")


def _build_verifier_label_index(labels: Mapping[str, VerifierLabel] | Sequence[VerifierLabel] | None) -> dict[str, VerifierLabel]:
    if labels is None:
        return {}
    if isinstance(labels, Mapping):
        out: dict[str, VerifierLabel] = {}
        for key, value in labels.items():
            if isinstance(value, VerifierLabel):
                out[str(key)] = value
        return out
    out = {}
    for label in labels:
        if isinstance(label, VerifierLabel):
            out[label.branch_id] = label
    return out


def _attach_label(input_record: _NormalizedInput, label: VerifierLabel | None) -> _NormalizedInput:
    if label is None:
        return input_record
    input_record.outcome.verifier_score = max(input_record.outcome.verifier_score, _clamp01(label.composite_score))
    input_record.outcome.logical_consistency = max(input_record.outcome.logical_consistency, _clamp01(label.logical_consistency))
    input_record.outcome.completeness = max(input_record.outcome.completeness, _clamp01(label.completeness))
    input_record.outcome.repairability = max(input_record.outcome.repairability, _clamp01(label.repairability))
    input_record.outcome.symbolic_score = max(input_record.outcome.symbolic_score, _clamp01(label.symbolic_agreement))
    input_record.outcome.symbolic_valid = input_record.outcome.symbolic_valid or bool(label.symbolic_agreement >= 0.5)
    if label.final_answer_correct in {0, 1}:
        input_record.outcome.final_answer_correct = bool(label.final_answer_correct)
    if not input_record.outcome.failure_type and label.failure_type:
        input_record.outcome.failure_type = _normalize_text(label.failure_type) or None
    if not input_record.outcome.failure_location and label.failure_location:
        input_record.outcome.failure_location = _normalize_text(label.failure_location) or None
    input_record.metadata["verifier_label"] = _as_json_value(label.model_dump(mode="json"))
    return input_record


def _should_keep(input_record: _NormalizedInput, config: DistillationConfig) -> tuple[bool, str | None]:
    if len(input_record.steps) < config.min_step_count:
        return False, "too_few_steps"
    reasoning_chars = sum(len(step.text) for step in input_record.steps)
    if reasoning_chars < config.min_reasoning_chars:
        return False, "too_short"
    if not input_record.outcome.answer_raw and not config.preserve_rejected_without_answer and input_record.outcome.failure_type is None:
        return False, "no_answer_or_failure"
    return True, None


def _classification(input_record: _NormalizedInput, config: DistillationConfig) -> _ClassificationDecision:
    outcome = input_record.outcome

    has_answer = bool(_normalize_text(outcome.answer_raw))
    explicit_correct = outcome.final_answer_correct
    verifier_good = outcome.verifier_score >= config.weak_verifier_threshold
    symbolic_good = outcome.symbolic_valid or outcome.symbolic_score >= config.weak_symbolic_threshold
    repaired = outcome.repaired or outcome.repair_count > 0

    inferred_success = has_answer and verifier_good and symbolic_good
    success = explicit_correct if explicit_correct is not None else inferred_success

    if not symbolic_good:
        if repaired:
            return _ClassificationDecision(DistilledTraceClass.REPAIRED_FAIL, TraceQualityTier.REPAIRED)
        return _ClassificationDecision(DistilledTraceClass.SYMBOLIC_CONTRADICTED, TraceQualityTier.FAILED)

    if repaired:
        if success:
            return _ClassificationDecision(DistilledTraceClass.REPAIRED_SUCCESS, TraceQualityTier.REPAIRED)
        return _ClassificationDecision(DistilledTraceClass.REPAIRED_FAIL, TraceQualityTier.REPAIRED)

    if success:
        is_strong = (
            outcome.verifier_score >= config.strong_verifier_threshold
            and outcome.symbolic_score >= config.strong_symbolic_threshold
            and outcome.completeness >= config.strong_completeness_threshold
        )
        if is_strong:
            return _ClassificationDecision(DistilledTraceClass.CORRECT_STRONG, TraceQualityTier.STRONG)
        return _ClassificationDecision(DistilledTraceClass.CORRECT_WEAK, TraceQualityTier.WEAK)

    return _ClassificationDecision(DistilledTraceClass.VERIFIER_REJECTED, TraceQualityTier.FAILED)


def _assign_split(problem_id: str, config: DistillationConfig) -> SplitName:
    ratios = config.normalized_split_ratios()
    token = f"{config.split_salt}::{problem_id}"
    bucket = int(sha1(token.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    train_cut = ratios[0]
    valid_cut = ratios[0] + ratios[1]
    if bucket < train_cut:
        return SplitName.TRAIN
    if bucket < valid_cut:
        return SplitName.VALID
    return SplitName.CALIBRATION


def _record_id(input_record: _NormalizedInput, classification: DistilledTraceClass) -> str:
    payload = {
        "problem_id": input_record.problem_id,
        "branch_id": input_record.branch_id,
        "classification": classification.value,
        "answer": input_record.outcome.answer_canonical,
        "operator_sequence": input_record.operator_sequence,
        "steps": [{
            "index": step.step_index,
            "kind": step.kind,
            "phase": step.phase,
            "text": step.text,
            "operator_name": step.operator_name,
        } for step in input_record.steps],
    }
    return _stable_hash("distilled_trace", payload)


def distill_traces(
    sources: Sequence[BranchTrace | BranchState | Mapping[str, Any]],
    *,
    verifier_labels: Mapping[str, VerifierLabel] | Sequence[VerifierLabel] | None = None,
    config: DistillationConfig | None = None,
) -> tuple[DistilledTraceArtifact, TraceDistillationReport]:
    """Distill structured trajectories into a deterministic training artifact."""

    resolved_config = config or DistillationConfig()
    label_index = _build_verifier_label_index(verifier_labels)

    kept_records: list[DistilledTraceRecord] = []
    dropped_reasons: dict[str, int] = {}

    normalized_inputs = [normalize_trace_source(source) for source in sources]
    for input_record in normalized_inputs:
        input_record = _attach_label(input_record, label_index.get(input_record.branch_id))
        keep, reason = _should_keep(input_record, resolved_config)
        if not keep:
            dropped_reasons[reason or "filtered"] = dropped_reasons.get(reason or "filtered", 0) + 1
            continue

        decision = _classification(input_record, resolved_config)
        split = _assign_split(input_record.problem_id, resolved_config)
        source_payload = input_record.model_dump(mode="json")
        source_digest = _stable_hash("trace_source", source_payload)
        provenance = TraceProvenanceRecord(
            source_type=input_record.source_type,
            source_record_id=input_record.trace_id,
            problem_id=input_record.problem_id,
            branch_id=input_record.branch_id,
            root_branch_id=input_record.root_branch_id,
            parent_branch_id=input_record.parent_branch_id,
            lineage=list(input_record.lineage),
            route_snapshot=input_record.route_snapshot,
            source_digest=source_digest,
            metadata={
                "normalization_metadata": _as_json_value(input_record.metadata),
            },
        )
        kept_records.append(
            DistilledTraceRecord(
                record_id=_record_id(input_record, decision.classification),
                trace_id=input_record.trace_id,
                problem_id=input_record.problem_id,
                branch_id=input_record.branch_id,
                classification=decision.classification,
                quality_tier=decision.quality_tier,
                split=split,
                operator_sequence=list(dict.fromkeys(input_record.operator_sequence)),
                archetypes=sorted(dict.fromkeys(input_record.archetypes)),
                steps=list(input_record.steps),
                evidence=list(input_record.evidence),
                outcome=input_record.outcome,
                provenance=provenance,
                metadata={
                    "step_count": len(input_record.steps),
                    "evidence_count": len(input_record.evidence),
                    "reasoning_char_count": sum(len(step.text) for step in input_record.steps),
                    "classification_basis": {
                        "verifier_score": input_record.outcome.verifier_score,
                        "symbolic_score": input_record.outcome.symbolic_score,
                        "completeness": input_record.outcome.completeness,
                        "repaired": input_record.outcome.repaired,
                        "repair_count": input_record.outcome.repair_count,
                        "final_answer_correct": input_record.outcome.final_answer_correct,
                        "outcome_metadata": _as_json_value(input_record.outcome.metadata),
                        "route_compute_signals": _as_json_value(input_record.route_snapshot.get("compute_signals", {})),
                    },
                },
            )
        )

    kept_records.sort(key=lambda item: (item.problem_id, item.branch_id, item.classification.value, item.record_id))

    class_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for record in kept_records:
        class_counts[record.classification.value] = class_counts.get(record.classification.value, 0) + 1
        split_counts[record.split.value] = split_counts.get(record.split.value, 0) + 1

    source_digest = _stable_hash(
        "distilled_sources",
        [{"problem_id": rec.problem_id, "branch_id": rec.branch_id, "record_id": rec.record_id} for rec in kept_records],
    )
    config_digest = _stable_hash("distillation_config", resolved_config.model_dump(mode="json"))
    artifact_id = _stable_hash(
        "distilled_trace_artifact",
        {
            "artifact_version": resolved_config.artifact_version,
            "config_digest": config_digest,
            "record_ids": [record.record_id for record in kept_records],
        },
    )
    artifact = DistilledTraceArtifact(
        metadata=DistilledTraceArtifactMetadata(
            artifact_version=resolved_config.artifact_version,
            schema_version=resolved_config.schema_version,
            artifact_id=artifact_id,
            record_count=len(kept_records),
            problem_count=len({record.problem_id for record in kept_records}),
            source_count=len(normalized_inputs),
            config_digest=config_digest,
            class_counts=class_counts,
            split_counts=split_counts,
            source_digest=source_digest,
        ),
        records=kept_records,
    )
    report = TraceDistillationReport(
        kept_count=len(kept_records),
        dropped_count=len(normalized_inputs) - len(kept_records),
        class_counts=class_counts,
        split_counts=split_counts,
        dropped_reasons=dropped_reasons,
    )
    return artifact, report


def artifact_records_by_split(artifact: DistilledTraceArtifact) -> dict[SplitName, list[DistilledTraceRecord]]:
    out: dict[SplitName, list[DistilledTraceRecord]] = {
        SplitName.TRAIN: [],
        SplitName.VALID: [],
        SplitName.CALIBRATION: [],
    }
    for record in artifact.records:
        out[record.split].append(record)
    return out


def write_distilled_artifact(
    artifact: DistilledTraceArtifact,
    output_dir: str | Path,
    *,
    include_split_files: bool = True,
) -> DistilledArtifactWriteResult:
    """Write manifest + deterministic JSONL export for downstream offline jobs."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = out_dir / artifact.metadata.artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = artifact_dir / "manifest.json"
    records_path = artifact_dir / "records.jsonl"

    manifest_payload = artifact.metadata.model_dump(mode="json")
    manifest_path.write_text(_stable_json(manifest_payload) + "\n", encoding="utf-8")

    with records_path.open("w", encoding="utf-8") as handle:
        for record in artifact.records:
            handle.write(_stable_json(record.model_dump(mode="json")) + "\n")

    split_paths: dict[str, str] = {}
    if include_split_files:
        for split, records in artifact_records_by_split(artifact).items():
            split_path = artifact_dir / f"{split.value}.jsonl"
            with split_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(_stable_json(record.model_dump(mode="json")) + "\n")
            split_paths[split.value] = str(split_path)

    return DistilledArtifactWriteResult(
        artifact_dir=str(artifact_dir),
        manifest_path=str(manifest_path),
        records_path=str(records_path),
        split_paths=split_paths,
    )


def load_distilled_artifact(path: str | Path) -> DistilledTraceArtifact:
    """Load a written artifact from its manifest directory or manifest path."""

    input_path = Path(path)
    if input_path.is_dir():
        manifest_path = input_path / "manifest.json"
        records_path = input_path / "records.jsonl"
    elif input_path.name == "manifest.json":
        manifest_path = input_path
        records_path = input_path.with_name("records.jsonl")
    else:
        raise ValueError(f"Unsupported artifact path: {path}")

    metadata = DistilledTraceArtifactMetadata.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    records: list[DistilledTraceRecord] = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(DistilledTraceRecord.model_validate_json(stripped))
    return DistilledTraceArtifact(metadata=metadata, records=records)


def build_trace_lookup(artifact: DistilledTraceArtifact) -> dict[str, DistilledTraceRecord]:
    return {record.record_id: record for record in artifact.records}


__all__ = [
    "ARTIFACT_KIND",
    "DEFAULT_ARTIFACT_VERSION",
    "SCHEMA_VERSION",
    "DistillationConfig",
    "DistilledArtifactWriteResult",
    "DistilledEvidenceRecord",
    "DistilledTraceArtifact",
    "DistilledTraceArtifactMetadata",
    "DistilledTraceClass",
    "DistilledTraceRecord",
    "DistilledTraceStep",
    "InputSourceType",
    "SplitName",
    "TraceDistillationReport",
    "TraceOutcomeRecord",
    "TraceProvenanceRecord",
    "TraceQualityTier",
    "artifact_records_by_split",
    "build_trace_lookup",
    "distill_traces",
    "load_distilled_artifact",
    "normalize_trace_source",
    "write_distilled_artifact",
]
