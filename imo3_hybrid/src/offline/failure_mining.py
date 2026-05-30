from __future__ import annotations

"""Deterministic offline failure mining for hard-case learning loops.

This module turns historical solve outcomes into typed failure records,
slice metrics, and ranked repair suggestions that are directly actionable for
verifier, symbolic, routing, retrieval, operator, and search improvements.
"""

from collections import defaultdict
from enum import Enum
from hashlib import sha1
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.aggregation.canonicalize import canonicalize_competition_answer
from src.common.schemas import (
    BranchTrace,
    FailureBucket,
    FailureCause,
    FailureRecord,
    HistoricalSolveRecord,
)
from src.offline.trace_distillation import DistilledTraceArtifact, DistilledTraceRecord, load_distilled_artifact


SCHEMA_VERSION = "failure_mining.v1"
ARTIFACT_KIND = "failure_mining_run"


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        populate_by_name=True,
    )


class FailureMiningStatus(str, Enum):
    COMPLETED = "completed"
    INSUFFICIENT_INPUT = "insufficient_input"


class FailureMiningConfig(StrictModel):
    min_hard_problems: int = Field(default=20, ge=1)
    max_records: int | None = Field(default=None, ge=1)
    require_gold_for_failure: bool = True
    hard_difficulty_labels: tuple[str, ...] = ("hard", "very_hard")
    hard_branch_budget_threshold: int = 20
    hard_depth_threshold: int = 4
    shallow_step_threshold: int = 3
    verifier_gap_threshold: float = 0.08
    confidence_threshold: float = 0.55
    calibration_overconfidence_threshold: float = 0.75
    artifact_version: str = "v1"


class FailurePatternSummary(StrictModel):
    failure_bucket: FailureBucket
    subtype: str
    count: int
    average_weight: float
    average_confidence: float
    estimated_score_gain: float
    recommended_fix: str


class SuggestedRepair(StrictModel):
    failure_type: str
    target_file: str
    exact_fix_direction: str
    expected_roi: float


class FailureMiningManifest(StrictModel):
    artifact_kind: str = ARTIFACT_KIND
    schema_version: str = SCHEMA_VERSION
    artifact_version: str = "v1"
    run_id: str
    status: FailureMiningStatus
    total_records: int
    labeled_records: int
    failed_records: int
    skipped_missing_gold: int
    notes: list[str] = Field(default_factory=list)


class FailureMiningResult(StrictModel):
    manifest: FailureMiningManifest
    failures: tuple[FailureRecord, ...] = Field(default_factory=tuple)
    failure_summary: dict[str, Any] = Field(default_factory=dict)
    slice_metrics: dict[str, Any] = Field(default_factory=dict)
    suggested_repairs: tuple[SuggestedRepair, ...] = Field(default_factory=tuple)


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _stable_hash(prefix: str, payload: Any) -> str:
    raw = f"{prefix}::{_stable_json(payload)}"
    return f"{prefix}_{sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}


def _parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _canonical_answer(value: str | None) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    canonical = canonicalize_competition_answer(text, require_strong=False)
    if canonical:
        return canonical
    return text.replace(",", "")


def _problem_text_from_row(row: Mapping[str, Any]) -> str:
    for key in ("raw_problem_text", "problem_text", "problem", "question"):
        value = row.get(key)
        if value is not None and _normalize_text(value):
            return _normalize_text(value)
    return ""


def _gold_answer_from_row(row: Mapping[str, Any], final_prediction: Mapping[str, Any]) -> str | None:
    for key in ("gold_answer", "expected_answer", "target", "gold"):
        if row.get(key) is not None:
            return str(row.get(key))
    if final_prediction.get("gold_answer") is not None:
        return str(final_prediction.get("gold_answer"))
    return None


def _predicted_answer_from_row(row: Mapping[str, Any], final_prediction: Mapping[str, Any]) -> str | None:
    for key in ("predicted_answer", "final_answer", "answer", "prediction"):
        if row.get(key) is not None:
            return str(row.get(key))
    if final_prediction.get("final_answer") is not None:
        return str(final_prediction.get("final_answer"))
    return None


def _route_snapshot_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("route_snapshot", "route", "route_decision"):
        value = row.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _symbolic_snapshot_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("symbolic_snapshot", "symbolic", "symbolic_outputs"):
        value = row.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _verifier_snapshot_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("verifier_snapshot", "verifier", "verifier_scores"):
        value = row.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _selector_snapshot_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("final_selector_diagnostics", "selector", "aggregation", "selection"):
        value = row.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _final_prediction_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("final_prediction")
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _coerce_branch_trace(problem_id: str, row: Mapping[str, Any], fallback_index: int) -> BranchTrace:
    branch_id = str(row.get("branch_id") or row.get("id") or f"{problem_id}::branch::{fallback_index:04d}")
    operator_sequence = row.get("operator_sequence")
    if not isinstance(operator_sequence, Sequence) or isinstance(operator_sequence, (str, bytes)):
        operator_sequence = []
    verifier_decomposition = row.get("verifier_decomposition")
    if not isinstance(verifier_decomposition, Mapping):
        verifier_decomposition = {}
    metadata = dict(row.get("metadata") or {})
    for key in ("symbolic_status", "contradicted_obligation_count", "discharged_obligation_count"):
        if key in row and key not in metadata:
            metadata[key] = row.get(key)
    if "route_compute_signals" not in metadata:
        route_compute_signals = row.get("route_compute_signals") or row.get("compute_signals") or {}
        if isinstance(route_compute_signals, Mapping):
            metadata["route_compute_signals"] = dict(route_compute_signals)
    if "phase" in row and "phase" not in metadata:
        metadata["phase"] = row.get("phase")
    return BranchTrace(
        branch_id=branch_id,
        problem_id=problem_id,
        steps=[],
        full_reasoning=str(row.get("full_reasoning") or row.get("reasoning") or row.get("trace") or ""),
        answer=None if row.get("answer") is None else str(row.get("answer")),
        answer_canonical=None if row.get("answer_canonical") is None else str(row.get("answer_canonical")),
        symbolic_valid=_parse_bool(row.get("symbolic_valid", False)),
        branch_score=_parse_float(row.get("branch_score", row.get("composite_score", 0.0))),
        verifier_score=_parse_float(row.get("verifier_score", row.get("probability", 0.0))),
        tool_consistency=_parse_float(row.get("tool_consistency", 0.0)),
        logical_consistency=_parse_float(row.get("logical_consistency", 0.0)),
        completeness=_parse_float(row.get("completeness", 0.0)),
        repairability=_parse_float(row.get("repairability", 0.0)),
        answer_correctness_likelihood=_parse_float(row.get("answer_correctness_likelihood", 0.0)),
        symbolic_agreement=_parse_float(row.get("symbolic_agreement", 0.0)),
        step_quality=_parse_float(row.get("step_quality", row.get("prm_step_quality", 0.0))),
        prefix_quality=_parse_float(row.get("prefix_quality", row.get("prm_prefix_quality", 0.0))),
        prm_prefix_quality=_parse_float(row.get("prm_prefix_quality", row.get("prefix_quality", 0.0))),
        prm_step_quality=_parse_float(row.get("prm_step_quality", 0.0)),
        retrieval_compatibility=_parse_float(row.get("retrieval_compatibility", 0.0)),
        retrieval_support=_parse_float(row.get("retrieval_support", 0.0)),
        operator_reliability=_parse_float(row.get("operator_reliability", 0.0)),
        discharge_fraction=_parse_float(row.get("discharge_fraction", 0.0)),
        open_obligation_burden=_parse_float(row.get("open_obligation_burden", 0.0)),
        verifier_decomposition={str(k): _parse_float(v) for k, v in verifier_decomposition.items()},
        self_critiqued=_parse_bool(row.get("self_critiqued", False)),
        repaired=_parse_bool(row.get("repaired", False)),
        repair_count=_parse_int(row.get("repair_count", 0)),
        operator_sequence=[str(item) for item in operator_sequence if _normalize_text(item)],
        archetype_used=None if row.get("archetype_used") is None else str(row.get("archetype_used")),
        retrieval_used=_parse_bool(row.get("retrieval_used", False)),
        generation_time_sec=_parse_float(row.get("generation_time_sec", 0.0)),
        failure_type=row.get("failure_type"),
        source_split=None if row.get("source_split") is None else str(row.get("source_split")),
        metadata=metadata,
    )


def _record_from_mapping(row: Mapping[str, Any]) -> HistoricalSolveRecord:
    problem_id = _normalize_text(row.get("problem_id") or row.get("id"))
    if not problem_id:
        raise ValueError("Historical solve row is missing problem_id")
    final_prediction = _final_prediction_payload(row)
    branches_payload = row.get("branch_traces")
    if not isinstance(branches_payload, list):
        branches_payload = row.get("branches")
    if not isinstance(branches_payload, list):
        branches_payload = row.get("all_branches")
    if not isinstance(branches_payload, list):
        branches_payload = []
    branches = [
        _coerce_branch_trace(problem_id, item, index)
        for index, item in enumerate(branches_payload)
        if isinstance(item, Mapping)
    ]
    difficulty = (
        _normalize_text(row.get("difficulty"))
        or _normalize_text(_route_snapshot_from_row(row).get("difficulty"))
        or _normalize_text(final_prediction.get("provenance", {}).get("difficulty") if isinstance(final_prediction.get("provenance"), Mapping) else "")
        or "unknown"
    )
    domain = (
        _normalize_text(row.get("domain"))
        or _normalize_text(_route_snapshot_from_row(row).get("domain"))
        or _normalize_text(final_prediction.get("provenance", {}).get("domain") if isinstance(final_prediction.get("provenance"), Mapping) else "")
        or "unknown"
    )
    route_snapshot = _route_snapshot_from_row(row)
    verifier_snapshot = _verifier_snapshot_from_row(row)
    symbolic_snapshot = _symbolic_snapshot_from_row(row)
    selector_snapshot = _selector_snapshot_from_row(row)
    provenance = final_prediction.get("provenance")
    if isinstance(provenance, Mapping):
        if not route_snapshot and isinstance(provenance.get("route_snapshot"), Mapping):
            route_snapshot = dict(provenance.get("route_snapshot") or {})
        if not verifier_snapshot and isinstance(provenance.get("verifier_snapshot"), Mapping):
            verifier_snapshot = dict(provenance.get("verifier_snapshot") or {})
        if not symbolic_snapshot and isinstance(provenance.get("symbolic_snapshot"), Mapping):
            symbolic_snapshot = dict(provenance.get("symbolic_snapshot") or {})
        if not selector_snapshot:
            selector_snapshot = {
                "confidence": _parse_float(final_prediction.get("confidence", 0.0), 0.0),
                "method_used": str(final_prediction.get("method_used", "")),
                "signal_decomposition": dict(final_prediction.get("signal_decomposition") or {}),
                "calibration_summary": dict(final_prediction.get("calibration_summary") or {}),
            }
    if not route_snapshot:
        route_snapshot = {"status": "unavailable"}
    if not verifier_snapshot:
        verifier_snapshot = {"status": "unavailable"}
    if not symbolic_snapshot:
        symbolic_snapshot = {"status": "unavailable"}
    if not selector_snapshot:
        selector_snapshot = {"status": "unavailable"}

    return HistoricalSolveRecord(
        problem_id=problem_id,
        raw_problem_text=_problem_text_from_row(row),
        normalized_problem_text=_normalize_text(
            row.get("normalized_problem_text") or _problem_text_from_row(row)
        ),
        domain=domain or "unknown",
        difficulty=difficulty or "unknown",
        gold_answer=_gold_answer_from_row(row, final_prediction),
        predicted_answer=_predicted_answer_from_row(row, final_prediction),
        branch_traces=branches,
        route_snapshot=route_snapshot,
        verifier_snapshot=verifier_snapshot,
        symbolic_snapshot=symbolic_snapshot,
        final_selector_diagnostics=selector_snapshot,
        metadata={
            **dict(row.get("metadata") or {}),
            **({"final_prediction": final_prediction} if final_prediction else {}),
        },
    )


def _record_from_distilled_problem(problem_id: str, records: Sequence[DistilledTraceRecord]) -> HistoricalSolveRecord:
    sorted_records = sorted(
        records,
        key=lambda item: (
            -float(item.outcome.branch_score or 0.0),
            -float(item.outcome.verifier_score or 0.0),
            item.branch_id,
        ),
    )
    predicted = sorted_records[0].outcome.answer_canonical if sorted_records else None
    branches: list[BranchTrace] = []
    difficulty = "unknown"
    domain = "unknown"
    for item in sorted_records:
        metadata = dict(item.outcome.metadata or {})
        branches.append(
            BranchTrace(
                branch_id=item.branch_id,
                problem_id=item.problem_id,
                steps=[],
                full_reasoning=" ".join(step.text for step in item.steps),
                answer=item.outcome.answer_raw,
                answer_canonical=item.outcome.answer_canonical,
                symbolic_valid=bool(item.outcome.symbolic_valid),
                branch_score=float(item.outcome.branch_score or 0.0),
                verifier_score=float(item.outcome.verifier_score or 0.0),
                tool_consistency=float(item.outcome.symbolic_score or 0.0),
                logical_consistency=float(item.outcome.logical_consistency or 0.0),
                completeness=float(item.outcome.completeness or 0.0),
                repairability=float(item.outcome.repairability or 0.0),
                answer_correctness_likelihood=float(metadata.get("answer_correctness_likelihood", 0.0) or 0.0),
                symbolic_agreement=float(metadata.get("symbolic_agreement", item.outcome.symbolic_score) or 0.0),
                step_quality=float(metadata.get("step_quality", 0.0) or 0.0),
                prefix_quality=float(metadata.get("prefix_quality", 0.0) or 0.0),
                prm_prefix_quality=float(metadata.get("prm_prefix_quality", 0.0) or 0.0),
                prm_step_quality=float(metadata.get("prm_step_quality", 0.0) or 0.0),
                retrieval_compatibility=float(metadata.get("retrieval_compatibility", 0.0) or 0.0),
                retrieval_support=float(metadata.get("retrieval_support", 0.0) or 0.0),
                operator_reliability=float(metadata.get("operator_reliability", 0.0) or 0.0),
                discharge_fraction=float(metadata.get("discharge_fraction", 0.0) or 0.0),
                open_obligation_burden=float(metadata.get("open_obligation_burden", 0.0) or 0.0),
                verifier_decomposition=dict(metadata.get("verifier_decomposition") or {}),
                self_critiqued=bool(item.outcome.self_critiqued),
                repaired=bool(item.outcome.repaired),
                repair_count=int(item.outcome.repair_count or 0),
                operator_sequence=list(item.operator_sequence),
                archetype_used=item.archetypes[0] if item.archetypes else None,
                retrieval_used=bool(item.outcome.retrieval_used),
                failure_type=item.outcome.failure_type,
                failure_location=item.outcome.failure_location,
                metadata={
                    "phase": item.classification.value,
                    "source_split": item.split.value,
                    "route_compute_signals": dict(item.metadata.get("classification_basis", {}).get("route_compute_signals", {})),
                },
            )
        )
        if item.provenance.route_snapshot:
            difficulty = _normalize_text(item.provenance.route_snapshot.get("difficulty")) or difficulty
            domain = _normalize_text(item.provenance.route_snapshot.get("domain")) or domain
    return HistoricalSolveRecord(
        problem_id=problem_id,
        raw_problem_text="",
        normalized_problem_text="",
        domain=domain or "unknown",
        difficulty=difficulty or "unknown",
        gold_answer=None,
        predicted_answer=predicted,
        branch_traces=branches,
        route_snapshot={},
        verifier_snapshot={},
        symbolic_snapshot={},
        final_selector_diagnostics={},
        metadata={"source": "distilled_trace_artifact"},
    )


def normalize_failure_source(
    source: HistoricalSolveRecord | DistilledTraceArtifact | DistilledTraceRecord | Mapping[str, Any],
) -> tuple[HistoricalSolveRecord, ...]:
    if isinstance(source, HistoricalSolveRecord):
        return (source,)
    if isinstance(source, DistilledTraceArtifact):
        grouped: dict[str, list[DistilledTraceRecord]] = defaultdict(list)
        for record in source.records:
            grouped[record.problem_id].append(record)
        return tuple(_record_from_distilled_problem(problem_id, grouped[problem_id]) for problem_id in sorted(grouped))
    if isinstance(source, DistilledTraceRecord):
        return (_record_from_distilled_problem(source.problem_id, [source]),)
    if isinstance(source, Mapping):
        return (_record_from_mapping(source),)
    raise TypeError(f"Unsupported failure mining source type: {type(source)!r}")


def _branch_rank(branch: BranchTrace) -> float:
    answer_prob = _clamp01(branch.answer_correctness_likelihood or branch.verifier_score)
    shallow_penalty = max(0.0, float(branch.prefix_quality or 0.0) - float(branch.step_quality or 0.0))
    contradiction_pressure = max(
        _parse_float(branch.metadata.get("contradicted_obligation_count", 0.0), 0.0) * 0.15,
        float(branch.open_obligation_burden or 0.0) * 0.10,
    )
    return (
        0.30 * float(branch.branch_score or 0.0)
        + 0.24 * float(branch.verifier_score or 0.0)
        + 0.18 * answer_prob
        + 0.10 * float(branch.logical_consistency or 0.0)
        + 0.08 * float(branch.symbolic_agreement or 0.0)
        + 0.05 * float(branch.prefix_quality or 0.0)
        + 0.05 * float(branch.step_quality or 0.0)
        - 0.08 * float(branch.open_obligation_burden or 0.0)
        - 0.06 * contradiction_pressure
        - 0.04 * shallow_penalty
    )


def _selector_confidence(record: HistoricalSolveRecord) -> float:
    selector = record.final_selector_diagnostics
    if isinstance(selector, Mapping):
        for key in ("confidence", "winner_confidence", "final_confidence"):
            if key in selector:
                return _clamp01(selector.get(key))
    return 0.0


def _route_budget(record: HistoricalSolveRecord, key: str, default: Any = 0) -> Any:
    route = record.route_snapshot
    if key in route:
        return route.get(key, default)
    budget = route.get("budget_plan")
    if isinstance(budget, Mapping):
        return budget.get(key, default)
    return default


def _top_branch(branches: Sequence[BranchTrace], *, answer_key: str | None = None) -> BranchTrace | None:
    filtered = []
    for branch in branches:
        branch_answer_key = _canonical_answer(branch.answer_canonical or branch.answer)
        if answer_key is not None and branch_answer_key != answer_key:
            continue
        filtered.append(branch)
    if not filtered:
        return None
    return sorted(
        filtered,
        key=lambda branch: (-_branch_rank(branch), branch.branch_id),
    )[0]


def _symbolic_status(branch: BranchTrace, record: HistoricalSolveRecord) -> str:
    branch_status = _normalize_text(branch.metadata.get("symbolic_status", ""))
    if branch_status:
        return branch_status.lower()
    snapshot = record.symbolic_snapshot
    for key in ("symbolic_status", "status"):
        value = snapshot.get(key)
        if value is not None:
            return _normalize_text(value).lower()
    if not branch.symbolic_valid:
        return "unsupported"
    return "success"


def _has_parser_symbolic_issue(record: HistoricalSolveRecord, branch: BranchTrace | None) -> bool:
    texts = []
    texts.extend(
        [
            _normalize_text(record.symbolic_snapshot.get("summary", "")),
            _normalize_text(record.symbolic_snapshot.get("error", "")),
            _normalize_text(record.symbolic_snapshot.get("message", "")),
        ]
    )
    if branch is not None:
        texts.extend(
            [
                _normalize_text(branch.metadata.get("symbolic_summary", "")),
                _normalize_text(branch.metadata.get("symbolic_error", "")),
            ]
        )
    joined = " ".join(texts).lower()
    status = _symbolic_status(branch, record) if branch is not None else _normalize_text(record.symbolic_snapshot.get("status", "")).lower()
    return status in {"malformed_input", "execution_failure"} or any(
        token in joined for token in ("parse", "parser", "syntax", "sympy solve import failure")
    )


def _recommended_fix_text(bucket: FailureBucket, subtype: str) -> str:
    mapping = {
        (FailureBucket.SEARCH_FAILURE, "never_explored_correct_path"): "Increase bounded exploration around structurally compatible operators before late verifier gating.",
        (FailureBucket.SEARCH_FAILURE, "pruned_too_early"): "Loosen prune pressure for branches with better symbolic/proof-obligation support than their current verifier rank.",
        (FailureBucket.SEARCH_FAILURE, "insufficient_depth"): "Raise search depth on hard problems when obligation burden stays high after shallow progress.",
        (FailureBucket.SEARCH_FAILURE, "budget_starvation"): "Allocate larger branch/node budgets for hard or high-uncertainty problems.",
        (FailureBucket.VERIFIER_FAILURE, "wrong_branch_scored_high"): "Retrain or recalibrate verifier slices where wrong smooth branches outrank structurally stronger alternatives.",
        (FailureBucket.VERIFIER_FAILURE, "contradiction_under_penalized"): "Increase contradiction and obligation-conflict penalties inside verifier scoring.",
        (FailureBucket.VERIFIER_FAILURE, "shallow_chain_overrated"): "Penalize high-prefix but low-step-quality traces more aggressively.",
        (FailureBucket.VERIFIER_FAILURE, "correct_branch_under_ranked"): "Reweight verifier decomposition to preserve structurally messy but correct branches.",
        (FailureBucket.SYMBOLIC_FAILURE, "simplification_wrong"): "Tighten bounded algebra simplification and cross-check exact equivalence before pass-through.",
        (FailureBucket.SYMBOLIC_FAILURE, "contradiction_missed"): "Surface contradiction evidence into branch-time gating instead of passive symbolic approval.",
        (FailureBucket.SYMBOLIC_FAILURE, "integer_reasoning_miss"): "Strengthen modular, divisibility, and residue-class reasoning paths in number theory validation.",
        (FailureBucket.SYMBOLIC_FAILURE, "parser_syntax_issue"): "Harden symbolic parsing and failure typing so parser/runtime issues become explicit unsupported failures.",
        (FailureBucket.RETRIEVAL_FAILURE, "wrong_hints"): "Down-rank retrieval traces whose operator or failure-mode support misleads final branch selection.",
        (FailureBucket.RETRIEVAL_FAILURE, "missing_relevant_trace"): "Improve compatibility-aware retrieval for hard cases with low useful support coverage.",
        (FailureBucket.RETRIEVAL_FAILURE, "over_conditioned_on_noise"): "Compress retrieval-conditioned prompts and reduce overweighting on noisy support.",
        (FailureBucket.OPERATOR_FAILURE, "useful_operator_absent"): "Promote recurrent repair-neighbor operators into the active library for this slice.",
        (FailureBucket.OPERATOR_FAILURE, "wrong_operator_repeatedly_selected"): "Lower operator priors for repeated losing transforms in this archetype/domain slice.",
        (FailureBucket.OPERATOR_FAILURE, "mined_operator_low_quality"): "Re-mine low-purity operator templates with stronger provenance and promotion thresholds.",
        (FailureBucket.AGGREGATION_FAILURE, "correct_cluster_existed_but_not_selected"): "Adjust selector weighting so correct answer families are not discarded after branch-level success exists.",
        (FailureBucket.AGGREGATION_FAILURE, "answer_family_fragmentation"): "Improve answer canonicalization and clustering so semantically same answers merge earlier.",
        (FailureBucket.AGGREGATION_FAILURE, "confidence_miscalibration"): "Recalibrate selector confidence so wrong winners do not present as high-certainty.",
        (FailureBucket.ROUTING_BUDGET_FAILURE, "under_budgeted_hard_problem"): "Raise route-conditioned compute allocation for hard slices before search begins.",
        (FailureBucket.ROUTING_BUDGET_FAILURE, "wrong_reasoning_mode"): "Switch to stronger reasoning/verifier mode for hard or structurally uncertain problems.",
        (FailureBucket.ROUTING_BUDGET_FAILURE, "route_uncertainty_mishandled"): "Make high route uncertainty trigger widening, critique, or safer branch retention.",
        (FailureBucket.ROUTING_BUDGET_FAILURE, "critique_disabled_too_early"): "Keep critique active on hard problems when no clean symbolic/verifier convergence exists.",
        (FailureBucket.UNKNOWN_FAILURE, "unknown_failure"): "Collect richer runtime diagnostics for this slice before retuning search or verifier behavior.",
    }
    return mapping.get((bucket, subtype), "Inspect this failure slice and add a typed remediation path.")


def _repair_target_and_direction(bucket: FailureBucket, subtype: str) -> tuple[str, str, float]:
    mapping = {
        (FailureBucket.SEARCH_FAILURE, "never_explored_correct_path"): ("src/branches/branch_controller.py", "Increase value-guided exploration around compatible operators and retained proof-obligation branches.", 0.90),
        (FailureBucket.SEARCH_FAILURE, "pruned_too_early"): ("src/branches/branch_controller.py", "Delay prune decisions when symbolic trust and obligation discharge are stronger than current verifier rank.", 0.95),
        (FailureBucket.SEARCH_FAILURE, "insufficient_depth"): ("src/branches/branch_controller.py", "Raise bounded depth/node budgets on hard slices with unresolved obligation burden.", 0.82),
        (FailureBucket.SEARCH_FAILURE, "budget_starvation"): ("src/branches/branch_controller.py", "Increase route-conditioned branch and node budgets for hard high-uncertainty problems.", 0.90),
        (FailureBucket.VERIFIER_FAILURE, "wrong_branch_scored_high"): ("src/verifier/scoring.py", "Retrain or recalibrate failure slices where wrong smooth branches outrank stronger alternatives.", 0.96),
        (FailureBucket.VERIFIER_FAILURE, "contradiction_under_penalized"): ("src/verifier/scoring.py", "Increase contradiction and obligation-conflict penalties in verifier fusion.", 0.98),
        (FailureBucket.VERIFIER_FAILURE, "shallow_chain_overrated"): ("src/verifier/scoring.py", "Penalize low step-quality but high smoothness traces more aggressively.", 0.92),
        (FailureBucket.VERIFIER_FAILURE, "correct_branch_under_ranked"): ("src/verifier/scoring.py", "Protect messy-but-correct branches by reweighting prefix/step/symbolic tradeoffs.", 0.90),
        (FailureBucket.SYMBOLIC_FAILURE, "simplification_wrong"): ("src/symbolic/algebra.py", "Tighten bounded simplification and exact equivalence checks before success labeling.", 0.84),
        (FailureBucket.SYMBOLIC_FAILURE, "contradiction_missed"): ("src/symbolic/validators.py", "Escalate missed contradiction evidence into explicit trust-anchor failure outputs.", 0.94),
        (FailureBucket.SYMBOLIC_FAILURE, "integer_reasoning_miss"): ("src/symbolic/number_theory.py", "Improve modular, divisibility, and residue contradiction coverage for integer tasks.", 0.90),
        (FailureBucket.SYMBOLIC_FAILURE, "parser_syntax_issue"): ("src/symbolic/validators.py", "Preserve parser/runtime failures as explicit unsupported or execution-failure results.", 0.82),
        (FailureBucket.RETRIEVAL_FAILURE, "wrong_hints"): ("src/retrieval/retrieval_policy.py", "Reduce misleading retrieval influence through stronger compatibility-aware reweighting.", 0.80),
        (FailureBucket.RETRIEVAL_FAILURE, "missing_relevant_trace"): ("src/retrieval/query.py", "Increase structure-aware retrieval recall for hard failure slices.", 0.78),
        (FailureBucket.RETRIEVAL_FAILURE, "over_conditioned_on_noise"): ("src/retrieval/trace_prompting.py", "Trim noisy retrieved context and emphasize support rationale over raw trace text.", 0.76),
        (FailureBucket.OPERATOR_FAILURE, "useful_operator_absent"): ("src/operators/operator_library.py", "Promote recurrent successful repair-neighbor moves into the active operator library.", 0.79),
        (FailureBucket.OPERATOR_FAILURE, "wrong_operator_repeatedly_selected"): ("src/operators/priors.py", "Down-weight repeated low-yield operators by failure slice and archetype.", 0.86),
        (FailureBucket.OPERATOR_FAILURE, "mined_operator_low_quality"): ("src/operators/operator_miner.py", "Raise purity thresholds and provenance requirements for promoted operators.", 0.81),
        (FailureBucket.AGGREGATION_FAILURE, "correct_cluster_existed_but_not_selected"): ("src/aggregation/final_selector.py", "Reweight selector inputs so existing correct clusters survive final choice.", 0.93),
        (FailureBucket.AGGREGATION_FAILURE, "answer_family_fragmentation"): ("src/aggregation/canonicalize.py", "Strengthen canonicalization and cluster-family merging for equivalent answers.", 0.77),
        (FailureBucket.AGGREGATION_FAILURE, "confidence_miscalibration"): ("src/aggregation/final_selector.py", "Recalibrate selector confidence against failure slices with wrong high-confidence winners.", 0.85),
        (FailureBucket.ROUTING_BUDGET_FAILURE, "under_budgeted_hard_problem"): ("src/routing/difficulty_estimator.py", "Increase compute allocation for hard slices before search starts.", 0.87),
        (FailureBucket.ROUTING_BUDGET_FAILURE, "wrong_reasoning_mode"): ("src/routing/difficulty_estimator.py", "Choose stronger reasoning mode on hard or structurally uncertain problems.", 0.73),
        (FailureBucket.ROUTING_BUDGET_FAILURE, "route_uncertainty_mishandled"): ("src/routing/difficulty_estimator.py", "Make route uncertainty widen compute and safer retention choices.", 0.79),
        (FailureBucket.ROUTING_BUDGET_FAILURE, "critique_disabled_too_early"): ("src/branches/branch_controller.py", "Keep critique enabled until symbolic/verifier convergence is stronger.", 0.74),
        (FailureBucket.UNKNOWN_FAILURE, "unknown_failure"): ("src/offline/failure_mining.py", "Collect richer typed diagnostics so this slice stops collapsing into unknown failure.", 0.45),
    }
    return mapping.get((bucket, subtype), ("src/offline/failure_mining.py", "Collect richer diagnostics for this slice.", 0.40))


def _domain_lower(record: HistoricalSolveRecord) -> str:
    return _normalize_text(record.domain).lower()


def _difficulty_lower(record: HistoricalSolveRecord) -> str:
    return _normalize_text(record.difficulty).lower()


def _add_cause(
    causes: list[FailureCause],
    *,
    bucket: FailureBucket,
    subtype: str,
    weight: float,
    rationale: str,
    branch_id: str | None = None,
    evidence_refs: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> None:
    causes.append(
        FailureCause(
            failure_bucket=bucket,
            subtype=subtype,
            weight=round(_clamp01(weight), 4),
            rationale=rationale,
            branch_id=branch_id,
            evidence_refs=list(evidence_refs),
            metadata=dict(metadata or {}),
        )
    )


def _mine_failure_for_record(
    record: HistoricalSolveRecord,
    *,
    config: FailureMiningConfig,
) -> FailureRecord | None:
    gold_key = _canonical_answer(record.gold_answer)
    predicted_key = _canonical_answer(record.predicted_answer)
    if not gold_key:
        return None
    if gold_key == predicted_key:
        return None

    branches = list(record.branch_traces)
    correct_branches = [branch for branch in branches if _canonical_answer(branch.answer_canonical or branch.answer) == gold_key]
    wrong_branches = [branch for branch in branches if _canonical_answer(branch.answer_canonical or branch.answer) and _canonical_answer(branch.answer_canonical or branch.answer) != gold_key]
    predicted_branches = [branch for branch in branches if _canonical_answer(branch.answer_canonical or branch.answer) == predicted_key]

    top_correct = _top_branch(correct_branches)
    top_wrong = _top_branch(predicted_branches) or _top_branch(wrong_branches) or _top_branch(branches)
    causes: list[FailureCause] = []

    difficulty = _difficulty_lower(record)
    domain = _domain_lower(record)
    hard_labels = {label.lower() for label in config.hard_difficulty_labels}
    branch_budget = _parse_int(_route_budget(record, "branch_budget", 0), 0)
    max_search_depth = _parse_int(_route_budget(record, "max_search_depth", 0), 0)
    max_search_nodes = _parse_int(_route_budget(record, "max_search_nodes", 0), 0)
    retrieval_depth = _parse_int(_route_budget(record, "retrieval_depth", 0), 0)
    critique_top_k = _parse_int(_route_budget(record, "critique_top_k", 0), 0)
    route_uncertainty = _parse_float(
        record.route_snapshot.get("route_uncertainty", record.route_snapshot.get("compute_signals", {}).get("route_uncertainty", 0.0)),
        0.0,
    )
    selector_confidence = _selector_confidence(record)
    max_step_count = max((len(branch.steps) for branch in branches), default=0)

    if top_correct is not None:
        wrong_rank = _branch_rank(top_wrong) if top_wrong is not None else 0.0
        correct_rank = _branch_rank(top_correct)
        verifier_gap = (float(top_wrong.verifier_score or 0.0) - float(top_correct.verifier_score or 0.0)) if top_wrong is not None else 0.0

        if verifier_gap >= float(config.verifier_gap_threshold):
            _add_cause(
                causes,
                bucket=FailureBucket.VERIFIER_FAILURE,
                subtype="wrong_branch_scored_high",
                weight=0.92,
                rationale="A gold-answer branch existed, but a wrong branch carried a materially higher verifier score.",
                branch_id=top_wrong.branch_id if top_wrong is not None else None,
                evidence_refs=[item for item in [top_wrong.branch_id if top_wrong else None, top_correct.branch_id] if item],
                metadata={"verifier_gap": round(verifier_gap, 4)},
            )
        elif wrong_rank > correct_rank:
            _add_cause(
                causes,
                bucket=FailureBucket.VERIFIER_FAILURE,
                subtype="correct_branch_under_ranked",
                weight=0.78,
                rationale="A correct branch existed but remained below the selected wrong branch after verifier/search ranking.",
                branch_id=top_correct.branch_id,
                evidence_refs=[top_correct.branch_id, top_wrong.branch_id if top_wrong else ""],
                metadata={"rank_gap": round(wrong_rank - correct_rank, 4)},
            )

        correct_phase = _normalize_text(top_correct.metadata.get("phase", "")).lower()
        if correct_phase in {"pruned", "failed"}:
            _add_cause(
                causes,
                bucket=FailureBucket.SEARCH_FAILURE,
                subtype="pruned_too_early",
                weight=0.86,
                rationale="A correct-answer branch existed but was already pruned or failed before final selection.",
                branch_id=top_correct.branch_id,
                evidence_refs=[top_correct.branch_id],
            )

        _add_cause(
            causes,
            bucket=FailureBucket.AGGREGATION_FAILURE,
            subtype="correct_cluster_existed_but_not_selected",
            weight=0.88,
            rationale="The correct answer family existed in the branch set, but final selection still chose a different answer.",
            branch_id=top_correct.branch_id,
            evidence_refs=[top_correct.branch_id, top_wrong.branch_id if top_wrong else ""],
        )
        if selector_confidence >= float(config.calibration_overconfidence_threshold):
            _add_cause(
                causes,
                bucket=FailureBucket.AGGREGATION_FAILURE,
                subtype="confidence_miscalibration",
                weight=0.62,
                rationale="Final selector expressed high confidence on a wrong answer despite a correct answer family being present.",
                branch_id=top_wrong.branch_id if top_wrong is not None else None,
                evidence_refs=[top_wrong.branch_id] if top_wrong is not None else (),
                metadata={"selector_confidence": round(selector_confidence, 4)},
            )
        gold_variants = {
            _normalize_text(branch.answer or "")
            for branch in correct_branches
            if _normalize_text(branch.answer or "")
        }
        if len(gold_variants) >= 2:
            _add_cause(
                causes,
                bucket=FailureBucket.AGGREGATION_FAILURE,
                subtype="answer_family_fragmentation",
                weight=0.48,
                rationale="Multiple raw answer forms mapped to the correct family, increasing fragmentation pressure before selection.",
                branch_id=top_correct.branch_id,
                evidence_refs=[top_correct.branch_id],
                metadata={"gold_variants": sorted(gold_variants)},
            )
    else:
        if difficulty in hard_labels and (
            branch_budget > 0 and branch_budget <= int(config.hard_branch_budget_threshold)
            or max_search_nodes > 0 and max_search_nodes <= 48
        ):
            _add_cause(
                causes,
                bucket=FailureBucket.ROUTING_BUDGET_FAILURE,
                subtype="under_budgeted_hard_problem",
                weight=0.86,
                rationale="Hard problem received a constrained branch/node budget and never surfaced a correct branch.",
                metadata={"branch_budget": branch_budget, "max_search_nodes": max_search_nodes},
            )
        if max_search_depth > 0 and max_search_depth <= int(config.hard_depth_threshold) or max_step_count <= int(config.shallow_step_threshold):
            _add_cause(
                causes,
                bucket=FailureBucket.SEARCH_FAILURE,
                subtype="insufficient_depth",
                weight=0.78,
                rationale="Search stayed shallow on a failed problem and never reached a deeper constructive path.",
                metadata={"max_search_depth": max_search_depth, "max_step_count": max_step_count},
            )
        if not causes:
            _add_cause(
                causes,
                bucket=FailureBucket.SEARCH_FAILURE,
                subtype="never_explored_correct_path",
                weight=0.72,
                rationale="No branch matched the gold answer, suggesting the correct path was not explored at all.",
            )

    if top_wrong is not None:
        contradiction_count = _parse_int(top_wrong.metadata.get("contradicted_obligation_count", 0), 0)
        if (
            contradiction_count > 0 or float(top_wrong.open_obligation_burden or 0.0) >= 0.45 or _symbolic_status(top_wrong, record) == "contradiction"
        ) and float(top_wrong.verifier_score or 0.0) >= 0.55:
            _add_cause(
                causes,
                bucket=FailureBucket.VERIFIER_FAILURE,
                subtype="contradiction_under_penalized",
                weight=0.95,
                rationale="The selected wrong branch carried contradiction or obligation-conflict pressure without enough verifier penalty.",
                branch_id=top_wrong.branch_id,
                evidence_refs=[top_wrong.branch_id],
                metadata={"contradicted_obligation_count": contradiction_count},
            )
        if (
            float(top_wrong.logical_consistency or 0.0) >= 0.70
            and float(top_wrong.prefix_quality or 0.0) >= 0.68
            and float(top_wrong.step_quality or 0.0) <= 0.45
        ):
            _add_cause(
                causes,
                bucket=FailureBucket.VERIFIER_FAILURE,
                subtype="shallow_chain_overrated",
                weight=0.84,
                rationale="A smooth prefix with low step quality still ranked highly, indicating shallow reasoning was overrated.",
                branch_id=top_wrong.branch_id,
                evidence_refs=[top_wrong.branch_id],
            )

        if _has_parser_symbolic_issue(record, top_wrong):
            _add_cause(
                causes,
                bucket=FailureBucket.SYMBOLIC_FAILURE,
                subtype="parser_syntax_issue",
                weight=0.82,
                rationale="Symbolic parsing or runtime failure likely degraded trust-anchor behavior on the selected branch.",
                branch_id=top_wrong.branch_id,
                evidence_refs=[top_wrong.branch_id],
            )
        elif domain in {"number_theory", "nt"} and not correct_branches:
            _add_cause(
                causes,
                bucket=FailureBucket.SYMBOLIC_FAILURE,
                subtype="integer_reasoning_miss",
                weight=0.58,
                rationale="Number-theory failure slice shows no correct branch despite symbolic execution staying in play.",
                branch_id=top_wrong.branch_id,
                evidence_refs=[top_wrong.branch_id],
            )
        elif domain == "algebra" and top_wrong.symbolic_valid and not correct_branches:
            _add_cause(
                causes,
                bucket=FailureBucket.SYMBOLIC_FAILURE,
                subtype="simplification_wrong",
                weight=0.46,
                rationale="Algebraic branch passed symbolic checks but still converged to the wrong answer.",
                branch_id=top_wrong.branch_id,
                evidence_refs=[top_wrong.branch_id],
            )
        elif top_wrong.symbolic_valid and not correct_branches and float(top_wrong.verifier_score or 0.0) >= 0.60:
            _add_cause(
                causes,
                bucket=FailureBucket.SYMBOLIC_FAILURE,
                subtype="contradiction_missed",
                weight=0.63,
                rationale="Wrong branch retained symbolic approval without exposing a contradiction signal.",
                branch_id=top_wrong.branch_id,
                evidence_refs=[top_wrong.branch_id],
            )

    retrieval_signal = max((float(branch.retrieval_support or 0.0) for branch in branches), default=0.0)
    retrieval_compatibility = max((float(branch.retrieval_compatibility or 0.0) for branch in branches), default=0.0)
    if retrieval_depth > 0 and retrieval_signal <= 0.15:
        _add_cause(
            causes,
            bucket=FailureBucket.RETRIEVAL_FAILURE,
            subtype="missing_relevant_trace",
            weight=0.57,
            rationale="Retrieval was enabled but contributed very little usable support to any failed branch.",
            metadata={"retrieval_depth": retrieval_depth, "retrieval_signal": round(retrieval_signal, 4)},
        )
    elif top_wrong is not None and top_wrong.retrieval_support >= 0.55 and top_wrong.retrieval_compatibility <= 0.30:
        _add_cause(
            causes,
            bucket=FailureBucket.RETRIEVAL_FAILURE,
            subtype="over_conditioned_on_noise",
            weight=0.61,
            rationale="The selected wrong branch leaned on retrieval support that was not structurally compatible.",
            branch_id=top_wrong.branch_id,
            evidence_refs=[top_wrong.branch_id],
        )
    elif top_wrong is not None and top_wrong.retrieval_support >= 0.50 and not correct_branches:
        _add_cause(
            causes,
            bucket=FailureBucket.RETRIEVAL_FAILURE,
            subtype="wrong_hints",
            weight=0.49,
            rationale="Retrieval support was strong on wrong branches but did not recover the correct answer family.",
            branch_id=top_wrong.branch_id,
            evidence_refs=[top_wrong.branch_id],
        )

    operator_sequences = [tuple(branch.operator_sequence) for branch in branches if branch.operator_sequence]
    operator_counts: dict[str, int] = defaultdict(int)
    for sequence in operator_sequences:
        for operator_name in sequence:
            operator_counts[operator_name] += 1
    if operator_counts:
        dominant_operator, dominant_count = sorted(operator_counts.items(), key=lambda item: (-item[1], item[0]))[0]
        dominance = dominant_count / max(1, sum(operator_counts.values()))
        if dominance >= 0.45 and not correct_branches:
            _add_cause(
                causes,
                bucket=FailureBucket.OPERATOR_FAILURE,
                subtype="wrong_operator_repeatedly_selected",
                weight=0.64,
                rationale="Failed branches repeatedly concentrated on the same operator family without finding the correct path.",
                metadata={"dominant_operator": dominant_operator, "dominance": round(dominance, 4)},
            )
        elif top_wrong is not None and float(top_wrong.operator_reliability or 0.0) <= 0.20:
            _add_cause(
                causes,
                bucket=FailureBucket.OPERATOR_FAILURE,
                subtype="mined_operator_low_quality",
                weight=0.52,
                rationale="The selected branch relied on an operator with very low learned reliability.",
                branch_id=top_wrong.branch_id,
                evidence_refs=[top_wrong.branch_id],
            )
    elif not correct_branches:
        _add_cause(
            causes,
            bucket=FailureBucket.OPERATOR_FAILURE,
            subtype="useful_operator_absent",
            weight=0.40,
            rationale="No meaningful operator sequence was recorded on failed branches, suggesting operator coverage is still weak.",
        )

    if difficulty in hard_labels and branch_budget <= int(config.hard_branch_budget_threshold):
        _add_cause(
            causes,
            bucket=FailureBucket.ROUTING_BUDGET_FAILURE,
            subtype="under_budgeted_hard_problem",
            weight=0.72,
            rationale="Hard problem carried a low branch budget relative to the intended search regime.",
            metadata={"branch_budget": branch_budget},
        )
    verifier_mode = _normalize_text(record.route_snapshot.get("verifier_mode", ""))
    if difficulty in hard_labels and verifier_mode in {"default", "fast"}:
        _add_cause(
            causes,
            bucket=FailureBucket.ROUTING_BUDGET_FAILURE,
            subtype="wrong_reasoning_mode",
            weight=0.42,
            rationale="Hard problem stayed in a generic reasoning/verifier mode instead of a stronger setting.",
            metadata={"verifier_mode": verifier_mode},
        )
    if route_uncertainty >= 0.45 and branch_budget <= 24:
        _add_cause(
            causes,
            bucket=FailureBucket.ROUTING_BUDGET_FAILURE,
            subtype="route_uncertainty_mishandled",
            weight=0.58,
            rationale="Route uncertainty was high but the downstream search budget did not widen enough.",
            metadata={"route_uncertainty": round(route_uncertainty, 4), "branch_budget": branch_budget},
        )
    if critique_top_k <= 0 and not any(branch.self_critiqued for branch in branches):
        _add_cause(
            causes,
            bucket=FailureBucket.ROUTING_BUDGET_FAILURE,
            subtype="critique_disabled_too_early",
            weight=0.44,
            rationale="No critique pass was available on a failed problem, reducing the chance of local correction.",
        )

    if not causes:
        _add_cause(
            causes,
            bucket=FailureBucket.UNKNOWN_FAILURE,
            subtype="unknown_failure",
            weight=0.40,
            rationale="No strong structured failure slice could be attributed from the available diagnostics.",
        )

    causes.sort(key=lambda cause: (-cause.weight, cause.failure_bucket.value, cause.subtype, cause.branch_id or ""))
    primary = causes[0]
    secondaries = [f"{cause.failure_bucket.value}:{cause.subtype}" for cause in causes[1:4]]
    recommended_fix = _recommended_fix_text(primary.failure_bucket, primary.subtype)

    branch_snapshot = {
        "branch_count": len(branches),
        "correct_branch_present": bool(correct_branches),
        "predicted_branch_present": bool(predicted_branches),
        "max_step_count": max_step_count,
        "winning_branch_id": top_wrong.branch_id if top_wrong is not None else None,
        "correct_branch_id": top_correct.branch_id if top_correct is not None else None,
    }
    verifier_snapshot = {
        **dict(record.verifier_snapshot),
        "top_wrong_verifier_score": round(float(top_wrong.verifier_score or 0.0), 6) if top_wrong is not None else 0.0,
        "top_correct_verifier_score": round(float(top_correct.verifier_score or 0.0), 6) if top_correct is not None else 0.0,
        "top_wrong_rank_score": round(_branch_rank(top_wrong), 6) if top_wrong is not None else 0.0,
        "top_correct_rank_score": round(_branch_rank(top_correct), 6) if top_correct is not None else 0.0,
    }
    symbolic_snapshot = {
        **dict(record.symbolic_snapshot),
        "selected_symbolic_status": _symbolic_status(top_wrong, record) if top_wrong is not None else "",
        "selected_symbolic_valid": bool(top_wrong.symbolic_valid) if top_wrong is not None else False,
        "parser_issue_detected": _has_parser_symbolic_issue(record, top_wrong),
    }
    route_snapshot = {
        **dict(record.route_snapshot),
        "difficulty": record.difficulty,
        "domain": record.domain,
        "route_uncertainty": round(route_uncertainty, 6),
        "branch_budget": branch_budget,
        "retrieval_depth": retrieval_depth,
        "max_search_depth": max_search_depth,
        "max_search_nodes": max_search_nodes,
        "critique_top_k": critique_top_k,
    }
    confidence = round(_clamp01(sum(cause.weight for cause in causes[:2]) / max(1, min(2, len(causes)))), 4)
    return FailureRecord(
        problem_id=record.problem_id,
        gold_answer=record.gold_answer,
        predicted_answer=record.predicted_answer,
        primary_failure=primary.failure_bucket,
        secondary_failures=secondaries,
        confidence_of_attribution=confidence,
        route_snapshot=route_snapshot,
        verifier_snapshot=verifier_snapshot,
        symbolic_snapshot=symbolic_snapshot,
        branch_snapshot=branch_snapshot,
        recommended_fix=recommended_fix,
        attributions=causes,
        metadata={
            "primary_subtype": primary.subtype,
            "domain": record.domain,
            "difficulty": record.difficulty,
            "selector_confidence": selector_confidence,
            "correct_branch_present": bool(correct_branches),
        },
    )


def _estimate_pattern_gain(count: int, average_weight: float, base_roi: float) -> float:
    return round(count * max(0.15, average_weight) * base_roi * 0.35, 4)


def mine_failures(
    sources: Sequence[HistoricalSolveRecord | DistilledTraceArtifact | DistilledTraceRecord | Mapping[str, Any]],
    *,
    config: FailureMiningConfig | None = None,
) -> FailureMiningResult:
    resolved_config = config or FailureMiningConfig()
    records: list[HistoricalSolveRecord] = []
    for source in sources:
        records.extend(normalize_failure_source(source))
    records.sort(key=lambda item: item.problem_id)
    if resolved_config.max_records is not None:
        records = records[: resolved_config.max_records]

    hard_records = [
        record
        for record in records
        if _difficulty_lower(record) in {label.lower() for label in resolved_config.hard_difficulty_labels}
    ]
    skipped_missing_gold = sum(1 for record in records if not _canonical_answer(record.gold_answer))
    labeled_records = [record for record in records if _canonical_answer(record.gold_answer)]
    failures: list[FailureRecord] = []
    for record in labeled_records:
        mined = _mine_failure_for_record(record, config=resolved_config)
        if mined is not None:
            failures.append(mined)
    failures.sort(key=lambda item: (item.problem_id, item.primary_failure.value, item.metadata.get("primary_subtype", "")))

    notes: list[str] = []
    status = FailureMiningStatus.COMPLETED
    if len(hard_records) < resolved_config.min_hard_problems:
        status = FailureMiningStatus.INSUFFICIENT_INPUT
        notes.append(
            f"Only {len(hard_records)} hard problems were provided; configured minimum is {resolved_config.min_hard_problems}."
        )
    if skipped_missing_gold:
        notes.append(f"Skipped {skipped_missing_gold} records without gold answers for failure attribution.")

    pattern_buckets: dict[tuple[str, str], list[FailureRecord]] = defaultdict(list)
    bucket_counts: dict[str, int] = defaultdict(int)
    domain_modes: dict[str, int] = defaultdict(int)
    difficulty_modes: dict[str, int] = defaultdict(int)
    for failure in failures:
        bucket_counts[failure.primary_failure.value] += 1
        domain_modes[str(failure.metadata.get("domain", "unknown"))] += 1
        difficulty_modes[str(failure.metadata.get("difficulty", "unknown"))] += 1
        pattern_buckets[
            (
                failure.primary_failure.value,
                str(failure.metadata.get("primary_subtype", "unknown_failure")),
            )
        ].append(failure)

    top_patterns: list[FailurePatternSummary] = []
    suggested_repairs: list[SuggestedRepair] = []
    estimated_score_gain_by_fix: dict[str, float] = {}
    for (bucket_name, subtype), items in sorted(pattern_buckets.items()):
        bucket = FailureBucket(bucket_name)
        avg_weight = sum(
            next(
                (
                    cause.weight
                    for cause in item.attributions
                    if cause.failure_bucket is bucket and cause.subtype == subtype
                ),
                0.0,
            )
            for item in items
        ) / len(items)
        avg_confidence = sum(item.confidence_of_attribution for item in items) / len(items)
        target_file, fix_direction, base_roi = _repair_target_and_direction(bucket, subtype)
        estimated_gain = _estimate_pattern_gain(len(items), avg_weight, base_roi)
        repair_key = f"{bucket.value}:{subtype}"
        top_patterns.append(
            FailurePatternSummary(
                failure_bucket=bucket,
                subtype=subtype,
                count=len(items),
                average_weight=round(avg_weight, 4),
                average_confidence=round(avg_confidence, 4),
                estimated_score_gain=estimated_gain,
                recommended_fix=_recommended_fix_text(bucket, subtype),
            )
        )
        suggested_repairs.append(
            SuggestedRepair(
                failure_type=repair_key,
                target_file=target_file,
                exact_fix_direction=fix_direction,
                expected_roi=estimated_gain,
            )
        )
        estimated_score_gain_by_fix[repair_key] = estimated_gain

    top_patterns.sort(
        key=lambda item: (-item.estimated_score_gain, -item.count, item.failure_bucket.value, item.subtype)
    )
    suggested_repairs.sort(key=lambda item: (-item.expected_roi, item.failure_type, item.target_file))

    failure_summary = {
        "counts_by_failure_type": dict(sorted(bucket_counts.items())),
        "top_patterns": [item.model_dump(mode="json") for item in top_patterns[:12]],
        "repeated_problem_modes": {
            "by_domain": dict(sorted(domain_modes.items(), key=lambda item: (-item[1], item[0]))),
            "by_difficulty": dict(sorted(difficulty_modes.items(), key=lambda item: (-item[1], item[0]))),
        },
        "estimated_score_gain_by_fix": estimated_score_gain_by_fix,
    }

    slice_metrics = {
        "total_records": len(records),
        "labeled_records": len(labeled_records),
        "failed_records": len(failures),
        "skipped_missing_gold": skipped_missing_gold,
        "hard_problem_count": len(hard_records),
        "metrics_by_failure_type": {
            key: {
                "count": len(items),
                "avg_confidence": round(sum(item.confidence_of_attribution for item in items) / len(items), 4),
                "avg_selector_confidence": round(
                    sum(_parse_float(item.metadata.get("selector_confidence", 0.0), 0.0) for item in items) / len(items),
                    4,
                ),
            }
            for key, items in sorted(
                ((key, [item for item in failures if item.primary_failure.value == key]) for key in bucket_counts),
                key=lambda item: item[0],
            )
        },
        "metrics_by_domain": {
            key: {
                "count": len(items),
                "failure_rate": round(len(items) / max(1, len([record for record in labeled_records if _domain_lower(record) == key.lower()])), 4),
            }
            for key, items in sorted(
                ((key, [item for item in failures if str(item.metadata.get("domain", "unknown")) == key]) for key in domain_modes),
                key=lambda item: item[0],
            )
        },
        "metrics_by_difficulty": {
            key: {
                "count": len(items),
                "failure_rate": round(len(items) / max(1, len([record for record in labeled_records if _difficulty_lower(record) == key.lower()])), 4),
            }
            for key, items in sorted(
                ((key, [item for item in failures if str(item.metadata.get("difficulty", "unknown")) == key]) for key in difficulty_modes),
                key=lambda item: item[0],
            )
        },
        "dominant_slices": [item.model_dump(mode="json") for item in top_patterns[:8]],
    }

    manifest = FailureMiningManifest(
        artifact_version=resolved_config.artifact_version,
        run_id=_stable_hash(
            "failure_mining_run",
            {
                "problem_ids": [record.problem_id for record in records],
                "failed_problem_ids": [failure.problem_id for failure in failures],
                "config": resolved_config.model_dump(mode="json"),
            },
        ),
        status=status,
        total_records=len(records),
        labeled_records=len(labeled_records),
        failed_records=len(failures),
        skipped_missing_gold=skipped_missing_gold,
        notes=notes,
    )
    return FailureMiningResult(
        manifest=manifest,
        failures=tuple(failures),
        failure_summary=failure_summary,
        slice_metrics=slice_metrics,
        suggested_repairs=tuple(suggested_repairs),
    )


def build_failure_dataset(
    result: FailureMiningResult | Sequence[FailureRecord],
) -> tuple[dict[str, Any], ...]:
    failures = result.failures if isinstance(result, FailureMiningResult) else tuple(result)
    rows: list[dict[str, Any]] = []
    for failure in failures:
        rows.append(
            {
                "problem_id": failure.problem_id,
                "gold_answer": failure.gold_answer,
                "predicted_answer": failure.predicted_answer,
                "primary_failure": failure.primary_failure.value,
                "secondary_failures": list(failure.secondary_failures),
                "primary_subtype": failure.metadata.get("primary_subtype", "unknown_failure"),
                "confidence_of_attribution": failure.confidence_of_attribution,
                "route_snapshot": dict(failure.route_snapshot),
                "verifier_snapshot": dict(failure.verifier_snapshot),
                "symbolic_snapshot": dict(failure.symbolic_snapshot),
                "branch_snapshot": dict(failure.branch_snapshot),
                "recommended_fix": failure.recommended_fix,
                "attributions": [item.model_dump(mode="json") for item in failure.attributions],
                "metadata": dict(failure.metadata),
            }
        )
    rows.sort(key=lambda item: (item["problem_id"], item["primary_failure"], item["primary_subtype"]))
    return tuple(rows)


def write_failure_artifacts(result: FailureMiningResult, output_dir: str | Path) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures_path = out_dir / "failures.jsonl"
    failure_summary_path = out_dir / "failure_summary.json"
    slice_metrics_path = out_dir / "slice_metrics.json"
    suggested_repairs_path = out_dir / "suggested_repairs.json"
    manifest_path = out_dir / "manifest.json"

    with failures_path.open("w", encoding="utf-8") as handle:
        for row in build_failure_dataset(result):
            handle.write(_stable_json(row) + "\n")
    failure_summary_path.write_text(_stable_json(result.failure_summary) + "\n", encoding="utf-8")
    slice_metrics_path.write_text(_stable_json(result.slice_metrics) + "\n", encoding="utf-8")
    suggested_repairs_path.write_text(
        _stable_json([item.model_dump(mode="json") for item in result.suggested_repairs]) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(_stable_json(result.manifest.model_dump(mode="json")) + "\n", encoding="utf-8")
    return {
        "manifest": str(manifest_path),
        "failures": str(failures_path),
        "failure_summary": str(failure_summary_path),
        "slice_metrics": str(slice_metrics_path),
        "suggested_repairs": str(suggested_repairs_path),
    }


def load_failure_input_path(path: str | Path) -> tuple[HistoricalSolveRecord, ...]:
    resolved = Path(path)
    if resolved.is_dir():
        manifest_path = resolved / "manifest.json"
        records_path = resolved / "records.jsonl"
        if manifest_path.exists():
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_payload.get("artifact_kind") == "distilled_trace_corpus":
                return tuple(normalize_failure_source(load_distilled_artifact(resolved)))
            manifest_records_path = manifest_payload.get("records_path")
            if isinstance(manifest_records_path, str) and manifest_records_path.strip():
                resolved = (
                    (resolved / manifest_records_path).resolve()
                    if not Path(manifest_records_path).is_absolute()
                    else Path(manifest_records_path)
                )
            elif records_path.exists():
                resolved = records_path
        elif records_path.exists():
            resolved = records_path
    if resolved.name == "manifest.json":
        manifest_payload = json.loads(resolved.read_text(encoding="utf-8"))
        if manifest_payload.get("artifact_kind") == "distilled_trace_corpus":
            return tuple(normalize_failure_source(load_distilled_artifact(resolved.parent)))
        records_path = manifest_payload.get("records_path")
        if isinstance(records_path, str) and records_path.strip():
            resolved = (resolved.parent / records_path).resolve() if not Path(records_path).is_absolute() else Path(records_path)
        else:
            sibling_records = resolved.with_name("records.jsonl")
            if sibling_records.exists():
                resolved = sibling_records
    if resolved.suffix.lower() == ".jsonl":
        rows: list[HistoricalSolveRecord] = []
        with resolved.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                rows.extend(normalize_failure_source(json.loads(stripped)))
        return tuple(rows)
    if resolved.suffix.lower() == ".json":
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows: list[HistoricalSolveRecord] = []
            for item in payload:
                if isinstance(item, Mapping):
                    rows.extend(normalize_failure_source(item))
            return tuple(rows)
        if isinstance(payload, Mapping):
            if payload.get("artifact_kind") == "distilled_trace_corpus":
                return tuple(
                    normalize_failure_source(
                        load_distilled_artifact(resolved.parent if resolved.name == "manifest.json" else resolved)
                    )
                )
            return tuple(normalize_failure_source(payload))
    raise ValueError(f"Unsupported failure mining input path: {path}")


__all__ = [
    "ARTIFACT_KIND",
    "FailureMiningConfig",
    "FailureMiningManifest",
    "FailureMiningResult",
    "FailureMiningStatus",
    "FailurePatternSummary",
    "SuggestedRepair",
    "build_failure_dataset",
    "load_failure_input_path",
    "mine_failures",
    "normalize_failure_source",
    "write_failure_artifacts",
]
