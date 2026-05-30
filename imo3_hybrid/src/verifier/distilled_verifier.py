"""
Typed runtime distilled verifier with explicit artifact availability handling.

This module is intentionally self-contained because the current repository state
still has empty `scoring.py` / `verifier_labels.py` stubs and no verifier
artifacts checked in. The implementation therefore:

- exposes a stable typed runtime interface for branch-controller usage
- packs branch + symbolic evidence into a bounded verifier input contract
- attempts to use offline-produced artifacts when they exist
- falls back deterministically when artifacts are missing or invalid
- labels fallback usage explicitly instead of pretending a trained model exists
- delegates to `scoring.py` / `verifier_labels.py` automatically once those
  modules are populated later
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha1
import json
import math
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.aggregation.canonicalize import canonicalize_competition_answer
from src.branches.branch_state import BranchPhase, BranchState
from src.common.constants import (
    MAX_REPAIR_ATTEMPTS,
)
from src.common.schemas import ParsedProblem, RouteDecision, VerifierLabel as SchemaVerifierLabel
from src.prm import ProcessObligationInput, ProcessScoreInput, ProcessStepInput, score_process
from src.state_graph.node import ConstraintRecord, GoalRecord, InvariantRecord, ReasoningStateNode

from . import scoring as scoring_module
from . import verifier_labels as verifier_labels_module

if TYPE_CHECKING:
    from src.branches.branch_controller import GenerationResult, SymbolicHookResult, VerifierHookResult

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str | None) -> str:
    return _WHITESPACE_RE.sub(" ", (text or "").strip())


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(float(v) for v in values) / len(values)


def _stable_hash(prefix: str, payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"{prefix}_{sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _coerce_scalar_score(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return _clamp01(float(value))
    if isinstance(value, Mapping):
        for key in ("branch_score", "composite_score", "score", "verifier_score"):
            if key in value and isinstance(value[key], (int, float)):
                return _clamp01(float(value[key]))
        return None
    for attr in ("branch_score", "composite_score", "score", "verifier_score"):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, (int, float)):
            return _clamp01(float(candidate))
    return None


def _constraint_summaries(records: Sequence[ConstraintRecord], limit: int) -> tuple[str, ...]:
    return tuple(record.summary() for record in records[:limit] if record.summary())


def _goal_summaries(records: Sequence[GoalRecord], limit: int) -> tuple[str, ...]:
    return tuple(record.normalized_text for record in records[:limit] if record.normalized_text)


def _invariant_summaries(records: Sequence[InvariantRecord], limit: int) -> tuple[str, ...]:
    return tuple(record.normalized_expression for record in records[:limit] if record.normalized_expression)


def _phase_progress(phase: BranchPhase) -> float:
    ordered = {
        BranchPhase.INITIALIZED: 0.05,
        BranchPhase.RETRIEVED: 0.18,
        BranchPhase.REASONING: 0.45,
        BranchPhase.VERIFIED: 0.70,
        BranchPhase.CRITIQUED: 0.82,
        BranchPhase.REPAIRED: 0.62,
        BranchPhase.ROLLED_BACK: 0.40,
        BranchPhase.SOLVED: 1.00,
        BranchPhase.FAILED: 0.10,
        BranchPhase.PRUNED: 0.12,
    }
    return ordered.get(phase, 0.0)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )


class VerifierArtifactState(str, Enum):
    READY = "ready"
    MISSING = "missing"
    INVALID = "invalid"


class VerifierExecutionMode(str, Enum):
    MODEL = "model"
    FALLBACK = "fallback"


class VerifierDiagnosticSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class VerifierDiagnostic(StrictModel):
    code: str
    message: str
    severity: VerifierDiagnosticSeverity = VerifierDiagnosticSeverity.INFO
    payload: dict[str, Any] = Field(default_factory=dict)


class DistilledVerifierConfig(StrictModel):
    checkpoint_path: str | None = None
    search_default_artifacts: bool = True
    allow_fallback: bool = True
    deterministic: bool = True
    max_steps: int = Field(default=12, ge=1)
    max_constraints: int = Field(default=12, ge=1)
    max_invariants: int = Field(default=8, ge=1)
    max_goals: int = Field(default=6, ge=1)
    max_operator_history: int = Field(default=12, ge=1)
    max_reasoning_chars: int = Field(default=1200, ge=64)


class VerifierRuntimeStatus(StrictModel):
    backend_name: str
    artifact_state: VerifierArtifactState
    artifact_path: str | None = None
    ready: bool = False
    reason: str = ""
    diagnostics: tuple[VerifierDiagnostic, ...] = Field(default_factory=tuple)


class PackedVerifierStep(StrictModel):
    index: int = Field(ge=0)
    description: str
    operator_name: str | None = None
    phase: str
    state_fingerprint: str | None = None
    summary_text: str | None = None
    proof_obligation_ids: tuple[str, ...] = Field(default_factory=tuple)


class PackedVerifierInput(StrictModel):
    problem_id: str
    branch_id: str
    branch_phase: str
    domain: str
    difficulty: str
    route_uncertainty: float = Field(ge=0.0, le=1.0)
    repair_threshold: float = Field(ge=0.0, le=1.0)
    answer_raw: str | None = None
    answer_canonical: str | None = None
    answer_in_range: bool = False
    reasoning_excerpt: str = ""
    step_count: int = Field(ge=0)
    retrieval_count: int = Field(ge=0)
    symbolic_count: int = Field(ge=0)
    verifier_count: int = Field(ge=0)
    critique_count: int = Field(ge=0)
    repair_count: int = Field(ge=0)
    candidate_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    generation_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    latest_operator: str | None = None
    operator_sequence: tuple[str, ...] = Field(default_factory=tuple)
    constraints: tuple[str, ...] = Field(default_factory=tuple)
    invariants: tuple[str, ...] = Field(default_factory=tuple)
    goals: tuple[str, ...] = Field(default_factory=tuple)
    steps: tuple[PackedVerifierStep, ...] = Field(default_factory=tuple)
    symbolic_passed: bool = False
    symbolic_score: float = Field(default=0.0, ge=0.0, le=1.0)
    symbolic_exact_match: bool = False
    symbolic_summary: str = ""
    retrieval_support: float = Field(default=0.0, ge=0.0, le=1.0)
    retrieval_answer_support: float = Field(default=0.0, ge=0.0, le=1.0)
    prior_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    existing_verifier_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    contradiction_flag: bool = False
    proof_obligation_count: int = Field(default=0, ge=0)
    open_proof_obligation_count: int = Field(default=0, ge=0)
    contradicted_proof_obligation_count: int = Field(default=0, ge=0)
    proof_state_fingerprint: str | None = None
    truncation_notes: tuple[str, ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def feature_vector(self) -> dict[str, float]:
        step_density = _clamp01(_safe_ratio(self.step_count, max(1, len(self.steps) or self.step_count or 1)))
        reasoning_present = 1.0 if self.reasoning_excerpt else 0.0
        answer_present = 1.0 if self.answer_canonical else 0.0
        answer_in_range = 1.0 if self.answer_in_range else 0.0
        repair_pressure = _clamp01(_safe_ratio(self.repair_count, MAX_REPAIR_ATTEMPTS))
        verifier_pressure = _clamp01(_safe_ratio(self.verifier_count, 4.0))
        critique_pressure = _clamp01(_safe_ratio(self.critique_count, 3.0))
        route_certainty = _clamp01(1.0 - self.route_uncertainty)
        contradiction = 1.0 if self.contradiction_flag else 0.0
        exact_symbolic = 1.0 if self.symbolic_exact_match else 0.0
        return {
            "answer_present": answer_present,
            "answer_in_range": answer_in_range,
            "reasoning_present": reasoning_present,
            "step_density": step_density,
            "constraint_coverage": _clamp01(_safe_ratio(len(self.constraints), 8.0)),
            "invariant_coverage": _clamp01(_safe_ratio(len(self.invariants), 6.0)),
            "goal_coverage": _clamp01(_safe_ratio(len(self.goals), 4.0)),
            "candidate_confidence": self.candidate_confidence,
            "generation_confidence": self.generation_confidence,
            "symbolic_score": self.symbolic_score,
            "symbolic_pass": 1.0 if self.symbolic_passed else 0.0,
            "symbolic_exact": exact_symbolic,
            "retrieval_support": self.retrieval_support,
            "retrieval_answer_support": self.retrieval_answer_support,
            "route_certainty": route_certainty,
            "prior_confidence": self.prior_confidence,
            "phase_progress": _phase_progress(BranchPhase(self.branch_phase)),
            "repair_pressure": repair_pressure,
            "verifier_pressure": verifier_pressure,
            "critique_pressure": critique_pressure,
            "existing_verifier_probability": self.existing_verifier_probability,
            "contradiction_flag": contradiction,
            "open_obligation_burden": _clamp01(_safe_ratio(self.open_proof_obligation_count, max(1, self.proof_obligation_count))),
            "proof_state_present": 1.0 if self.proof_state_fingerprint else 0.0,
        }


class VerifierScoreBundle(StrictModel):
    problem_id: str
    branch_id: str
    backend_name: str
    artifact_state: VerifierArtifactState
    mode: VerifierExecutionMode
    probability: float = Field(ge=0.0, le=1.0)
    logical_consistency: float = Field(ge=0.0, le=1.0)
    symbolic_consistency: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    repairability: float = Field(ge=0.0, le=1.0)
    step_quality: float = Field(ge=0.0, le=1.0)
    prefix_quality: float = Field(ge=0.0, le=1.0)
    open_obligation_burden: float = Field(ge=0.0, le=1.0)
    branch_score: float = Field(ge=0.0, le=1.0)
    summary: str
    label: SchemaVerifierLabel
    feature_vector: dict[str, float] = Field(default_factory=dict)
    diagnostics: tuple[VerifierDiagnostic, ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def answer_correctness_likelihood(self) -> float:
        return self.probability

    @property
    def symbolic_agreement(self) -> float:
        return self.symbolic_consistency

    @property
    def overall_score(self) -> float:
        return self.branch_score

    def to_branch_controller_hook_result(self) -> "VerifierHookResult":
        from src.branches.branch_controller import VerifierHookResult

        return VerifierHookResult(
            probability=self.probability,
            logical_consistency=self.logical_consistency,
            completeness=self.completeness,
            repairability=self.repairability,
            summary=self.summary,
            symbolic_agreement=self.symbolic_consistency,
            answer_correctness_likelihood=self.answer_correctness_likelihood,
            branch_score=self.branch_score,
            step_quality=self.step_quality,
            prefix_quality=self.prefix_quality,
            open_obligation_burden=self.open_obligation_burden,
            metadata={
                "symbolic_consistency": self.symbolic_consistency,
                "branch_score": self.branch_score,
                "artifact_state": self.artifact_state.value,
                "mode": self.mode.value,
                "backend_name": self.backend_name,
                "step_quality": self.step_quality,
                "prefix_quality": self.prefix_quality,
                "open_obligation_burden": self.open_obligation_burden,
                "verifier_decomposition": {
                    "logical_consistency": self.logical_consistency,
                    "symbolic_agreement": self.symbolic_consistency,
                    "completeness": self.completeness,
                    "answer_correctness_likelihood": self.answer_correctness_likelihood,
                    "repairability": self.repairability,
                    "step_quality": self.step_quality,
                    "prefix_quality": self.prefix_quality,
                    "open_obligation_burden": self.open_obligation_burden,
                },
                "label": self.label.model_dump(mode="json"),
                "diagnostics": [item.model_dump(mode="json") for item in self.diagnostics],
                **dict(self.metadata),
            },
        )

    def to_scoring_payload(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "branch_id": self.branch_id,
            "backend_name": self.backend_name,
            "artifact_state": self.artifact_state.value,
            "mode": self.mode.value,
            "verifier_probability": self.probability,
            "answer_correctness_likelihood": self.answer_correctness_likelihood,
            "logical_consistency": self.logical_consistency,
            "symbolic_agreement": self.symbolic_agreement,
            "symbolic_consistency": self.symbolic_consistency,
            "completeness": self.completeness,
            "repairability": self.repairability,
            "step_quality": self.step_quality,
            "prefix_quality": self.prefix_quality,
            "open_obligation_burden": self.open_obligation_burden,
            "branch_score": self.branch_score,
            "overall_score": self.overall_score,
            "feature_vector": dict(self.feature_vector),
            "label": self.label.model_dump(mode="json"),
        }


@dataclass(frozen=True)
class RuntimePrediction:
    probability: float
    logical_consistency: float
    symbolic_consistency: float
    completeness: float
    repairability: float
    summary: str
    metadata: Mapping[str, Any]


class RuntimeModel(Protocol):
    def status(self) -> VerifierRuntimeStatus: ...

    def predict(self, packed: PackedVerifierInput) -> RuntimePrediction: ...


class UnavailableRuntimeModel:
    def __init__(
        self,
        *,
        artifact_state: VerifierArtifactState,
        reason: str,
        artifact_path: str | None = None,
        diagnostics: Sequence[VerifierDiagnostic] = (),
    ) -> None:
        self._status = VerifierRuntimeStatus(
            backend_name="unavailable",
            artifact_state=artifact_state,
            artifact_path=artifact_path,
            ready=False,
            reason=reason,
            diagnostics=tuple(diagnostics),
        )

    def status(self) -> VerifierRuntimeStatus:
        return self._status

    def predict(self, packed: PackedVerifierInput) -> RuntimePrediction:
        raise RuntimeError(self._status.reason or "Runtime verifier artifact unavailable.")


class JsonLinearRuntimeModel:
    def __init__(self, manifest_path: Path, manifest: Mapping[str, Any]) -> None:
        self._manifest_path = manifest_path
        self._manifest = dict(manifest)
        self._feature_order = tuple(
            str(item).strip()
            for item in manifest.get("feature_order", ())
            if str(item).strip()
        )
        self._heads = self._normalize_heads(manifest)
        self._status = VerifierRuntimeStatus(
            backend_name=str(manifest.get("backend", "linear-json")),
            artifact_state=VerifierArtifactState.READY,
            artifact_path=str(manifest_path),
            ready=True,
            reason="loaded",
        )

    def status(self) -> VerifierRuntimeStatus:
        return self._status

    def predict(self, packed: PackedVerifierInput) -> RuntimePrediction:
        features = packed.feature_vector()
        scores = {
            name: self._head_score(self._heads[name], features)
            for name in ("probability", "logical_consistency", "symbolic_consistency", "completeness", "repairability")
        }
        return RuntimePrediction(
            probability=scores["probability"],
            logical_consistency=scores["logical_consistency"],
            symbolic_consistency=scores["symbolic_consistency"],
            completeness=scores["completeness"],
            repairability=scores["repairability"],
            summary=f"artifact_runtime::{self._manifest_path.name}",
            metadata={
                "artifact_path": str(self._manifest_path),
                "feature_order": list(self._feature_order),
            },
        )

    def _normalize_heads(self, manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        default_head = {
            "weights": manifest.get("weights", {}),
            "bias": manifest.get("bias", 0.0),
            "scale": manifest.get("scale", 1.0),
        }
        heads = manifest.get("heads")
        if not isinstance(heads, Mapping):
            return {
                "probability": default_head,
                "logical_consistency": default_head,
                "symbolic_consistency": default_head,
                "completeness": default_head,
                "repairability": default_head,
            }
        normalized: dict[str, Mapping[str, Any]] = {}
        for name in ("probability", "logical_consistency", "symbolic_consistency", "completeness", "repairability"):
            head = heads.get(name, default_head)
            normalized[name] = {
                "weights": head.get("weights", default_head["weights"]),
                "bias": head.get("bias", default_head["bias"]),
                "scale": head.get("scale", default_head["scale"]),
            }
        return normalized

    def _head_score(self, head: Mapping[str, Any], features: Mapping[str, float]) -> float:
        raw_weights = head.get("weights", {})
        if isinstance(raw_weights, Mapping):
            weights = {str(key): float(value) for key, value in raw_weights.items()}
        elif isinstance(raw_weights, Sequence):
            weights = {
                feature_name: float(raw_weights[index])
                for index, feature_name in enumerate(self._feature_order)
                if index < len(raw_weights)
            }
        else:
            weights = {}
        linear = float(head.get("bias", 0.0))
        for feature_name, feature_value in features.items():
            linear += float(weights.get(feature_name, 0.0)) * float(feature_value)
        linear *= float(head.get("scale", 1.0))
        return _clamp01(1.0 / (1.0 + math.exp(-linear)))


class DistilledVerifier:
    """
    Lightweight runtime verifier.

    The verifier always returns a typed bundle. When trained artifacts are
    missing, the bundle is explicitly marked as fallback + missing-artifact.
    """

    def __init__(self, config: DistilledVerifierConfig | None = None) -> None:
        self.config = config or DistilledVerifierConfig()
        self._runtime_model = self._load_runtime_model()

    def status(self) -> VerifierRuntimeStatus:
        return self._runtime_model.status()

    def load_checkpoint(self, path: str | Path) -> VerifierRuntimeStatus:
        self.config = self.config.model_copy(update={"checkpoint_path": str(path)})
        self._runtime_model = self._load_runtime_model()
        return self.status()

    def score_trace(
        self,
        trace_batch: PackedVerifierInput | Sequence[PackedVerifierInput],
    ) -> VerifierScoreBundle | tuple[VerifierScoreBundle, ...]:
        if isinstance(trace_batch, PackedVerifierInput):
            return self.verify_packed(trace_batch)
        return tuple(self.verify_packed(item) for item in trace_batch)

    def pack_input(
        self,
        *,
        problem: ParsedProblem,
        route: RouteDecision,
        branch: BranchState,
        node: ReasoningStateNode,
        symbolic: "SymbolicHookResult",
        generation: "GenerationResult",
    ) -> PackedVerifierInput:
        active_steps = list(branch.active_steps())
        current_candidate = branch.current_candidate()
        latest_answer = (
            (current_candidate.raw_answer if current_candidate is not None else None)
            or generation.answer
        )
        canonical_answer = (
            (current_candidate.canonical_answer if current_candidate is not None else None)
            or generation.answer_canonical
            or canonicalize_competition_answer(latest_answer)
        )
        reasoning_text = _normalize_text(
            generation.reasoning
            or generation.summary
            or " ".join(step.description for step in active_steps)
        )
        truncation_notes: list[str] = []
        if len(reasoning_text) > self.config.max_reasoning_chars:
            reasoning_text = reasoning_text[: self.config.max_reasoning_chars].rstrip()
            truncation_notes.append("reasoning_excerpt_truncated")

        packed_steps = [
            PackedVerifierStep(
                index=step.index,
                description=_normalize_text(step.description),
                operator_name=_normalize_text(step.operator_name) or None,
                phase=step.phase.value,
                state_fingerprint=step.state_fingerprint,
                summary_text=_normalize_text(step.summary_text) or None,
                proof_obligation_ids=tuple(step.proof_obligation_ids),
            )
            for step in active_steps[: self.config.max_steps]
        ]
        if len(active_steps) > self.config.max_steps:
            truncation_notes.append("step_list_truncated")

        constraints = _constraint_summaries(node.constraints, self.config.max_constraints)
        invariants = _invariant_summaries(node.invariants, self.config.max_invariants)
        goals = _goal_summaries(node.goals, self.config.max_goals)

        if len(node.constraints) > self.config.max_constraints:
            truncation_notes.append("constraint_list_truncated")
        if len(node.invariants) > self.config.max_invariants:
            truncation_notes.append("invariant_list_truncated")
        if len(node.goals) > self.config.max_goals:
            truncation_notes.append("goal_list_truncated")

        retrieval_scores = [float(item.score) for item in branch.active_retrieval_evidence()]
        canonicalized_retrieval_answers = [
            canonicalize_competition_answer(item.answer) for item in branch.active_retrieval_evidence()
        ]
        retrieval_answer_support = _safe_ratio(
            sum(1 for value in canonicalized_retrieval_answers if value and value == canonical_answer),
            max(1, len(canonicalized_retrieval_answers)),
        )

        contradiction_flag = (
            "contradiction" in _normalize_text(symbolic.summary).lower()
            or "violates" in _normalize_text(symbolic.summary).lower()
            or bool(symbolic.metadata.get("symbolic_result", {}).get("metadata", {}).get("contradiction_found", False))
        )

        latest_operator = None
        operator_sequence = tuple(
            _normalize_text(step.operator_name)
            for step in active_steps
            if _normalize_text(step.operator_name)
        )
        if operator_sequence:
            latest_operator = operator_sequence[-1]

        existing_verifier_probability = 0.0
        active_verifier = branch.active_verifier_evidence()
        if active_verifier:
            existing_verifier_probability = _clamp01(active_verifier[-1].probability)

        prior_confidence = 0.0
        if latest_operator:
            prior_confidence = _clamp01(float((route.operator_prior or {}).get(latest_operator, 0.0)))
        proof_obligations = list(branch.active_proof_obligations())
        open_proof_obligation_count = sum(
            1 for item in proof_obligations if item.status.value in {"open", "partial", "blocked"}
        )
        contradicted_proof_obligation_count = sum(
            1 for item in proof_obligations if item.status.value == "contradicted"
        )

        return PackedVerifierInput(
            problem_id=problem.problem_id,
            branch_id=branch.branch_id,
            branch_phase=branch.phase.value,
            domain=getattr(problem.domain, "value", str(problem.domain)),
            difficulty=route.difficulty.value,
            route_uncertainty=_clamp01(route.route_uncertainty),
            repair_threshold=_clamp01(route.repair_threshold),
            answer_raw=latest_answer,
            answer_canonical=canonical_answer,
            answer_in_range=canonical_answer is not None,
            reasoning_excerpt=reasoning_text,
            step_count=len(active_steps),
            retrieval_count=branch.active_cursor.retrieval_count,
            symbolic_count=branch.active_cursor.symbolic_count,
            verifier_count=branch.active_cursor.verifier_count,
            critique_count=branch.active_cursor.critique_count,
            repair_count=branch.active_cursor.repair_count,
            candidate_confidence=_clamp01(current_candidate.confidence if current_candidate is not None else 0.0),
            generation_confidence=_clamp01(generation.confidence),
            latest_operator=latest_operator,
            operator_sequence=operator_sequence[: self.config.max_operator_history],
            constraints=constraints,
            invariants=invariants,
            goals=goals,
            steps=tuple(packed_steps),
            symbolic_passed=bool(symbolic.passed),
            symbolic_score=_clamp01(symbolic.score),
            symbolic_exact_match=bool(symbolic.exact_match),
            symbolic_summary=_normalize_text(symbolic.summary),
            retrieval_support=_clamp01(_mean(retrieval_scores)),
            retrieval_answer_support=_clamp01(retrieval_answer_support),
            prior_confidence=prior_confidence,
            existing_verifier_probability=existing_verifier_probability,
            contradiction_flag=contradiction_flag,
            proof_obligation_count=len(proof_obligations),
            open_proof_obligation_count=open_proof_obligation_count,
            contradicted_proof_obligation_count=contradicted_proof_obligation_count,
            proof_state_fingerprint=branch.proof_state_fingerprint,
            truncation_notes=tuple(truncation_notes),
            metadata={
                "generation_summary": _normalize_text(generation.summary),
                "node_id": node.node_id,
                "state_fingerprint": node.state_fingerprint,
                "problem_target": _normalize_text(problem.target),
                "proof_obligation_ids": [item.obligation_id for item in proof_obligations],
            },
        )

    def verify_packed(self, packed: PackedVerifierInput) -> VerifierScoreBundle:
        runtime_status = self._runtime_model.status()
        diagnostics = list(runtime_status.diagnostics)
        mode = VerifierExecutionMode.MODEL
        prm_summary = self._prm_summary_from_packed(packed)

        if runtime_status.ready:
            prediction = self._runtime_model.predict(packed)
        else:
            if not self.config.allow_fallback:
                raise RuntimeError(runtime_status.reason or "Distilled verifier runtime unavailable and fallback disabled.")
            mode = VerifierExecutionMode.FALLBACK
            diagnostics.append(
                VerifierDiagnostic(
                    code="verifier_fallback_active",
                    message="Using deterministic fallback because runtime verifier artifacts are unavailable.",
                    severity=VerifierDiagnosticSeverity.WARNING,
                    payload={
                        "artifact_state": runtime_status.artifact_state.value,
                        "artifact_path": runtime_status.artifact_path,
                    },
                )
            )
            prediction = self._fallback_predict(packed, runtime_status)

        probability = _clamp01(prediction.probability)
        logical_consistency = _clamp01(prediction.logical_consistency)
        symbolic_consistency = _clamp01(prediction.symbolic_consistency)
        completeness = _clamp01(prediction.completeness)
        repairability = _clamp01(prediction.repairability)
        step_quality = prm_summary.mean_step_quality
        prefix_quality = prm_summary.prefix_quality
        open_obligation_burden = prm_summary.open_obligation_burden

        label = self._build_label(
            packed=packed,
            probability=probability,
            logical_consistency=logical_consistency,
            symbolic_consistency=symbolic_consistency,
            completeness=completeness,
            repairability=repairability,
        )
        branch_score, scoring_source = self._score_bundle(
            packed=packed,
            probability=probability,
            logical_consistency=logical_consistency,
            symbolic_consistency=symbolic_consistency,
            completeness=completeness,
            repairability=repairability,
            step_quality=step_quality,
            prefix_quality=prefix_quality,
            open_obligation_burden=open_obligation_burden,
            mode=mode,
            artifact_state=runtime_status.artifact_state,
            label=label,
        )
        label = label.model_copy(update={"composite_score": branch_score})

        if scoring_source != "internal_fallback":
            diagnostics.append(
                VerifierDiagnostic(
                    code="scoring_adapter_used",
                    message=f"Score obtained from scoring adapter `{scoring_source}`.",
                    severity=VerifierDiagnosticSeverity.INFO,
                )
            )
        elif mode is VerifierExecutionMode.FALLBACK:
            diagnostics.append(
                VerifierDiagnostic(
                    code="fallback_score_downweighted",
                    message="Verifier score was downweighted because the runtime model is unavailable.",
                    severity=VerifierDiagnosticSeverity.WARNING,
                )
            )

        return VerifierScoreBundle(
            problem_id=packed.problem_id,
            branch_id=packed.branch_id,
            backend_name=runtime_status.backend_name,
            artifact_state=runtime_status.artifact_state,
            mode=mode,
            probability=probability,
            logical_consistency=logical_consistency,
            symbolic_consistency=symbolic_consistency,
            completeness=completeness,
            repairability=repairability,
            step_quality=step_quality,
            prefix_quality=prefix_quality,
            open_obligation_burden=open_obligation_burden,
            branch_score=branch_score,
            summary=prediction.summary,
            label=label,
            feature_vector=packed.feature_vector(),
            diagnostics=tuple(diagnostics),
            metadata={
                "artifact_path": runtime_status.artifact_path,
                "scoring_source": scoring_source,
                "truncation_notes": list(packed.truncation_notes),
                "proof_obligation_count": packed.proof_obligation_count,
                "open_proof_obligation_count": packed.open_proof_obligation_count,
                "contradicted_proof_obligation_count": packed.contradicted_proof_obligation_count,
                **dict(prediction.metadata),
                **dict(prm_summary.metadata),
            },
        )

    def verify(
        self,
        *,
        problem: ParsedProblem,
        route: RouteDecision,
        branch: BranchState,
        node: ReasoningStateNode,
        symbolic: "SymbolicHookResult",
        generation: "GenerationResult",
    ) -> VerifierScoreBundle:
        packed = self.pack_input(
            problem=problem,
            route=route,
            branch=branch,
            node=node,
            symbolic=symbolic,
            generation=generation,
        )
        return self.verify_packed(packed)

    def build_branch_controller_hook(self):
        def _hook(*, problem, route, branch, node, symbolic, generation):
            bundle = self.verify(
                problem=problem,
                route=route,
                branch=branch,
                node=node,
                symbolic=symbolic,
                generation=generation,
            )
            return bundle.to_branch_controller_hook_result()

        return _hook

    def _prm_summary_from_packed(self, packed: PackedVerifierInput):
        proof_ids = list(packed.metadata.get("proof_obligation_ids", []) or [])
        obligations = []
        for index, obligation_id in enumerate(proof_ids):
            status = "discharged"
            if index < packed.contradicted_proof_obligation_count:
                status = "contradicted"
            elif index < packed.contradicted_proof_obligation_count + packed.open_proof_obligation_count:
                status = "open"
            obligations.append(ProcessObligationInput(obligation_id=obligation_id, status=status))
        if not obligations and packed.proof_obligation_count > 0:
            for index in range(packed.proof_obligation_count):
                status = "contradicted" if index < packed.contradicted_proof_obligation_count else "open"
                if index >= packed.contradicted_proof_obligation_count + packed.open_proof_obligation_count:
                    status = "discharged"
                obligations.append(ProcessObligationInput(obligation_id=f"{packed.branch_id}::obl::{index}", status=status))
        return score_process(
            ProcessScoreInput(
                problem_id=packed.problem_id,
                branch_id=packed.branch_id,
                steps=tuple(
                    ProcessStepInput(
                        step_index=step.index + 1,
                        description=step.description,
                        operator_name=step.operator_name,
                        symbolic_valid=packed.symbolic_passed,
                        proof_obligation_ids=tuple(step.proof_obligation_ids),
                        state_fingerprint=step.state_fingerprint,
                    )
                    for step in packed.steps
                ),
                proof_obligations=tuple(obligations),
                route_uncertainty=packed.route_uncertainty,
                retrieval_support=packed.retrieval_support,
                symbolic_score=packed.symbolic_score,
                symbolic_passed=packed.symbolic_passed,
                exact_symbolic_match=packed.symbolic_exact_match,
                answer_present=bool(packed.answer_canonical),
                contradiction_count=packed.contradicted_proof_obligation_count if packed.contradicted_proof_obligation_count else int(packed.contradiction_flag),
                metadata=dict(packed.metadata),
            )
        )

    def _load_runtime_model(self) -> RuntimeModel:
        explicit = Path(self.config.checkpoint_path) if self.config.checkpoint_path else None
        candidates: list[Path] = []
        if explicit is not None:
            candidates.append(explicit)
        if self.config.search_default_artifacts:
            repo_root = Path(__file__).resolve().parents[2]
            candidates.extend(
                [
                    repo_root / "models" / "distilled_small_verifier" / "manifest.json",
                    repo_root / "models" / "distilled_small_verifier" / "model.json",
                    repo_root / "artifacts" / "verifier" / "distilled_manifest.json",
                    repo_root / "artifacts" / "verifier" / "runtime_model.json",
                ]
            )

        first_existing: Path | None = None
        seen: set[Path] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists():
                first_existing = candidate
                break

        if first_existing is None:
            return UnavailableRuntimeModel(
                artifact_state=VerifierArtifactState.MISSING,
                reason="No distilled verifier checkpoint/manifest was found.",
                artifact_path=str(explicit) if explicit is not None else None,
                diagnostics=(
                    VerifierDiagnostic(
                        code="missing_verifier_artifact",
                        message="No runtime verifier artifact is present; deterministic fallback will be used.",
                        severity=VerifierDiagnosticSeverity.WARNING,
                    ),
                ),
            )

        try:
            manifest = json.loads(first_existing.read_text(encoding="utf-8"))
        except Exception as exc:
            return UnavailableRuntimeModel(
                artifact_state=VerifierArtifactState.INVALID,
                reason=f"Failed to parse verifier artifact: {exc}",
                artifact_path=str(first_existing),
                diagnostics=(
                    VerifierDiagnostic(
                        code="invalid_verifier_artifact_json",
                        message=f"Verifier artifact exists but could not be parsed: {exc}",
                        severity=VerifierDiagnosticSeverity.ERROR,
                    ),
                ),
            )

        backend = str(manifest.get("backend", "linear-json")).strip().lower()
        if backend != "linear-json":
            return UnavailableRuntimeModel(
                artifact_state=VerifierArtifactState.INVALID,
                reason=f"Unsupported verifier artifact backend `{backend}`.",
                artifact_path=str(first_existing),
                diagnostics=(
                    VerifierDiagnostic(
                        code="unsupported_verifier_backend",
                        message=f"Verifier artifact backend `{backend}` is not supported by this runtime.",
                        severity=VerifierDiagnosticSeverity.ERROR,
                    ),
                ),
            )
        return JsonLinearRuntimeModel(first_existing, manifest)

    def _fallback_predict(
        self,
        packed: PackedVerifierInput,
        runtime_status: VerifierRuntimeStatus,
    ) -> RuntimePrediction:
        features = packed.feature_vector()
        answer_present = features["answer_present"]
        answer_in_range = features["answer_in_range"]
        route_certainty = features["route_certainty"]
        repair_pressure = features["repair_pressure"]
        contradiction = features["contradiction_flag"]
        open_obligation_burden = features["open_obligation_burden"]

        symbolic_consistency = _clamp01(
            0.72 * features["symbolic_score"]
            + 0.12 * answer_in_range
            + 0.10 * features["retrieval_answer_support"]
            + 0.06 * (1.0 - contradiction)
            - 0.10 * open_obligation_burden
        )
        logical_consistency = _clamp01(
            0.23 * features["reasoning_present"]
            + 0.16 * features["step_density"]
            + 0.18 * features["candidate_confidence"]
            + 0.12 * features["generation_confidence"]
            + 0.14 * route_certainty
            + 0.10 * features["prior_confidence"]
            + 0.07 * features["phase_progress"]
            - 0.22 * repair_pressure
            - 0.18 * contradiction
            - 0.08 * open_obligation_burden
        )
        completeness = _clamp01(
            0.30 * answer_present
            + 0.24 * features["reasoning_present"]
            + 0.18 * features["goal_coverage"]
            + 0.14 * features["constraint_coverage"]
            + 0.08 * features["step_density"]
            + 0.06 * features["retrieval_support"]
            - 0.10 * repair_pressure
            - 0.16 * open_obligation_burden
        )

        raw_probability = (
            0.38 * symbolic_consistency
            + 0.30 * logical_consistency
            + 0.20 * completeness
            + 0.07 * features["retrieval_support"]
            + 0.05 * answer_in_range
        )
        evidence_scale = 0.86 if runtime_status.artifact_state is not VerifierArtifactState.READY else 1.0
        probability = _clamp01(raw_probability * evidence_scale)

        if not packed.answer_canonical:
            probability = min(probability, 0.42)
        if not packed.symbolic_passed:
            probability = min(probability, 0.38 + 0.25 * logical_consistency)
        if contradiction >= 1.0:
            probability = min(probability, 0.12)

        if packed.symbolic_passed and probability >= max(0.45, packed.repair_threshold):
            repairability = _clamp01(0.12 + 0.10 * (1.0 - probability))
        else:
            repairability = _clamp01(
                0.72
                - 0.26 * symbolic_consistency
                - 0.12 * answer_present
                + 0.10 * (1.0 - logical_consistency)
                - 0.12 * repair_pressure
                + 0.08 * features["retrieval_support"]
            )

        summary = (
            "deterministic_fallback_verifier"
            f"::{runtime_status.artifact_state.value}"
        )
        return RuntimePrediction(
            probability=probability,
            logical_consistency=logical_consistency,
            symbolic_consistency=symbolic_consistency,
            completeness=completeness,
            repairability=repairability,
            summary=summary,
            metadata={
                "artifact_reason": runtime_status.reason,
                "heuristic_features": features,
            },
        )

    def _build_label(
        self,
        *,
        packed: PackedVerifierInput,
        probability: float,
        logical_consistency: float,
        symbolic_consistency: float,
        completeness: float,
        repairability: float,
    ) -> SchemaVerifierLabel:
        label = self._label_from_module(
            packed=packed,
            probability=probability,
            logical_consistency=logical_consistency,
            symbolic_consistency=symbolic_consistency,
            completeness=completeness,
            repairability=repairability,
        )
        if label is not None:
            return label

        failing_tail = 0
        if symbolic_consistency < 0.55:
            failing_tail = 1
        if logical_consistency < 0.40 and packed.step_count > 1:
            failing_tail = max(failing_tail, 2)
        valid_prefix = max(0, packed.step_count - failing_tail)
        step_correctness = [1] * valid_prefix + [0] * max(0, packed.step_count - valid_prefix)

        if not step_correctness and packed.reasoning_excerpt:
            step_correctness = [1 if probability >= 0.5 else 0]

        failure_type = None
        if not packed.symbolic_passed:
            failure_type = "symbolic_mismatch"
        elif completeness < 0.45 or not packed.answer_canonical:
            failure_type = "coverage_gap"
        elif logical_consistency < 0.45:
            failure_type = "logic"

        return SchemaVerifierLabel(
            branch_id=packed.branch_id,
            step_correctness=step_correctness,
            symbolic_agreement=symbolic_consistency,
            logical_consistency=logical_consistency,
            completeness=completeness,
            final_answer_correct=1 if probability >= 0.70 and packed.answer_in_range else 0,
            final_answer_correct_probability=probability,
            repairability=repairability,
            repair_type="local_repair" if repairability >= 0.45 else None,
            failure_type=failure_type,
            failure_location=f"step_{max(1, len(step_correctness))}" if failure_type and step_correctness else None,
            composite_score=0.0,
        )

    def _label_from_module(
        self,
        *,
        packed: PackedVerifierInput,
        probability: float,
        logical_consistency: float,
        symbolic_consistency: float,
        completeness: float,
        repairability: float,
    ) -> SchemaVerifierLabel | None:
        for name in ("build_verifier_label", "create_verifier_label", "compose_verifier_label", "label_trace"):
            fn = getattr(verifier_labels_module, name, None)
            if not callable(fn):
                continue
            payload = {
                "packed": packed,
                "probability": probability,
                "logical_consistency": logical_consistency,
                "symbolic_consistency": symbolic_consistency,
                "completeness": completeness,
                "repairability": repairability,
            }
            for candidate in (
                lambda: fn(payload),
                lambda: fn(packed),
                lambda: fn(
                    packed=packed,
                    probability=probability,
                    logical_consistency=logical_consistency,
                    symbolic_consistency=symbolic_consistency,
                    completeness=completeness,
                    repairability=repairability,
                ),
            ):
                try:
                    result = candidate()
                except TypeError:
                    continue
                except Exception:
                    break
                if isinstance(result, SchemaVerifierLabel):
                    return result
                if isinstance(result, Mapping):
                    try:
                        return SchemaVerifierLabel.model_validate(result)
                    except Exception:
                        return None
                return None
        return None

    def _score_bundle(
        self,
        *,
        packed: PackedVerifierInput,
        probability: float,
        logical_consistency: float,
        symbolic_consistency: float,
        completeness: float,
        repairability: float,
        step_quality: float,
        prefix_quality: float,
        open_obligation_burden: float,
        mode: VerifierExecutionMode,
        artifact_state: VerifierArtifactState,
        label: SchemaVerifierLabel,
    ) -> tuple[float, str]:
        payload = {
            "problem_id": packed.problem_id,
            "branch_id": packed.branch_id,
            "verifier_probability": probability,
            "logical_consistency": logical_consistency,
            "symbolic_consistency": symbolic_consistency,
            "completeness": completeness,
            "repairability": repairability,
            "step_quality": step_quality,
            "prefix_quality": prefix_quality,
            "open_obligation_burden": open_obligation_burden,
            "artifact_state": artifact_state.value,
            "mode": mode.value,
            "label": label.model_dump(mode="json"),
        }

        for name in ("score_verifier_bundle", "score_branch", "score_bundle", "compute_verifier_score"):
            fn = getattr(scoring_module, name, None)
            if not callable(fn):
                continue
            for candidate in (
                lambda: fn(payload),
                lambda: fn(label),
                lambda: fn(payload, label.model_dump(mode="json")),
                lambda: fn(
                    verifier_probability=probability,
                    logical_consistency=logical_consistency,
                    symbolic_consistency=symbolic_consistency,
                    completeness=completeness,
                    repairability=repairability,
                ),
            ):
                try:
                    result = candidate()
                except TypeError:
                    continue
                except Exception:
                    break
                score = _coerce_scalar_score(result)
                if score is not None:
                    return score, name

        evidence_scale = 1.0
        if mode is VerifierExecutionMode.FALLBACK:
            evidence_scale *= 0.88
        if artifact_state is not VerifierArtifactState.READY:
            evidence_scale *= 0.95
        if packed.contradiction_flag:
            evidence_scale *= 0.70
        if not packed.symbolic_passed and probability > 0.55:
            evidence_scale *= 0.75

        internal = (
            0.45 * probability
            + 0.22 * symbolic_consistency
            + 0.18 * logical_consistency
            + 0.10 * completeness
            + 0.05 * (1.0 - repairability)
        )
        internal += 0.05 * packed.retrieval_answer_support
        internal += 0.03 * _clamp01(1.0 - _safe_ratio(packed.repair_count, MAX_REPAIR_ATTEMPTS + 1))
        internal += 0.02 * (1.0 if packed.symbolic_exact_match else packed.symbolic_score)
        internal += 0.04 * step_quality + 0.06 * prefix_quality
        internal -= 0.10 * open_obligation_burden
        return _clamp01(internal * evidence_scale), "internal_fallback"


def build_distilled_verifier(config: DistilledVerifierConfig | None = None) -> DistilledVerifier:
    return DistilledVerifier(config=config)


def build_branch_controller_verifier_hook(
    *,
    config: DistilledVerifierConfig | None = None,
    verifier: DistilledVerifier | None = None,
):
    runtime = verifier or DistilledVerifier(config=config)
    return runtime.build_branch_controller_hook()


__all__ = [
    "DistilledVerifier",
    "DistilledVerifierConfig",
    "PackedVerifierInput",
    "PackedVerifierStep",
    "RuntimePrediction",
    "VerifierArtifactState",
    "VerifierDiagnostic",
    "VerifierDiagnosticSeverity",
    "VerifierExecutionMode",
    "VerifierRuntimeStatus",
    "VerifierScoreBundle",
    "build_branch_controller_verifier_hook",
    "build_distilled_verifier",
]
