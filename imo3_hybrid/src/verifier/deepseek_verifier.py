from __future__ import annotations

import inspect
import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from pydantic import BaseModel, Field

from src.aggregation.canonicalize import canonicalize_competition_answer
from src.common.constants import VERIFIER_PASS_THRESHOLD
from src.common.schemas import BranchTrace, FailureType, ParsedProblem, ReasoningStep, RouteDecision, VerifierLabel
from src.prm import ProcessObligationInput, ProcessScoreInput, ProcessStepInput, score_process
from src.state_graph.proof_obligations import ProofObligation

from . import scoring as scoring_module
from . import verifier_labels as verifier_labels_module


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_ASSUMPTION_RE = re.compile(r"\b(assume|suppose|obvious|obviously|wlog|without loss)\b", re.IGNORECASE)
_CASE_RE = re.compile(r"\bcase(s)?\b", re.IGNORECASE)


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))


def _normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def _model_dump(instance: BaseModel) -> dict[str, Any]:
    if hasattr(instance, "model_dump"):
        return instance.model_dump(mode="json")
    return instance.dict()


def _pack_proof_obligation(value: Any) -> PackedProofObligation | None:
    if isinstance(value, PackedProofObligation):
        return value
    if isinstance(value, ProofObligation):
        return PackedProofObligation(
            obligation_id=value.obligation_id,
            claim=value.claim,
            evidence_kind_required=value.evidence_kind_required.value,
            status=value.status.value,
            originating_node_id=value.originating_node_id,
            target_goal_id=value.target_goal_id,
            discharged_by=list(value.discharged_by),
        )
    if isinstance(value, Mapping):
        obligation_id = _normalize_text(value.get("obligation_id", ""))
        if not obligation_id:
            return None
        return PackedProofObligation(
            obligation_id=obligation_id,
            claim=_normalize_text(value.get("claim", "")),
            evidence_kind_required=_normalize_text(value.get("evidence_kind_required", "other")) or "other",
            status=_normalize_text(value.get("status", "open")) or "open",
            originating_node_id=_normalize_text(value.get("originating_node_id", "")),
            target_goal_id=_normalize_text(value.get("target_goal_id", "")) or None,
            discharged_by=[
                _normalize_text(item)
                for item in list(value.get("discharged_by", []) or [])
                if _normalize_text(item)
            ],
        )
    return None


class VerifierRuntimeStatus(str, Enum):
    MODEL = "model"
    MODEL_UNAVAILABLE = "model_unavailable"
    MODEL_ERROR = "model_error"
    PARSE_ERROR = "parse_error"
    DETERMINISTIC_FALLBACK = "deterministic_fallback"


class VerifierTaskType(str, Enum):
    TRACE = "trace"
    STEP = "step"


class VerifierEvidenceKind(str, Enum):
    MODEL = "model"
    STEP = "step"
    SYMBOLIC = "symbolic"
    ANSWER = "answer"
    COVERAGE = "coverage"
    ASSUMPTION = "assumption"
    FAILURE = "failure"
    RETRIEVAL = "retrieval"
    REASONING = "reasoning"


class StepVerificationStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    UNCERTAIN = "uncertain"


class RepairStrategy(str, Enum):
    NONE = "none"
    LOCAL_REPAIR = "local_repair"
    CASE_COMPLETION = "case_completion"
    RESAMPLE = "resample"


class VerifierModelProtocol(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class PackedReasoningStep(BaseModel):
    step_index: int = Field(ge=1)
    description: str
    operator_used: str | None = None
    symbolic_valid: bool = True
    proof_obligation_ids: list[str] = Field(default_factory=list)


class PackedProofObligation(BaseModel):
    obligation_id: str
    claim: str
    evidence_kind_required: str
    status: str
    originating_node_id: str = ""
    target_goal_id: str | None = None
    discharged_by: list[str] = Field(default_factory=list)


class VerifierTraceInput(BaseModel):
    task_type: VerifierTaskType = VerifierTaskType.TRACE
    problem_id: str
    branch_id: str
    raw_problem_text: str = ""
    target: str = ""
    domain: str = "unknown"
    route_problem_type: dict[str, float] = Field(default_factory=dict)
    route_archetypes: dict[str, float] = Field(default_factory=dict)
    route_uncertainty: float = Field(0.0, ge=0.0, le=1.0)
    node_summary: str = ""
    reasoning: str = ""
    answer: str | None = None
    answer_canonical: str | None = None
    generation_confidence: float = Field(0.0, ge=0.0, le=1.0)
    symbolic_passed: bool = False
    symbolic_score: float = Field(0.0, ge=0.0, le=1.0)
    symbolic_summary: str = ""
    exact_symbolic_match: bool = False
    retrieval_support: float = Field(0.0, ge=0.0, le=1.0)
    failure_hint: str | None = None
    branch_phase: str = ""
    steps: list[PackedReasoningStep] = Field(default_factory=list)
    proof_obligations: list[PackedProofObligation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerifierEvidenceItem(BaseModel):
    kind: VerifierEvidenceKind
    message: str
    step_index: int | None = None
    score: float = Field(0.0, ge=0.0, le=1.0)
    source: str = "verifier"


class StepVerificationResult(BaseModel):
    step_index: int = Field(ge=1)
    status: StepVerificationStatus
    score: float = Field(0.0, ge=0.0, le=1.0)
    logical_consistency: float = Field(0.0, ge=0.0, le=1.0)
    symbolic_agreement: float = Field(0.0, ge=0.0, le=1.0)
    completeness: float = Field(0.0, ge=0.0, le=1.0)
    step_quality: float = Field(0.0, ge=0.0, le=1.0)
    prefix_quality: float = Field(0.0, ge=0.0, le=1.0)
    contradiction_flags: list[str] = Field(default_factory=list)
    repairability: float = Field(0.0, ge=0.0, le=1.0)
    evidence: list[VerifierEvidenceItem] = Field(default_factory=list)
    failure_type: str | None = None


class VerifierEvidenceBundle(BaseModel):
    runtime_status: VerifierRuntimeStatus
    degraded_mode: bool = False
    summary_tags: list[str] = Field(default_factory=list)
    items: list[VerifierEvidenceItem] = Field(default_factory=list)


class ModelVerifierPayload(BaseModel):
    logical_consistency: float = Field(0.0, ge=0.0, le=1.0)
    symbolic_agreement: float = Field(0.0, ge=0.0, le=1.0)
    completeness: float = Field(0.0, ge=0.0, le=1.0)
    answer_correctness_likelihood: float = Field(0.0, ge=0.0, le=1.0)
    repairability: float = Field(0.0, ge=0.0, le=1.0)
    failure_type: str = "none"
    repair_type: str = RepairStrategy.NONE.value
    summary_tags: list[str] = Field(default_factory=list)
    step_correctness: list[int] = Field(default_factory=list)
    step_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    prefix_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    failure_step_index: int | None = None
    failure_obligation_id: str | None = None
    evidence: list[VerifierEvidenceItem] = Field(default_factory=list)
    overall_score: float | None = Field(default=None, ge=0.0, le=1.0)


class VerifierHookPayload(BaseModel):
    probability: float = Field(0.0, ge=0.0, le=1.0)
    logical_consistency: float = Field(0.0, ge=0.0, le=1.0)
    completeness: float = Field(0.0, ge=0.0, le=1.0)
    repairability: float = Field(0.0, ge=0.0, le=1.0)
    summary: str
    symbolic_agreement: float = Field(0.0, ge=0.0, le=1.0)
    answer_correctness_likelihood: float = Field(0.0, ge=0.0, le=1.0)
    branch_score: float = Field(0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerifierOutput(BaseModel):
    branch_id: str
    problem_id: str
    runtime_status: VerifierRuntimeStatus
    degraded_mode: bool = False
    model_name: str = "deepseek"
    logical_consistency: float = Field(0.0, ge=0.0, le=1.0)
    symbolic_agreement: float = Field(0.0, ge=0.0, le=1.0)
    completeness: float = Field(0.0, ge=0.0, le=1.0)
    answer_correctness_likelihood: float = Field(0.0, ge=0.0, le=1.0)
    repairability: float = Field(0.0, ge=0.0, le=1.0)
    step_quality: float = Field(0.0, ge=0.0, le=1.0)
    prefix_quality: float = Field(0.0, ge=0.0, le=1.0)
    open_obligation_burden: float = Field(0.0, ge=0.0, le=1.0)
    probability: float = Field(0.0, ge=0.0, le=1.0)
    overall_score: float = Field(0.0, ge=0.0, le=1.0)
    failure_type: str | None = None
    repair_type: str | None = None
    step_correctness: list[int] = Field(default_factory=list)
    step_results: list[StepVerificationResult] = Field(default_factory=list)
    evidence_bundle: VerifierEvidenceBundle
    summary: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def symbolic_consistency(self) -> float:
        return self.symbolic_agreement

    def to_label(self) -> VerifierLabel:
        payload = {
            "branch_id": self.branch_id,
            "step_correctness": list(self.step_correctness),
            "symbolic_agreement": self.symbolic_agreement,
            "logical_consistency": self.logical_consistency,
            "completeness": self.completeness,
            "answer_correctness_likelihood": self.answer_correctness_likelihood,
            "repairability": self.repairability,
            "repair_type": self.repair_type,
            "failure_type": self.failure_type,
            "composite_score": self.overall_score,
            "answer_in_range": self.metadata.get("answer_in_range", True),
        }
        for name in ("build_verifier_label", "create_verifier_label", "compose_verifier_label", "label_trace"):
            fn = getattr(verifier_labels_module, name, None)
            if not callable(fn):
                continue
            try:
                result = fn(payload)
            except Exception:
                continue
            if isinstance(result, VerifierLabel):
                return result
            if isinstance(result, Mapping):
                try:
                    return VerifierLabel.model_validate(result)
                except Exception:
                    continue

        return VerifierLabel(
            branch_id=self.branch_id,
            step_correctness=list(self.step_correctness),
            symbolic_agreement=self.symbolic_agreement,
            logical_consistency=self.logical_consistency,
            completeness=self.completeness,
            final_answer_correct=1 if self.answer_correctness_likelihood >= VERIFIER_PASS_THRESHOLD else 0,
            final_answer_correct_probability=self.answer_correctness_likelihood,
            repairability=self.repairability,
            repair_type=self.repair_type,
            failure_type=self.failure_type,
            composite_score=self.overall_score,
        )

    def to_hook_payload(self) -> VerifierHookPayload:
        return VerifierHookPayload(
            probability=self.answer_correctness_likelihood,
            logical_consistency=self.logical_consistency,
            completeness=self.completeness,
            repairability=self.repairability,
            summary=self.summary,
            symbolic_agreement=self.symbolic_agreement,
            answer_correctness_likelihood=self.answer_correctness_likelihood,
            branch_score=self.overall_score,
            metadata={
                "runtime_status": self.runtime_status.value,
                "degraded_mode": self.degraded_mode,
                "symbolic_agreement": self.symbolic_agreement,
                "answer_correctness_likelihood": self.answer_correctness_likelihood,
                "branch_score": self.overall_score,
                "overall_score": self.overall_score,
                "logical_consistency": self.logical_consistency,
                "completeness": self.completeness,
                "repairability": self.repairability,
                "step_quality": self.step_quality,
                "prefix_quality": self.prefix_quality,
                "open_obligation_burden": self.open_obligation_burden,
                "verifier_decomposition": {
                    "logical_consistency": self.logical_consistency,
                    "symbolic_agreement": self.symbolic_agreement,
                    "completeness": self.completeness,
                    "answer_correctness_likelihood": self.answer_correctness_likelihood,
                    "repairability": self.repairability,
                    "step_quality": self.step_quality,
                    "prefix_quality": self.prefix_quality,
                    "open_obligation_burden": self.open_obligation_burden,
                },
                "repair_type": self.repair_type,
                "failure_type": self.failure_type,
                "summary_tags": list(self.evidence_bundle.summary_tags),
                "supporting_obligation_ids": list(self.metadata.get("supporting_obligation_ids", []) or []),
                "open_proof_obligation_count": self.metadata.get("open_proof_obligation_count", 0),
                "contradicted_proof_obligation_count": self.metadata.get("contradicted_proof_obligation_count", 0),
                "failure_step_index": self.metadata.get("failure_step_index"),
                "failure_step_id": self.metadata.get("failure_step_id"),
                "failure_node_id": self.metadata.get("failure_node_id"),
                "failure_obligation_id": self.metadata.get("failure_obligation_id"),
            },
        )


class VerifierConfig(BaseModel):
    model_name: str = "deepseek"
    enable_model: bool = True
    max_problem_chars: int = Field(1400, ge=256)
    max_reasoning_chars: int = Field(2200, ge=256)
    max_node_summary_chars: int = Field(320, ge=64)
    max_steps: int = Field(8, ge=1)
    max_evidence_items: int = Field(6, ge=1)
    max_prompt_chars: int = Field(6500, ge=512)
    prompt_temperature: float = Field(0.0, ge=0.0, le=1.0)
    prompt_max_tokens: int = Field(900, ge=64)
    strict_json_only: bool = True
    allow_model_for_step_scoring: bool = False


class _ModelUnavailableError(RuntimeError):
    pass


class DeepSeekVerifier:
    """
    Stronger teacher-style verifier with deterministic degraded mode.

    The public contract is intentionally shaped to remain swappable with the
    intended distilled verifier interface:
    - load_checkpoint(...)
    - score_trace(...)
    - score_step(...)
    - predict_failure_type(...)
    - predict_repairability(...)
    - callable branch-controller hook via __call__(...)
    """

    def __init__(
        self,
        *,
        model: VerifierModelProtocol | Any | None = None,
        config: VerifierConfig | None = None,
        model_name: str | None = None,
    ) -> None:
        self.model = model
        self.config = config or VerifierConfig()
        if model_name:
            self.config.model_name = model_name

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        model: VerifierModelProtocol | Any | None = None,
        config: VerifierConfig | None = None,
    ) -> "DeepSeekVerifier":
        instance = cls(model=model, config=config, model_name=str(path))
        return instance

    def __call__(
        self,
        *,
        problem: ParsedProblem,
        route: RouteDecision,
        branch: Any,
        node: Any,
        symbolic: Any,
        generation: Any,
    ) -> VerifierHookPayload:
        output = self.verify_branch(
            problem=problem,
            route=route,
            branch=branch,
            node=node,
            symbolic=symbolic,
            generation=generation,
        )
        return output.to_hook_payload()

    def verify_branch(
        self,
        *,
        problem: ParsedProblem,
        route: RouteDecision,
        branch: Any,
        node: Any,
        symbolic: Any,
        generation: Any,
    ) -> VerifierOutput:
        trace_input = self._trace_input_from_hook(
            problem=problem,
            route=route,
            branch=branch,
            node=node,
            symbolic=symbolic,
            generation=generation,
        )
        return self.verify_trace(trace_input)

    def verify_trace(self, trace: VerifierTraceInput | BranchTrace | Mapping[str, Any]) -> VerifierOutput:
        packed = self._coerce_trace_input(trace)
        prompt = self.build_prompt(packed)

        if self.config.enable_model and self.model is not None:
            try:
                raw = self._call_model(prompt)
            except _ModelUnavailableError as exc:
                return self._deterministic_fallback_output(
                    packed,
                    runtime_status=VerifierRuntimeStatus.MODEL_UNAVAILABLE,
                    reason=str(exc),
                )
            except Exception as exc:
                return self._deterministic_fallback_output(
                    packed,
                    runtime_status=VerifierRuntimeStatus.MODEL_ERROR,
                    reason=f"model_error::{type(exc).__name__}",
                )

            try:
                payload = self._parse_model_payload(raw, expected_step_count=len(packed.steps))
                return self._output_from_model_payload(packed, payload)
            except Exception as exc:
                return self._deterministic_fallback_output(
                    packed,
                    runtime_status=VerifierRuntimeStatus.PARSE_ERROR,
                    reason=f"parse_error::{type(exc).__name__}",
                )

        return self._deterministic_fallback_output(
            packed,
            runtime_status=VerifierRuntimeStatus.MODEL_UNAVAILABLE,
            reason="model_not_configured",
        )

    def score_trace(
        self,
        trace_batch: Sequence[VerifierTraceInput | BranchTrace | Mapping[str, Any]],
    ) -> tuple[VerifierOutput, ...]:
        return tuple(self.verify_trace(trace) for trace in trace_batch)

    def score_step(
        self,
        step_batch: Sequence[PackedReasoningStep | ReasoningStep | Mapping[str, Any]],
    ) -> tuple[StepVerificationResult, ...]:
        return tuple(self._score_step(step, use_model=self.config.allow_model_for_step_scoring) for step in step_batch)

    def predict_failure_type(
        self,
        trace_or_step: VerifierTraceInput | BranchTrace | PackedReasoningStep | ReasoningStep | Mapping[str, Any],
    ) -> str | None:
        if self._looks_like_step(trace_or_step):
            return self._score_step(trace_or_step, use_model=False).failure_type
        return self.verify_trace(trace_or_step).failure_type

    def predict_repairability(
        self,
        trace_or_step: VerifierTraceInput | BranchTrace | PackedReasoningStep | ReasoningStep | Mapping[str, Any],
    ) -> float:
        if self._looks_like_step(trace_or_step):
            return self._score_step(trace_or_step, use_model=False).repairability
        return self.verify_trace(trace_or_step).repairability

    def build_prompt(self, trace: VerifierTraceInput) -> str:
        payload = self._pack_trace_input(trace)
        schema = {
            "logical_consistency": "float[0,1]",
            "symbolic_agreement": "float[0,1]",
            "completeness": "float[0,1]",
            "answer_correctness_likelihood": "float[0,1]",
            "repairability": "float[0,1]",
            "step_quality": "float[0,1]|null",
            "prefix_quality": "float[0,1]|null",
            "failure_type": "none|logic_error|missing_case|symbolic_mismatch|false_assumption|coverage_gap|arithmetic",
            "repair_type": "none|local_repair|case_completion|resample",
            "failure_step_index": "int|null",
            "failure_obligation_id": "string|null",
            "summary_tags": ["string"],
            "step_correctness": [0, 1],
            "evidence": [
                {
                    "kind": "model|step|symbolic|answer|coverage|assumption|failure|retrieval|reasoning",
                    "message": "short string",
                    "step_index": "int|null",
                    "score": "float[0,1]",
                    "source": "string",
                }
            ],
            "proof_obligations": [
                {
                    "obligation_id": "string",
                    "claim": "string",
                    "status": "open|partial|discharged|contradicted|blocked",
                }
            ],
            "overall_score": "float[0,1]|null",
        }
        prompt = (
            "You are the stronger verifier for one mathematical reasoning branch.\n"
            "Judge only the provided branch. Do not solve from scratch.\n"
            "Required heads:\n"
            "- logical consistency\n"
            "- symbolic agreement with provided symbolic evidence\n"
            "- completeness\n"
            "- answer correctness likelihood\n"
            "- repairability\n"
            "Return strict JSON only using this schema:\n"
            f"{json.dumps(schema, ensure_ascii=True)}\n"
            "Branch payload:\n"
            f"{json.dumps(_model_dump(payload), ensure_ascii=True)}"
        )
        return prompt[: self.config.max_prompt_chars]

    def _trace_input_from_hook(
        self,
        *,
        problem: ParsedProblem,
        route: RouteDecision,
        branch: Any,
        node: Any,
        symbolic: Any,
        generation: Any,
    ) -> VerifierTraceInput:
        steps = []
        for raw_step in list(getattr(branch, "active_steps", lambda: ())())[-self.config.max_steps :]:
            steps.append(
                PackedReasoningStep(
                    step_index=max(1, int(getattr(raw_step, "index", 0)) + 1),
                    description=_normalize_text(getattr(raw_step, "description", "")),
                    operator_used=_normalize_text(getattr(raw_step, "operator_name", "")) or None,
                    symbolic_valid=bool(getattr(raw_step, "kind", None) is None or getattr(raw_step, "phase", None) is not None),
                    proof_obligation_ids=list(getattr(raw_step, "proof_obligation_ids", ()) or ()),
                )
            )

        retrieval = list(getattr(branch, "active_retrieval_evidence", lambda: ())())
        retrieval_support = 0.0
        if retrieval:
            retrieval_support = _clamp01(sum(float(getattr(item, "score", 0.0) or 0.0) for item in retrieval) / len(retrieval))

        candidate = getattr(branch, "current_candidate", lambda: None)()
        raw_answer = getattr(generation, "answer", None) or getattr(candidate, "raw_answer", None)
        canonical_answer = getattr(generation, "answer_canonical", None) or getattr(candidate, "canonical_answer", None)
        reasoning = _normalize_text(getattr(generation, "reasoning", ""))
        proof_obligations = [
            item
            for item in (
                _pack_proof_obligation(value)
                for value in (
                    list(getattr(node, "proof_obligations", ()) or ())
                    + list(getattr(symbolic, "metadata", {}).get("proof_obligations", []) or [])
                )
            )
            if item is not None
        ]

        return VerifierTraceInput(
            problem_id=problem.problem_id,
            branch_id=str(getattr(branch, "branch_id", problem.problem_id)),
            raw_problem_text=(problem.raw_text or "")[: self.config.max_problem_chars],
            target=problem.target,
            domain=getattr(problem.domain, "value", str(problem.domain)),
            route_problem_type={k: float(v) for k, v in dict(route.problem_type).items()},
            route_archetypes={k: float(v) for k, v in dict(route.archetypes).items()},
            route_uncertainty=_clamp01(route.route_uncertainty),
            node_summary=_normalize_text(getattr(node, "summary_text", ""))[: self.config.max_node_summary_chars],
            reasoning=reasoning[: self.config.max_reasoning_chars],
            answer=_normalize_text(raw_answer) or None,
            answer_canonical=self._normalize_answer(canonical_answer or raw_answer),
            generation_confidence=_clamp01(getattr(generation, "confidence", 0.0)),
            symbolic_passed=bool(getattr(symbolic, "passed", False)),
            symbolic_score=_clamp01(getattr(symbolic, "score", 0.0)),
            symbolic_summary=_normalize_text(getattr(symbolic, "summary", "")),
            exact_symbolic_match=bool(getattr(symbolic, "exact_match", False)),
            retrieval_support=retrieval_support,
            failure_hint=_normalize_text(getattr(branch, "metadata", {}).get("failure_reason", "")) or None,
            branch_phase=_normalize_text(getattr(getattr(branch, "phase", None), "value", getattr(branch, "phase", ""))),
            steps=steps,
            proof_obligations=proof_obligations,
            metadata={
                "operator_history": [
                    _normalize_text(getattr(step, "operator_name", "")) for step in getattr(branch, "active_steps", lambda: ())()
                    if _normalize_text(getattr(step, "operator_name", ""))
                ],
                "generation_summary": _normalize_text(getattr(generation, "summary", "")),
                "discharged_obligation_ids": list(getattr(symbolic, "metadata", {}).get("discharged_obligation_ids", []) or []),
                "contradicted_obligation_ids": list(getattr(symbolic, "metadata", {}).get("contradicted_obligation_ids", []) or []),
            },
        )

    def _coerce_trace_input(self, trace: VerifierTraceInput | BranchTrace | Mapping[str, Any]) -> VerifierTraceInput:
        if isinstance(trace, VerifierTraceInput):
            return self._pack_trace_input(trace)

        if isinstance(trace, BranchTrace):
            steps = [
                PackedReasoningStep(
                    step_index=max(1, int(step.step_num)),
                    description=_normalize_text(step.description),
                    operator_used=_normalize_text(step.operator_used) or None,
                    symbolic_valid=bool(step.symbolic_valid),
                )
                for step in trace.steps[: self.config.max_steps]
            ]
            return self._pack_trace_input(
                VerifierTraceInput(
                    problem_id=trace.problem_id,
                    branch_id=trace.branch_id,
                    reasoning=(trace.full_reasoning or "")[: self.config.max_reasoning_chars],
                    answer=trace.answer,
                    answer_canonical=self._normalize_answer(trace.answer_canonical or trace.answer),
                    symbolic_passed=bool(trace.symbolic_valid),
                    symbolic_score=_clamp01(trace.tool_consistency),
                    exact_symbolic_match=bool(trace.symbolic_valid),
                    retrieval_support=1.0 if trace.retrieval_used else 0.0,
                    branch_phase="critiqued" if trace.self_critiqued else "reasoning",
                    failure_hint=trace.failure_type.value if trace.failure_type else None,
                    steps=steps,
                    proof_obligations=[
                        item
                        for item in (_pack_proof_obligation(value) for value in list(trace.proof_obligations or []))
                        if item is not None
                    ],
                    metadata={
                        "branch_score": float(trace.branch_score or 0.0),
                        "verifier_score": float(trace.verifier_score or 0.0),
                        "logical_consistency": float(trace.logical_consistency or 0.0),
                        "completeness": float(trace.completeness or 0.0),
                        "repairability": float(trace.repairability or 0.0),
                        "step_quality": float(trace.step_quality or 0.0),
                        "prefix_quality": float(trace.prefix_quality or 0.0),
                        "open_obligation_burden": float(trace.open_obligation_burden or 0.0),
                        "proof_state_fingerprint": trace.proof_state_fingerprint,
                        "failure_step_index": trace.failure_step_index,
                        "failure_step_id": trace.failure_step_id,
                        "failure_node_id": trace.failure_node_id,
                        "failure_obligation_id": trace.failure_obligation_id,
                        "verifier_decomposition": dict(trace.verifier_decomposition or {}),
                    },
                )
            )

        payload = dict(trace)
        steps = [self._coerce_step_input(item) for item in payload.get("steps", [])]
        return self._pack_trace_input(
            VerifierTraceInput(
                problem_id=str(payload.get("problem_id", "unknown_problem")),
                branch_id=str(payload.get("branch_id", payload.get("problem_id", "unknown_branch"))),
                raw_problem_text=str(payload.get("raw_problem_text", "")),
                target=str(payload.get("target", "")),
                domain=str(payload.get("domain", "unknown")),
                route_problem_type={str(k): float(v) for k, v in dict(payload.get("route_problem_type") or {}).items()},
                route_archetypes={str(k): float(v) for k, v in dict(payload.get("route_archetypes") or {}).items()},
                route_uncertainty=_clamp01(payload.get("route_uncertainty", 0.0)),
                node_summary=str(payload.get("node_summary", "")),
                reasoning=str(payload.get("reasoning", "")),
                answer=_normalize_text(payload.get("answer", "")) or None,
                answer_canonical=self._normalize_answer(payload.get("answer_canonical", payload.get("answer"))),
                generation_confidence=_clamp01(payload.get("generation_confidence", 0.0)),
                symbolic_passed=bool(payload.get("symbolic_passed", False)),
                symbolic_score=_clamp01(payload.get("symbolic_score", 0.0)),
                symbolic_summary=str(payload.get("symbolic_summary", "")),
                exact_symbolic_match=bool(payload.get("exact_symbolic_match", False)),
                retrieval_support=_clamp01(payload.get("retrieval_support", 0.0)),
                failure_hint=_normalize_text(payload.get("failure_hint", "")) or None,
                branch_phase=str(payload.get("branch_phase", "")),
                steps=steps,
                proof_obligations=[
                    item
                    for item in (
                        _pack_proof_obligation(value)
                        for value in list(payload.get("proof_obligations", []) or [])
                    )
                    if item is not None
                ],
                metadata=dict(payload.get("metadata") or {}),
            )
        )

    def _pack_trace_input(self, trace: VerifierTraceInput) -> VerifierTraceInput:
        packed_steps = trace.steps[: self.config.max_steps]
        return trace.model_copy(
            update={
                "raw_problem_text": (trace.raw_problem_text or "")[: self.config.max_problem_chars],
                "node_summary": (trace.node_summary or "")[: self.config.max_node_summary_chars],
                "reasoning": (trace.reasoning or "")[: self.config.max_reasoning_chars],
                "answer": _normalize_text(trace.answer) or None,
                "answer_canonical": self._normalize_answer(trace.answer_canonical or trace.answer),
                "steps": packed_steps,
                "proof_obligations": trace.proof_obligations[:12],
                "route_uncertainty": _clamp01(trace.route_uncertainty),
                "generation_confidence": _clamp01(trace.generation_confidence),
                "symbolic_score": _clamp01(trace.symbolic_score),
                "retrieval_support": _clamp01(trace.retrieval_support),
            }
        )

    def _prm_summary_from_trace(self, trace: VerifierTraceInput):
        contradiction_count = sum(1 for item in trace.proof_obligations if item.status == "contradicted")
        return score_process(
            ProcessScoreInput(
                problem_id=trace.problem_id,
                branch_id=trace.branch_id,
                steps=tuple(
                    ProcessStepInput(
                        step_index=step.step_index,
                        description=step.description,
                        operator_name=step.operator_used,
                        symbolic_valid=step.symbolic_valid,
                        proof_obligation_ids=tuple(step.proof_obligation_ids),
                    )
                    for step in trace.steps
                ),
                proof_obligations=tuple(
                    ProcessObligationInput(
                        obligation_id=item.obligation_id,
                        status=item.status,
                        claim=item.claim,
                        evidence_kind_required=item.evidence_kind_required,
                        originating_node_id=item.originating_node_id,
                        target_goal_id=item.target_goal_id,
                    )
                    for item in trace.proof_obligations
                ),
                route_uncertainty=trace.route_uncertainty,
                retrieval_support=trace.retrieval_support,
                symbolic_score=trace.symbolic_score,
                symbolic_passed=trace.symbolic_passed,
                exact_symbolic_match=trace.exact_symbolic_match,
                answer_present=bool(self._normalize_answer(trace.answer_canonical or trace.answer)),
                contradiction_count=contradiction_count,
                metadata=dict(trace.metadata),
            )
        )

    def _coerce_step_input(
        self,
        step: PackedReasoningStep | ReasoningStep | Mapping[str, Any],
    ) -> PackedReasoningStep:
        if isinstance(step, PackedReasoningStep):
            return step
        if isinstance(step, ReasoningStep):
            return PackedReasoningStep(
                step_index=max(1, int(step.step_num)),
                description=_normalize_text(step.description),
                operator_used=_normalize_text(step.operator_used) or None,
                symbolic_valid=bool(step.symbolic_valid),
            )
        return PackedReasoningStep(
            step_index=max(1, int(step.get("step_index", step.get("step_num", 1)))),
            description=_normalize_text(step.get("description", "")),
            operator_used=_normalize_text(step.get("operator_used", "")) or None,
            symbolic_valid=bool(step.get("symbolic_valid", True)),
            proof_obligation_ids=[
                _normalize_text(item)
                for item in list(step.get("proof_obligation_ids", []) or [])
                if _normalize_text(item)
            ],
        )

    def _score_step(
        self,
        step: PackedReasoningStep | ReasoningStep | Mapping[str, Any],
        *,
        use_model: bool,
    ) -> StepVerificationResult:
        packed = self._coerce_step_input(step)
        prm_summary = score_process(
            ProcessScoreInput(
                problem_id="step_only",
                branch_id="step_only",
                steps=(
                    ProcessStepInput(
                        step_index=packed.step_index,
                        description=packed.description,
                        operator_name=packed.operator_used,
                        symbolic_valid=packed.symbolic_valid,
                        proof_obligation_ids=tuple(packed.proof_obligation_ids),
                    ),
                ),
                proof_obligations=tuple(
                    ProcessObligationInput(obligation_id=item, status="open")
                    for item in packed.proof_obligation_ids
                ),
                symbolic_score=1.0 if packed.symbolic_valid else 0.0,
                symbolic_passed=packed.symbolic_valid,
                answer_present=False,
            )
        )
        step_quality = prm_summary.mean_step_quality
        prefix_quality = prm_summary.prefix_quality
        if use_model and self.model is not None:
            trace_input = VerifierTraceInput(
                task_type=VerifierTaskType.STEP,
                problem_id="step_only",
                branch_id="step_only",
                reasoning=packed.description,
                steps=[packed],
                symbolic_passed=packed.symbolic_valid,
                symbolic_score=1.0 if packed.symbolic_valid else 0.0,
            )
            output = self.verify_trace(trace_input)
            if output.step_results:
                return output.step_results[0]

        failure_type = None
        flags: list[str] = []
        evidence: list[VerifierEvidenceItem] = []
        logical = 0.82 if packed.symbolic_valid else 0.24
        symbolic = 1.0 if packed.symbolic_valid else 0.0
        completeness = 0.55 if packed.description else 0.10

        if not packed.symbolic_valid:
            failure_type = FailureType.SYMBOLIC_MISMATCH.value
            flags.append("symbolic_mismatch")
            evidence.append(
                VerifierEvidenceItem(
                    kind=VerifierEvidenceKind.SYMBOLIC,
                    message="symbolic mismatch",
                    step_index=packed.step_index,
                    score=0.0,
                    source="deterministic_fallback",
                )
            )

        if _ASSUMPTION_RE.search(packed.description):
            failure_type = failure_type or FailureType.FALSE_ASSUMPTION.value
            flags.append("unsupported_assumption")
            logical = min(logical, 0.48)
            evidence.append(
                VerifierEvidenceItem(
                    kind=VerifierEvidenceKind.ASSUMPTION,
                    message="unsupported assumption language",
                    step_index=packed.step_index,
                    score=0.40,
                    source="deterministic_fallback",
                )
            )

        if packed.proof_obligation_ids and packed.symbolic_valid:
            completeness = min(1.0, completeness + 0.08)
            evidence.append(
                VerifierEvidenceItem(
                    kind=VerifierEvidenceKind.COVERAGE,
                    message="step linked to formal proof obligation",
                    step_index=packed.step_index,
                    score=0.72,
                    source="deterministic_fallback",
                )
            )
        elif packed.proof_obligation_ids and not packed.symbolic_valid:
            failure_type = failure_type or FailureType.SYMBOLIC_MISMATCH.value
            flags.append("obligation_unresolved")

        if packed.symbolic_valid and not flags:
            status = StepVerificationStatus.VALID
            repairability = 0.20
        elif packed.symbolic_valid:
            status = StepVerificationStatus.UNCERTAIN
            repairability = 0.55
        else:
            status = StepVerificationStatus.INVALID
            repairability = 0.74

        score = _clamp01(0.45 * logical + 0.35 * symbolic + 0.20 * completeness)
        return StepVerificationResult(
            step_index=packed.step_index,
            status=status,
            score=score,
            logical_consistency=_clamp01(logical),
            symbolic_agreement=_clamp01(symbolic),
            completeness=_clamp01(completeness),
            step_quality=step_quality,
            prefix_quality=prefix_quality,
            contradiction_flags=flags,
            repairability=_clamp01(repairability),
            evidence=evidence[: self.config.max_evidence_items],
            failure_type=failure_type,
        )

    def _call_model(self, prompt: str) -> str:
        if self.model is None:
            raise _ModelUnavailableError("model_not_configured")

        candidates = []
        if hasattr(self.model, "generate"):
            candidates.append(getattr(self.model, "generate"))
        if hasattr(self.model, "complete"):
            candidates.append(getattr(self.model, "complete"))
        if hasattr(self.model, "chat"):
            candidates.append(getattr(self.model, "chat"))
        if callable(self.model):
            candidates.append(self.model)
        if not candidates:
            raise _ModelUnavailableError("model_has_no_callable_interface")

        last_error: Exception | None = None
        for candidate in candidates:
            try:
                signature = inspect.signature(candidate)
            except (TypeError, ValueError):
                signature = None

            kwargs: dict[str, Any] = {}
            if signature is not None:
                if "prompt" in signature.parameters:
                    kwargs["prompt"] = prompt
                if "max_tokens" in signature.parameters:
                    kwargs["max_tokens"] = self.config.prompt_max_tokens
                if "temperature" in signature.parameters:
                    kwargs["temperature"] = self.config.prompt_temperature
                if "timeout" in signature.parameters:
                    kwargs["timeout"] = 0

            try:
                if kwargs:
                    result = candidate(**kwargs)
                else:
                    result = candidate(prompt)
            except TypeError:
                try:
                    result = candidate(prompt=prompt)
                except Exception as exc:
                    last_error = exc
                    continue
            except Exception as exc:
                last_error = exc
                continue

            if result is None:
                last_error = RuntimeError("model_returned_none")
                continue
            return str(result)

        if last_error is None:
            raise _ModelUnavailableError("model_callable_failed")
        raise last_error

    def _parse_model_payload(self, raw: str, *, expected_step_count: int) -> ModelVerifierPayload:
        raw_text = str(raw or "").strip()
        match = _JSON_BLOCK_RE.search(raw_text)
        if match is None:
            raise ValueError("no_json_payload")
        payload = json.loads(match.group(0))
        if not isinstance(payload, dict):
            raise ValueError("json_payload_not_object")

        evidence: list[VerifierEvidenceItem] = []
        for item in list(payload.get("evidence", []))[: self.config.max_evidence_items]:
            if isinstance(item, str):
                evidence.append(
                    VerifierEvidenceItem(
                        kind=VerifierEvidenceKind.MODEL,
                        message=_normalize_text(item),
                        score=0.50,
                        source="model",
                    )
                )
                continue
            evidence.append(
                VerifierEvidenceItem(
                    kind=VerifierEvidenceKind(str(item.get("kind", VerifierEvidenceKind.MODEL.value))),
                    message=_normalize_text(item.get("message", "")),
                    step_index=int(item["step_index"]) if item.get("step_index") is not None else None,
                    score=_clamp01(item.get("score", 0.5)),
                    source=_normalize_text(item.get("source", "model")) or "model",
                )
            )

        step_correctness = [1 if int(value) > 0 else 0 for value in payload.get("step_correctness", [])[:expected_step_count]]
        if len(step_correctness) < expected_step_count:
            step_correctness.extend([0] * (expected_step_count - len(step_correctness)))

        return ModelVerifierPayload(
            logical_consistency=_clamp01(payload.get("logical_consistency", 0.0)),
            symbolic_agreement=_clamp01(payload.get("symbolic_agreement", 0.0)),
            completeness=_clamp01(payload.get("completeness", 0.0)),
            answer_correctness_likelihood=_clamp01(payload.get("answer_correctness_likelihood", 0.0)),
            repairability=_clamp01(payload.get("repairability", 0.0)),
            failure_type=_normalize_text(payload.get("failure_type", "none")) or "none",
            repair_type=_normalize_text(payload.get("repair_type", RepairStrategy.NONE.value)) or RepairStrategy.NONE.value,
            summary_tags=[_normalize_text(tag) for tag in payload.get("summary_tags", []) if _normalize_text(tag)],
            step_correctness=step_correctness,
            step_quality=_clamp01(payload.get("step_quality", 0.0)) if payload.get("step_quality") is not None else None,
            prefix_quality=_clamp01(payload.get("prefix_quality", 0.0)) if payload.get("prefix_quality") is not None else None,
            failure_step_index=int(payload["failure_step_index"]) if payload.get("failure_step_index") is not None else None,
            failure_obligation_id=_normalize_text(payload.get("failure_obligation_id", "")) or None,
            evidence=evidence,
            overall_score=payload.get("overall_score"),
        )

    def _output_from_model_payload(self, trace: VerifierTraceInput, payload: ModelVerifierPayload) -> VerifierOutput:
        probability = _clamp01(payload.answer_correctness_likelihood)
        prm_summary = self._prm_summary_from_trace(trace)
        step_quality = _clamp01(payload.step_quality if payload.step_quality is not None else prm_summary.mean_step_quality)
        prefix_quality = _clamp01(payload.prefix_quality if payload.prefix_quality is not None else prm_summary.prefix_quality)
        step_results = self._step_results_from_step_correctness(
            trace.steps,
            payload.step_correctness,
            runtime_status=VerifierRuntimeStatus.MODEL,
            prm_summary=prm_summary,
        )
        summary_tags = payload.summary_tags or self._default_summary_tags(
            failure_type=payload.failure_type,
            completeness=payload.completeness,
            repairability=payload.repairability,
        )
        summary = self._build_summary(
            runtime_status=VerifierRuntimeStatus.MODEL,
            failure_type=None if payload.failure_type == "none" else payload.failure_type,
            tags=summary_tags,
        )
        branch_score, scoring_metadata = self._score_output(
            trace=trace,
            model_confidence=payload.overall_score,
            logical_consistency=payload.logical_consistency,
            symbolic_agreement=payload.symbolic_agreement,
            completeness=payload.completeness,
            answer_correctness=payload.answer_correctness_likelihood,
            repairability=payload.repairability,
            step_quality=step_quality,
            prefix_quality=prefix_quality,
            open_obligation_burden=prm_summary.open_obligation_burden,
            runtime_status=VerifierRuntimeStatus.MODEL,
            degraded_mode=False,
            failure_type=None if payload.failure_type == "none" else payload.failure_type,
            repair_type=None if payload.repair_type == RepairStrategy.NONE.value else payload.repair_type,
        )
        return VerifierOutput(
            branch_id=trace.branch_id,
            problem_id=trace.problem_id,
            runtime_status=VerifierRuntimeStatus.MODEL,
            degraded_mode=False,
            model_name=self.config.model_name,
            logical_consistency=payload.logical_consistency,
            symbolic_agreement=payload.symbolic_agreement,
            completeness=payload.completeness,
            answer_correctness_likelihood=payload.answer_correctness_likelihood,
            repairability=payload.repairability,
            step_quality=step_quality,
            prefix_quality=prefix_quality,
            open_obligation_burden=prm_summary.open_obligation_burden,
            probability=probability,
            overall_score=branch_score,
            failure_type=None if payload.failure_type == "none" else payload.failure_type,
            repair_type=None if payload.repair_type == RepairStrategy.NONE.value else payload.repair_type,
            step_correctness=list(payload.step_correctness),
            step_results=step_results,
            evidence_bundle=VerifierEvidenceBundle(
                runtime_status=VerifierRuntimeStatus.MODEL,
                degraded_mode=False,
                summary_tags=summary_tags,
                items=payload.evidence[: self.config.max_evidence_items],
            ),
            summary=summary,
            metadata={
                "task_type": trace.task_type.value,
                "prompt_mode": "strict_json",
                "step_count": len(trace.steps),
                "answer_in_range": trace.answer_canonical is not None,
                "proof_obligation_count": len(trace.proof_obligations),
                "open_proof_obligation_count": sum(
                    1 for item in trace.proof_obligations if item.status in {"open", "partial"}
                ),
                "supporting_obligation_ids": [item.obligation_id for item in trace.proof_obligations],
                "failure_step_index": payload.failure_step_index,
                "failure_step_id": trace.metadata.get("failure_step_id"),
                "failure_node_id": trace.metadata.get("failure_node_id"),
                "failure_obligation_id": payload.failure_obligation_id,
                "step_quality": step_quality,
                "prefix_quality": prefix_quality,
                "open_obligation_burden": prm_summary.open_obligation_burden,
                **scoring_metadata,
            },
        )

    def _deterministic_fallback_output(
        self,
        trace: VerifierTraceInput,
        *,
        runtime_status: VerifierRuntimeStatus,
        reason: str,
    ) -> VerifierOutput:
        answer_canonical = self._normalize_answer(trace.answer_canonical or trace.answer)
        has_answer = answer_canonical is not None
        reasoning_signal = min(1.0, len(trace.reasoning) / 180.0) if trace.reasoning else 0.0
        structural_signal = min(1.0, len(trace.steps) / max(1, self.config.max_steps))
        exact_match_bonus = 1.0 if trace.exact_symbolic_match else 0.0
        open_obligations = [item for item in trace.proof_obligations if item.status in {"open", "partial"}]
        contradicted_obligations = [item for item in trace.proof_obligations if item.status == "contradicted"]
        prm_summary = self._prm_summary_from_trace(trace)

        logical = _clamp01(
            0.34 * trace.symbolic_score
            + 0.26 * trace.generation_confidence
            + 0.18 * (1.0 - trace.route_uncertainty)
            + 0.12 * structural_signal
            + 0.10 * reasoning_signal
        )
        symbolic_agreement = _clamp01(0.82 * trace.symbolic_score + 0.18 * (1.0 if trace.symbolic_passed else 0.0))
        completeness = _clamp01(
            0.15
            + 0.30 * (1.0 if has_answer else 0.0)
            + 0.20 * structural_signal
            + 0.20 * reasoning_signal
            + 0.15 * (1.0 if trace.reasoning else 0.0)
        )
        if open_obligations:
            completeness = _clamp01(completeness - min(0.25, 0.05 * len(open_obligations)))
        if contradicted_obligations:
            logical = _clamp01(logical - min(0.45, 0.18 * len(contradicted_obligations)))
        answer_correctness = _clamp01(
            0.35 * logical
            + 0.30 * symbolic_agreement
            + 0.20 * completeness
            + 0.10 * (1.0 if has_answer else 0.0)
            + 0.05 * trace.retrieval_support
        )

        failure_type, repair_type, summary_tags, evidence_items = self._infer_failure_and_evidence(
            trace=trace,
            logical=logical,
            symbolic_agreement=symbolic_agreement,
            completeness=completeness,
            has_answer=has_answer,
            reason=reason,
        )
        if trace.exact_symbolic_match:
            answer_correctness = _clamp01(max(answer_correctness, 0.80))
            evidence_items.append(
                VerifierEvidenceItem(
                    kind=VerifierEvidenceKind.SYMBOLIC,
                    message="exact symbolic match",
                    score=1.0,
                    source="deterministic_fallback",
                )
            )
        if exact_match_bonus:
            logical = _clamp01(max(logical, 0.72))

        repairability = self._repairability_from_failure(
            failure_type=failure_type,
            trace=trace,
            completeness=completeness,
            symbolic_agreement=symbolic_agreement,
            has_answer=has_answer,
        )
        probability = _clamp01(answer_correctness)
        branch_score, scoring_metadata = self._score_output(
            trace=trace,
            model_confidence=probability,
            logical_consistency=logical,
            symbolic_agreement=symbolic_agreement,
            completeness=completeness,
            answer_correctness=answer_correctness,
            repairability=repairability,
            step_quality=prm_summary.mean_step_quality,
            prefix_quality=prm_summary.prefix_quality,
            open_obligation_burden=prm_summary.open_obligation_burden,
            runtime_status=runtime_status,
            degraded_mode=True,
            failure_type=failure_type,
            repair_type=repair_type,
        )
        step_results = []
        for index, step in enumerate(trace.steps):
            result = self._score_step(step, use_model=False)
            if index < len(prm_summary.step_scores):
                prm_step = prm_summary.step_scores[index]
                result = result.model_copy(
                    update={
                        "step_quality": prm_step.step_quality,
                        "prefix_quality": prm_step.prefix_quality,
                    }
                )
            step_results.append(result)
        step_correctness = [1 if result.status is StepVerificationStatus.VALID else 0 for result in step_results]
        summary = self._build_summary(runtime_status=runtime_status, failure_type=failure_type, tags=summary_tags)

        return VerifierOutput(
            branch_id=trace.branch_id,
            problem_id=trace.problem_id,
            runtime_status=runtime_status,
            degraded_mode=True,
            model_name=self.config.model_name,
            logical_consistency=logical,
            symbolic_agreement=symbolic_agreement,
            completeness=completeness,
            answer_correctness_likelihood=answer_correctness,
            repairability=repairability,
            step_quality=prm_summary.mean_step_quality,
            prefix_quality=prm_summary.prefix_quality,
            open_obligation_burden=prm_summary.open_obligation_burden,
            probability=probability,
            overall_score=branch_score,
            failure_type=failure_type,
            repair_type=repair_type,
            step_correctness=step_correctness,
            step_results=step_results,
            evidence_bundle=VerifierEvidenceBundle(
                runtime_status=runtime_status,
                degraded_mode=True,
                summary_tags=summary_tags,
                items=evidence_items[: self.config.max_evidence_items],
            ),
            summary=summary,
            metadata={
                "task_type": trace.task_type.value,
                "fallback_reason": reason,
                "step_count": len(trace.steps),
                "has_answer": has_answer,
                "answer_in_range": has_answer,
                "proof_obligation_count": len(trace.proof_obligations),
                "open_proof_obligation_count": len(open_obligations),
                "contradicted_proof_obligation_count": len(contradicted_obligations),
                "supporting_obligation_ids": [item.obligation_id for item in trace.proof_obligations],
                "failure_step_index": trace.metadata.get("failure_step_index"),
                "failure_step_id": trace.metadata.get("failure_step_id"),
                "failure_node_id": trace.metadata.get("failure_node_id"),
                "failure_obligation_id": trace.metadata.get("failure_obligation_id"),
                "step_quality": prm_summary.mean_step_quality,
                "prefix_quality": prm_summary.prefix_quality,
                "open_obligation_burden": prm_summary.open_obligation_burden,
                **scoring_metadata,
            },
        )

    def _score_output(
        self,
        *,
        trace: VerifierTraceInput,
        model_confidence: float | None,
        logical_consistency: float,
        symbolic_agreement: float,
        completeness: float,
        answer_correctness: float,
        repairability: float,
        step_quality: float,
        prefix_quality: float,
        open_obligation_burden: float,
        runtime_status: VerifierRuntimeStatus,
        degraded_mode: bool,
        failure_type: str | None,
        repair_type: str | None,
    ) -> tuple[float, dict[str, Any]]:
        payload = {
            "problem_id": trace.problem_id,
            "branch_id": trace.branch_id,
            "model_confidence": _clamp01(
                model_confidence if model_confidence is not None else answer_correctness
            ),
            "logical_consistency": logical_consistency,
            "symbolic_agreement": symbolic_agreement,
            "completeness": completeness,
            "answer_correctness_likelihood": answer_correctness,
            "repairability": repairability,
            "step_quality": step_quality,
            "prefix_quality": prefix_quality,
            "open_obligation_burden": open_obligation_burden,
            "route_uncertainty": trace.route_uncertainty,
            "step_count": len(trace.steps),
            "artifact_state": "ready" if runtime_status is VerifierRuntimeStatus.MODEL else runtime_status.value,
            "mode": "llm_backed" if runtime_status is VerifierRuntimeStatus.MODEL else "heuristic",
            "failure_type": failure_type,
            "repair_type": repair_type,
            "metadata": {
                "runtime_status": runtime_status.value,
                "degraded_mode": degraded_mode,
            },
        }
        try:
            bundle = scoring_module.compute_verifier_score(payload)
        except Exception as exc:
            return (
                _clamp01(
                    0.34 * answer_correctness
                    + 0.28 * logical_consistency
                    + 0.22 * symbolic_agreement
                    + 0.10 * completeness
                    + 0.06 * repairability
                ),
                {
                    "scoring_source": "deepseek_local_fallback",
                    "scoring_error": f"{type(exc).__name__}",
                    "step_quality": step_quality,
                    "prefix_quality": prefix_quality,
                    "open_obligation_burden": open_obligation_burden,
                },
            )

        branch_score = getattr(bundle, "branch_score", None)
        if not isinstance(branch_score, (int, float)):
            branch_score = answer_correctness

        metadata: dict[str, Any] = {"scoring_source": "scoring.compute_verifier_score"}
        summary = getattr(bundle, "summary", None)
        if isinstance(summary, str) and summary:
            metadata["scoring_summary"] = summary
        verdict = getattr(bundle, "verdict", None)
        verdict_value = getattr(verdict, "value", verdict)
        if isinstance(verdict_value, str) and verdict_value:
            metadata["scoring_verdict"] = verdict_value
        flags = getattr(bundle, "flags", None)
        if isinstance(flags, BaseModel):
            metadata["scoring_flags"] = _model_dump(flags)
        uncertainty = getattr(bundle, "uncertainty", None)
        if isinstance(uncertainty, BaseModel):
            metadata["scoring_uncertainty"] = _model_dump(uncertainty)
        return _clamp01(branch_score), metadata

    def _infer_failure_and_evidence(
        self,
        *,
        trace: VerifierTraceInput,
        logical: float,
        symbolic_agreement: float,
        completeness: float,
        has_answer: bool,
        reason: str,
    ) -> tuple[str | None, str | None, list[str], list[VerifierEvidenceItem]]:
        evidence: list[VerifierEvidenceItem] = [
            VerifierEvidenceItem(
                kind=VerifierEvidenceKind.MODEL,
                message=reason,
                score=0.0,
                source="deterministic_fallback",
            )
        ]
        tags = [reason]
        failure_type: str | None = None
        repair_type: str | None = None

        contradicted_obligations = [item for item in trace.proof_obligations if item.status == "contradicted"]
        open_obligations = [item for item in trace.proof_obligations if item.status in {"open", "partial"}]

        if contradicted_obligations:
            failure_type = FailureType.SYMBOLIC_MISMATCH.value
            repair_type = RepairStrategy.LOCAL_REPAIR.value
            tags.extend(["contradicted obligation", "formal contradiction"])
            evidence.append(
                VerifierEvidenceItem(
                    kind=VerifierEvidenceKind.FAILURE,
                    message=f"contradicted_obligation::{contradicted_obligations[0].obligation_id}",
                    score=0.0,
                    source="deterministic_fallback",
                )
            )
        elif open_obligations and completeness < 0.80:
            failure_type = FailureType.COVERAGE_GAP.value
            repair_type = RepairStrategy.LOCAL_REPAIR.value
            tags.extend(["open proof obligation", "incomplete justification"])
            evidence.append(
                VerifierEvidenceItem(
                    kind=VerifierEvidenceKind.COVERAGE,
                    message=f"open_obligation::{open_obligations[0].obligation_id}",
                    score=max(0.0, 1.0 - completeness),
                    source="deterministic_fallback",
                )
            )

        if (not trace.symbolic_passed or symbolic_agreement < 0.45) and failure_type is None:
            failure_type = FailureType.SYMBOLIC_MISMATCH.value
            repair_type = RepairStrategy.LOCAL_REPAIR.value
            tags.extend(["symbolic mismatch", "local repair"])
            evidence.append(
                VerifierEvidenceItem(
                    kind=VerifierEvidenceKind.SYMBOLIC,
                    message=trace.symbolic_summary or "symbolic mismatch",
                    score=symbolic_agreement,
                    source="deterministic_fallback",
                )
            )

        assumption_hit = _ASSUMPTION_RE.search(trace.reasoning) or any(_ASSUMPTION_RE.search(step.description) for step in trace.steps)
        if assumption_hit and failure_type is None:
            failure_type = FailureType.FALSE_ASSUMPTION.value
            repair_type = RepairStrategy.LOCAL_REPAIR.value
            tags.extend(["unsupported assumption", "unjustified"])
            evidence.append(
                VerifierEvidenceItem(
                    kind=VerifierEvidenceKind.ASSUMPTION,
                    message="unsupported assumption language",
                    score=max(0.0, 1.0 - logical),
                    source="deterministic_fallback",
                )
            )

        if (not has_answer or completeness < 0.48) and failure_type is None:
            if _CASE_RE.search(trace.reasoning):
                failure_type = FailureType.MISSING_CASE.value
                repair_type = RepairStrategy.CASE_COMPLETION.value
                tags.extend(["missing case", "coverage gap"])
                evidence.append(
                    VerifierEvidenceItem(
                        kind=VerifierEvidenceKind.COVERAGE,
                        message="missing case coverage",
                        score=max(0.0, 1.0 - completeness),
                        source="deterministic_fallback",
                    )
                )
            else:
                failure_type = FailureType.COVERAGE_GAP.value
                repair_type = RepairStrategy.LOCAL_REPAIR.value
                tags.extend(["coverage gap", "partial"])
                evidence.append(
                    VerifierEvidenceItem(
                        kind=VerifierEvidenceKind.COVERAGE,
                        message="coverage gap",
                        score=max(0.0, 1.0 - completeness),
                        source="deterministic_fallback",
                    )
                )

        if failure_type is None and logical < 0.40:
            failure_type = FailureType.LOGIC_ERROR.value
            repair_type = RepairStrategy.RESAMPLE.value
            tags.extend(["logic gap", "verifier rejection"])
            evidence.append(
                VerifierEvidenceItem(
                    kind=VerifierEvidenceKind.FAILURE,
                    message="logic gap",
                    score=max(0.0, 1.0 - logical),
                    source="deterministic_fallback",
                )
            )

        if has_answer:
            evidence.append(
                VerifierEvidenceItem(
                    kind=VerifierEvidenceKind.ANSWER,
                    message=f"answer_candidate::{trace.answer_canonical}",
                    score=0.85,
                    source="deterministic_fallback",
                )
            )
        else:
            evidence.append(
                VerifierEvidenceItem(
                    kind=VerifierEvidenceKind.ANSWER,
                    message="answer_missing",
                    score=0.0,
                    source="deterministic_fallback",
                )
            )

        if trace.retrieval_support > 0.0:
            evidence.append(
                VerifierEvidenceItem(
                    kind=VerifierEvidenceKind.RETRIEVAL,
                    message="retrieval support observed",
                    score=trace.retrieval_support,
                    source="deterministic_fallback",
                )
            )

        if failure_type is None:
            tags.append("consistent")

        return failure_type, repair_type, tags[:6], evidence

    def _repairability_from_failure(
        self,
        *,
        failure_type: str | None,
        trace: VerifierTraceInput,
        completeness: float,
        symbolic_agreement: float,
        has_answer: bool,
    ) -> float:
        if failure_type == FailureType.SYMBOLIC_MISMATCH.value:
            return _clamp01(0.82 - 0.30 * symbolic_agreement)
        if failure_type == FailureType.FALSE_ASSUMPTION.value:
            return 0.62
        if failure_type == FailureType.MISSING_CASE.value:
            return _clamp01(0.78 - 0.20 * completeness)
        if failure_type == FailureType.COVERAGE_GAP.value:
            return _clamp01(0.72 - 0.15 * completeness)
        if failure_type == FailureType.LOGIC_ERROR.value:
            return 0.32
        if not has_answer and trace.reasoning:
            return 0.58
        return 0.18

    def _step_results_from_step_correctness(
        self,
        steps: Sequence[PackedReasoningStep],
        step_correctness: Sequence[int],
        *,
        runtime_status: VerifierRuntimeStatus,
        prm_summary=None,
    ) -> list[StepVerificationResult]:
        out: list[StepVerificationResult] = []
        for index, step in enumerate(steps):
            value = int(step_correctness[index]) if index < len(step_correctness) else 0
            status = StepVerificationStatus.VALID if value > 0 else StepVerificationStatus.INVALID
            prm_step = None
            if prm_summary is not None and index < len(getattr(prm_summary, "step_scores", ())):
                prm_step = prm_summary.step_scores[index]
            out.append(
                StepVerificationResult(
                    step_index=step.step_index,
                    status=status,
                    score=1.0 if value > 0 else 0.0,
                    logical_consistency=1.0 if value > 0 else 0.0,
                    symbolic_agreement=1.0 if value > 0 and step.symbolic_valid else 0.0,
                    completeness=0.80 if value > 0 else 0.30,
                    step_quality=_clamp01(getattr(prm_step, "step_quality", 1.0 if value > 0 else 0.0)),
                    prefix_quality=_clamp01(getattr(prm_step, "prefix_quality", 1.0 if value > 0 else 0.0)),
                    contradiction_flags=[] if value > 0 else ["model_flagged_invalid"],
                    repairability=0.20 if value > 0 else 0.70,
                    evidence=[
                        VerifierEvidenceItem(
                            kind=VerifierEvidenceKind.STEP,
                            message=f"{runtime_status.value} step verdict",
                            step_index=step.step_index,
                            score=1.0 if value > 0 else 0.0,
                            source=runtime_status.value,
                        )
                    ],
                    failure_type=None if value > 0 else FailureType.LOGIC_ERROR.value,
                )
            )
        return out

    def _fuse_probability(
        self,
        *,
        logical_consistency: float,
        symbolic_agreement: float,
        completeness: float,
        answer_correctness: float,
    ) -> float:
        return _clamp01(
            0.30 * logical_consistency
            + 0.25 * symbolic_agreement
            + 0.15 * completeness
            + 0.30 * answer_correctness
        )

    def _build_summary(
        self,
        *,
        runtime_status: VerifierRuntimeStatus,
        failure_type: str | None,
        tags: Sequence[str],
    ) -> str:
        normalized_tags = [_normalize_text(str(tag).replace("_", " ")) for tag in tags if _normalize_text(tag)]
        if failure_type:
            normalized_failure = _normalize_text(str(failure_type).replace("_", " "))
            normalized_tags = [normalized_failure] + [tag for tag in normalized_tags if tag != normalized_failure]
        return f"verifier::{runtime_status.value}|" + "|".join(normalized_tags[:5] or ["consistent"])

    def _default_summary_tags(
        self,
        *,
        failure_type: str,
        completeness: float,
        repairability: float,
    ) -> list[str]:
        tags: list[str] = []
        if failure_type and failure_type != "none":
            tags.append(str(failure_type).replace("_", " "))
        if completeness < 0.5:
            tags.append("coverage gap")
        if repairability >= 0.5:
            tags.append("local repair")
        if not tags:
            tags.append("consistent")
        return tags

    def _normalize_answer(self, answer: Any) -> str | None:
        return canonicalize_competition_answer(answer)

    def _looks_like_step(self, value: Any) -> bool:
        if isinstance(value, (PackedReasoningStep, ReasoningStep)):
            return True
        if isinstance(value, Mapping):
            return "description" in value and ("step_index" in value or "step_num" in value)
        return False


def build_deepseek_verifier(
    *,
    model: VerifierModelProtocol | Any | None = None,
    config: VerifierConfig | None = None,
    model_name: str | None = None,
) -> DeepSeekVerifier:
    return DeepSeekVerifier(model=model, config=config, model_name=model_name)


__all__ = [
    "DeepSeekVerifier",
    "ModelVerifierPayload",
    "PackedReasoningStep",
    "PackedProofObligation",
    "RepairStrategy",
    "StepVerificationResult",
    "StepVerificationStatus",
    "VerifierConfig",
    "VerifierEvidenceBundle",
    "VerifierEvidenceItem",
    "VerifierEvidenceKind",
    "VerifierHookPayload",
    "VerifierOutput",
    "VerifierRuntimeStatus",
    "VerifierTaskType",
    "VerifierTraceInput",
    "build_deepseek_verifier",
]
