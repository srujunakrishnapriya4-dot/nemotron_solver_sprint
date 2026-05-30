"""Compile verified proof artifacts into auditable Pass 5 traces."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
import re
from typing import Any, Mapping

from nemotron_engine.core.schemas import ExecutableProof, VerificationStatus, stable_hash
from nemotron_engine.scoring.answer_extractor import AnswerExtractionError, extract_boxed_answer, find_boxed_spans
from nemotron_engine.solvers.base import SolverResult

from .templates import (
    render_direct_answer_trace,
    render_hard_rejection_trace,
    render_minimal_induced_rule_trace,
    render_minimal_known_rule_trace,
)


class TraceCompilationError(ValueError):
    """Raised when a trace cannot be compiled safely."""


class TraceType(str, Enum):
    DIRECT_ANSWER = "direct_answer"
    MINIMAL_KNOWN_RULE = "minimal_known_rule"
    MINIMAL_INDUCED_RULE = "minimal_induced_rule"
    HARD_REJECTION = "hard_rejection"


@dataclass(frozen=True)
class TraceRecord:
    trace_id: str
    problem_id: str
    trace_type: TraceType
    prompt: str | None
    completion: str
    boxed_answer: str | None
    source_proof_hash: str | None
    source_program_hash: str | None
    solver_name: str | None
    target_output: str | None
    is_positive: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    trace_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.trace_id, str) or not self.trace_id.strip():
            raise TraceCompilationError("trace_id must be non-empty.")
        if not isinstance(self.problem_id, str) or not self.problem_id.strip():
            raise TraceCompilationError("problem_id must be non-empty.")
        if not isinstance(self.completion, str) or not self.completion.strip():
            raise TraceCompilationError("completion must be non-empty.")
        object.__setattr__(self, "trace_type", _coerce_trace_type(self.trace_type))
        object.__setattr__(self, "metadata", dict(self.metadata))
        expected_id = make_trace_id(
            problem_id=self.problem_id,
            trace_type=self.trace_type,
            source_proof_hash=self.source_proof_hash,
            source_program_hash=self.source_program_hash,
            completion=self.completion,
            metadata=self.metadata,
        )
        if self.trace_id != expected_id:
            raise TraceCompilationError("trace_id does not match deterministic trace payload.")
        expected_hash = _trace_payload_hash(self)
        if not self.trace_hash:
            object.__setattr__(self, "trace_hash", expected_hash)
        elif self.trace_hash != expected_hash:
            raise TraceCompilationError("trace_hash does not match trace payload.")

        spans = find_boxed_spans(self.completion)
        if self.is_positive:
            if self.trace_type is TraceType.HARD_REJECTION:
                raise TraceCompilationError("positive traces cannot use hard_rejection trace_type.")
            if not self.boxed_answer or not str(self.boxed_answer).strip():
                raise TraceCompilationError("positive traces require boxed_answer.")
            if not self.target_output or not str(self.target_output).strip():
                raise TraceCompilationError("positive traces require target_output.")
            if len(spans) != 1:
                raise TraceCompilationError("positive traces require exactly one boxed answer.")
            extracted = extract_boxed_answer(self.completion).raw
            if extracted != self.boxed_answer:
                raise TraceCompilationError("completion boxed answer must equal boxed_answer.")
            if self.target_output is not None and self.boxed_answer != self.target_output:
                raise TraceCompilationError("boxed_answer must equal target_output.")
        else:
            if self.trace_type is not TraceType.HARD_REJECTION:
                raise TraceCompilationError("non-positive traces must use hard_rejection trace_type.")
            if self.boxed_answer is not None:
                raise TraceCompilationError("hard rejection traces must not set boxed_answer.")
            if spans:
                raise TraceCompilationError("hard rejection traces must not contain boxed answers.")
            if not _has_rejection_reason(self.completion, self.metadata):
                raise TraceCompilationError("hard rejection traces require a non-empty reason.")


@dataclass(frozen=True)
class TraceAuditReport:
    trace_id: str
    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "errors", tuple(str(item) for item in self.errors))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(self, "metrics", dict(self.metrics))


def compile_trace_from_proof(
    proof: ExecutableProof,
    *,
    trace_type: TraceType | None = None,
    solver_name: str | None = None,
    prompt: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> TraceRecord:
    _require_safe_proof(proof)
    answer = _target_output(proof)
    selected = _infer_trace_type(proof, trace_type, solver_name)
    meta = dict(metadata or {})
    meta.setdefault("example_count", len(proof.example_executions))
    meta.setdefault("program_id", proof.locked_program.program_id)
    completion = _render_positive(selected, proof, answer)
    return _make_trace_record(
        problem_id=proof.problem_id,
        trace_type=selected,
        prompt=prompt,
        completion=completion,
        boxed_answer=answer,
        source_proof_hash=stable_hash(proof),
        source_program_hash=proof.locked_program.program_hash,
        solver_name=solver_name,
        target_output=answer,
        is_positive=True,
        metadata=meta,
    )


def compile_trace_from_solver_result(
    result: SolverResult,
    *,
    prompt: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> TraceRecord:
    if result.best_attempt is not None and result.best_attempt.proof is not None and result.best_attempt.proof.is_training_safe:
        meta = dict(metadata or {})
        meta.setdefault("solver_result_attempts", len(result.attempts))
        return compile_trace_from_proof(
            result.best_attempt.proof,
            solver_name=result.solver_name,
            prompt=prompt,
            metadata=meta,
        )
    reason = _solver_rejection_reason(result)
    return compile_rejection_trace(
        result.problem_id,
        reason,
        solver_name=result.solver_name,
        prompt=prompt,
        metadata={**dict(metadata or {}), "reason": reason},
    )


def compile_rejection_trace(
    problem_id: str,
    reason: str,
    *,
    solver_name: str | None = None,
    prompt: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> TraceRecord:
    if not isinstance(reason, str) or not reason.strip():
        raise TraceCompilationError("rejection reason must be non-empty.")
    meta = dict(metadata or {})
    meta["reason"] = reason.strip()
    completion = render_hard_rejection_trace(reason=reason)
    return _make_trace_record(
        problem_id=problem_id,
        trace_type=TraceType.HARD_REJECTION,
        prompt=prompt,
        completion=completion,
        boxed_answer=None,
        source_proof_hash=None,
        source_program_hash=None,
        solver_name=solver_name,
        target_output=None,
        is_positive=False,
        metadata=meta,
    )


def make_trace_id(
    *,
    problem_id: str,
    trace_type: TraceType,
    source_proof_hash: str | None,
    source_program_hash: str | None,
    completion: str,
    metadata: Mapping[str, Any],
) -> str:
    payload = {
        "problem_id": problem_id,
        "trace_type": trace_type.value,
        "source_proof_hash": source_proof_hash,
        "source_program_hash": source_program_hash,
        "completion": completion,
        "metadata": dict(metadata),
    }
    return f"trace-{stable_hash(payload)[:24]}"


def _make_trace_record(
    *,
    problem_id: str,
    trace_type: TraceType,
    prompt: str | None,
    completion: str,
    boxed_answer: str | None,
    source_proof_hash: str | None,
    source_program_hash: str | None,
    solver_name: str | None,
    target_output: str | None,
    is_positive: bool,
    metadata: dict[str, Any],
) -> TraceRecord:
    trace_id = make_trace_id(
        problem_id=problem_id,
        trace_type=trace_type,
        source_proof_hash=source_proof_hash,
        source_program_hash=source_program_hash,
        completion=completion,
        metadata=metadata,
    )
    initial = TraceRecord(
        trace_id=trace_id,
        problem_id=problem_id,
        trace_type=trace_type,
        prompt=prompt,
        completion=completion,
        boxed_answer=boxed_answer,
        source_proof_hash=source_proof_hash,
        source_program_hash=source_program_hash,
        solver_name=solver_name,
        target_output=target_output,
        is_positive=is_positive,
        metadata=metadata,
    )
    return initial


def _render_positive(trace_type: TraceType, proof: ExecutableProof, answer: str) -> str:
    rule = _program_summary(proof)
    checks = f"{len(proof.example_executions)} examples passed"
    target = "program output computed"
    if trace_type is TraceType.DIRECT_ANSWER:
        return render_direct_answer_trace(answer=answer)
    if trace_type is TraceType.MINIMAL_KNOWN_RULE:
        return render_minimal_known_rule_trace(rule_summary=rule, example_checks=checks, target_application=target, answer=answer)
    if trace_type is TraceType.MINIMAL_INDUCED_RULE:
        return render_minimal_induced_rule_trace(rule_summary=rule, example_checks=checks, target_application=target, answer=answer)
    raise TraceCompilationError("positive proof cannot compile as hard_rejection.")


def _program_summary(proof: ExecutableProof) -> str:
    primitives = [step.primitive for step in proof.locked_program.steps]
    return " -> ".join(primitives) or proof.locked_program.program_id


def _require_safe_proof(proof: ExecutableProof) -> None:
    if not isinstance(proof, ExecutableProof):
        raise TraceCompilationError("proof must be an ExecutableProof.")
    if proof.is_training_safe is not True:
        raise TraceCompilationError("positive trace requires training-safe proof.")
    _target_output(proof)


def _target_output(proof: ExecutableProof) -> str:
    if proof.target_execution is None or proof.target_execution.output_value is None:
        raise TraceCompilationError("proof target execution must have output.")
    answer = str(proof.target_execution.output_value)
    if not answer:
        raise TraceCompilationError("proof target output must be non-empty.")
    return answer


def _infer_trace_type(proof: ExecutableProof, trace_type: TraceType | None, solver_name: str | None) -> TraceType:
    if trace_type is not None:
        selected = _coerce_trace_type(trace_type)
        if selected is TraceType.HARD_REJECTION:
            raise TraceCompilationError("training-safe proof cannot compile as hard_rejection.")
        return selected
    primitives = [step.primitive for step in proof.locked_program.steps]
    if primitives in (["raw"], ["identity", "raw"]):
        return TraceType.DIRECT_ANSWER
    name = (solver_name or "").lower()
    if not name or "synth" in name or name == "universal_synthesizer":
        return TraceType.MINIMAL_INDUCED_RULE
    return TraceType.MINIMAL_KNOWN_RULE


def _solver_rejection_reason(result: SolverResult) -> str:
    if result.ambiguity_report.ambiguous:
        return result.ambiguity_report.reason or "ambiguous_or_no_safe_solution"
    for attempt in result.rejected_attempts:
        if attempt.reason:
            return attempt.reason
    return "no_training_safe_best_attempt"


def _trace_payload_hash(trace: TraceRecord) -> str:
    return stable_hash({item.name: getattr(trace, item.name) for item in fields(TraceRecord) if item.name != "trace_hash"})


def _has_rejection_reason(completion: str, metadata: Mapping[str, Any]) -> bool:
    reason = metadata.get("reason")
    if isinstance(reason, str) and reason.strip():
        return True
    match = re.search(r"^Reason:\s*(.+)$", completion, flags=re.I | re.M)
    return bool(match and match.group(1).strip())


def _coerce_trace_type(value: TraceType | str) -> TraceType:
    if isinstance(value, TraceType):
        return value
    try:
        return TraceType(str(value))
    except ValueError as exc:
        raise TraceCompilationError(f"invalid trace_type: {value!r}") from exc


__all__ = [
    "TraceAuditReport",
    "TraceCompilationError",
    "TraceRecord",
    "TraceType",
    "compile_rejection_trace",
    "compile_trace_from_proof",
    "compile_trace_from_solver_result",
    "make_trace_id",
]
