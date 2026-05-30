"""Trace-level negative generation for Pass 6 DPO builders."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Sequence

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.traces.trace_auditor import audit_trace
from nemotron_engine.traces.trace_compiler import TraceRecord, TraceType, compile_rejection_trace, make_trace_id


ALLOWED_NEGATIVE_TYPES = (
    "false_rule",
    "fake_verification",
    "decoded_vs_encoded",
    "format_box",
    "target_leak",
    "overlong_reasoning",
    "unnecessary_ambiguity",
)


class NegativeGenerationError(ValueError):
    """Raised when a trace-level negative cannot be generated safely."""


@dataclass(frozen=True)
class NegativeTraceCandidate:
    negative_id: str
    problem_id: str
    source_trace_id: str
    negative_type: str
    completion: str
    trace: TraceRecord
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    negative_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("negative_id", "problem_id", "source_trace_id", "negative_type", "completion", "reason"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise NegativeGenerationError(f"{name} must be a non-empty string.")
        if self.negative_type not in ALLOWED_NEGATIVE_TYPES:
            raise NegativeGenerationError("unsupported_negative_type")
        if not isinstance(self.trace, TraceRecord):
            raise NegativeGenerationError("trace must be a TraceRecord.")
        if self.trace.problem_id != self.problem_id:
            raise NegativeGenerationError("trace problem_id mismatch.")
        if self.trace.completion != self.completion:
            raise NegativeGenerationError("candidate completion must equal trace completion.")
        object.__setattr__(self, "metadata", dict(self.metadata))
        expected_id = make_negative_id(
            problem_id=self.problem_id,
            source_trace_id=self.source_trace_id,
            negative_type=self.negative_type,
            completion=self.completion,
            metadata=self.metadata,
        )
        if self.negative_id != expected_id:
            raise NegativeGenerationError("negative_id does not match deterministic payload.")
        expected_hash = _negative_payload_hash(self)
        if not self.negative_hash:
            object.__setattr__(self, "negative_hash", expected_hash)
        elif self.negative_hash != expected_hash:
            raise NegativeGenerationError("negative_hash does not match payload.")


@dataclass(frozen=True)
class NegativeGenerationReport:
    input_count: int
    generated_count: int
    skipped_count: int
    skipped_reasons: Mapping[str, int] = field(default_factory=dict)
    negative_type_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if min(self.input_count, self.generated_count, self.skipped_count) < 0:
            raise NegativeGenerationError("negative generation counts must be non-negative.")
        object.__setattr__(self, "skipped_reasons", dict(self.skipped_reasons))
        object.__setattr__(self, "negative_type_counts", dict(self.negative_type_counts))
        if sum(self.skipped_reasons.values()) != self.skipped_count:
            raise NegativeGenerationError("skipped reason totals must match skipped_count.")
        if sum(self.negative_type_counts.values()) != self.generated_count:
            raise NegativeGenerationError("negative type totals must match generated_count.")


def generate_negative_for_trace(trace: TraceRecord, negative_type: str) -> NegativeTraceCandidate:
    _require_source_trace(trace)
    if negative_type not in ALLOWED_NEGATIVE_TYPES:
        raise NegativeGenerationError("unsupported_negative_type")
    metadata = {
        "negative_type": negative_type,
        "is_negative": True,
        "source_trace_id": trace.trace_id,
    }
    reason = _negative_reason(negative_type)
    if negative_type in {"false_rule", "decoded_vs_encoded", "unnecessary_ambiguity"}:
        negative_trace = compile_rejection_trace(
            trace.problem_id,
            reason,
            solver_name=trace.solver_name,
            prompt=trace.prompt,
            metadata=metadata,
        )
    else:
        completion = _positive_shaped_negative_completion(trace, negative_type)
        negative_trace = _make_positive_shaped_negative(trace, completion, metadata)
    if negative_trace.completion == trace.completion:
        raise NegativeGenerationError("negative completion must differ from source completion.")
    negative_id = make_negative_id(
        problem_id=trace.problem_id,
        source_trace_id=trace.trace_id,
        negative_type=negative_type,
        completion=negative_trace.completion,
        metadata=metadata,
    )
    return NegativeTraceCandidate(
        negative_id=negative_id,
        problem_id=trace.problem_id,
        source_trace_id=trace.trace_id,
        negative_type=negative_type,
        completion=negative_trace.completion,
        trace=negative_trace,
        reason=reason,
        metadata=metadata,
    )


def generate_negative_traces(
    positive_traces: Sequence[TraceRecord],
    *,
    negative_types: Sequence[str] | None = None,
    max_negatives_per_trace: int = 4,
) -> tuple[tuple[NegativeTraceCandidate, ...], NegativeGenerationReport]:
    if not isinstance(max_negatives_per_trace, int) or isinstance(max_negatives_per_trace, bool) or max_negatives_per_trace < 0:
        raise NegativeGenerationError("max_negatives_per_trace must be a non-negative integer.")
    selected = tuple(negative_types) if negative_types is not None else ALLOWED_NEGATIVE_TYPES
    for item in selected:
        if item not in ALLOWED_NEGATIVE_TYPES:
            raise NegativeGenerationError("unsupported_negative_type")
    generated: list[NegativeTraceCandidate] = []
    skipped: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    materialized = tuple(positive_traces)
    for trace in materialized:
        try:
            _require_source_trace(trace)
        except NegativeGenerationError as exc:
            skipped[str(exc)] = skipped.get(str(exc), 0) + 1
            continue
        for index, negative_type in enumerate(selected):
            if index >= max_negatives_per_trace:
                skipped["max_negatives_per_trace"] = skipped.get("max_negatives_per_trace", 0) + 1
                continue
            candidate = generate_negative_for_trace(trace, negative_type)
            generated.append(candidate)
            type_counts[negative_type] = type_counts.get(negative_type, 0) + 1
    report = NegativeGenerationReport(
        input_count=len(materialized),
        generated_count=len(generated),
        skipped_count=sum(skipped.values()),
        skipped_reasons=skipped,
        negative_type_counts=type_counts,
    )
    return tuple(generated), report


def make_negative_id(
    *,
    problem_id: str,
    source_trace_id: str,
    negative_type: str,
    completion: str,
    metadata: Mapping[str, Any],
) -> str:
    return "neg-" + stable_hash(
        {
            "problem_id": problem_id,
            "source_trace_id": source_trace_id,
            "negative_type": negative_type,
            "completion": completion,
            "metadata": dict(metadata),
        }
    )[:24]


def _require_source_trace(trace: TraceRecord) -> None:
    if not isinstance(trace, TraceRecord):
        raise NegativeGenerationError("trace must be a TraceRecord.")
    if trace.is_positive is not True:
        raise NegativeGenerationError("source_trace_not_positive")
    if trace.metadata.get("negative_type") or trace.metadata.get("is_negative") is True:
        raise NegativeGenerationError("source_trace_marked_negative")
    if not trace.source_proof_hash or not trace.source_program_hash:
        raise NegativeGenerationError("source_trace_missing_verified_provenance")
    if not trace.boxed_answer or not trace.target_output:
        raise NegativeGenerationError("source_trace_missing_answer")
    if trace.boxed_answer != trace.target_output:
        raise NegativeGenerationError("source_trace_answer_mismatch")
    if audit_trace(trace).valid is not True:
        raise NegativeGenerationError("source_trace_audit_invalid")


def _make_positive_shaped_negative(trace: TraceRecord, completion: str, metadata: Mapping[str, Any]) -> TraceRecord:
    trace_id = make_trace_id(
        problem_id=trace.problem_id,
        trace_type=TraceType.MINIMAL_INDUCED_RULE,
        source_proof_hash=trace.source_proof_hash,
        source_program_hash=trace.source_program_hash,
        completion=completion,
        metadata=metadata,
    )
    return TraceRecord(
        trace_id=trace_id,
        problem_id=trace.problem_id,
        trace_type=TraceType.MINIMAL_INDUCED_RULE,
        prompt=trace.prompt,
        completion=completion,
        boxed_answer=trace.boxed_answer,
        source_proof_hash=trace.source_proof_hash,
        source_program_hash=trace.source_program_hash,
        solver_name=trace.solver_name,
        target_output=trace.target_output,
        is_positive=True,
        metadata=metadata,
    )


def _positive_shaped_negative_completion(trace: TraceRecord, negative_type: str) -> str:
    answer = str(trace.boxed_answer)
    if negative_type == "target_leak":
        return "\n".join(("Target output appears first: " + answer, "Apply to target.", rf"\boxed{{{answer}}}"))
    if negative_type == "format_box":
        return "\n".join(("Final answer without application marker.", rf"\boxed{{{answer}}}"))
    if negative_type == "fake_verification":
        return "\n".join(("Verified on examples: fabricated checks.", "Apply to target.", rf"\boxed{{{answer}}}"))
    if negative_type == "overlong_reasoning":
        repeated = ["boilerplate verification step"] * 40
        return "\n".join((*repeated, "Apply to target.", rf"\boxed{{{answer}}}"))
    raise NegativeGenerationError("unsupported positive-shaped negative type")


def _negative_reason(negative_type: str) -> str:
    return {
        "false_rule": "false_rule: candidate rule disagrees with verified trace.",
        "fake_verification": "fake_verification: completion claims unsupported verification.",
        "decoded_vs_encoded": "decoded_vs_encoded: answer encoding policy is wrong.",
        "format_box": "format_box: final answer format/application marker is invalid.",
        "target_leak": "target_leak: answer appears before target application.",
        "overlong_reasoning": "overlong_reasoning: completion is intentionally too long.",
        "unnecessary_ambiguity": "unnecessary_ambiguity: rejects a verified unique answer.",
    }[negative_type]


def _negative_payload_hash(candidate: NegativeTraceCandidate) -> str:
    return stable_hash({item.name: getattr(candidate, item.name) for item in fields(NegativeTraceCandidate) if item.name != "negative_hash"})


__all__ = [
    "ALLOWED_NEGATIVE_TYPES",
    "NegativeGenerationError",
    "NegativeGenerationReport",
    "NegativeTraceCandidate",
    "generate_negative_for_trace",
    "generate_negative_traces",
    "make_negative_id",
]
