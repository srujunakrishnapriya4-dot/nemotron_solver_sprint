"""Deterministic dedupe for Pass 5 trace records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from nemotron_engine.core.schemas import stable_hash

from .trace_compiler import TraceRecord


@dataclass(frozen=True)
class TraceDedupeReport:
    input_count: int
    output_count: int
    removed_trace_ids: tuple[str, ...] = ()
    duplicate_keys: tuple[str, ...] = ()
    conflicts: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.input_count < 0 or self.output_count < 0:
            raise ValueError("dedupe counts must be non-negative.")
        if self.output_count > self.input_count:
            raise ValueError("output_count cannot exceed input_count.")
        if len(self.removed_trace_ids) > self.input_count:
            raise ValueError("removed_trace_ids cannot exceed input_count.")
        object.__setattr__(self, "removed_trace_ids", tuple(str(item) for item in self.removed_trace_ids))
        object.__setattr__(self, "duplicate_keys", tuple(str(item) for item in self.duplicate_keys))
        object.__setattr__(
            self,
            "conflicts",
            tuple((str(left), str(right)) for left, right in self.conflicts),
        )


def compute_trace_hash(trace: TraceRecord) -> str:
    return stable_hash(
        {
            "trace_id": trace.trace_id,
            "problem_id": trace.problem_id,
            "trace_type": trace.trace_type,
            "prompt": trace.prompt,
            "completion": trace.completion,
            "boxed_answer": trace.boxed_answer,
            "source_proof_hash": trace.source_proof_hash,
            "source_program_hash": trace.source_program_hash,
            "solver_name": trace.solver_name,
            "target_output": trace.target_output,
            "is_positive": trace.is_positive,
            "metadata": trace.metadata,
        }
    )


def dedupe_traces(traces: Sequence[TraceRecord]) -> tuple[tuple[TraceRecord, ...], TraceDedupeReport]:
    output: list[TraceRecord] = []
    seen_hashes: set[str] = set()
    seen_semantic: dict[tuple[str, str | None, str], TraceRecord] = {}
    positive_by_problem: dict[str, TraceRecord] = {}
    removed: list[str] = []
    duplicate_keys: list[str] = []
    conflicts: list[tuple[str, str]] = []

    for trace in traces:
        semantic_key = (trace.problem_id, trace.boxed_answer, trace.trace_type.value)
        if trace.trace_hash in seen_hashes:
            removed.append(trace.trace_id)
            duplicate_keys.append(f"hash:{trace.trace_hash}")
            continue
        if semantic_key in seen_semantic:
            removed.append(trace.trace_id)
            duplicate_keys.append(f"semantic:{trace.problem_id}:{trace.trace_type.value}:{trace.boxed_answer}")
            continue
        if trace.is_positive and trace.boxed_answer is not None:
            prior = positive_by_problem.get(trace.problem_id)
            if prior is not None and prior.boxed_answer != trace.boxed_answer:
                conflicts.append((prior.trace_id, trace.trace_id))
            else:
                positive_by_problem[trace.problem_id] = trace
        seen_hashes.add(trace.trace_hash)
        seen_semantic[semantic_key] = trace
        output.append(trace)

    report = TraceDedupeReport(
        input_count=len(traces),
        output_count=len(output),
        removed_trace_ids=tuple(removed),
        duplicate_keys=tuple(duplicate_keys),
        conflicts=tuple(conflicts),
    )
    return tuple(output), report


__all__ = ["TraceDedupeReport", "compute_trace_hash", "dedupe_traces"]
