from __future__ import annotations

from dataclasses import replace

from nemotron_engine.parsing import canonicalize_prompt
from nemotron_engine.solvers import solve_numeric
import pytest

from nemotron_engine.traces.dedupe import TraceDedupeReport, compute_trace_hash, dedupe_traces
from nemotron_engine.traces.trace_compiler import TraceRecord, compile_rejection_trace, compile_trace_from_proof, make_trace_id


def trace():
    result = solve_numeric(canonicalize_prompt("p", "1 -> 2\n2 -> 3\n4 -> ?"))
    return compile_trace_from_proof(result.best_attempt.proof)


def rebuild_trace(base: TraceRecord, **changes) -> TraceRecord:
    payload = {
        "problem_id": base.problem_id,
        "trace_type": base.trace_type,
        "prompt": base.prompt,
        "completion": base.completion,
        "boxed_answer": base.boxed_answer,
        "source_proof_hash": base.source_proof_hash,
        "source_program_hash": base.source_program_hash,
        "solver_name": base.solver_name,
        "target_output": base.target_output,
        "is_positive": base.is_positive,
        "metadata": base.metadata,
    }
    payload.update(changes)
    payload["trace_id"] = make_trace_id(
        problem_id=payload["problem_id"],
        trace_type=payload["trace_type"],
        source_proof_hash=payload["source_proof_hash"],
        source_program_hash=payload["source_program_hash"],
        completion=payload["completion"],
        metadata=payload["metadata"],
    )
    return TraceRecord(**payload)


def test_duplicate_trace_hash_removed() -> None:
    first = trace()
    deduped, report = dedupe_traces([first, first])

    assert deduped == (first,)
    assert report.removed_trace_ids == (first.trace_id,)


def test_same_problem_answer_type_duplicate_removed_without_dropping_first() -> None:
    first = trace()
    second = rebuild_trace(first, metadata={"variant": 2})

    deduped, report = dedupe_traces([first, second])

    assert deduped[0] == first
    assert second.trace_id in report.removed_trace_ids


def test_conflicting_positive_traces_report_both_trace_ids() -> None:
    first = trace()
    second = rebuild_trace(first, boxed_answer="999", target_output="999", completion="Apply to target.\n\\boxed{999}")

    deduped, report = dedupe_traces([first, second])

    assert len(deduped) == 2
    assert report.conflicts == ((first.trace_id, second.trace_id),)


def test_compute_trace_hash_matches_record_hash() -> None:
    first = compile_rejection_trace("p", "bad")

    assert compute_trace_hash(first) == first.trace_hash


def test_dedupe_report_rejects_forged_inconsistent_counts() -> None:
    with pytest.raises(ValueError):
        TraceDedupeReport(input_count=-1, output_count=0)
    with pytest.raises(ValueError):
        TraceDedupeReport(input_count=1, output_count=2)
    with pytest.raises(ValueError):
        TraceDedupeReport(input_count=1, output_count=0, removed_trace_ids=("a", "b"))
