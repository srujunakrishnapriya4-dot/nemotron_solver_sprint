from __future__ import annotations


def test_pass5_imports_work() -> None:
    from nemotron_engine.traces import (
        TraceAuditReport,
        TraceRecord,
        audit_trace,
        compile_trace_from_proof,
        dedupe_traces,
    )

    assert TraceAuditReport
    assert TraceRecord
    assert audit_trace
    assert compile_trace_from_proof
    assert dedupe_traces


def test_representative_locked_imports_still_work() -> None:
    from nemotron_engine.core import ExecutableProof
    from nemotron_engine.data import detect_cross_split_contamination
    from nemotron_engine.programs import execute_program
    from nemotron_engine.scoring import extract_boxed_answer
    from nemotron_engine.solvers import solve_numeric

    assert ExecutableProof
    assert detect_cross_split_contamination
    assert execute_program
    assert extract_boxed_answer
    assert solve_numeric

