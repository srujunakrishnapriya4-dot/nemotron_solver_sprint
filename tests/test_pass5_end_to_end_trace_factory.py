from __future__ import annotations

from nemotron_engine.parsing import canonicalize_prompt
from nemotron_engine.solvers import solve_numeric
from nemotron_engine.solvers.known_binary import solve as solve_binary
from nemotron_engine.traces import audit_trace, compile_trace_from_solver_result, dedupe_traces


def test_end_to_end_trace_factory_from_pass3_solver() -> None:
    problem = canonicalize_prompt("p", "1 -> 2\n2 -> 3\n4 -> ?")
    result = solve_numeric(problem)

    trace = compile_trace_from_solver_result(result, prompt=problem.raw_prompt)
    report = audit_trace(trace, proof=result.best_attempt.proof)
    deduped, dedupe_report = dedupe_traces([trace])

    assert report.valid
    assert deduped == (trace,)
    assert dedupe_report.conflicts == ()


def test_ambiguous_solver_result_produces_rejection_trace_with_no_box() -> None:
    problem = canonicalize_prompt("amb", "00 -> 00\n01 -> ?")
    result = solve_binary(problem)

    trace = compile_trace_from_solver_result(result, prompt=problem.raw_prompt)
    report = audit_trace(trace)

    assert not trace.is_positive
    assert trace.boxed_answer is None
    assert "\\boxed" not in trace.completion
    assert report.valid

