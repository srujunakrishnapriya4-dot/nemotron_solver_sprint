from __future__ import annotations

from dataclasses import replace

import pytest

from nemotron_engine.core.schemas import FormatReport, LeakageReport
from nemotron_engine.parsing import canonicalize_prompt
from nemotron_engine.solvers import solve_numeric
from nemotron_engine.traces.trace_compiler import (
    TraceCompilationError,
    TraceRecord,
    TraceType,
    compile_rejection_trace,
    compile_trace_from_proof,
    compile_trace_from_solver_result,
    make_trace_id,
)


def safe_proof():
    result = solve_numeric(canonicalize_prompt("p", "1 -> 2\n2 -> 3\n4 -> ?"))
    assert result.best_attempt is not None
    assert result.best_attempt.proof is not None
    return result.best_attempt.proof


def solver_result():
    return solve_numeric(canonicalize_prompt("p", "1 -> 2\n2 -> 3\n4 -> ?"))


def trace_id_for(
    *,
    problem_id: str = "p",
    trace_type: TraceType = TraceType.MINIMAL_KNOWN_RULE,
    source_proof_hash: str | None = "proof",
    source_program_hash: str | None = "program",
    completion: str,
    metadata: dict | None = None,
) -> str:
    return make_trace_id(
        problem_id=problem_id,
        trace_type=trace_type,
        source_proof_hash=source_proof_hash,
        source_program_hash=source_program_hash,
        completion=completion,
        metadata=metadata or {},
    )


def test_compile_positive_trace_from_training_safe_proof() -> None:
    proof = safe_proof()

    trace = compile_trace_from_proof(proof, solver_name="known_numeric")

    assert trace.is_positive
    assert trace.boxed_answer == proof.target_execution.output_value
    assert trace.source_program_hash == proof.locked_program.program_hash


def test_trace_record_rejects_positive_trace_with_no_box() -> None:
    proof = safe_proof()
    completion = "Apply to target.\nanswer"

    with pytest.raises(TraceCompilationError):
        TraceRecord(
            trace_id_for(completion=completion, source_program_hash=proof.locked_program.program_hash),
            "p",
            TraceType.MINIMAL_KNOWN_RULE,
            None,
            completion,
            "answer",
            "proof",
            proof.locked_program.program_hash,
            "solver",
            "answer",
            True,
        )


def test_trace_record_rejects_positive_box_different_from_target_output() -> None:
    completion = r"Apply to target.\n\boxed{5}"
    with pytest.raises(TraceCompilationError):
        TraceRecord(
            trace_id_for(completion=completion),
            "p",
            TraceType.MINIMAL_KNOWN_RULE,
            None,
            completion,
            "5",
            "proof",
            "program",
            "solver",
            "6",
            True,
        )


def test_trace_record_rejects_hard_rejection_with_boxed_answer() -> None:
    completion = r"Reason: bad\n\boxed{5}"
    with pytest.raises(TraceCompilationError):
        TraceRecord(
            trace_id_for(
                trace_type=TraceType.HARD_REJECTION,
                source_proof_hash=None,
                source_program_hash=None,
                completion=completion,
                metadata={"reason": "bad"},
            ),
            "p",
            TraceType.HARD_REJECTION,
            None,
            completion,
            "5",
            None,
            None,
            "solver",
            None,
            False,
            {"reason": "bad"},
        )


def test_trace_record_rejects_tampered_trace_id() -> None:
    completion = r"Apply to target.\n\boxed{5}"

    with pytest.raises(TraceCompilationError):
        TraceRecord(
            "trace-forged",
            "p",
            TraceType.MINIMAL_KNOWN_RULE,
            None,
            completion,
            "5",
            "proof",
            "program",
            "solver",
            "5",
            True,
        )


def test_trace_record_rejects_non_positive_non_hard_rejection_type() -> None:
    completion = "Reason: rejected"

    with pytest.raises(TraceCompilationError):
        TraceRecord(
            trace_id_for(trace_type=TraceType.MINIMAL_KNOWN_RULE, completion=completion),
            "p",
            TraceType.MINIMAL_KNOWN_RULE,
            None,
            completion,
            None,
            "proof",
            "program",
            "solver",
            None,
            False,
        )


def test_trace_record_rejects_non_positive_non_rejection_with_box() -> None:
    completion = r"Reason: rejected\n\boxed{5}"

    with pytest.raises(TraceCompilationError):
        TraceRecord(
            trace_id_for(trace_type=TraceType.MINIMAL_KNOWN_RULE, completion=completion),
            "p",
            TraceType.MINIMAL_KNOWN_RULE,
            None,
            completion,
            None,
            "proof",
            "program",
            "solver",
            None,
            False,
        )


def test_trace_record_rejects_tampered_trace_hash() -> None:
    trace = compile_rejection_trace("p", "bad")

    with pytest.raises(TraceCompilationError):
        replace(trace, trace_hash="tampered")


def test_compile_trace_from_proof_rejects_empty_target_output() -> None:
    proof = safe_proof()
    bad_target = replace(proof.target_execution, output_value="")
    bad = replace(proof, target_execution=bad_target)

    with pytest.raises(TraceCompilationError):
        compile_trace_from_proof(bad)


def test_compile_trace_from_proof_rejects_unsafe_proof() -> None:
    proof = safe_proof()
    unsafe = replace(proof, shadow_verifier_pass=False)

    with pytest.raises(TraceCompilationError):
        compile_trace_from_proof(unsafe)


def test_compile_trace_from_proof_rejects_leaky_ambiguous_invalid_format() -> None:
    proof = safe_proof()

    with pytest.raises(TraceCompilationError):
        compile_trace_from_proof(replace(proof, leakage_report=LeakageReport(True, ("x",), ("x",))))
    with pytest.raises(TraceCompilationError):
        compile_trace_from_proof(replace(proof, ambiguity_report=replace(proof.ambiguity_report, unique_target_output=False, ambiguous=True)))
    with pytest.raises(TraceCompilationError):
        compile_trace_from_proof(replace(proof, format_report=FormatReport(False, errors=("bad",))))


def test_compile_from_solver_result_best_attempt() -> None:
    result = solver_result()

    trace = compile_trace_from_solver_result(result)

    assert trace.is_positive
    assert trace.boxed_answer == result.best_attempt.proof.target_execution.output_value


def test_solver_result_without_safe_best_attempt_returns_rejection() -> None:
    result = solve_numeric(canonicalize_prompt("amb", "1 -> 2\n1 -> 3\n4 -> ?"))

    trace = compile_trace_from_solver_result(result)

    assert not trace.is_positive
    assert trace.boxed_answer is None
    assert r"\boxed" not in trace.completion


def test_metadata_order_does_not_change_trace_id() -> None:
    proof = safe_proof()

    first = compile_trace_from_proof(proof, metadata={"a": 1, "b": 2})
    second = compile_trace_from_proof(proof, metadata={"b": 2, "a": 1})

    assert first.trace_id == second.trace_id
    assert first.trace_hash == second.trace_hash
