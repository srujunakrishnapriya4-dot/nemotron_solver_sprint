from __future__ import annotations

from dataclasses import replace

import pytest

from nemotron_engine.core.schemas import AmbiguityReport, FormatReport
from nemotron_engine.parsing import canonicalize_prompt
from nemotron_engine.solvers import solve_numeric
from nemotron_engine.traces.trace_auditor import TraceAuditError, assert_trace_valid, audit_trace
from nemotron_engine.traces.trace_compiler import TraceRecord, TraceType, compile_rejection_trace, compile_trace_from_proof, make_trace_id


def safe_proof():
    result = solve_numeric(canonicalize_prompt("p", "1 -> 2\n2 -> 3\n4 -> ?"))
    assert result.best_attempt is not None
    assert result.best_attempt.proof is not None
    return result.best_attempt.proof


def rebuild_trace(trace: TraceRecord, **changes) -> TraceRecord:
    payload = {
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


def test_valid_compiled_trace_passes_audit() -> None:
    proof = safe_proof()
    trace = compile_trace_from_proof(proof, solver_name="known_numeric")

    report = audit_trace(trace, proof=proof)

    assert report.valid


def test_forged_non_positive_non_hard_trace_with_box_rejected_by_schema() -> None:
    completion = r"Reason: bad\n\boxed{5}"

    with pytest.raises(Exception):
        TraceRecord(
            make_trace_id(
                problem_id="p",
                trace_type=TraceType.MINIMAL_KNOWN_RULE,
                source_proof_hash="proof",
                source_program_hash="program",
                completion=completion,
                metadata={},
            ),
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


def test_positive_trace_with_verified_language_without_proof_warns() -> None:
    proof = safe_proof()
    trace = compile_trace_from_proof(proof, solver_name="known_numeric")

    report = audit_trace(trace)

    assert report.valid
    assert "verification_claim_without_proof" in report.warnings


def test_multiple_boxes_rejected() -> None:
    proof = safe_proof()

    with pytest.raises(Exception):
        replace(compile_trace_from_proof(proof), completion="Apply to target.\n\\boxed{5}\n\\boxed{5}", trace_hash="")


def test_no_box_in_positive_trace_rejected_by_construction() -> None:
    proof = safe_proof()

    with pytest.raises(Exception):
        replace(compile_trace_from_proof(proof), completion="Apply to target.\n5", trace_hash="")


def test_box_mismatch_rejected() -> None:
    proof = safe_proof()

    with pytest.raises(Exception):
        replace(compile_trace_from_proof(proof), completion="Apply to target.\n\\boxed{999}", boxed_answer="999", trace_hash="")


def test_source_hash_mismatch_rejected() -> None:
    proof = safe_proof()
    trace = rebuild_trace(compile_trace_from_proof(proof), source_proof_hash="bad")

    report = audit_trace(trace, proof=proof)

    assert "source_proof_hash_mismatch" in report.errors


def test_program_hash_mismatch_rejected() -> None:
    proof = safe_proof()
    trace = rebuild_trace(compile_trace_from_proof(proof), source_program_hash="bad")

    report = audit_trace(trace, proof=proof)

    assert "source_program_hash_mismatch" in report.errors


def test_positive_trace_with_answer_before_apply_marker_rejected() -> None:
    proof = safe_proof()
    answer = proof.target_execution.output_value
    trace = rebuild_trace(
        compile_trace_from_proof(proof),
        completion=f"The answer is {answer}.\nApply to target.\n\\boxed{{{answer}}}",
    )

    report = audit_trace(trace, proof=proof)

    assert "answer_leakage_before_application" in report.errors


def test_positive_trace_with_no_application_marker_rejected() -> None:
    proof = safe_proof()
    answer = proof.target_execution.output_value

    trace = rebuild_trace(compile_trace_from_proof(proof), completion=f"Rule checked.\n\\boxed{{{answer}}}")

    report = audit_trace(trace, proof=proof)

    assert "missing_application_marker" in report.errors


def test_positive_trace_with_unsafe_proof_rejected() -> None:
    proof = safe_proof()
    unsafe = replace(proof, shadow_verifier_pass=False)
    trace = compile_trace_from_proof(proof)

    report = audit_trace(trace, proof=unsafe)

    assert "proof_not_training_safe" in report.errors


def test_rejection_trace_with_box_rejected_by_construction() -> None:
    with pytest.raises(Exception):
        replace(compile_rejection_trace("p", "bad"), completion="Reason: bad\n\\boxed{5}", trace_hash="")


def test_fake_verified_examples_text_rejected_when_example_failed() -> None:
    proof = safe_proof()
    failed_execution = replace(proof.example_executions[0], passed=False)
    failed = replace(proof, example_executions=(failed_execution,) + proof.example_executions[1:])
    trace = compile_trace_from_proof(proof, solver_name="known_numeric")

    report = audit_trace(trace, proof=failed)

    assert "fake_verification_claim" in report.errors


def test_too_long_trace_rejected() -> None:
    proof = safe_proof()
    trace = compile_trace_from_proof(proof)

    report = audit_trace(trace, proof=proof, max_chars=10)

    assert "trace_too_long" in report.errors


def test_excessive_boilerplate_rejected() -> None:
    proof = safe_proof()
    answer = proof.target_execution.output_value
    completion = "\n".join(["repeat"] * 10 + ["Apply to target.", f"\\boxed{{{answer}}}"])
    trace = rebuild_trace(compile_trace_from_proof(proof), completion=completion)

    report = audit_trace(trace, proof=proof)

    assert "excessive_boilerplate" in report.errors


def test_assert_trace_valid_raises_on_invalid() -> None:
    proof = safe_proof()
    trace = compile_trace_from_proof(proof)

    with pytest.raises(TraceAuditError):
        assert_trace_valid(trace, proof=replace(proof, format_report=FormatReport(False)))
