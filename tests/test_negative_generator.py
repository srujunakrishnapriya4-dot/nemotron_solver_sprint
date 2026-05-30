from __future__ import annotations

import pytest

from nemotron_engine.parsing import canonicalize_prompt
from nemotron_engine.solvers import solve_numeric
from nemotron_engine.traces import audit_trace, compile_trace_from_solver_result
from nemotron_engine.training.negative_generator import (
    ALLOWED_NEGATIVE_TYPES,
    NegativeGenerationError,
    NegativeGenerationReport,
    generate_negative_for_trace,
    generate_negative_traces,
)
from nemotron_engine.training.sft_builder import SFTBuildError, build_sft_row


def valid_trace():
    problem = canonicalize_prompt("p", "1 -> 2\n2 -> 3\n4 -> ?")
    result = solve_numeric(problem)
    return compile_trace_from_solver_result(result, prompt=problem.raw_prompt)


def forged_trace(base, **updates: object):
    forged = object.__new__(type(base))
    for name in type(base).__dataclass_fields__:
        object.__setattr__(forged, name, getattr(base, name))
    for key, value in updates.items():
        object.__setattr__(forged, key, value)
    return forged


def test_each_required_negative_type_generated_with_reason() -> None:
    trace = valid_trace()

    candidates = [generate_negative_for_trace(trace, kind) for kind in ALLOWED_NEGATIVE_TYPES]

    assert {candidate.negative_type for candidate in candidates} == set(ALLOWED_NEGATIVE_TYPES)
    assert all(candidate.reason for candidate in candidates)
    assert all(candidate.completion != trace.completion for candidate in candidates)


def test_negative_trace_is_non_positive_or_fails_audit() -> None:
    trace = valid_trace()

    for kind in ALLOWED_NEGATIVE_TYPES:
        negative = generate_negative_for_trace(trace, kind).trace
        report = audit_trace(negative)
        assert negative.is_positive is False or not report.valid or report.warnings


def test_target_leak_and_format_box_are_audit_invalid() -> None:
    trace = valid_trace()

    target_leak = generate_negative_for_trace(trace, "target_leak").trace
    format_box = generate_negative_for_trace(trace, "format_box").trace

    assert "answer_leakage_before_application" in audit_trace(target_leak).errors
    assert "missing_application_marker" in audit_trace(format_box).errors


def test_no_generated_negative_can_pass_sft_builder() -> None:
    trace = valid_trace()
    candidates, _ = generate_negative_traces([trace], negative_types=ALLOWED_NEGATIVE_TYPES, max_negatives_per_trace=7)

    for candidate in candidates:
        with pytest.raises(SFTBuildError):
            build_sft_row(candidate.trace)


def test_fake_verification_not_sft_safe() -> None:
    trace = valid_trace()
    fake = generate_negative_for_trace(trace, "fake_verification").trace

    assert audit_trace(fake).warnings or not audit_trace(fake).valid
    with pytest.raises(SFTBuildError):
        build_sft_row(fake)


def test_overlong_reasoning_exceeds_audit_length() -> None:
    trace = valid_trace()
    overlong = generate_negative_for_trace(trace, "overlong_reasoning").trace

    assert not audit_trace(overlong, max_lines=30).valid


def test_generated_ids_are_stable_across_calls() -> None:
    trace = valid_trace()

    first = generate_negative_for_trace(trace, "target_leak")
    second = generate_negative_for_trace(trace, "target_leak")

    assert first.negative_id == second.negative_id
    assert first.negative_hash == second.negative_hash
    assert first.trace.trace_id == second.trace.trace_id


def test_unsupported_negative_type_rejected() -> None:
    with pytest.raises(NegativeGenerationError, match="unsupported_negative_type"):
        generate_negative_for_trace(valid_trace(), "unknown")


def test_rejects_invalid_or_forged_source_traces() -> None:
    trace = valid_trace()
    answer = trace.boxed_answer

    with pytest.raises(NegativeGenerationError, match="source_trace_marked_negative"):
        generate_negative_for_trace(forged_trace(trace, metadata={"negative_type": "false_rule", "is_negative": True}), "false_rule")
    with pytest.raises(NegativeGenerationError, match="source_trace_missing_verified_provenance"):
        generate_negative_for_trace(forged_trace(trace, source_proof_hash=None), "false_rule")
    with pytest.raises(NegativeGenerationError, match="source_trace_missing_verified_provenance"):
        generate_negative_for_trace(forged_trace(trace, source_program_hash=None), "false_rule")
    with pytest.raises(NegativeGenerationError, match="source_trace_audit_invalid"):
        generate_negative_for_trace(forged_trace(trace, completion=f"Final only\n\\boxed{{{answer}}}"), "false_rule")


def test_negative_generation_report_rejects_forged_counts() -> None:
    with pytest.raises(NegativeGenerationError, match="skipped reason totals"):
        NegativeGenerationReport(input_count=1, generated_count=0, skipped_count=1, skipped_reasons={"bad": 2})
    with pytest.raises(NegativeGenerationError, match="negative type totals"):
        NegativeGenerationReport(input_count=1, generated_count=1, skipped_count=0, negative_type_counts={"false_rule": 2})
