"""Primary verifier for executable proof-spine programs."""

from __future__ import annotations

from nemotron_engine.core.schemas import (
    AmbiguityReport,
    CandidateProgram,
    CanonicalProblem,
    ExecutableProof,
    ExecutionTrace,
    FormatReport,
    Program,
)

from .ambiguity import build_ambiguity_report
from .executor import execute_program
from .leakage import detect_target_leakage


def verify_program_on_problem(
    problem: CanonicalProblem,
    program: Program,
    *,
    expected_target_output: str | None = None,
) -> ExecutableProof:
    example_traces: list[ExecutionTrace] = []
    for example in problem.examples:
        trace = execute_program(program, example.input_value)
        passed = trace.error is None and str(trace.output_value) == str(example.output_value)
        example_traces.append(
            ExecutionTrace(
                input_value=trace.input_value,
                output_value=trace.output_value,
                expected_value=example.output_value,
                passed=passed,
                intermediate_values=trace.intermediate_values,
                error=trace.error,
            )
        )

    target_trace = execute_program(program, problem.target.input_value)
    target_output = target_trace.output_value
    if expected_target_output is not None:
        target_passed = target_trace.error is None and str(target_output) == str(expected_target_output)
        target_trace = ExecutionTrace(
            input_value=target_trace.input_value,
            output_value=target_output,
            expected_value=expected_target_output,
            passed=target_passed,
            intermediate_values=target_trace.intermediate_values,
            error=target_trace.error,
        )
    candidate = CandidateProgram(program=program, example_executions=tuple(example_traces), target_execution=target_trace)
    ambiguity = build_ambiguity_report([candidate])
    leakage_target = expected_target_output if expected_target_output is not None else problem.target.expected_output
    leakage = detect_target_leakage(problem.raw_prompt, leakage_target)
    format_report = _build_format_report(target_trace)
    primary_pass = (
        all(trace.passed is True for trace in example_traces)
        and target_trace.error is None
        and target_trace.output_value not in {None, ""}
        and format_report.valid
    )
    return ExecutableProof(
        problem_id=problem.problem_id,
        parse_hash=problem.parse_hash,
        program_hash=program.program_hash,
        locked_program=program,
        rejected_programs=(),
        example_executions=tuple(example_traces),
        target_execution=target_trace,
        ambiguity_report=ambiguity,
        leakage_report=leakage,
        format_report=format_report,
        primary_verifier_pass=primary_pass,
        shadow_verifier_pass=False,
    )


def _build_format_report(target_trace: ExecutionTrace) -> FormatReport:
    if target_trace.error is not None:
        return FormatReport(valid=False, errors=(target_trace.error,))
    if target_trace.output_value in {None, ""}:
        return FormatReport(valid=False, errors=("Target output is missing.",))
    return FormatReport(valid=True, normalized_output=str(target_trace.output_value))


__all__ = ["verify_program_on_problem"]
