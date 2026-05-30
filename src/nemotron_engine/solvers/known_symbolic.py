"""Simple binary integer symbolic operator solver."""

from __future__ import annotations

from nemotron_engine.core.schemas import CanonicalProblem, Program
from nemotron_engine.programs.executor import _execute_step
from nemotron_engine.programs.primitives import make_format_step, make_parse_step, make_program, make_transform_step

from .base import SolverResult, build_solver_result, rejected_attempt, verify_candidate_program


SOLVER_NAME = "known_symbolic"
OPS = ("add", "sub", "abs_sub", "mul_small")


def solve(problem: CanonicalProblem) -> SolverResult:
    programs, rejection = enumerate_candidate_programs_with_rejection(problem)
    if not programs:
        return build_solver_result(problem, SOLVER_NAME, [rejected_attempt(SOLVER_NAME, rejection or "no_symbolic_rule")])
    attempts = [verify_candidate_program(problem, program, solver_name=SOLVER_NAME) for program in programs]
    return build_solver_result(problem, SOLVER_NAME, attempts)


def enumerate_candidate_programs(problem: CanonicalProblem) -> list[Program]:
    return enumerate_candidate_programs_with_rejection(problem)[0]


def enumerate_candidate_programs_with_rejection(problem: CanonicalProblem) -> tuple[list[Program], str | None]:
    parsed_examples = []
    for example in problem.examples:
        try:
            parsed = _execute_step("parse_binary_int_expr", {}, {"current": example.input_value})
        except Exception:
            return [], "malformed_binary_expression"
        parsed_examples.append((parsed, example.output_value))
    operators = {str(parsed["op"]) for parsed, _expected in parsed_examples}
    if len(operators) != 1:
        return [], "mixed_visible_operator_tokens"
    expected_operator = next(iter(operators))
    try:
        target_parsed = _execute_step("parse_binary_int_expr", {}, {"current": problem.target.input_value})
    except Exception:
        return [], "malformed_target_expression"
    if str(target_parsed["op"]) != expected_operator:
        return [], "target_operator_mismatch"
    programs = []
    for op in OPS:
        if all(_op_matches(parsed, expected, op) for parsed, expected in parsed_examples):
            programs.append(_binary_program(op))
    if not programs:
        return [], "inconsistent_operator_alias"
    return programs, None


def _op_matches(parsed: dict[str, int | str], expected: str, op: str) -> bool:
    try:
        result = _execute_step("binary_op", {"op": op}, {"current": parsed})
    except Exception:
        return False
    return str(result) == str(expected)


def _binary_program(op: str) -> Program:
    return make_program(
        f"symbolic_{op}",
        (
            make_parse_step("parse_binary_int_expr", output_key="expr"),
            make_transform_step("binary_op", {"op": op}, "value"),
            make_format_step("raw", output_key="out"),
        ),
        output_key="out",
    )


solve_symbolic = solve


__all__ = ["enumerate_candidate_programs", "solve", "solve_symbolic"]
