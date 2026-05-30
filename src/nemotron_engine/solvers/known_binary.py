"""High-precision bitstring solver templates."""

from __future__ import annotations

from nemotron_engine.core.schemas import CanonicalProblem, Program
from nemotron_engine.programs.primitives import make_format_step, make_parse_step, make_program, make_transform_step
from nemotron_engine.programs.type_system import is_bitstring

from .base import SolverResult, build_solver_result, rejected_attempt, verify_candidate_program


SOLVER_NAME = "known_binary"


def solve(problem: CanonicalProblem) -> SolverResult:
    reason = _preflight(problem)
    if reason is not None:
        return build_solver_result(problem, SOLVER_NAME, [rejected_attempt(SOLVER_NAME, reason)])
    attempts = [verify_candidate_program(problem, program, solver_name=SOLVER_NAME) for program in enumerate_candidate_programs(problem)]
    return build_solver_result(problem, SOLVER_NAME, attempts)


def enumerate_candidate_programs(problem: CanonicalProblem) -> list[Program]:
    return [
        _bit_program("binary_identity", ()),
        _bit_program("binary_reverse", (make_transform_step("reverse", output_key="rev"),)),
        _bit_program("binary_bit_not", (make_transform_step("bit_not", output_key="not_bits"),)),
    ]


def _bit_program(program_id: str, transforms: tuple) -> Program:
    steps = [make_parse_step("parse_bitstring", output_key="bits")]
    steps.extend(transforms)
    steps.append(make_format_step("bitstring", output_key="out"))
    return make_program(program_id, steps, output_key="out")


def _preflight(problem: CanonicalProblem) -> str | None:
    values = [example.input_value for example in problem.examples] + [example.output_value for example in problem.examples]
    values.append(problem.target.input_value)
    if any(not is_bitstring(value) for value in values):
        return "non_bitstring_value"
    widths = {len(value) for value in values}
    if len(widths) != 1:
        return "variable_length_bitstrings"
    return None


solve_binary = solve


__all__ = ["enumerate_candidate_programs", "solve", "solve_binary"]
