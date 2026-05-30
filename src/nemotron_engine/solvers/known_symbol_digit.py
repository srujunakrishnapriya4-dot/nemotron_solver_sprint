"""Exact symbol/digit bijection solver."""

from __future__ import annotations

from nemotron_engine.core.schemas import CanonicalProblem, Program
from nemotron_engine.programs.primitives import make_format_step, make_parse_step, make_program, make_transform_step

from .base import SolverResult, build_solver_result, rejected_attempt, verify_candidate_program


SOLVER_NAME = "known_symbol_digit"


def solve(problem: CanonicalProblem) -> SolverResult:
    program, reason = _program_from_mapping(problem)
    if program is None:
        return build_solver_result(problem, SOLVER_NAME, [rejected_attempt(SOLVER_NAME, reason or "no_symbol_digit_mapping")])
    return build_solver_result(problem, SOLVER_NAME, [verify_candidate_program(problem, program, solver_name=SOLVER_NAME)])


def enumerate_candidate_programs(problem: CanonicalProblem) -> list[Program]:
    program, _reason = _program_from_mapping(problem)
    return [] if program is None else [program]


def _program_from_mapping(problem: CanonicalProblem) -> tuple[Program | None, str | None]:
    inputs = [example.input_value for example in problem.examples]
    outputs = [example.output_value for example in problem.examples]
    if not all(len(i) == len(o) for i, o in zip(inputs, outputs)):
        return None, "length_mismatch"
    symbol_to_digit = all(text.isalpha() for text in inputs) and all(text.isdigit() for text in outputs)
    digit_to_symbol = all(text.isdigit() for text in inputs) and all(text.isalpha() for text in outputs)
    if not (symbol_to_digit or digit_to_symbol):
        return None, "unsupported_symbol_digit_direction"
    mapping: dict[str, str] = {}
    reverse: dict[str, str] = {}
    for source, target in zip(inputs, outputs):
        for src_char, dst_char in zip(source, target):
            if src_char in mapping and mapping[src_char] != dst_char:
                return None, "mapping_collision"
            if dst_char in reverse and reverse[dst_char] != src_char:
                return None, "non_bijective_mapping"
            mapping[src_char] = dst_char
            reverse[dst_char] = src_char
    if any(char not in mapping for char in problem.target.input_value):
        return None, "unseen_target_symbol"
    return _mapping_program("symbol_digit_bijection", mapping), None


def _mapping_program(program_id: str, mapping: dict[str, str]) -> Program:
    return make_program(
        program_id,
        (
            make_parse_step("parse_symbols", output_key="symbols"),
            make_transform_step("symbol_bijection", {"mapping": mapping}, "mapped"),
            make_format_step("raw", output_key="out"),
        ),
        output_key="out",
    )


solve_symbol_digit = solve


__all__ = ["enumerate_candidate_programs", "solve", "solve_symbol_digit"]
