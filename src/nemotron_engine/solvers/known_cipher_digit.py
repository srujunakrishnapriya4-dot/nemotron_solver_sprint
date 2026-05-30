"""Exact single-token cipher/digit mapping solver."""

from __future__ import annotations

from nemotron_engine.core.schemas import CanonicalProblem, Program
from nemotron_engine.programs.primitives import make_format_step, make_parse_step, make_program, make_transform_step

from .base import SolverResult, build_solver_result, rejected_attempt, verify_candidate_program


SOLVER_NAME = "known_cipher_digit"


def solve(problem: CanonicalProblem) -> SolverResult:
    program, reason = _program_from_mapping(problem)
    if program is None:
        return build_solver_result(problem, SOLVER_NAME, [rejected_attempt(SOLVER_NAME, reason or "no_cipher_digit_mapping")])
    return build_solver_result(problem, SOLVER_NAME, [verify_candidate_program(problem, program, solver_name=SOLVER_NAME)])


def enumerate_candidate_programs(problem: CanonicalProblem) -> list[Program]:
    program, _reason = _program_from_mapping(problem)
    return [] if program is None else [program]


def _program_from_mapping(problem: CanonicalProblem) -> tuple[Program | None, str | None]:
    if any(_is_multi_token(example.input_value) or _is_multi_token(example.output_value) for example in problem.examples):
        return None, "unsupported_multi_token_delimiter_preserving_output"
    if _is_multi_token(problem.target.input_value):
        return None, "unsupported_multi_token_delimiter_preserving_output"
    mapping: dict[str, str] = {}
    reverse: dict[str, str] = {}
    for example in problem.examples:
        source = example.input_value
        target = example.output_value
        if source in mapping and mapping[source] != target:
            return None, "mapping_collision"
        if target in reverse and reverse[target] != source:
            return None, "non_bijective_mapping"
        mapping[source] = target
        reverse[target] = source
    if problem.target.input_value not in mapping:
        return None, "unseen_target_token"
    return _mapping_program("cipher_digit_bijection", mapping), None


def _mapping_program(program_id: str, mapping: dict[str, str]) -> Program:
    return make_program(
        program_id,
        (
            make_parse_step("parse_token_sequence", output_key="tokens"),
            make_transform_step("symbol_bijection", {"mapping": mapping}, "mapped"),
            make_format_step("raw", output_key="out"),
        ),
        output_key="out",
    )


def _is_multi_token(value: str) -> bool:
    return len(str(value).split()) != 1


solve_cipher_digit = solve


__all__ = ["enumerate_candidate_programs", "solve", "solve_cipher_digit"]
