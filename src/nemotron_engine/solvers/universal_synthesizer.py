"""Tiny bounded synthesizer over Pass 2/2.5 executable templates."""

from __future__ import annotations

from collections.abc import Iterable

from nemotron_engine.core.schemas import CanonicalProblem, Program
from nemotron_engine.programs.primitives import make_format_step, make_parse_step, make_program, make_select_identity_step, make_transform_step

from .base import SolverAttempt, SolverResult, build_solver_result, rejected_attempt, verify_candidate_program
from .known_symbol_digit import enumerate_candidate_programs as enumerate_symbol_digit_programs


SOLVER_NAME = "universal_synthesizer"
DEFAULT_MAX_CANDIDATES = 200


def synthesize(
    problem: CanonicalProblem,
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    expected_target_output: str | None = None,
    evaluation_mode: bool = False,
) -> SolverResult:
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int):
        raise ValueError("max_candidates must be an integer.")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive.")
    limit = min(max_candidates, DEFAULT_MAX_CANDIDATES)
    attempts: list[SolverAttempt] = []
    seen: set[str] = set()
    exhausted = False
    enumerated_candidates = 0
    for program in _candidate_templates(problem):
        if program.program_hash in seen:
            continue
        seen.add(program.program_hash)
        if len(seen) > limit:
            exhausted = True
            break
        enumerated_candidates += 1
        expected = expected_target_output if evaluation_mode else None
        attempts.append(verify_candidate_program(problem, program, solver_name=SOLVER_NAME, expected_target_output=expected))
    if exhausted:
        attempts.append(
            rejected_attempt(
                SOLVER_NAME,
                "candidate_budget_exhausted",
                metadata={
                    "max_candidates": limit,
                    "enumerated_candidates": enumerated_candidates,
                    "skipped_due_budget": True,
                },
            )
        )
    return build_solver_result(problem, SOLVER_NAME, attempts)


def solve(problem: CanonicalProblem) -> SolverResult:
    return synthesize(problem)


def _candidate_templates(problem: CanonicalProblem) -> Iterable[Program]:
    yield make_program(
        "synth_identity",
        (make_select_identity_step(output_key="value"), make_format_step("raw", output_key="out")),
        output_key="out",
    )
    yield make_program(
        "synth_reverse",
        (make_transform_step("reverse", output_key="value"), make_format_step("raw", output_key="out")),
        output_key="out",
    )
    yield make_program(
        "synth_bit_not",
        (
            make_parse_step("parse_bitstring", output_key="bits"),
            make_transform_step("bit_not", output_key="value"),
            make_format_step("bitstring", output_key="out"),
        ),
        output_key="out",
    )
    for primitive in ("add_const", "sub_const", "mul_const"):
        for const in _small_ints(10):
            yield make_program(
                f"synth_{primitive}_{const}",
                (
                    make_parse_step("parse_int", output_key="x"),
                    make_transform_step(primitive, {"const": const}, "value"),
                    make_format_step("raw", output_key="out"),
                ),
                output_key="out",
            )
    yield make_program(
        "synth_digit_sum",
        (
            make_parse_step("parse_digits", output_key="digits"),
            make_transform_step("digit_sum", output_key="value"),
            make_format_step("raw", output_key="out"),
        ),
        output_key="out",
    )
    for op in ("add", "sub", "abs_sub", "mul_small"):
        yield make_program(
            f"synth_binary_{op}",
            (
                make_parse_step("parse_binary_int_expr", output_key="expr"),
                make_transform_step("binary_op", {"op": op}, "value"),
                make_format_step("raw", output_key="out"),
            ),
            output_key="out",
        )
    for program in enumerate_symbol_digit_programs(problem):
        yield program
    for a_value in _small_ints(5):
        for b_value in _small_ints(20):
            yield make_program(
                f"synth_affine_{a_value}_{b_value}",
                (
                    make_parse_step("parse_int", output_key="x"),
                    make_transform_step("affine_small", {"a": a_value, "b": b_value}, "value"),
                    make_format_step("raw", output_key="out"),
                ),
                output_key="out",
            )


def _small_ints(limit: int) -> list[int]:
    values = [0]
    for value in range(1, limit + 1):
        values.extend((value, -value))
    return values


__all__ = ["DEFAULT_MAX_CANDIDATES", "solve", "synthesize"]
