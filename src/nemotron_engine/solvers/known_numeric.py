"""High-precision integer and digit-sequence solver templates."""

from __future__ import annotations

from nemotron_engine.core.schemas import CanonicalProblem, Program, VerificationStatus
from nemotron_engine.programs.primitives import make_format_step, make_parse_step, make_program, make_transform_step
from nemotron_engine.programs.type_system import is_decimal_string, is_digit_sequence, is_integer_string

from .base import SolverAttempt, SolverResult, build_solver_result, rejected_attempt, verify_candidate_program


SOLVER_NAME = "known_numeric"


def solve(problem: CanonicalProblem) -> SolverResult:
    programs, rejected = enumerate_candidate_programs_with_rejections(problem)
    attempts = list(rejected)
    attempts.extend(verify_candidate_program(problem, program, solver_name=SOLVER_NAME) for program in programs)
    attempts = _reject_conflicting_verified_outputs(attempts)
    return build_solver_result(problem, SOLVER_NAME, attempts)


def enumerate_candidate_programs(problem: CanonicalProblem) -> list[Program]:
    return enumerate_candidate_programs_with_rejections(problem)[0]


def enumerate_candidate_programs_with_rejections(problem: CanonicalProblem):
    values = [example.input_value for example in problem.examples] + [example.output_value for example in problem.examples]
    values.append(problem.target.input_value)
    if any(is_decimal_string(value) for value in values):
        return [], [rejected_attempt(SOLVER_NAME, "decimals_not_supported")]
    if len(problem.examples) < 2:
        if all(is_integer_string(value) or is_digit_sequence(value) for value in values):
            return [], [rejected_attempt(SOLVER_NAME, "insufficient_examples")]
        return [], [rejected_attempt(SOLVER_NAME, "no_supported_numeric_rule")]
    programs: list[Program] = []
    integer_examples = [(int(example.input_value), int(example.output_value)) for example in problem.examples if is_integer_string(example.input_value) and is_integer_string(example.output_value)]
    if len(integer_examples) == len(problem.examples) and is_integer_string(problem.target.input_value):
        programs.extend(_integer_programs(integer_examples))
    digit_sum = _digit_sum_program(problem)
    if digit_sum is not None:
        programs.append(digit_sum)
    if not programs:
        return [], [rejected_attempt(SOLVER_NAME, "no_supported_numeric_rule")]
    return _dedupe_programs(programs), []


def _integer_programs(examples: list[tuple[int, int]]) -> list[Program]:
    programs: list[Program] = []
    if all(x == y for x, y in examples):
        programs.append(_int_program("numeric_identity", None, {}))
    deltas = {y - x for x, y in examples}
    if len(deltas) == 1:
        delta = next(iter(deltas))
        if delta >= 0:
            programs.append(_int_program(f"numeric_add_{delta}", "add_const", {"const": delta}))
        else:
            programs.append(_int_program(f"numeric_sub_{abs(delta)}", "sub_const", {"const": abs(delta)}))
    multipliers: set[int] = set()
    valid_mul = True
    for x_value, y_value in examples:
        if x_value == 0:
            if y_value != 0:
                valid_mul = False
                break
            continue
        if y_value % x_value != 0:
            valid_mul = False
            break
        multipliers.add(y_value // x_value)
    if valid_mul and len(multipliers) == 1:
        multiplier = next(iter(multipliers))
        programs.append(_int_program(f"numeric_mul_{multiplier}", "mul_const", {"const": multiplier}))
    affine = _fit_unique_affine(examples)
    if affine is not None:
        a_value, b_value = affine
        programs.append(_int_program(f"numeric_affine_{a_value}_{b_value}", "affine_small", {"a": a_value, "b": b_value}))
    return programs


def _int_program(program_id: str, transform: str | None, args: dict[str, int]) -> Program:
    steps = [make_parse_step("parse_int", output_key="x")]
    if transform is not None:
        steps.append(make_transform_step(transform, args, "y"))
    steps.append(make_format_step("raw", output_key="out"))
    return make_program(program_id, steps, output_key="out")


def _digit_sum_program(problem: CanonicalProblem) -> Program | None:
    if not all(is_digit_sequence(example.input_value) and is_integer_string(example.output_value) for example in problem.examples):
        return None
    if not is_digit_sequence(problem.target.input_value):
        return None
    if not all(sum(int(char) for char in example.input_value) == int(example.output_value) for example in problem.examples):
        return None
    return make_program(
        "numeric_digit_sum",
        (
            make_parse_step("parse_digits", output_key="digits"),
            make_transform_step("digit_sum", output_key="sum"),
            make_format_step("raw", output_key="out"),
        ),
        output_key="out",
    )


def _dedupe_programs(programs: list[Program]) -> list[Program]:
    seen: set[str] = set()
    unique: list[Program] = []
    for program in programs:
        if program.program_hash not in seen:
            seen.add(program.program_hash)
            unique.append(program)
    return unique


def _fit_unique_affine(examples: list[tuple[int, int]]) -> tuple[int, int] | None:
    distinct_inputs = sorted({x for x, _y in examples})
    if len(distinct_inputs) < 2:
        return None
    first_x = distinct_inputs[0]
    second_x = next(x for x in distinct_inputs if x != first_x)
    y_for_first = {y for x, y in examples if x == first_x}
    y_for_second = {y for x, y in examples if x == second_x}
    if len(y_for_first) != 1 or len(y_for_second) != 1:
        return None
    first_y = next(iter(y_for_first))
    second_y = next(iter(y_for_second))
    dx = second_x - first_x
    dy = second_y - first_y
    if dy % dx != 0:
        return None
    a_value = dy // dx
    b_value = first_y - a_value * first_x
    if abs(a_value) > 10 or abs(b_value) > 1000:
        return None
    if all(a_value * x + b_value == y for x, y in examples):
        return a_value, b_value
    return None


def _reject_conflicting_verified_outputs(attempts: list[SolverAttempt]) -> list[SolverAttempt]:
    accepted = [attempt for attempt in attempts if attempt.status is VerificationStatus.PASS and attempt.proof is not None]
    outputs = {
        str(attempt.proof.target_execution.output_value)
        for attempt in accepted
        if attempt.proof.target_execution is not None and attempt.proof.target_execution.output_value not in {None, ""}
    }
    if len(outputs) <= 1:
        return attempts
    rewritten: list[SolverAttempt] = []
    for attempt in attempts:
        if attempt.status is VerificationStatus.PASS:
            rewritten.append(
                SolverAttempt(
                    solver_name=attempt.solver_name,
                    program=attempt.program,
                    proof=attempt.proof,
                    status=VerificationStatus.AMBIGUOUS,
                    reason="conflicting_numeric_target_outputs",
                    score=attempt.score,
                    metadata={**attempt.metadata, "conflicting_target_outputs": sorted(outputs)},
                )
            )
        else:
            rewritten.append(attempt)
    return rewritten


solve_numeric = solve


__all__ = ["enumerate_candidate_programs", "solve", "solve_numeric"]
