"""Conservative modular arithmetic solver for Sprint 1."""

from __future__ import annotations

import math
import re

from nemotron_engine.scoring.answer_extractor import normalize_answer_candidate

from .solver_base import ParsedProblem, SolverCandidate, SolverResult, verify_candidate_on_examples


SOLVER_NAME = "modular_solver"
_INT_RE = re.compile(r"^[+-]?\d+$")


def solve_modular_problem(parsed: ParsedProblem) -> SolverResult:
    return _verified_result(parsed, enumerate_modular_candidates(parsed))


def enumerate_modular_candidates(parsed: ParsedProblem) -> tuple[SolverCandidate, ...]:
    parsed_ints = _parse_integer_problem(parsed)
    if isinstance(parsed_ints, str):
        return (_diagnostic_candidate(parsed, "parse_integer", parsed_ints),)
    pairs, target = parsed_ints
    if len({x for x, _ in pairs}) < 2:
        return (_diagnostic_candidate(parsed, "modular", "insufficient_examples"),)
    candidates: list[SolverCandidate] = []
    expected = tuple(y for _, y in pairs)
    _append_if_examples_match(candidates, parsed, expected, "parity", tuple(x % 2 for x, _ in pairs), target % 2)
    _append_if_examples_match(candidates, parsed, expected, "last_digit", tuple(x % 10 for x, _ in pairs), target % 10)
    _append_if_examples_match(candidates, parsed, expected, "last_two_digits", tuple(x % 100 for x, _ in pairs), target % 100)
    _append_if_examples_match(candidates, parsed, expected, "digital_root", tuple(_digital_root(x) for x, _ in pairs), _digital_root(target))
    for m in range(2, 101):
        _append_if_examples_match(candidates, parsed, expected, f"x_mod_{m}", tuple(x % m for x, _ in pairs), target % m)
        _append_if_examples_match(candidates, parsed, expected, f"digit_sum_mod_{m}", tuple(_digit_sum(x) % m for x, _ in pairs), _digit_sum(target) % m)
        _append_if_examples_match(candidates, parsed, expected, f"digit_product_mod_{m}", tuple(_digit_product(x) % m for x, _ in pairs), _digit_product(target) % m)
    for m in range(2, 51):
        for a in range(-20, 21):
            for b in range(-100, 101):
                example_outputs = tuple(((a * x) + b) % m for x, _ in pairs)
                _append_if_examples_match(
                    candidates,
                    parsed,
                    expected,
                    f"affine_mod_{a}_{b}_{m}",
                    example_outputs,
                    ((a * target) + b) % m,
                )
    return tuple(candidates)


def _verified_result(parsed: ParsedProblem, candidates: tuple[SolverCandidate, ...]) -> SolverResult:
    verified: list[SolverCandidate] = []
    rejected: list[SolverCandidate] = []
    for candidate in candidates:
        if candidate.metadata.get("rejection_reason"):
            rejected.append(candidate)
        elif verify_candidate_on_examples(parsed, candidate):
            verified.append(candidate)
        else:
            rejected.append(_with_reason(candidate, "failed_example_or_target_validation"))
    if not verified:
        return SolverResult(status="abstain", prediction=None, rejected_candidates=tuple(rejected))
    predictions = {candidate.target_prediction for candidate in verified}
    if len(predictions) == 1:
        return SolverResult(
            status="solved",
            prediction=next(iter(predictions)),
            verified_candidates=tuple(verified),
            rejected_candidates=tuple(rejected),
        )
    return SolverResult(
        status="disagreement",
        prediction=None,
        verified_candidates=tuple(verified),
        rejected_candidates=tuple(rejected),
        errors=("ambiguous_moduli", "verified_candidates_disagree"),
    )


def _parse_integer_problem(parsed: ParsedProblem) -> tuple[tuple[tuple[int, int], ...], int] | str:
    if not _INT_RE.fullmatch(parsed.target_input):
        return "non_integer_input"
    pairs: list[tuple[int, int]] = []
    for example in parsed.examples:
        if not _INT_RE.fullmatch(example.input_value):
            return "non_integer_input"
        if not _INT_RE.fullmatch(example.output_value):
            return "invalid_output_format"
        try:
            output_value = normalize_answer_candidate(example.output_value)
        except ValueError:
            return "invalid_output_format"
        pairs.append((int(example.input_value), output_value))
    return tuple(pairs), int(parsed.target_input)


def _candidate(parsed: ParsedProblem, rule_id: str, example_outputs: tuple[int, ...], target_output: int) -> SolverCandidate:
    metadata = {"rule_id": rule_id}
    try:
        target_prediction = str(normalize_answer_candidate(target_output))
    except ValueError:
        target_prediction = str(target_output)
        metadata["rejection_reason"] = "invalid_output_format"
    return SolverCandidate(
        solver_name=SOLVER_NAME,
        example_predictions=tuple(str(value) for value in example_outputs),
        target_prediction=target_prediction,
        metadata=metadata,
    )


def _append_if_examples_match(
    candidates: list[SolverCandidate],
    parsed: ParsedProblem,
    expected: tuple[int, ...],
    rule_id: str,
    example_outputs: tuple[int, ...],
    target_output: int,
) -> None:
    if example_outputs == expected:
        candidates.append(_candidate(parsed, rule_id, example_outputs, target_output))


def _diagnostic_candidate(parsed: ParsedProblem, rule_id: str, reason: str) -> SolverCandidate:
    return SolverCandidate(
        solver_name=SOLVER_NAME,
        example_predictions=tuple("__rejected__" for _ in parsed.examples),
        target_prediction="__rejected__",
        metadata={"rule_id": rule_id, "rejection_reason": reason},
    )


def _with_reason(candidate: SolverCandidate, reason: str) -> SolverCandidate:
    metadata = dict(candidate.metadata)
    metadata.setdefault("rejection_reason", reason)
    return SolverCandidate(candidate.solver_name, candidate.example_predictions, candidate.target_prediction, metadata)


def _digital_root(value: int) -> int:
    value = abs(value)
    if value == 0:
        return 0
    return 1 + ((value - 1) % 9)


def _digit_sum(value: int) -> int:
    return sum(int(char) for char in str(abs(value)))


def _digit_product(value: int) -> int:
    return math.prod(int(char) for char in str(abs(value)))


__all__ = ["enumerate_modular_candidates", "solve_modular_problem"]
