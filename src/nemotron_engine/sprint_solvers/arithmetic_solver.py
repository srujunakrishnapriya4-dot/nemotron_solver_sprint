"""Conservative arithmetic solver for Sprint 1."""

from __future__ import annotations

from collections.abc import Callable
import math
import re

from .solver_base import ParsedProblem, SolverCandidate, SolverResult, verify_candidate_on_examples


SOLVER_NAME = "arithmetic_solver"
_INT_RE = re.compile(r"^[+-]?\d+$")
_BINARY_RE = re.compile(r"^\s*([+-]?\d+)\s*([^\s\d]+)\s*([+-]?\d+)\s*$")


def solve_arithmetic_problem(parsed: ParsedProblem) -> SolverResult:
    return _verified_result(parsed, enumerate_arithmetic_candidates(parsed))


def enumerate_arithmetic_candidates(parsed: ParsedProblem) -> tuple[SolverCandidate, ...]:
    single = _parse_single_examples(parsed)
    if single is not None:
        return _single_integer_candidates(parsed, single)
    lists = _parse_list_examples(parsed)
    if lists is not None:
        return _list_candidates(parsed, lists)
    binary = _parse_binary_examples(parsed)
    if binary is not None:
        return _binary_candidates(parsed, binary)
    return ()


def _verified_result(parsed: ParsedProblem, candidates: tuple[SolverCandidate, ...]) -> SolverResult:
    verified: list[SolverCandidate] = []
    rejected: list[SolverCandidate] = []
    for candidate in candidates:
        if verify_candidate_on_examples(parsed, candidate):
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
        errors=("verified_candidates_disagree",),
    )


def _single_integer_candidates(parsed: ParsedProblem, data: tuple[tuple[int, int], int]) -> tuple[SolverCandidate, ...]:
    pairs, target = data
    candidates: list[SolverCandidate] = []
    candidates.extend(
        _candidate(parsed, rule_id, tuple(func(x) for x, _ in pairs), func(target))
        for rule_id, func in (
            ("identity", lambda x: x),
            ("square", lambda x: x * x),
            ("cube", lambda x: x * x * x),
            ("digit_sum", _digit_sum),
            ("digit_product", _digit_product),
            ("reverse_digits", _reverse_digits),
            ("count_digits", _count_digits),
        )
    )
    distinct_inputs = {x for x, _ in pairs}
    if len(distinct_inputs) >= 2:
        add = {y - x for x, y in pairs}
        if len(add) == 1:
            c = next(iter(add))
            candidates.append(_candidate(parsed, f"add_const_{c}", tuple(x + c for x, _ in pairs), target + c))
        sub = {x - y for x, y in pairs}
        if len(sub) == 1:
            c = next(iter(sub))
            candidates.append(_candidate(parsed, f"sub_const_{c}", tuple(x - c for x, _ in pairs), target - c))
        mul = _infer_mul_const(pairs)
        if mul is not None:
            candidates.append(_candidate(parsed, f"mul_const_{mul}", tuple(x * mul for x, _ in pairs), target * mul))
        affine = _infer_affine(pairs)
        if affine is not None:
            a, b = affine
            candidates.append(_candidate(parsed, f"affine_{a}_{b}", tuple(a * x + b for x, _ in pairs), a * target + b))
    return tuple(candidates)


def _list_candidates(parsed: ParsedProblem, data: tuple[tuple[tuple[int, ...], int], tuple[int, ...]]) -> tuple[SolverCandidate, ...]:
    pairs, target = data
    rules: tuple[tuple[str, Callable[[tuple[int, ...]], int]], ...] = (
        ("sum", sum),
        ("product", math.prod),
        ("max", max),
        ("min", min),
        ("max_minus_min", lambda xs: max(xs) - min(xs)),
        ("length", len),
        ("count_even", lambda xs: sum(1 for item in xs if item % 2 == 0)),
        ("count_odd", lambda xs: sum(1 for item in xs if item % 2 != 0)),
    )
    return tuple(_candidate(parsed, rule_id, tuple(func(xs) for xs, _ in pairs), func(target)) for rule_id, func in rules)


def _binary_candidates(parsed: ParsedProblem, data: tuple[tuple[tuple[int, str, int, int], ...], tuple[int, str, int]]) -> tuple[SolverCandidate, ...]:
    pairs, target = data
    operator = pairs[0][1]
    rules = _binary_rules_for_operator(operator, pairs, target)
    candidates: list[SolverCandidate] = []
    target_a, _, target_b = target
    for rule_id, func in rules:
        example_outputs = tuple(func(a, b) for a, _, b, _ in pairs)
        target_output = func(target_a, target_b)
        if target_output is None or any(output is None for output in example_outputs):
            continue
        candidates.append(_candidate(parsed, rule_id, example_outputs, target_output))
    return tuple(candidates)


def _binary_rules_for_operator(
    operator: str,
    pairs: tuple[tuple[int, str, int, int], ...],
    target: tuple[int, str, int],
) -> tuple[tuple[str, Callable[[int, int], int | None]], ...]:
    if operator == "+":
        return (("a_plus_b", lambda a, b: a + b),)
    if operator == "-":
        return (
            ("a_minus_b", lambda a, b: a - b),
            ("b_minus_a", lambda a, b: b - a),
            ("abs_a_minus_b", lambda a, b: abs(a - b)),
        )
    if operator == "*":
        return (("a_times_b", lambda a, b: a * b),)
    if operator == "/":
        if target[2] == 0 or any(b == 0 or a % b != 0 for a, _, b, _ in pairs):
            return ()
        return (("a_exact_div_b", lambda a, b: None if b == 0 or a % b != 0 else a // b),)
    if operator == "%":
        if target[2] == 0 or any(b == 0 for _, _, b, _ in pairs):
            return ()
        return (("a_mod_b", lambda a, b: None if b == 0 else a % b),)

    rules: list[tuple[str, Callable[[int, int], int | None]]] = [
        ("a_plus_b", lambda a, b: a + b),
        ("a_minus_b", lambda a, b: a - b),
        ("b_minus_a", lambda a, b: b - a),
        ("abs_a_minus_b", lambda a, b: abs(a - b)),
        ("a_times_b", lambda a, b: a * b),
        ("a_mod_b", lambda a, b: None if b == 0 else a % b),
    ]
    if all(b != 0 and a % b == 0 for a, _, b, _ in pairs):
        rules.append(("a_exact_div_b", lambda a, b: None if b == 0 or a % b != 0 else a // b))
    return tuple(rules)


def _parse_single_examples(parsed: ParsedProblem) -> tuple[tuple[tuple[int, int], ...], int] | None:
    if not _INT_RE.fullmatch(parsed.target_input):
        return None
    pairs: list[tuple[int, int]] = []
    for example in parsed.examples:
        if not _INT_RE.fullmatch(example.input_value) or not _INT_RE.fullmatch(example.output_value):
            return None
        pairs.append((int(example.input_value), int(example.output_value)))
    return tuple(pairs), int(parsed.target_input)


def _parse_list_examples(parsed: ParsedProblem) -> tuple[tuple[tuple[int, ...], int], ...] | tuple[tuple[tuple[tuple[int, ...], int], ...], tuple[int, ...]] | None:
    target = _parse_int_list(parsed.target_input)
    if target is None or len(target) < 2:
        return None
    pairs: list[tuple[tuple[int, ...], int]] = []
    for example in parsed.examples:
        values = _parse_int_list(example.input_value)
        if values is None or len(values) < 2 or not _INT_RE.fullmatch(example.output_value):
            return None
        pairs.append((values, int(example.output_value)))
    return tuple(pairs), target


def _parse_binary_examples(parsed: ParsedProblem) -> tuple[tuple[tuple[int, str, int, int], ...], tuple[int, str, int]] | None:
    target = _parse_binary(parsed.target_input)
    if target is None:
        return None
    pairs: list[tuple[int, str, int, int]] = []
    for example in parsed.examples:
        parsed_input = _parse_binary(example.input_value)
        if parsed_input is None or not _INT_RE.fullmatch(example.output_value):
            return None
        pairs.append((*parsed_input, int(example.output_value)))
    ops = {op for _, op, _, _ in pairs}
    if len(ops) != 1 or target[1] not in ops:
        return None
    return tuple(pairs), target


def _parse_binary(text: str) -> tuple[int, str, int] | None:
    match = _BINARY_RE.fullmatch(text)
    if not match:
        return None
    return int(match.group(1)), match.group(2), int(match.group(3))


def _parse_int_list(text: str) -> tuple[int, ...] | None:
    normalized = text.replace(",", " ")
    parts = normalized.split()
    if len(parts) < 2 or any(not _INT_RE.fullmatch(part) for part in parts):
        return None
    return tuple(int(part) for part in parts)


def _infer_mul_const(pairs: tuple[tuple[int, int], ...]) -> int | None:
    constants = set()
    saw_nonzero = False
    for x, y in pairs:
        if x == 0:
            if y != 0:
                return None
            continue
        saw_nonzero = True
        if y % x != 0:
            return None
        constants.add(y // x)
    return next(iter(constants)) if saw_nonzero and len(constants) == 1 else None


def _infer_affine(pairs: tuple[tuple[int, int], ...]) -> tuple[int, int] | None:
    distinct = sorted({x for x, _ in pairs})
    first_x = distinct[0]
    second_x = next(x for x in distinct if x != first_x)
    first_y = next(y for x, y in pairs if x == first_x)
    second_y = next(y for x, y in pairs if x == second_x)
    dx = second_x - first_x
    dy = second_y - first_y
    if dy % dx != 0:
        return None
    a = dy // dx
    b = first_y - (a * first_x)
    if abs(a) > 20 or abs(b) > 10000:
        return None
    if all((a * x) + b == y for x, y in pairs):
        return a, b
    return None


def _digit_sum(value: int) -> int:
    return sum(int(char) for char in str(abs(value)))


def _digit_product(value: int) -> int:
    return math.prod(int(char) for char in str(abs(value)))


def _reverse_digits(value: int) -> int:
    sign = -1 if value < 0 else 1
    return sign * int(str(abs(value))[::-1])


def _count_digits(value: int) -> int:
    return len(str(abs(value)))


def _candidate(parsed: ParsedProblem, rule_id: str, example_outputs: tuple[int | None, ...], target_output: int | None) -> SolverCandidate:
    return SolverCandidate(
        solver_name=SOLVER_NAME,
        example_predictions=tuple(str(output) for output in example_outputs),
        target_prediction=str(target_output),
        metadata={"rule_id": rule_id},
    )


def _with_reason(candidate: SolverCandidate, reason: str) -> SolverCandidate:
    metadata = dict(candidate.metadata)
    metadata.setdefault("rejection_reason", reason)
    return SolverCandidate(candidate.solver_name, candidate.example_predictions, candidate.target_prediction, metadata)


__all__ = ["enumerate_arithmetic_candidates", "solve_arithmetic_problem"]
