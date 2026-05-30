"""Conservative integer sequence solver for Sprint 1."""

from __future__ import annotations

from collections.abc import Callable
import math
import re

from nemotron_engine.scoring.answer_extractor import normalize_answer_candidate

from .solver_base import ParsedProblem, SolverCandidate, SolverResult, verify_candidate_on_examples


SOLVER_NAME = "sequence_solver"
_INT_RE = re.compile(r"^[+-]?\d+$")


def solve_sequence_problem(parsed: ParsedProblem) -> SolverResult:
    return _verified_result(parsed, enumerate_sequence_candidates(parsed))


def enumerate_sequence_candidates(parsed: ParsedProblem) -> tuple[SolverCandidate, ...]:
    parse_error = _sequence_parse_error(parsed)
    if parse_error is not None:
        return (_diagnostic_candidate(parsed, "parse_sequence", parse_error),)
    parsed_sequences = _parse_sequence_problem(parsed)
    if parsed_sequences is not None:
        return _sequence_candidates(parsed, parsed_sequences)
    parsed_index = _parse_index_problem(parsed)
    if parsed_index is not None:
        return _index_candidates(parsed, parsed_index)
    if _looks_sequence_like(parsed):
        return (_diagnostic_candidate(parsed, "parse_sequence", "non_integer_sequence"),)
    return ()


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
        errors=("verified_candidates_disagree",),
    )


def _sequence_candidates(parsed: ParsedProblem, data: tuple[tuple[tuple[int, ...], int], tuple[int, ...]]) -> tuple[SolverCandidate, ...]:
    examples, target = data
    if any(len(seq) < 3 for seq, _ in examples) or len(target) < 3:
        return (_diagnostic_candidate(parsed, "sequence", "insufficient_terms"),)
    candidates: list[SolverCandidate] = []
    rules: tuple[tuple[str, Callable[[tuple[int, ...]], int]], ...] = (
        ("arithmetic_progression_next", _arithmetic_next),
        ("geometric_progression_next", _geometric_next),
        ("fibonacci_like_next", _fibonacci_next),
        ("constant_difference_sum", _constant_difference_sum),
        ("polynomial_degree_2_next", _degree2_next),
        ("repeat_cycle_next", _repeat_cycle_next),
        ("sequence_length", len),
        ("sequence_sum", sum),
        ("sequence_product", math.prod),
        ("sequence_max", max),
        ("sequence_min", min),
        ("max_minus_min", lambda seq: max(seq) - min(seq)),
        ("count_even", lambda seq: sum(1 for value in seq if value % 2 == 0)),
        ("count_odd", lambda seq: sum(1 for value in seq if value % 2 != 0)),
    )
    for rule_id, func in rules:
        candidate = _candidate_from_sequence_rule(parsed, examples, target, rule_id, func)
        if candidate is not None:
            candidates.append(candidate)
    if any(len(seq) < 4 for seq, _ in examples) or len(target) < 4:
        candidates.append(_diagnostic_candidate(parsed, "polynomial_degree_2_next", "insufficient_terms_degree2"))
    if _has_non_integral_ratio(examples, target):
        candidates.append(_diagnostic_candidate(parsed, "geometric_progression_next", "non_integral_ratio"))
    return tuple(candidates)


def _index_candidates(parsed: ParsedProblem, data: tuple[tuple[tuple[int, int], ...], int]) -> tuple[SolverCandidate, ...]:
    pairs, target = data
    if len(pairs) < 3 or len({x for x, _ in pairs}) < 3:
        return (_diagnostic_candidate(parsed, "index_sequence", "insufficient_examples"),)
    ordered = tuple(sorted(pairs))
    xs = tuple(x for x, _ in ordered)
    ys = tuple(y for _, y in ordered)
    lookup = dict(ordered)
    if target < max(xs) and target not in lookup:
        return (_diagnostic_candidate(parsed, "index_sequence", "unsafe_backward_index_extrapolation"),)
    if any(right - left != 1 for left, right in zip(xs, xs[1:])):
        return ()
    candidates: list[SolverCandidate] = []
    index_rules: list[tuple[str, Callable[[int], int] | None]] = [
        ("arithmetic_progression_next", _index_arithmetic_rule(ordered)),
    ]
    if len(pairs) >= 4:
        index_rules.append(("polynomial_degree_2_next", _index_degree2_rule(ordered)))
    else:
        candidates.append(_diagnostic_candidate(parsed, "polynomial_degree_2_next", "insufficient_examples_degree2"))
    for rule_id, func in index_rules:
        if func is None:
            continue
        candidates.append(_candidate(parsed, rule_id, tuple(func(x) for x, _ in pairs), func(target)))
    if _is_fibonacci_like(ys):
        if target < ordered[-1][0] and target not in dict(ordered):
            candidates.append(_diagnostic_candidate(parsed, "fibonacci_like_next", "unsafe_backward_index_extrapolation"))
        else:
            try:
                candidates.append(_candidate(parsed, "fibonacci_like_next", tuple(_lookup_or_fib_next(ordered, x) for x, _ in pairs), _lookup_or_fib_next(ordered, target)))
            except ValueError:
                candidates.append(_diagnostic_candidate(parsed, "fibonacci_like_next", "unsafe_backward_index_extrapolation"))
    return tuple(candidates)


def _parse_sequence_problem(parsed: ParsedProblem) -> tuple[tuple[tuple[int, ...], int], tuple[int, ...]] | None:
    target = _parse_int_sequence(parsed.target_input)
    if target is None:
        return None
    examples: list[tuple[tuple[int, ...], int]] = []
    for example in parsed.examples:
        seq = _parse_int_sequence(example.input_value)
        if seq is None or not _INT_RE.fullmatch(example.output_value):
            return None
        examples.append((seq, int(example.output_value)))
    return tuple(examples), target


def _parse_index_problem(parsed: ParsedProblem) -> tuple[tuple[tuple[int, int], ...], int] | None:
    if not _INT_RE.fullmatch(parsed.target_input):
        return None
    pairs: list[tuple[int, int]] = []
    for example in parsed.examples:
        if not _INT_RE.fullmatch(example.input_value) or not _INT_RE.fullmatch(example.output_value):
            return None
        pairs.append((int(example.input_value), int(example.output_value)))
    return tuple(pairs), int(parsed.target_input)


def _parse_int_sequence(text: str) -> tuple[int, ...] | None:
    if "," not in text and not re.search(r"\s", text.strip()):
        return None
    parts = tuple(part for part in text.replace(",", " ").split() if part)
    if len(parts) < 2 or any(not _INT_RE.fullmatch(part) for part in parts):
        return None
    return tuple(int(part) for part in parts)


def _sequence_parse_error(parsed: ParsedProblem) -> str | None:
    if _looks_sequence_like(parsed):
        sequence_values = [parsed.target_input, *(example.input_value for example in parsed.examples)]
        if any(_parse_int_sequence(value) is None for value in sequence_values):
            return "non_integer_sequence"
        if any(not _INT_RE.fullmatch(example.output_value) for example in parsed.examples):
            return "invalid_output_format"
    return None


def _looks_sequence_like(parsed: ParsedProblem) -> bool:
    values = [parsed.target_input, *(example.input_value for example in parsed.examples)]
    return any("," in value or re.search(r"\s", value.strip()) for value in values)


def _candidate_from_sequence_rule(
    parsed: ParsedProblem,
    examples: tuple[tuple[tuple[int, ...], int], ...],
    target: tuple[int, ...],
    rule_id: str,
    func: Callable[[tuple[int, ...]], int],
) -> SolverCandidate | None:
    try:
        return _candidate(parsed, rule_id, tuple(func(seq) for seq, _ in examples), func(target))
    except Exception as exc:
        reason = str(exc)
        if reason not in {"insufficient_terms_degree2", "non_integral_ratio"}:
            reason = "candidate_exception"
        return _diagnostic_candidate(parsed, rule_id, reason)


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


def _arithmetic_next(seq: tuple[int, ...]) -> int:
    diffs = tuple(right - left for left, right in zip(seq, seq[1:]))
    if len(set(diffs)) != 1:
        raise ValueError("not arithmetic")
    return seq[-1] + diffs[-1]


def _geometric_next(seq: tuple[int, ...]) -> int:
    if any(value == 0 for value in seq[:-1]):
        raise ValueError("zero ratio base")
    ratios: list[int] = []
    for left, right in zip(seq, seq[1:]):
        if right % left != 0:
            raise ValueError("non-integral ratio")
        ratios.append(right // left)
    if len(set(ratios)) != 1:
        raise ValueError("not geometric")
    return seq[-1] * ratios[-1]


def _fibonacci_next(seq: tuple[int, ...]) -> int:
    if not _is_fibonacci_like(seq):
        raise ValueError("not fibonacci-like")
    return seq[-1] + seq[-2]


def _constant_difference_sum(seq: tuple[int, ...]) -> int:
    diffs = tuple(right - left for left, right in zip(seq, seq[1:]))
    if len(set(diffs)) != 1:
        raise ValueError("not constant difference")
    return sum(diffs)


def _degree2_next(seq: tuple[int, ...]) -> int:
    if len(seq) < 4:
        raise ValueError("insufficient_terms_degree2")
    diffs = tuple(right - left for left, right in zip(seq, seq[1:]))
    second = tuple(right - left for left, right in zip(diffs, diffs[1:]))
    if not second or len(set(second)) != 1:
        raise ValueError("not degree 2")
    return seq[-1] + diffs[-1] + second[-1]


def _repeat_cycle_next(seq: tuple[int, ...]) -> int:
    for period in range(1, (len(seq) // 2) + 1):
        if all(value == seq[index % period] for index, value in enumerate(seq)):
            return seq[len(seq) % period]
    raise ValueError("not repeating")


def _has_non_integral_ratio(examples: tuple[tuple[tuple[int, ...], int], ...], target: tuple[int, ...]) -> bool:
    for seq, _ in (*examples, (target, 0)):
        if any(left != 0 and right % left != 0 for left, right in zip(seq, seq[1:])):
            return True
    return False


def _index_arithmetic_rule(pairs: tuple[tuple[int, int], ...]) -> Callable[[int], int] | None:
    diffs = tuple(right[1] - left[1] for left, right in zip(pairs, pairs[1:]))
    if len(set(diffs)) != 1:
        return None
    first_x, first_y = pairs[0]
    step = diffs[0]
    return lambda x: first_y + ((x - first_x) * step)


def _index_degree2_rule(pairs: tuple[tuple[int, int], ...]) -> Callable[[int], int] | None:
    if len(pairs) < 4:
        return None
    diffs = tuple(right[1] - left[1] for left, right in zip(pairs, pairs[1:]))
    second = tuple(right - left for left, right in zip(diffs, diffs[1:]))
    if not second or len(set(second)) != 1:
        return None
    x0, y0 = pairs[0]
    d0 = diffs[0]
    dd = second[0]

    def predict(x: int) -> int:
        n = x - x0
        return y0 + (n * d0) + ((n * (n - 1) // 2) * dd)

    return predict


def _is_fibonacci_like(seq: tuple[int, ...]) -> bool:
    return len(seq) >= 3 and all(seq[index] == seq[index - 1] + seq[index - 2] for index in range(2, len(seq)))


def _lookup_or_fib_next(pairs: tuple[tuple[int, int], ...], x: int) -> int:
    lookup = dict(pairs)
    if x in lookup:
        return lookup[x]
    if x < pairs[-1][0]:
        raise ValueError("unsafe backward extrapolation")
    ordered = tuple(value for _, value in pairs)
    current_index = pairs[-1][0]
    values = list(ordered)
    while current_index < x:
        values.append(values[-1] + values[-2])
        current_index += 1
    return values[-1]


__all__ = ["enumerate_sequence_candidates", "solve_sequence_problem"]
