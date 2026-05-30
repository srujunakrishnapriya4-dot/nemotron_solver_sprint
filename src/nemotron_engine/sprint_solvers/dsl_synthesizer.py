"""Conservative shallow DSL synthesizer for Sprint 1."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import re

from nemotron_engine.scoring.answer_extractor import normalize_answer_candidate

from .solver_base import ParsedProblem, SolverCandidate, SolverResult, verify_candidate_on_examples


SOLVER_NAME = "dsl_synthesizer"
_INT_RE = re.compile(r"^[+-]?\d+$")
_BINARY_RE = re.compile(r"^\s*([+-]?\d+)\s*([^\s\d]+)\s*([+-]?\d+)\s*$")
_SYMBOLIC_OPS = {"@", "#", "$", "?", "&"}
_DIGITS = set("0123456789")


@dataclass(frozen=True)
class _BuildResult:
    candidates: tuple[SolverCandidate, ...]
    budget_exhausted: bool


class _CandidateBudget:
    def __init__(self, max_candidates: int) -> None:
        self.max_candidates = max_candidates
        self.candidates: list[SolverCandidate] = []
        self.exhausted = False

    def add(self, candidate: SolverCandidate | None) -> None:
        if candidate is None:
            return
        if len(self.candidates) >= self.max_candidates:
            self.exhausted = True
            return
        self.candidates.append(candidate)


def solve_dsl_problem(parsed: ParsedProblem, *, max_candidates: int = 200) -> SolverResult:
    built = _build_candidates(parsed, max_candidates=max_candidates)
    if built.budget_exhausted:
        errors = ("budget_exhausted",)
        rejected = tuple(_with_reason(candidate, "budget_exhausted") for candidate in built.candidates) + (
            _diagnostic_candidate(parsed, "budget", "budget_exhausted"),
        )
        return SolverResult(status="abstain", prediction=None, rejected_candidates=rejected, errors=errors)
    return _verified_result(parsed, built.candidates)


def enumerate_dsl_candidates(parsed: ParsedProblem, *, max_candidates: int = 200) -> tuple[SolverCandidate, ...]:
    return _build_candidates(parsed, max_candidates=max_candidates).candidates


def _build_candidates(parsed: ParsedProblem, *, max_candidates: int) -> _BuildResult:
    if max_candidates < 1:
        budget = _CandidateBudget(0)
        budget.exhausted = True
        return _BuildResult(candidates=(), budget_exhausted=True)
    budget = _CandidateBudget(max_candidates)
    _add_string_candidates(parsed, budget)
    _add_bit_candidates(parsed, budget)
    _add_integer_candidates(parsed, budget)
    _add_sequence_candidates(parsed, budget)
    _add_binary_candidates(parsed, budget)
    _add_mapping_candidates(parsed, budget)
    return _BuildResult(candidates=tuple(budget.candidates), budget_exhausted=budget.exhausted)


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


def _add_string_candidates(parsed: ParsedProblem, budget: _CandidateBudget) -> None:
    budget.add(_string_rule_candidate(parsed, "identity", lambda text: text))
    budget.add(_string_rule_candidate(parsed, "reverse_string", lambda text: text[::-1]))
    budget.add(_string_rule_candidate(parsed, "reverse_digits", _reverse_digits_text))
    budget.add(_int_rule_candidate(parsed, "digit_sum", _digit_sum_text))
    budget.add(_int_rule_candidate(parsed, "digit_product", _digit_product_text))
    budget.add(_int_rule_candidate(parsed, "count_digits", lambda text: len(text) if _INT_RE.fullmatch(text) else _raise("non_integer_input")))
    budget.add(_int_rule_candidate(parsed, "reverse_digits_then_digit_sum", lambda text: _digit_sum_text(_reverse_digits_text(text))))
    if len(parsed.examples) >= 2:
        _add_mod_compositions(parsed, budget, "digit_sum_then_mod", lambda text: _digit_sum_text(text))
        _add_mod_compositions(parsed, budget, "digit_product_then_mod", lambda text: _digit_product_text(text))


def _add_bit_candidates(parsed: ParsedProblem, budget: _CandidateBudget) -> None:
    values = [parsed.target_input, *(example.input_value for example in parsed.examples), *(example.output_value for example in parsed.examples)]
    if not values or any(not value or set(value) - {"0", "1"} for value in values) or len({len(value) for value in values}) != 1:
        return
    budget.add(_bit_rule_candidate(parsed, "bit_not", lambda text: "".join("1" if char == "0" else "0" for char in text)))


def _add_integer_candidates(parsed: ParsedProblem, budget: _CandidateBudget) -> None:
    parsed_ints = _parse_integer_problem(parsed)
    if parsed_ints is None:
        if _looks_integer_problem(parsed):
            budget.add(_diagnostic_candidate(parsed, "integer_parse", "non_integer_input"))
        return
    pairs, target = parsed_ints
    if len({x for x, _ in pairs}) < 2:
        budget.add(_diagnostic_candidate(parsed, "add_const", "underdetermined_rule"))
        budget.add(_diagnostic_candidate(parsed, "mul_const", "underdetermined_rule"))
        budget.add(_diagnostic_candidate(parsed, "affine_small", "underdetermined_rule"))
        budget.add(_diagnostic_candidate(parsed, "modulo_const", "underdetermined_rule"))
    else:
        add_consts = {y - x for x, y in pairs}
        if len(add_consts) == 1:
            c = next(iter(add_consts))
            budget.add(_int_outputs_candidate(parsed, f"add_const_{c}", tuple(x + c for x, _ in pairs), target + c))
        mul = _infer_mul_const(pairs)
        if mul is not None:
            budget.add(_int_outputs_candidate(parsed, f"mul_const_{mul}", tuple(x * mul for x, _ in pairs), target * mul))
        affine = _infer_affine(pairs)
        if affine is not None:
            a, b = affine
            budget.add(_int_outputs_candidate(parsed, f"affine_small_{a}_{b}", tuple((a * x) + b for x, _ in pairs), (a * target) + b))
        for m in range(2, 51):
            outputs = tuple(x % m for x, _ in pairs)
            if outputs == tuple(y for _, y in pairs):
                budget.add(_int_outputs_candidate(parsed, f"modulo_const_{m}", outputs, target % m))
    budget.add(_int_outputs_candidate(parsed, "parity", tuple(x % 2 for x, _ in pairs), target % 2))
    budget.add(_int_outputs_candidate(parsed, "last_digit", tuple(x % 10 for x, _ in pairs), target % 10))
    budget.add(_int_outputs_candidate(parsed, "digital_root", tuple(_digital_root(x) for x, _ in pairs), _digital_root(target)))


def _add_sequence_candidates(parsed: ParsedProblem, budget: _CandidateBudget) -> None:
    parsed_sequences = _parse_sequence_problem(parsed)
    if parsed_sequences is None:
        return
    pairs, target = parsed_sequences
    budget.add(_sequence_rule_candidate(parsed, "sequence_sum", pairs, target, sum))
    budget.add(_sequence_rule_candidate(parsed, "sequence_product", pairs, target, math.prod))
    budget.add(_sequence_rule_candidate(parsed, "sequence_length", pairs, target, len))
    _add_sequence_mod_compositions(parsed, budget, pairs, target)


def _add_binary_candidates(parsed: ParsedProblem, budget: _CandidateBudget) -> None:
    parsed_binary = _parse_binary_problem(parsed)
    if parsed_binary is None:
        return
    pairs, target = parsed_binary
    operator = pairs[0][1]
    if operator in {"/", "%"} and (target[2] == 0 or any(b == 0 for _, _, b, _ in pairs)):
        rule_id = "binary_exact_div" if operator == "/" else "binary_mod"
        budget.add(_diagnostic_candidate(parsed, rule_id, "division_by_zero"))
        return
    if operator == "/" and (any(a % b != 0 for a, _, b, _ in pairs) or target[0] % target[2] != 0):
        budget.add(_diagnostic_candidate(parsed, "binary_exact_div", "non_exact_division"))
        return
    for rule_id, func in _binary_rules(operator, pairs, target):
        try:
            example_outputs = tuple(func(a, b) for a, _, b, _ in pairs)
            target_output = func(target[0], target[2])
        except ZeroDivisionError:
            budget.add(_diagnostic_candidate(parsed, rule_id, "division_by_zero"))
            continue
        if target_output is None or any(output is None for output in example_outputs):
            if "exact_div" in rule_id:
                reason = "division_by_zero" if target[2] == 0 or any(b == 0 for _, _, b, _ in pairs) else "non_exact_division"
            else:
                reason = "non_exact_division" if operator == "/" else "division_by_zero"
            budget.add(_diagnostic_candidate(parsed, rule_id, reason))
            continue
        budget.add(_int_outputs_candidate(parsed, rule_id, example_outputs, target_output))


def _add_mapping_candidates(parsed: ParsedProblem, budget: _CandidateBudget) -> None:
    all_values = [parsed.target_input, *(example.input_value for example in parsed.examples), *(example.output_value for example in parsed.examples)]
    if all_values and all(set(value) <= {"0", "1"} for value in all_values):
        return
    char_mapping, char_reason = _infer_bijection(tuple(tuple(example.input_value) for example in parsed.examples), tuple(tuple(example.output_value) for example in parsed.examples))
    if char_reason == "mapping_collision":
        budget.add(_diagnostic_candidate(parsed, "char_bijection", "mapping_collision"))
    elif char_mapping is not None:
        if any(char not in char_mapping for char in parsed.target_input):
            budget.add(_diagnostic_candidate(parsed, "char_bijection", "unseen_target_symbol"))
        else:
            budget.add(
                _raw_outputs_candidate(
                    parsed,
                    "char_bijection",
                    tuple("".join(char_mapping[char] for char in example.input_value) for example in parsed.examples),
                    "".join(char_mapping[char] for char in parsed.target_input),
                )
            )

    token_data = _tokenized_examples(parsed)
    if token_data is not None:
        parsed_examples, target_tokens, sep = token_data
        mapping, reason = _infer_bijection(tuple(item[0] for item in parsed_examples), tuple(item[1] for item in parsed_examples))
        if reason == "mapping_collision":
            budget.add(_diagnostic_candidate(parsed, "token_bijection", "mapping_collision"))
        elif mapping is not None:
            if any(token not in mapping for token in target_tokens):
                budget.add(_diagnostic_candidate(parsed, "token_bijection", "unseen_target_token"))
            else:
                budget.add(
                    _token_outputs_candidate(
                        parsed,
                        "token_bijection",
                        tuple(_join_tokens(tuple(mapping[token] for token in input_tokens), sep) for input_tokens, _ in parsed_examples),
                        tuple(mapping[token] for token in target_tokens),
                        _join_tokens(tuple(mapping[token] for token in target_tokens), sep),
                    )
                )


def _string_rule_candidate(parsed: ParsedProblem, rule_id: str, func: Callable[[str], str]) -> SolverCandidate:
    try:
        return _raw_outputs_candidate(parsed, rule_id, tuple(func(example.input_value) for example in parsed.examples), func(parsed.target_input))
    except Exception as exc:
        return _diagnostic_candidate(parsed, rule_id, _exception_reason(exc))


def _bit_rule_candidate(parsed: ParsedProblem, rule_id: str, func: Callable[[str], str]) -> SolverCandidate:
    try:
        example_outputs = tuple(func(example.input_value) for example in parsed.examples)
        raw_target = func(parsed.target_input)
        metadata = {"rule_id": rule_id, "raw_candidate_output": raw_target}
        try:
            target_prediction = str(normalize_answer_candidate(int(raw_target, 2)))
        except ValueError:
            target_prediction = str(int(raw_target, 2))
            metadata["rejection_reason"] = "invalid_output_format"
        return SolverCandidate(
            solver_name=SOLVER_NAME,
            example_predictions=example_outputs,
            target_prediction=target_prediction,
            metadata=metadata,
        )
    except Exception as exc:
        return _diagnostic_candidate(parsed, rule_id, _exception_reason(exc))


def _int_rule_candidate(parsed: ParsedProblem, rule_id: str, func: Callable[[str], int]) -> SolverCandidate:
    try:
        return _int_outputs_candidate(parsed, rule_id, tuple(func(example.input_value) for example in parsed.examples), func(parsed.target_input))
    except Exception as exc:
        return _diagnostic_candidate(parsed, rule_id, _exception_reason(exc))


def _sequence_rule_candidate(
    parsed: ParsedProblem,
    rule_id: str,
    pairs: tuple[tuple[tuple[int, ...], int], ...],
    target: tuple[int, ...],
    func: Callable[[tuple[int, ...]], int],
) -> SolverCandidate:
    try:
        return _int_outputs_candidate(parsed, rule_id, tuple(func(seq) for seq, _ in pairs), func(target))
    except Exception as exc:
        return _diagnostic_candidate(parsed, rule_id, _exception_reason(exc))


def _add_mod_compositions(parsed: ParsedProblem, budget: _CandidateBudget, prefix: str, func: Callable[[str], int]) -> None:
    try:
        transformed_examples = tuple(func(example.input_value) for example in parsed.examples)
        target_value = func(parsed.target_input)
    except Exception as exc:
        budget.add(_diagnostic_candidate(parsed, prefix, _exception_reason(exc)))
        return
    if len(transformed_examples) < 2 or len(set(transformed_examples)) < 2:
        budget.add(_diagnostic_candidate(parsed, prefix, "underdetermined_rule"))
        return
    for m in range(2, 11):
        budget.add(_int_outputs_candidate(parsed, f"{prefix}_{m}", tuple(value % m for value in transformed_examples), target_value % m))


def _add_sequence_mod_compositions(
    parsed: ParsedProblem,
    budget: _CandidateBudget,
    pairs: tuple[tuple[tuple[int, ...], int], ...],
    target: tuple[int, ...],
) -> None:
    transformed_examples = tuple(sum(seq) for seq, _ in pairs)
    target_sum = sum(target)
    if len(transformed_examples) < 2 or len(set(transformed_examples)) < 2:
        budget.add(_diagnostic_candidate(parsed, "sequence_sum_then_mod", "underdetermined_rule"))
    else:
        for m in range(2, 11):
            budget.add(_int_outputs_candidate(parsed, f"sequence_sum_then_mod_{m}", tuple(value % m for value in transformed_examples), target_sum % m))

    product_examples = tuple(math.prod(seq) for seq, _ in pairs)
    target_product = math.prod(target)
    if len(product_examples) < 2 or len(set(product_examples)) < 2:
        budget.add(_diagnostic_candidate(parsed, "sequence_product_then_mod", "underdetermined_rule"))
        return
    for m in range(2, 11):
        budget.add(_int_outputs_candidate(parsed, f"sequence_product_then_mod_{m}", tuple(value % m for value in product_examples), target_product % m))


def _raw_outputs_candidate(parsed: ParsedProblem, rule_id: str, example_outputs: tuple[str, ...], raw_target: str) -> SolverCandidate:
    metadata = {"rule_id": rule_id, "raw_candidate_output": str(raw_target)}
    try:
        target_prediction = str(normalize_answer_candidate(str(raw_target)))
    except ValueError:
        target_prediction = str(raw_target)
        metadata["rejection_reason"] = "invalid_output_format"
    return SolverCandidate(
        solver_name=SOLVER_NAME,
        example_predictions=tuple(str(output) for output in example_outputs),
        target_prediction=target_prediction,
        metadata=metadata,
    )


def _token_outputs_candidate(
    parsed: ParsedProblem,
    rule_id: str,
    example_outputs: tuple[str, ...],
    target_tokens: tuple[str, ...],
    raw_target: str,
) -> SolverCandidate:
    metadata = {"rule_id": rule_id, "raw_candidate_output": str(raw_target)}
    if all(_INT_RE.fullmatch(token) for token in target_tokens):
        target_value = "".join(target_tokens)
    else:
        target_value = str(raw_target)
    try:
        target_prediction = str(normalize_answer_candidate(target_value))
    except ValueError:
        target_prediction = target_value
        metadata["rejection_reason"] = "invalid_output_format"
    return SolverCandidate(
        solver_name=SOLVER_NAME,
        example_predictions=tuple(str(output) for output in example_outputs),
        target_prediction=target_prediction,
        metadata=metadata,
    )


def _int_outputs_candidate(parsed: ParsedProblem, rule_id: str, example_outputs: tuple[int | None, ...], target_output: int | None) -> SolverCandidate:
    if target_output is None or any(output is None for output in example_outputs):
        return _diagnostic_candidate(parsed, rule_id, "candidate_exception")
    return _raw_outputs_candidate(parsed, rule_id, tuple(str(output) for output in example_outputs), str(target_output))


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


def _parse_integer_problem(parsed: ParsedProblem) -> tuple[tuple[tuple[int, int], ...], int] | None:
    if not _INT_RE.fullmatch(parsed.target_input):
        return None
    pairs: list[tuple[int, int]] = []
    for example in parsed.examples:
        if not _INT_RE.fullmatch(example.input_value) or not _INT_RE.fullmatch(example.output_value):
            return None
        pairs.append((int(example.input_value), int(example.output_value)))
    return tuple(pairs), int(parsed.target_input)


def _looks_integer_problem(parsed: ParsedProblem) -> bool:
    return any(_INT_RE.fullmatch(value) for value in [parsed.target_input, *(example.input_value for example in parsed.examples)])


def _parse_sequence_problem(parsed: ParsedProblem) -> tuple[tuple[tuple[int, ...], int], tuple[int, ...]] | None:
    target = _parse_int_list(parsed.target_input)
    if target is None:
        return None
    pairs: list[tuple[tuple[int, ...], int]] = []
    for example in parsed.examples:
        values = _parse_int_list(example.input_value)
        if values is None or not _INT_RE.fullmatch(example.output_value):
            return None
        pairs.append((values, int(example.output_value)))
    return tuple(pairs), target


def _parse_int_list(text: str) -> tuple[int, ...] | None:
    if "," not in text and not re.search(r"\s", text.strip()):
        return None
    parts = tuple(part for part in text.replace(",", " ").split() if part)
    if len(parts) < 2 or any(not _INT_RE.fullmatch(part) for part in parts):
        return None
    return tuple(int(part) for part in parts)


def _parse_binary_problem(parsed: ParsedProblem) -> tuple[tuple[tuple[int, str, int, int], ...], tuple[int, str, int]] | None:
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


def _binary_rules(
    operator: str,
    pairs: tuple[tuple[int, str, int, int], ...],
    target: tuple[int, str, int],
) -> tuple[tuple[str, Callable[[int, int], int | None]], ...]:
    if operator == "+":
        return (("binary_add", lambda a, b: a + b),)
    if operator == "-":
        return (("binary_sub", lambda a, b: a - b),)
    if operator == "*":
        return (("binary_mul", lambda a, b: a * b),)
    if operator == "/":
        return (("binary_exact_div", lambda a, b: a // b),)
    if operator == "%":
        return (("binary_mod", lambda a, b: a % b),)
    if operator not in _SYMBOLIC_OPS:
        return ()
    return (
        ("symbolic_add", lambda a, b: a + b),
        ("symbolic_sub", lambda a, b: a - b),
        ("symbolic_mul", lambda a, b: a * b),
        ("symbolic_exact_div", lambda a, b: None if b == 0 or a % b != 0 else a // b),
        ("symbolic_mod", lambda a, b: None if b == 0 else a % b),
    )


def _tokenized_examples(parsed: ParsedProblem) -> tuple[list[tuple[tuple[str, ...], tuple[str, ...]]], tuple[str, ...], str] | None:
    parsed_examples: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    sep: str | None = None
    for example in parsed.examples:
        input_tokens, input_sep, input_explicit = _split_tokens(example.input_value)
        output_tokens, output_sep, output_explicit = _split_tokens(example.output_value)
        if not input_explicit or not output_explicit or len(input_tokens) != len(output_tokens) or input_sep != output_sep:
            return None
        if sep is None:
            sep = input_sep
        elif sep != input_sep:
            return None
        parsed_examples.append((input_tokens, output_tokens))
    target_tokens, target_sep, target_explicit = _split_tokens(parsed.target_input)
    if sep is None or not target_explicit or target_sep != sep:
        return None
    return parsed_examples, target_tokens, sep


def _split_tokens(text: str) -> tuple[tuple[str, ...], str, bool]:
    if "," in text:
        tokens = tuple(token.strip() for token in text.split(","))
        return (tokens if all(tokens) else (), ",", len(tokens) >= 2 and all(tokens))
    if re.search(r"\s", text.strip()):
        tokens = tuple(text.split())
        return (tokens if all(tokens) else (), " ", len(tokens) >= 2 and all(tokens))
    return ((text.strip(),), " ", False)


def _join_tokens(tokens: tuple[str, ...], sep: str) -> str:
    return ("," if sep == "," else " ").join(tokens)


def _infer_bijection(inputs: tuple[tuple[str, ...], ...], outputs: tuple[tuple[str, ...], ...]) -> tuple[dict[str, str] | None, str | None]:
    forward: dict[str, str] = {}
    reverse: dict[str, str] = {}
    for input_items, output_items in zip(inputs, outputs):
        if len(input_items) != len(output_items):
            return None, "unsupported_input"
        for input_item, output_item in zip(input_items, output_items):
            if input_item in forward and forward[input_item] != output_item:
                return None, "mapping_collision"
            if output_item in reverse and reverse[output_item] != input_item:
                return None, "mapping_collision"
            forward[input_item] = output_item
            reverse[output_item] = input_item
    return forward, None


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
    if len(distinct) < 2:
        return None
    first_x = distinct[0]
    second_x = distinct[1]
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


def _reverse_digits_text(text: str) -> str:
    if not _INT_RE.fullmatch(text):
        _raise("non_integer_input")
    sign = "-" if text.startswith("-") else ""
    digits = text[1:] if sign else text
    return sign + digits[::-1]


def _digit_sum_text(text: str) -> int:
    if not _INT_RE.fullmatch(text):
        _raise("non_integer_input")
    return sum(int(char) for char in text.lstrip("-"))


def _digit_product_text(text: str) -> int:
    if not _INT_RE.fullmatch(text):
        _raise("non_integer_input")
    return math.prod(int(char) for char in text.lstrip("-"))


def _digital_root(value: int) -> int:
    value = abs(value)
    if value == 0:
        return 0
    return 1 + ((value - 1) % 9)


def _exception_reason(exc: Exception) -> str:
    reason = str(exc)
    if reason in {
        "underdetermined_rule",
        "budget_exhausted",
        "invalid_output_format",
        "candidate_exception",
        "unseen_target_symbol",
        "unseen_target_token",
        "mapping_collision",
        "division_by_zero",
        "non_exact_division",
        "non_integer_input",
        "unsupported_input",
    }:
        return reason
    return "candidate_exception"


def _raise(reason: str) -> None:
    raise ValueError(reason)


__all__ = ["enumerate_dsl_candidates", "solve_dsl_problem"]
