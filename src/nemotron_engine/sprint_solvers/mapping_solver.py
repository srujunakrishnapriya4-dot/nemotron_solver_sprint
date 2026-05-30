"""Conservative mapping solver for Sprint 1."""

from __future__ import annotations

import re

from .solver_base import ParsedProblem, SolverCandidate, SolverResult, verify_candidate_on_examples


SOLVER_NAME = "mapping_solver"
_DIGITS = set("0123456789")


def solve_mapping_problem(parsed: ParsedProblem) -> SolverResult:
    return _verified_result(parsed, enumerate_mapping_candidates(parsed))


def enumerate_mapping_candidates(parsed: ParsedProblem) -> tuple[SolverCandidate, ...]:
    builders = (
        _exact_lookup_candidate,
        _char_bijection_candidate,
        _token_bijection_candidate,
        _alphabet_shift_candidate,
        _digit_symbol_bijection_candidate,
    )
    candidates: list[SolverCandidate] = []
    for builder in builders:
        built = builder(parsed)
        if built is None:
            continue
        candidates.extend(built)
    return tuple(candidates)


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


def _exact_lookup_candidate(parsed: ParsedProblem) -> tuple[SolverCandidate, ...] | None:
    lookup: dict[str, str] = {}
    for example in parsed.examples:
        if example.input_value in lookup and lookup[example.input_value] != example.output_value:
            return None
        lookup[example.input_value] = example.output_value
    if parsed.target_input not in lookup:
        return None
    return (_candidate(
        parsed,
        "exact_lookup",
        tuple(lookup[example.input_value] for example in parsed.examples),
        lookup[parsed.target_input],
    ),)


def _char_bijection_candidate(parsed: ParsedProblem) -> tuple[SolverCandidate, ...] | None:
    mapping, reason = _infer_bijection(
        tuple(tuple(example.input_value) for example in parsed.examples),
        tuple(tuple(example.output_value) for example in parsed.examples),
    )
    if reason == "mapping_collision":
        return (_diagnostic_candidate(parsed, "char_bijection", "mapping_collision"),)
    if mapping is None:
        return None
    if any(char not in mapping for char in parsed.target_input):
        return (_diagnostic_candidate(parsed, "char_bijection", "unseen_target_symbol"),)
    return (_candidate(
        parsed,
        "char_bijection",
        tuple("".join(mapping[char] for char in example.input_value) for example in parsed.examples),
        "".join(mapping[char] for char in parsed.target_input),
    ),)


def _token_bijection_candidate(parsed: ParsedProblem) -> tuple[SolverCandidate, ...] | None:
    parsed_examples: list[tuple[tuple[str, ...], tuple[str, ...], str]] = []
    for example in parsed.examples:
        input_tokens, sep, input_explicit = _split_tokens(example.input_value)
        output_tokens, output_sep, output_explicit = _split_tokens(example.output_value)
        if (
            not input_explicit
            or not output_explicit
            or len(input_tokens) < 2
            or len(output_tokens) < 2
            or len(input_tokens) != len(output_tokens)
            or sep != output_sep
        ):
            return None
        parsed_examples.append((input_tokens, output_tokens, sep))
    target_tokens, target_sep, target_explicit = _split_tokens(parsed.target_input)
    if not target_explicit or len(target_tokens) < 2:
        return None
    mapping, reason = _infer_bijection(tuple(item[0] for item in parsed_examples), tuple(item[1] for item in parsed_examples))
    if reason == "mapping_collision":
        return (_diagnostic_candidate(parsed, "token_bijection", "mapping_collision"),)
    if mapping is None:
        return None
    if any(token not in mapping for token in target_tokens):
        return (_diagnostic_candidate(parsed, "token_bijection", "unseen_target_token"),)
    if any(sep != target_sep for _, _, sep in parsed_examples):
        return None
    return (_candidate(
        parsed,
        "token_bijection",
        tuple(_join_tokens(tuple(mapping[token] for token in input_tokens), sep) for input_tokens, _, sep in parsed_examples),
        _join_tokens(tuple(mapping[token] for token in target_tokens), target_sep),
    ),)


def _alphabet_shift_candidate(parsed: ParsedProblem) -> tuple[SolverCandidate, ...] | None:
    shift: int | None = None
    aligned_positions = 0
    distinct_inputs: set[str] = set()
    for example in parsed.examples:
        if not _same_case_alpha(example.input_value, example.output_value):
            return None
        for input_char, output_char in zip(example.input_value, example.output_value):
            aligned_positions += 1
            distinct_inputs.add(input_char)
            current = (ord(output_char) - ord(input_char)) % 26
            if shift is None:
                shift = current
            elif shift != current:
                return None
    if shift is None or not _is_alpha_word(parsed.target_input):
        return None
    if aligned_positions < 2 or len(distinct_inputs) < 2:
        return (_diagnostic_candidate(parsed, "alphabet_shift", "insufficient_shift_evidence"),)
    first_input = parsed.examples[0].input_value
    if parsed.target_input.islower() != first_input.islower() or parsed.target_input.isupper() != first_input.isupper():
        return None
    return (_candidate(
        parsed,
        f"alphabet_shift_{shift}",
        tuple(_shift_word(example.input_value, shift) for example in parsed.examples),
        _shift_word(parsed.target_input, shift),
    ),)


def _digit_symbol_bijection_candidate(parsed: ParsedProblem) -> tuple[SolverCandidate, ...] | None:
    digit_to_symbol = all(set(example.input_value) <= _DIGITS and not (set(example.output_value) <= _DIGITS) for example in parsed.examples)
    symbol_to_digit = all(not (set(example.input_value) <= _DIGITS) and set(example.output_value) <= _DIGITS for example in parsed.examples)
    if not digit_to_symbol and not symbol_to_digit:
        return None
    mapping, reason = _infer_bijection(
        tuple(tuple(example.input_value) for example in parsed.examples),
        tuple(tuple(example.output_value) for example in parsed.examples),
    )
    if reason == "mapping_collision":
        return (_diagnostic_candidate(parsed, "digit_symbol_bijection", "mapping_collision"),)
    if mapping is None:
        return None
    if any(char not in mapping for char in parsed.target_input):
        return (_diagnostic_candidate(parsed, "digit_symbol_bijection", "unseen_target_symbol"),)
    if digit_to_symbol and not (set(parsed.target_input) <= _DIGITS):
        return None
    if symbol_to_digit and set(parsed.target_input) <= _DIGITS:
        return None
    return (_candidate(
        parsed,
        "digit_symbol_bijection",
        tuple("".join(mapping[char] for char in example.input_value) for example in parsed.examples),
        "".join(mapping[char] for char in parsed.target_input),
    ),)


def _infer_bijection(inputs: tuple[tuple[str, ...], ...], outputs: tuple[tuple[str, ...], ...]) -> tuple[dict[str, str] | None, str | None]:
    forward: dict[str, str] = {}
    reverse: dict[str, str] = {}
    for input_items, output_items in zip(inputs, outputs):
        if len(input_items) != len(output_items):
            return None, "length_mismatch"
        for input_item, output_item in zip(input_items, output_items):
            if input_item in forward and forward[input_item] != output_item:
                return None, "mapping_collision"
            if output_item in reverse and reverse[output_item] != input_item:
                return None, "mapping_collision"
            forward[input_item] = output_item
            reverse[output_item] = input_item
    return forward, None


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


def _same_case_alpha(left: str, right: str) -> bool:
    return len(left) == len(right) and _is_alpha_word(left) and _is_alpha_word(right) and (
        (left.islower() and right.islower()) or (left.isupper() and right.isupper())
    )


def _is_alpha_word(text: str) -> bool:
    return bool(text) and text.isalpha() and (text.islower() or text.isupper())


def _shift_word(text: str, shift: int) -> str:
    base = ord("a") if text.islower() else ord("A")
    return "".join(chr(base + ((ord(char) - base + shift) % 26)) for char in text)


def _candidate(parsed: ParsedProblem, rule_id: str, example_predictions: tuple[str, ...], target_prediction: str) -> SolverCandidate:
    return SolverCandidate(
        solver_name=SOLVER_NAME,
        example_predictions=example_predictions,
        target_prediction=target_prediction,
        metadata={"rule_id": rule_id},
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


__all__ = ["enumerate_mapping_candidates", "solve_mapping_problem"]
