"""Conservative string solver for Sprint 1."""

from __future__ import annotations

from collections.abc import Callable
import re

from nemotron_engine.scoring.answer_extractor import normalize_answer_candidate

from .solver_base import ParsedProblem, SolverCandidate, SolverResult, verify_candidate_on_examples


SOLVER_NAME = "string_solver"
_VOWELS = set("aeiouAEIOU")


def solve_string_problem(parsed: ParsedProblem) -> SolverResult:
    return _verified_result(parsed, enumerate_string_candidates(parsed))


def enumerate_string_candidates(parsed: ParsedProblem) -> tuple[SolverCandidate, ...]:
    candidates: list[SolverCandidate] = []
    int_rules: tuple[tuple[str, Callable[[str], int]], ...] = (
        ("length", len),
        ("count_vowels", lambda text: sum(1 for char in text if char in _VOWELS)),
        ("count_consonants", _count_consonants),
        ("count_digits", lambda text: sum(1 for char in text if char.isdigit())),
        ("count_letters", lambda text: sum(1 for char in text if char.isalpha())),
        ("first_char_alpha_index", _first_alpha_index),
        ("last_char_alpha_index", _last_alpha_index),
        ("ascii_sum", lambda text: sum(ord(char) for char in text)),
    )
    string_rules: tuple[tuple[str, Callable[[str], str]], ...] = (
        ("reverse_string", lambda text: text[::-1]),
        ("sort_chars", lambda text: "".join(sorted(text))),
        ("remove_vowels", lambda text: "".join(char for char in text if char not in _VOWELS)),
        ("duplicate_each_char", lambda text: "".join(char * 2 for char in text)),
        ("compress_runs", _compress_runs),
    )
    for rule_id, func in int_rules:
        candidates.append(_int_candidate(parsed, rule_id, func))
    for rule_id, func in string_rules:
        candidates.append(_string_candidate(parsed, rule_id, func))
    candidates.extend(_fixed_affix_candidates(parsed))
    candidates.append(_caesar_candidate(parsed))
    return tuple(candidate for candidate in candidates if candidate is not None)


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


def _int_candidate(parsed: ParsedProblem, rule_id: str, func: Callable[[str], int]) -> SolverCandidate:
    try:
        example_outputs = tuple(func(example.input_value) for example in parsed.examples)
        target_output = func(parsed.target_input)
    except Exception:
        return _diagnostic_candidate(parsed, rule_id, "candidate_exception")
    return _candidate(parsed, rule_id, tuple(str(value) for value in example_outputs), str(target_output), str(target_output))


def _string_candidate(parsed: ParsedProblem, rule_id: str, func: Callable[[str], str]) -> SolverCandidate:
    try:
        example_outputs = tuple(func(example.input_value) for example in parsed.examples)
        raw_target = func(parsed.target_input)
    except Exception:
        return _diagnostic_candidate(parsed, rule_id, "candidate_exception")
    return _candidate(parsed, rule_id, example_outputs, raw_target, raw_target)


def _fixed_affix_candidates(parsed: ParsedProblem) -> tuple[SolverCandidate, ...]:
    candidates: list[SolverCandidate] = []
    first = parsed.examples[0]
    affix_rules: list[tuple[str, str, Callable[[str, str], str]]] = []
    if first.input_value.endswith(first.output_value) and first.input_value != first.output_value:
        affix_rules.append(("fixed_prefix_remove", first.input_value[: -len(first.output_value)], lambda text, affix: text[len(affix) :] if text.startswith(affix) else _raise()))
    if first.input_value.startswith(first.output_value) and first.input_value != first.output_value:
        affix_rules.append(("fixed_suffix_remove", first.input_value[len(first.output_value) :], lambda text, affix: text[: -len(affix)] if text.endswith(affix) else _raise()))
    if first.output_value.endswith(first.input_value) and first.input_value != first.output_value:
        affix_rules.append(("fixed_prefix_add", first.output_value[: -len(first.input_value)], lambda text, affix: affix + text))
    if first.output_value.startswith(first.input_value) and first.input_value != first.output_value:
        affix_rules.append(("fixed_suffix_add", first.output_value[len(first.input_value) :], lambda text, affix: text + affix))
    for rule_id, affix, func in affix_rules:
        if affix:
            candidates.append(_string_candidate(parsed, f"{rule_id}_{affix}", lambda text, affix=affix, func=func: func(text, affix)))
    return tuple(candidates)


def _caesar_candidate(parsed: ParsedProblem) -> SolverCandidate:
    shift: int | None = None
    positions = 0
    distinct_inputs: set[str] = set()
    for example in parsed.examples:
        if len(example.input_value) != len(example.output_value) or not example.input_value.isalpha() or not example.output_value.isalpha():
            return _diagnostic_candidate(parsed, "caesar_shift", "unsupported_chars")
        for left, right in zip(example.input_value, example.output_value):
            if left.islower() != right.islower() or left.isupper() != right.isupper():
                return _diagnostic_candidate(parsed, "caesar_shift", "unsupported_chars")
            base = ord("a") if left.islower() else ord("A")
            current = (ord(right) - ord(left)) % 26
            if shift is None:
                shift = current
            elif shift != current:
                return _diagnostic_candidate(parsed, "caesar_shift", "candidate_exception")
            positions += 1
            distinct_inputs.add(left)
    if shift is None or positions < 2 or len(distinct_inputs) < 2:
        return _diagnostic_candidate(parsed, "caesar_shift", "insufficient_shift_evidence")
    if not parsed.target_input.isalpha():
        return _diagnostic_candidate(parsed, "caesar_shift", "unsupported_chars")
    return _string_candidate(parsed, f"caesar_shift_{shift}", lambda text, shift=shift: _shift_text(text, shift))


def _candidate(parsed: ParsedProblem, rule_id: str, example_outputs: tuple[str, ...], raw_target: str, target_value: str) -> SolverCandidate:
    metadata = {"rule_id": rule_id, "raw_candidate_output": raw_target}
    try:
        target_prediction = str(normalize_answer_candidate(target_value))
    except ValueError:
        target_prediction = str(target_value)
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


def _raise() -> str:
    raise ValueError("unsupported target")


def _count_consonants(text: str) -> int:
    return sum(1 for char in text if char.isalpha() and char not in _VOWELS)


def _first_alpha_index(text: str) -> int:
    if not text or not text[0].isalpha():
        raise ValueError("unsupported chars")
    return ord(text[0].lower()) - ord("a") + 1


def _last_alpha_index(text: str) -> int:
    if not text or not text[-1].isalpha():
        raise ValueError("unsupported chars")
    return ord(text[-1].lower()) - ord("a") + 1


def _compress_runs(text: str) -> str:
    if not text:
        return ""
    parts: list[str] = []
    current = text[0]
    count = 1
    for char in text[1:]:
        if char == current:
            count += 1
        else:
            parts.append(f"{current}{count}")
            current = char
            count = 1
    parts.append(f"{current}{count}")
    return "".join(parts)


def _shift_text(text: str, shift: int) -> str:
    chars: list[str] = []
    for char in text:
        if not char.isalpha():
            raise ValueError("unsupported chars")
        base = ord("a") if char.islower() else ord("A")
        chars.append(chr(base + ((ord(char) - base + shift) % 26)))
    return "".join(chars)


__all__ = ["enumerate_string_candidates", "solve_string_problem"]
