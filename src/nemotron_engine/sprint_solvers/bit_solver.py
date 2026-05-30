"""Conservative bitstring solver for Sprint 1."""

from __future__ import annotations

from collections.abc import Callable

from nemotron_engine.scoring.answer_extractor import normalize_answer_candidate

from .solver_base import ParsedProblem, SolverCandidate, SolverResult, verify_candidate_on_examples


SOLVER_NAME = "bit_solver"


def solve_bit_problem(parsed: ParsedProblem) -> SolverResult:
    errors = _validation_errors(parsed)
    if errors:
        return SolverResult(status="abstain", prediction=None, errors=errors)
    candidates = enumerate_bit_candidates(parsed)
    return _verified_result(parsed, candidates)


def enumerate_bit_candidates(parsed: ParsedProblem) -> tuple[SolverCandidate, ...]:
    if _validation_errors(parsed):
        return ()

    width = len(parsed.target_input)
    candidates: list[SolverCandidate] = []
    candidates.append(_candidate(parsed, "identity", lambda bits: bits))
    candidates.append(_candidate(parsed, "reverse_bits", lambda bits: bits[::-1]))
    candidates.append(_candidate(parsed, "bit_not", _bit_not))
    for k in range(1, width):
        candidates.append(_candidate(parsed, f"rotate_left_{k}", lambda bits, k=k: bits[k:] + bits[:k]))
        candidates.append(_candidate(parsed, f"rotate_right_{k}", lambda bits, k=k: bits[-k:] + bits[:-k]))
    if len(parsed.examples) >= 2:
        mask = _infer_xor_mask(parsed)
        if mask is not None:
            candidates.append(_candidate(parsed, f"xor_const_{mask}", lambda bits, mask=mask: _xor(bits, mask)))
        mask = _infer_and_mask(parsed)
        if mask is not None:
            candidates.append(_candidate(parsed, f"and_const_{mask}", lambda bits, mask=mask: _and(bits, mask)))
        mask = _infer_or_mask(parsed)
        if mask is not None:
            candidates.append(_candidate(parsed, f"or_const_{mask}", lambda bits, mask=mask: _or(bits, mask)))
    for k in range(1, width):
        candidates.append(_candidate(parsed, f"shift_left_{k}", lambda bits, k=k: bits[k:] + ("0" * k)))
        candidates.append(_candidate(parsed, f"shift_right_{k}", lambda bits, k=k: ("0" * k) + bits[: width - k]))
    return tuple(candidate for candidate in candidates if candidate is not None)


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


def _candidate(parsed: ParsedProblem, rule_id: str, transform: Callable[[str], str]) -> SolverCandidate | None:
    raw_target = transform(parsed.target_input)
    metadata = {"rule_id": rule_id, "raw_candidate_output": raw_target}
    try:
        target_prediction = str(normalize_answer_candidate(int(raw_target, 2)))
    except ValueError:
        target_prediction = str(int(raw_target, 2))
        metadata["rejection_reason"] = "target_prediction_outside_submission_range"
    return SolverCandidate(
        solver_name=SOLVER_NAME,
        example_predictions=tuple(transform(example.input_value) for example in parsed.examples),
        target_prediction=target_prediction,
        metadata=metadata,
    )


def _with_reason(candidate: SolverCandidate, reason: str) -> SolverCandidate:
    metadata = dict(candidate.metadata)
    metadata.setdefault("rejection_reason", reason)
    return SolverCandidate(candidate.solver_name, candidate.example_predictions, candidate.target_prediction, metadata)


def _validation_errors(parsed: ParsedProblem) -> tuple[str, ...]:
    values = [parsed.target_input]
    for example in parsed.examples:
        values.extend((example.input_value, example.output_value))
    if any(not value for value in values):
        return ("empty_bitstring",)
    if any(set(value) - {"0", "1"} for value in values):
        return ("non_bit_character",)
    widths = {len(value) for value in values}
    if len(widths) != 1:
        return ("mixed_width_bitstrings",)
    if next(iter(widths)) < 1:
        return ("empty_bitstring",)
    return ()


def _bit_not(bits: str) -> str:
    return "".join("1" if bit == "0" else "0" for bit in bits)


def _xor(bits: str, mask: str) -> str:
    return "".join("1" if bit != mask_bit else "0" for bit, mask_bit in zip(bits, mask))


def _and(bits: str, mask: str) -> str:
    return "".join("1" if bit == "1" and mask_bit == "1" else "0" for bit, mask_bit in zip(bits, mask))


def _or(bits: str, mask: str) -> str:
    return "".join("1" if bit == "1" or mask_bit == "1" else "0" for bit, mask_bit in zip(bits, mask))


def _infer_xor_mask(parsed: ParsedProblem) -> str | None:
    masks = {_xor(example.input_value, example.output_value) for example in parsed.examples}
    return next(iter(masks)) if len(masks) == 1 else None


def _infer_and_mask(parsed: ParsedProblem) -> str | None:
    width = len(parsed.target_input)
    mask: list[str | None] = [None] * width
    for example in parsed.examples:
        for index, (input_bit, output_bit) in enumerate(zip(example.input_value, example.output_value)):
            if input_bit == "0":
                if output_bit != "0":
                    return None
                continue
            if mask[index] is not None and mask[index] != output_bit:
                return None
            mask[index] = output_bit
    if any(bit is None for bit in mask):
        return None
    return "".join(bit or "0" for bit in mask)


def _infer_or_mask(parsed: ParsedProblem) -> str | None:
    width = len(parsed.target_input)
    mask: list[str | None] = [None] * width
    for example in parsed.examples:
        for index, (input_bit, output_bit) in enumerate(zip(example.input_value, example.output_value)):
            if input_bit == "1":
                if output_bit != "1":
                    return None
                continue
            if mask[index] is not None and mask[index] != output_bit:
                return None
            mask[index] = output_bit
    if any(bit is None for bit in mask):
        return None
    return "".join(bit or "0" for bit in mask)


__all__ = ["enumerate_bit_candidates", "solve_bit_problem"]
