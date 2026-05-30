"""Shared deterministic contracts for Sprint 1 symbolic solvers."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import re
from typing import Any, Mapping, Sequence

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.scoring.answer_extractor import normalize_answer_candidate


SPRINT_STATUSES = ("solved", "abstain", "disagreement", "error")
_INTEGER_RE = re.compile(r"^[+-]?\d+$")


class SprintSolverError(ValueError):
    """Raised when a sprint solver contract or hash invariant is violated."""


@dataclass(frozen=True)
class ExamplePair:
    input_value: str
    output_value: str
    pair_hash: str = ""

    def __post_init__(self) -> None:
        input_value = _require_text(self.input_value, "input_value")
        output_value = _require_text(self.output_value, "output_value")
        object.__setattr__(self, "input_value", input_value)
        object.__setattr__(self, "output_value", output_value)
        _set_or_check_hash(self, "pair_hash")


@dataclass(frozen=True)
class ParsedProblem:
    problem_id: str
    raw_prompt: str
    examples: tuple[ExamplePair, ...]
    target_input: str
    parser_warnings: tuple[str, ...] = ()
    parse_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem_id", _require_text(self.problem_id, "problem_id"))
        object.__setattr__(self, "raw_prompt", _require_text(self.raw_prompt, "raw_prompt"))
        examples = tuple(self.examples)
        if not examples:
            raise SprintSolverError("ParsedProblem requires at least one example.")
        for example in examples:
            if not isinstance(example, ExamplePair):
                raise SprintSolverError("examples must contain ExamplePair objects.")
            _validate_hash(example, "pair_hash")
        object.__setattr__(self, "examples", examples)
        object.__setattr__(self, "target_input", _require_text(self.target_input, "target_input"))
        object.__setattr__(self, "parser_warnings", tuple(str(item) for item in self.parser_warnings))
        _set_or_check_hash(self, "parse_hash")


@dataclass(frozen=True)
class SolverCandidate:
    solver_name: str
    example_predictions: tuple[str, ...]
    target_prediction: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    candidate_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "solver_name", _require_text(self.solver_name, "solver_name"))
        object.__setattr__(self, "example_predictions", tuple(str(item) for item in self.example_predictions))
        object.__setattr__(self, "target_prediction", _require_text(str(self.target_prediction), "target_prediction"))
        object.__setattr__(self, "metadata", dict(self.metadata))
        _set_or_check_hash(self, "candidate_hash")


@dataclass(frozen=True)
class SolverResult:
    status: str
    prediction: str | None
    verified_candidates: tuple[SolverCandidate, ...] = ()
    rejected_candidates: tuple[SolverCandidate, ...] = ()
    errors: tuple[str, ...] = ()
    result_hash: str = ""

    def __post_init__(self) -> None:
        status = str(self.status)
        if status not in SPRINT_STATUSES:
            raise SprintSolverError(f"status must be one of {SPRINT_STATUSES}.")
        if status == "solved" and not _has_text(self.prediction):
            raise SprintSolverError("solved SolverResult requires a prediction.")
        if status != "solved" and self.prediction is not None:
            raise SprintSolverError("non-solved SolverResult cannot carry a prediction.")
        verified = tuple(self.verified_candidates)
        rejected = tuple(self.rejected_candidates)
        for candidate in verified + rejected:
            if not isinstance(candidate, SolverCandidate):
                raise SprintSolverError("SolverResult candidates must be SolverCandidate objects.")
            _validate_hash(candidate, "candidate_hash")
        if len({id(candidate) for candidate in verified + rejected}) != len(verified) + len(rejected):
            raise SprintSolverError("verified_candidates and rejected_candidates must not overlap.")
        verified_predictions = tuple(candidate.target_prediction for candidate in verified)
        unique_verified_predictions = set(verified_predictions)
        if status == "solved":
            if not verified:
                raise SprintSolverError("solved SolverResult requires at least one verified candidate.")
            if any(not _has_text(prediction) for prediction in verified_predictions):
                raise SprintSolverError("solved SolverResult requires verified target predictions.")
            if len(unique_verified_predictions) != 1:
                raise SprintSolverError("solved SolverResult requires one unique verified target prediction.")
            if str(self.prediction) != next(iter(unique_verified_predictions)):
                raise SprintSolverError("solved SolverResult prediction must match verified target prediction.")
        if len(unique_verified_predictions) > 1 and status != "disagreement":
            raise SprintSolverError("disagreeing verified candidates require disagreement status.")
        if status == "disagreement" and self.prediction is not None:
            raise SprintSolverError("disagreement SolverResult cannot carry a prediction.")
        if status == "abstain" and verified:
            raise SprintSolverError("abstain SolverResult cannot have verified candidates.")
        if status == "error" and self.prediction is not None:
            raise SprintSolverError("error SolverResult cannot carry a prediction.")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "verified_candidates", verified)
        object.__setattr__(self, "rejected_candidates", rejected)
        object.__setattr__(self, "errors", tuple(str(item) for item in self.errors))
        if self.prediction is not None:
            object.__setattr__(self, "prediction", str(self.prediction))
        _set_or_check_hash(self, "result_hash")


@dataclass(frozen=True)
class RouterDecision:
    problem_id: str
    ordered_solvers: tuple[str, ...]
    matched_families: tuple[str, ...] = ()
    decision_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem_id", _require_text(self.problem_id, "problem_id"))
        ordered = tuple(_require_text(item, "ordered_solvers") for item in self.ordered_solvers)
        if not ordered:
            raise SprintSolverError("RouterDecision requires at least one solver.")
        if len(set(ordered)) != len(ordered):
            raise SprintSolverError("RouterDecision solvers must be unique.")
        if ordered[-1] != "dsl_synthesizer":
            raise SprintSolverError("dsl_synthesizer must be the final router fallback.")
        object.__setattr__(self, "ordered_solvers", ordered)
        object.__setattr__(self, "matched_families", tuple(str(item) for item in self.matched_families))
        _set_or_check_hash(self, "decision_hash")


def verify_candidate_on_examples(problem: ParsedProblem, candidate: SolverCandidate) -> bool:
    """Return whether a candidate exactly satisfies all examples and has a valid target."""

    _validate_hash(problem, "parse_hash")
    _validate_hash(candidate, "candidate_hash")
    if len(candidate.example_predictions) != len(problem.examples):
        return False
    for predicted, example in zip(candidate.example_predictions, problem.examples):
        if str(predicted) != example.output_value:
            return False
    return _target_prediction_is_valid(problem, candidate.target_prediction)


def choose_unique_verified_prediction(
    problem: ParsedProblem,
    candidates: Sequence[SolverCandidate],
) -> SolverResult:
    """Verify candidates and choose only a unique agreed prediction."""

    _validate_hash(problem, "parse_hash")
    verified: list[SolverCandidate] = []
    rejected: list[SolverCandidate] = []
    for candidate in candidates:
        _validate_hash(candidate, "candidate_hash")
        if verify_candidate_on_examples(problem, candidate):
            verified.append(candidate)
        else:
            rejected.append(candidate)

    if not verified:
        return SolverResult(status="abstain", prediction=None, rejected_candidates=tuple(rejected))

    predictions = {candidate.target_prediction for candidate in verified}
    if len(predictions) == 1:
        return SolverResult(status="solved", prediction=next(iter(predictions)), verified_candidates=tuple(verified), rejected_candidates=tuple(rejected))

    return SolverResult(
        status="disagreement",
        prediction=None,
        verified_candidates=tuple(verified),
        rejected_candidates=tuple(rejected),
        errors=("verified_candidates_disagree",),
    )


def _target_prediction_is_valid(problem: ParsedProblem, prediction: str) -> bool:
    if not _has_text(prediction):
        return False
    outputs = tuple(example.output_value for example in problem.examples)
    if outputs and all(_INTEGER_RE.fullmatch(item.strip()) for item in outputs):
        try:
            normalize_answer_candidate(prediction)
        except ValueError:
            return False
    return True


def _set_or_check_hash(instance: object, hash_field: str) -> None:
    expected = _payload_hash(instance, hash_field)
    current = getattr(instance, hash_field)
    if not current:
        object.__setattr__(instance, hash_field, expected)
    elif current != expected:
        raise SprintSolverError(f"{hash_field} does not match payload.")


def _validate_hash(instance: object, hash_field: str) -> None:
    if getattr(instance, hash_field) != _payload_hash(instance, hash_field):
        raise SprintSolverError(f"{hash_field} does not match payload.")


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


def _require_text(value: str, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise SprintSolverError(f"{field_name} must be a non-empty string.")
    return text


def _has_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


__all__ = [
    "ExamplePair",
    "ParsedProblem",
    "RouterDecision",
    "SPRINT_STATUSES",
    "SolverCandidate",
    "SolverResult",
    "SprintSolverError",
    "choose_unique_verified_prediction",
    "verify_candidate_on_examples",
]
