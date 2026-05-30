"""Distill verified symbolic solver outputs into SFT examples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from nemotron_engine.competition_sprint import CompetitionProblem, validate_competition_answer
from nemotron_engine.competition_sprint.competition_runner import _build_cipher_visible_vocabulary, _solve_problem
from nemotron_engine.core.schemas import stable_hash

from .sft_formatter import SFTExample, format_family_tagged_example


class SolverDistillationError(ValueError):
    """Raised when solver distillation state is inconsistent."""


@dataclass(frozen=True)
class SolverDistillationRecord:
    problem_id: str
    family: str
    answer: str
    solver_prediction: str | None
    solver_reason: str
    correct: bool
    diagnostics: Mapping[str, Any]
    record_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem_id", _text(self.problem_id, "problem_id"))
        object.__setattr__(self, "family", _text(self.family, "family"))
        object.__setattr__(self, "answer", validate_competition_answer(self.answer))
        if self.solver_prediction is not None:
            object.__setattr__(self, "solver_prediction", validate_competition_answer(self.solver_prediction))
        object.__setattr__(self, "solver_reason", _text(self.solver_reason, "solver_reason"))
        object.__setattr__(self, "correct", bool(self.correct))
        if self.correct and self.solver_prediction != self.answer:
            raise SolverDistillationError("correct records must match answer.")
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))
        _set_or_check_hash(self, "record_hash")


def distill_solver_predictions(problems: Sequence[CompetitionProblem], *, corpus_source_path: str | None = None) -> tuple[SolverDistillationRecord, ...]:
    """Run locked competition solvers and mark records that match train answers."""

    cipher_vocab = _build_cipher_visible_vocabulary(Path(corpus_source_path), include_sibling_train=False) if corpus_source_path else {}
    records: list[SolverDistillationRecord] = []
    for problem in problems:
        if problem.answer is None:
            raise SolverDistillationError("distillation requires labeled train problems.")
        prediction, reason, diagnostics = _solve_problem(problem, cipher_vocabulary=cipher_vocab)
        try:
            normalized_prediction = validate_competition_answer(prediction, expected_kind=problem.answer_kind) if prediction is not None else None
        except Exception:
            normalized_prediction = None
            reason = "invalid_solver_prediction"
        answer = validate_competition_answer(problem.answer, expected_kind=problem.answer_kind)
        records.append(
            SolverDistillationRecord(
                problem_id=problem.problem_id,
                family=problem.family,
                answer=answer,
                solver_prediction=normalized_prediction,
                solver_reason=reason,
                correct=normalized_prediction == answer,
                diagnostics=diagnostics,
            )
        )
    return tuple(records)


def build_solver_distilled_examples(problems: Sequence[CompetitionProblem], records: Sequence[SolverDistillationRecord]) -> tuple[SFTExample, ...]:
    """Build family-tagged SFT examples only for records where solver matched answer."""

    by_id = {record.problem_id: record for record in records}
    examples: list[SFTExample] = []
    for problem in problems:
        record = by_id.get(problem.problem_id)
        if record is not None and record.correct:
            examples.append(format_family_tagged_example(problem, source="solver_distilled"))
    return tuple(examples)


def _set_or_check_hash(instance: object, hash_field: str) -> None:
    expected = stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})
    current = getattr(instance, hash_field)
    if not current:
        object.__setattr__(instance, hash_field, expected)
    elif current != expected:
        raise SolverDistillationError(f"{hash_field} does not match payload.")


def _text(value: object, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise SolverDistillationError(f"{field_name} must be non-empty.")
    return text


__all__ = [
    "SolverDistillationError",
    "SolverDistillationRecord",
    "build_solver_distilled_examples",
    "distill_solver_predictions",
]
