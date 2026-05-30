"""Family labeling for competition adapter-training rows."""

from __future__ import annotations

from dataclasses import dataclass, fields

from nemotron_engine.competition_sprint import CompetitionProblem, parse_competition_prompt
from nemotron_engine.core.schemas import stable_hash


class AdapterFamilyLabelerError(ValueError):
    """Raised when a family label cannot be created safely."""


@dataclass(frozen=True)
class FamilyLabel:
    problem_id: str
    family: str
    answer_kind: str | None
    example_count: int
    label_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem_id", _text(self.problem_id, "problem_id"))
        object.__setattr__(self, "family", _text(self.family, "family"))
        if self.answer_kind is not None:
            object.__setattr__(self, "answer_kind", _text(self.answer_kind, "answer_kind"))
        count = int(self.example_count)
        if count <= 0:
            raise AdapterFamilyLabelerError("example_count must be positive.")
        object.__setattr__(self, "example_count", count)
        _set_or_check_hash(self, "label_hash")


def label_competition_family(problem_or_prompt: CompetitionProblem | str, *, problem_id: str = "problem", answer: str | None = None) -> FamilyLabel:
    """Return a deterministic family label from a parsed problem or raw prompt."""

    problem = problem_or_prompt if isinstance(problem_or_prompt, CompetitionProblem) else parse_competition_prompt(problem_id, problem_or_prompt, answer)
    return FamilyLabel(
        problem_id=problem.problem_id,
        family=problem.family,
        answer_kind=problem.answer_kind,
        example_count=len(problem.examples),
    )


def _set_or_check_hash(instance: object, hash_field: str) -> None:
    expected = stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})
    current = getattr(instance, hash_field)
    if not current:
        object.__setattr__(instance, hash_field, expected)
    elif current != expected:
        raise AdapterFamilyLabelerError(f"{hash_field} does not match payload.")


def _text(value: object, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise AdapterFamilyLabelerError(f"{field_name} must be non-empty.")
    return text


__all__ = ["AdapterFamilyLabelerError", "FamilyLabel", "label_competition_family"]
