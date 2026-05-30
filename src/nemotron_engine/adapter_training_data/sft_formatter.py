"""SFT chat formatting helpers for adapter training."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from nemotron_engine.competition_sprint import CompetitionProblem, validate_competition_answer
from nemotron_engine.core.schemas import stable_hash


class SFTFormatterError(ValueError):
    """Raised when an SFT row is invalid."""


@dataclass(frozen=True)
class SFTExample:
    problem_id: str
    family: str
    answer_kind: str
    source: str
    messages: tuple[Mapping[str, str], ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    example_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem_id", _text(self.problem_id, "problem_id"))
        object.__setattr__(self, "family", _text(self.family, "family"))
        object.__setattr__(self, "answer_kind", _text(self.answer_kind, "answer_kind"))
        object.__setattr__(self, "source", _text(self.source, "source"))
        messages = tuple(dict(message) for message in self.messages)
        if len(messages) != 2:
            raise SFTFormatterError("SFTExample requires exactly user and assistant messages.")
        if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
            raise SFTFormatterError("messages must be user then assistant.")
        for message in messages:
            if not _text(message.get("content"), "message content"):
                raise SFTFormatterError("message content must be non-empty.")
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "metadata", dict(self.metadata))
        _set_or_check_hash(self, "example_hash")

    def to_json_obj(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


def format_direct_answer_example(problem: CompetitionProblem, *, source: str = "direct_answer") -> SFTExample:
    """Format a direct-answer training example."""

    answer = _problem_answer(problem)
    return SFTExample(
        problem_id=problem.problem_id,
        family=problem.family,
        answer_kind=problem.answer_kind or "generic_string",
        source=source,
        messages=(
            {"role": "user", "content": f"{problem.raw_prompt}\n\nReturn only the final answer in \\boxed{{}}."},
            {"role": "assistant", "content": f"\\boxed{{{answer}}}"},
        ),
        metadata={"target_input": problem.target_input, "example_count": len(problem.examples)},
    )


def format_family_tagged_example(problem: CompetitionProblem, *, source: str = "family_tagged") -> SFTExample:
    """Format a family-tagged direct-answer training example."""

    answer = _problem_answer(problem)
    return SFTExample(
        problem_id=problem.problem_id,
        family=problem.family,
        answer_kind=problem.answer_kind or "generic_string",
        source=source,
        messages=(
            {"role": "user", "content": f"Family: {problem.family}.\n{problem.raw_prompt}\n\nReturn only the final answer in \\boxed{{}}."},
            {"role": "assistant", "content": f"\\boxed{{{answer}}}"},
        ),
        metadata={"target_input": problem.target_input, "example_count": len(problem.examples)},
    )


def _problem_answer(problem: CompetitionProblem) -> str:
    if problem.answer is None:
        raise SFTFormatterError("problem must include an answer for SFT formatting.")
    return validate_competition_answer(problem.answer, expected_kind=problem.answer_kind)


def _set_or_check_hash(instance: object, hash_field: str) -> None:
    expected = stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})
    current = getattr(instance, hash_field)
    if not current:
        object.__setattr__(instance, hash_field, expected)
    elif current != expected:
        raise SFTFormatterError(f"{hash_field} does not match payload.")


def _text(value: object, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise SFTFormatterError(f"{field_name} must be non-empty.")
    return text


__all__ = ["SFTExample", "SFTFormatterError", "format_direct_answer_example", "format_family_tagged_example"]
