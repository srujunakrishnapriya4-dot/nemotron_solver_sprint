"""Prompt adapter for NVIDIA Nemotron competition CSV rows."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import dataclass, field, fields
import re
from typing import Any

from nemotron_engine.core.schemas import stable_hash

from .competition_answer_policy import infer_answer_kind, normalize_competition_answer


class CompetitionPromptAdapterError(ValueError):
    """Raised when a competition prompt cannot be parsed safely."""


_ARROW_RE = re.compile(r"^\s*(?P<input>.+?)\s*->\s*(?P<output>.+?)\s*$")
_EQUATION_RE = re.compile(r"^\s*(?P<input>.+?)\s*=\s*(?P<output>.+?)\s*$")
_UNIT_EXAMPLE_RE = re.compile(r"^\s*(?P<input>[+-]?\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z]+)\s+becomes\s+(?P<output>[+-]?\d+(?:\.\d+)?)\s*$")
_GRAVITY_EXAMPLE_RE = re.compile(r"^\s*For\s+t\s*=\s*(?P<input>[+-]?\d+(?:\.\d+)?)s,\s*distance\s*=\s*(?P<output>[+-]?\d+(?:\.\d+)?)\s*m\s*$", re.IGNORECASE)
_LEAKAGE_RE = re.compile(
    r"\b(expected[_\s]+answer|gold[_\s]+answer|target[_\s]+answer|correct[_\s]+answer)\s*[:=]|\bcorrect\s+answer\s*:",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CompetitionExample:
    input: str
    output: str
    example_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "input", _text(self.input, "input"))
        object.__setattr__(self, "output", _text(self.output, "output"))
        _set_or_check_hash(self, "example_hash")


@dataclass(frozen=True)
class CompetitionProblem:
    problem_id: str
    raw_prompt: str
    family: str
    examples: tuple[CompetitionExample, ...]
    target_input: str
    answer: str | None = None
    answer_kind: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    problem_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem_id", _text(self.problem_id, "problem_id"))
        object.__setattr__(self, "raw_prompt", _text(self.raw_prompt, "raw_prompt"))
        object.__setattr__(self, "family", _text(self.family, "family"))
        examples = tuple(self.examples)
        if not examples:
            raise CompetitionPromptAdapterError("competition problem requires examples.")
        for example in examples:
            if not isinstance(example, CompetitionExample):
                raise CompetitionPromptAdapterError("examples must contain CompetitionExample objects.")
            _validate_hash(example, "example_hash")
        object.__setattr__(self, "examples", examples)
        object.__setattr__(self, "target_input", _text(self.target_input, "target_input"))
        if self.answer is not None:
            answer = normalize_competition_answer(self.answer)
            object.__setattr__(self, "answer", answer)
            object.__setattr__(self, "answer_kind", infer_answer_kind(answer))
        elif self.answer_kind is not None:
            object.__setattr__(self, "answer_kind", str(self.answer_kind))
        object.__setattr__(self, "metadata", dict(self.metadata))
        _set_or_check_hash(self, "problem_hash")


def parse_competition_prompt(problem_id: str, prompt: str, answer: str | None = None) -> CompetitionProblem:
    """Parse visible examples and target input from a competition prompt."""

    raw_prompt = _text(prompt, "prompt")
    if _LEAKAGE_RE.search(raw_prompt):
        raise CompetitionPromptAdapterError("prompt contains forbidden answer leakage.")
    family = _detect_family(raw_prompt)
    examples = _parse_examples(raw_prompt, family)
    target = _parse_target(raw_prompt, family)
    if not examples:
        raise CompetitionPromptAdapterError("no visible examples parsed.")
    if target is None:
        raise CompetitionPromptAdapterError("target input not parsed.")
    return CompetitionProblem(
        problem_id=problem_id,
        raw_prompt=raw_prompt,
        family=family,
        examples=tuple(examples),
        target_input=target,
        answer=answer,
        metadata={"example_count": len(examples)},
    )


def load_competition_csv(path: str, *, has_answers: bool) -> tuple[CompetitionProblem, ...]:
    """Load official competition CSV data into hash-checked problems."""

    problems: list[CompetitionProblem] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            if "id" not in row or "prompt" not in row:
                raise CompetitionPromptAdapterError("CSV must contain id and prompt columns.")
            answer = row.get("answer") if has_answers else None
            try:
                problems.append(parse_competition_prompt(str(row["id"]), str(row["prompt"]), answer))
            except Exception as exc:
                raise CompetitionPromptAdapterError(f"row {index} failed to parse: {exc}") from exc
    return tuple(problems)


def _detect_family(prompt: str) -> str:
    lowered = prompt.lower()
    if "bit manipulation" in lowered or "binary" in lowered and "xor" in lowered:
        return "bit_manipulation"
    if "secret encryption" in lowered or "decrypt the following text" in lowered or "cipher" in lowered:
        return "cipher_text"
    if "different numeral system" in lowered or "wonderland numeral system" in lowered or "roman" in lowered:
        return "roman_numeral"
    if "unit conversion" in lowered or "convert the following measurement" in lowered or "unit_conv" in lowered:
        return "unit_conversion"
    if "gravitational constant" in lowered or "gravity" in lowered:
        return "gravity_numeric"
    if ("transformation rules" in lowered and "equation" in lowered) or "symbol_digit" in lowered:
        return "equation_symbolic"
    return "unknown"


def _parse_examples(prompt: str, family: str) -> list[CompetitionExample]:
    examples: list[CompetitionExample] = []
    for raw_line in prompt.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("now,"):
            continue
        lowered = line.lower()
        if "example" in lowered or line.endswith(":"):
            continue
        if family in {"bit_manipulation", "cipher_text", "roman_numeral"}:
            match = _ARROW_RE.fullmatch(line)
            if match:
                examples.append(CompetitionExample(match.group("input").strip(), match.group("output").strip()))
        elif family == "unit_conversion":
            match = _UNIT_EXAMPLE_RE.fullmatch(line)
            if match:
                examples.append(CompetitionExample(f"{match.group('input')} {match.group('unit')}", match.group("output")))
        elif family == "gravity_numeric":
            match = _GRAVITY_EXAMPLE_RE.fullmatch(line)
            if match:
                examples.append(CompetitionExample(match.group("input"), match.group("output")))
        elif family == "equation_symbolic":
            match = _EQUATION_RE.fullmatch(line)
            if match:
                examples.append(CompetitionExample(match.group("input").strip(" `"), match.group("output").strip(" `")))
    return examples


def _parse_target(prompt: str, family: str) -> str | None:
    patterns = {
        "bit_manipulation": (r"Now,\s*determine\s+the\s+output\s+for:\s*(?P<target>.+?)\s*$",),
        "cipher_text": (r"Now,\s*decrypt\s+the\s+following\s+text:\s*(?P<target>.+?)\s*$",),
        "roman_numeral": (r"Now,\s*write\s+the\s+number\s+(?P<target>[+-]?\d+)\s+in\s+the\s+Wonderland\s+numeral\s+system\.",),
        "unit_conversion": (r"Now,\s*convert\s+the\s+following\s+measurement:\s*(?P<target>.+?)\s*$",),
        "gravity_numeric": (r"Now,\s*determine\s+the\s+falling\s+distance\s+for\s+t\s*=\s*(?P<target>[+-]?\d+(?:\.\d+)?)s",),
        "equation_symbolic": (r"Now,\s*determine\s+the\s+result\s+for:\s*(?P<target>.+?)\s*$",),
    }
    for pattern in patterns.get(family, ()):
        match = re.search(pattern, prompt, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group("target").strip(" `")
    return None


def _text(value: Any, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise CompetitionPromptAdapterError(f"{field_name} must be non-empty.")
    return text


def _set_or_check_hash(instance: object, hash_field: str) -> None:
    expected = _payload_hash(instance, hash_field)
    current = getattr(instance, hash_field)
    if not current:
        object.__setattr__(instance, hash_field, expected)
    elif current != expected:
        raise CompetitionPromptAdapterError(f"{hash_field} does not match payload.")


def _validate_hash(instance: object, hash_field: str) -> None:
    if getattr(instance, hash_field) != _payload_hash(instance, hash_field):
        raise CompetitionPromptAdapterError(f"{hash_field} does not match payload.")


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


__all__ = [
    "CompetitionExample",
    "CompetitionProblem",
    "CompetitionPromptAdapterError",
    "load_competition_csv",
    "parse_competition_prompt",
]
