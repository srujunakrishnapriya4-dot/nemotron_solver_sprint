"""Leakage-safe parser for Sprint 1 example-to-target prompts."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from .solver_base import ExamplePair, ParsedProblem, SprintSolverError


FORBIDDEN_INFERENCE_FIELDS = (
    "expected_answer",
    "gold",
    "target_answer",
    "correct_answer",
    "label",
    "is_correct",
    "correctness",
)

_PROMPT_KEYS = ("prompt", "question", "text", "problem", "raw_prompt")
_INPUT_OUTPUT_RE = re.compile(r"^\s*Input\s*:\s*(?P<input>.+?)\s*->\s*Output\s*:\s*(?P<output>.+?)\s*$", re.IGNORECASE)
_TARGET_RE = re.compile(r"^\s*Target\s*:\s*(?P<input>.+?)\s*->\s*(?P<output>.+?)\s*$", re.IGNORECASE)
_ARROW_RE = re.compile(r"^\s*(?P<input>.+?)\s*->\s*(?P<output>.+?)\s*$")
_FORBIDDEN_TEXT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bexpected[_\s]+answer\s*[:=]",
        r"\bgold[_\s]+answer\s*[:=]",
        r"\bgold\s*[:=]",
        r"\btarget[_\s]+answer\s*[:=]",
        r"\bcorrect[_\s]+answer\s*[:=]",
        r"\banswer\s+is\b",
        r"\blabel\s*[:=]",
        r"\bis_correct\b",
        r"\bcorrectness\b",
    )
)


def parse_problem(payload: str | Mapping[str, Any], *, problem_id: str | None = None) -> ParsedProblem:
    """Parse examples and a target without consulting forbidden answer fields."""

    prompt, resolved_problem_id = _extract_prompt(payload, problem_id=problem_id)
    _reject_prompt_leakage(prompt)

    examples: list[ExamplePair] = []
    target_input: str | None = None

    for line_number, raw_line in enumerate(prompt.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if "->" in line and line.count("->") != 1:
            raise SprintSolverError(f"malformed arrow on line {line_number}.")

        parsed = _parse_line(line)
        if parsed is None:
            _reject_malformed_labeled_line(line, line_number)
            continue

        kind, input_value, output_value = parsed
        input_value = input_value.strip()
        output_value = output_value.strip()
        if not input_value:
            raise SprintSolverError(f"empty input on line {line_number}.")
        if not output_value:
            raise SprintSolverError(f"empty output on line {line_number}.")
        if output_value == "?":
            if target_input is not None:
                raise SprintSolverError("multiple targets are not allowed.")
            target_input = input_value
        elif kind == "target":
            raise SprintSolverError("target with supplied output is forbidden.")
        else:
            examples.append(ExamplePair(input_value=input_value, output_value=output_value))

    if not examples:
        raise SprintSolverError("no examples found.")
    if target_input is None:
        raise SprintSolverError("no target found.")

    return ParsedProblem(
        problem_id=resolved_problem_id,
        raw_prompt=prompt,
        examples=tuple(examples),
        target_input=target_input,
    )


def forbidden_metadata_keys(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Return forbidden row-level metadata keys without exposing their values."""

    keys = {str(key).lower() for key in payload}
    return tuple(field for field in FORBIDDEN_INFERENCE_FIELDS if field in keys)


def _extract_prompt(payload: str | Mapping[str, Any], *, problem_id: str | None) -> tuple[str, str]:
    if isinstance(payload, Mapping):
        prompt = None
        for key in _PROMPT_KEYS:
            if key in payload:
                prompt = payload[key]
                break
        if prompt is None:
            raise SprintSolverError(f"payload must include one of {_PROMPT_KEYS}.")
        resolved_problem_id = str(problem_id or payload.get("id") or payload.get("problem_id") or "sprint-problem").strip()
    else:
        prompt = payload
        resolved_problem_id = str(problem_id or "sprint-problem").strip()
    if not resolved_problem_id:
        raise SprintSolverError("problem_id must be non-empty.")
    prompt_text = str(prompt).strip()
    if not prompt_text:
        raise SprintSolverError("prompt must be non-empty.")
    return prompt_text, resolved_problem_id


def _reject_prompt_leakage(prompt: str) -> None:
    for pattern in _FORBIDDEN_TEXT_PATTERNS:
        if pattern.search(prompt):
            raise SprintSolverError("prompt contains forbidden target/gold leakage.")


def _parse_line(line: str) -> tuple[str, str, str] | None:
    match = _TARGET_RE.fullmatch(line)
    if match:
        return ("target", match.group("input"), match.group("output"))
    match = _INPUT_OUTPUT_RE.fullmatch(line)
    if match:
        return ("pair", match.group("input"), match.group("output"))
    match = _ARROW_RE.fullmatch(line)
    if match:
        return ("pair", match.group("input"), match.group("output"))
    return None


def _reject_malformed_labeled_line(line: str, line_number: int) -> None:
    lowered = line.lower()
    if "->" in line or lowered.startswith(("input:", "output:", "target:")):
        raise SprintSolverError(f"malformed arrow on line {line_number}.")


__all__ = ["FORBIDDEN_INFERENCE_FIELDS", "forbidden_metadata_keys", "parse_problem"]
