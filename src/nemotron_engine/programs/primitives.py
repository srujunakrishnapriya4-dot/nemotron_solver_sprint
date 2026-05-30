"""Safe construction helpers for Pass 3 program templates."""

from __future__ import annotations

from typing import Any, Mapping

from nemotron_engine.core.schemas import Program, ProgramStep, ProgramStepKind


def make_parse_step(primitive: str, *, output_key: str = "parsed", args: Mapping[str, Any] | None = None) -> ProgramStep:
    return ProgramStep(ProgramStepKind.PARSE, primitive, dict(args or {}), output_key)


def make_select_identity_step(*, output_key: str = "value", input_key: str | None = None) -> ProgramStep:
    args = {"input_key": input_key} if input_key is not None else {}
    return ProgramStep(ProgramStepKind.SELECT, "identity", args, output_key)


def make_transform_step(
    primitive: str,
    args: Mapping[str, Any] | None = None,
    output_key: str = "value",
) -> ProgramStep:
    return ProgramStep(ProgramStepKind.TRANSFORM, primitive, dict(args or {}), output_key)


def make_format_step(primitive: str = "raw", *, output_key: str = "out", args: Mapping[str, Any] | None = None) -> ProgramStep:
    return ProgramStep(ProgramStepKind.FORMAT, primitive, dict(args or {}), output_key)


def make_program(program_id: str, steps: tuple[ProgramStep, ...] | list[ProgramStep], *, output_key: str = "out") -> Program:
    return Program(program_id=program_id, steps=tuple(steps), output_key=output_key)


__all__ = [
    "make_format_step",
    "make_parse_step",
    "make_program",
    "make_select_identity_step",
    "make_transform_step",
]
