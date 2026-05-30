"""Small typed DSL declarations for Pass 2 programs."""

from __future__ import annotations

from nemotron_engine.core.schemas import ALLOWED_PRIMITIVES, Program, ProgramStep, ProgramStepKind, SchemaValidationError


def validate_program(program: Program) -> Program:
    if not program.steps:
        raise SchemaValidationError("Program must contain at least one step.")
    for step in program.steps:
        if step.primitive not in ALLOWED_PRIMITIVES[step.kind]:
            raise SchemaValidationError(f"Invalid primitive {step.primitive!r} for {step.kind.value}.")
    if not program.output_key:
        raise SchemaValidationError("Program output_key must be non-empty.")
    return program


__all__ = [
    "ALLOWED_PRIMITIVES",
    "Program",
    "ProgramStep",
    "ProgramStepKind",
    "validate_program",
]
