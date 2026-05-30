from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.core.schemas import ProgramStepKind, SchemaValidationError  # noqa: E402
from nemotron_engine.programs.primitives import (  # noqa: E402
    make_format_step,
    make_parse_step,
    make_program,
    make_select_identity_step,
    make_transform_step,
)


def test_make_program_builds_deterministic_program() -> None:
    steps = (
        make_parse_step("parse_int", output_key="x"),
        make_transform_step("add_const", {"const": 1}, "y"),
        make_format_step("raw", output_key="out"),
    )

    assert make_program("p", steps, output_key="out").program_hash == make_program("p", steps, output_key="out").program_hash


def test_invalid_primitive_rejected() -> None:
    with pytest.raises(SchemaValidationError):
        make_transform_step("eval", output_key="out")


def test_output_key_required() -> None:
    with pytest.raises(SchemaValidationError):
        make_format_step("raw", output_key="")


def test_primitive_helpers_create_valid_steps() -> None:
    assert make_parse_step("parse_int").kind is ProgramStepKind.PARSE
    assert make_select_identity_step().kind is ProgramStepKind.SELECT
    assert make_transform_step("reverse").kind is ProgramStepKind.TRANSFORM
    assert make_format_step("raw").kind is ProgramStepKind.FORMAT
