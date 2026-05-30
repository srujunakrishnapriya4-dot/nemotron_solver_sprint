from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.core.schemas import Program, ProgramExecutionError, ProgramStep, ProgramStepKind, SchemaValidationError  # noqa: E402
from nemotron_engine.programs.executor import _execute_step, execute_program  # noqa: E402


def _program(*steps: ProgramStep, output_key: str) -> Program:
    return Program("p", tuple(steps), output_key)


def test_parse_int_add_const_raw() -> None:
    program = _program(
        ProgramStep(ProgramStepKind.PARSE, "parse_int", output_key="x"),
        ProgramStep(ProgramStepKind.TRANSFORM, "add_const", {"const": 1}, "y"),
        ProgramStep(ProgramStepKind.FORMAT, "raw", output_key="out"),
        output_key="out",
    )
    assert execute_program(program, "1").output_value == "2"


def test_digits_sum_bit_not_reverse_zero_pad_and_symbol_bijection() -> None:
    digit_sum = _program(
        ProgramStep(ProgramStepKind.PARSE, "parse_digits", output_key="d"),
        ProgramStep(ProgramStepKind.TRANSFORM, "digit_sum", output_key="s"),
        ProgramStep(ProgramStepKind.FORMAT, "raw", output_key="out"),
        output_key="out",
    )
    bit_not = _program(
        ProgramStep(ProgramStepKind.PARSE, "parse_bitstring", output_key="b"),
        ProgramStep(ProgramStepKind.TRANSFORM, "bit_not", output_key="n"),
        ProgramStep(ProgramStepKind.FORMAT, "bitstring", output_key="out"),
        output_key="out",
    )
    reverse = _program(ProgramStep(ProgramStepKind.TRANSFORM, "reverse", output_key="out"), output_key="out")
    pad = _program(ProgramStep(ProgramStepKind.FORMAT, "zero_pad", {"width": 3}, "out"), output_key="out")
    symbols = _program(
        ProgramStep(ProgramStepKind.PARSE, "parse_symbols", output_key="s"),
        ProgramStep(ProgramStepKind.TRANSFORM, "symbol_bijection", {"mapping": {"A": "Z"}}, "mapped"),
        ProgramStep(ProgramStepKind.FORMAT, "raw", output_key="out"),
        output_key="out",
    )

    assert execute_program(digit_sum, "123").output_value == "6"
    assert execute_program(bit_not, "101").output_value == "010"
    assert execute_program(reverse, "abc").output_value == "cba"
    assert execute_program(pad, "7").output_value == "007"
    assert execute_program(symbols, "A").output_value == "Z"
    assert _execute_step("reverse", {}, {"current": [1, 2, 3]}) == [3, 2, 1]


def test_invalid_primitive_rejected_and_invalid_args_error() -> None:
    with pytest.raises(SchemaValidationError):
        ProgramStep(ProgramStepKind.PARSE, "eval", output_key="x")
    with pytest.raises(ProgramExecutionError):
        _execute_step("add_const", {}, {"current": "1"})
    with pytest.raises(ProgramExecutionError):
        _execute_step("reverse", {}, {"current": 7})
    with pytest.raises(ProgramExecutionError):
        _execute_step("digit_i", {"index": 0}, {"current": 7})
    with pytest.raises(ProgramExecutionError):
        _execute_step("zero_pad", {"width": -1}, {"current": "7"})
    with pytest.raises(ProgramExecutionError):
        _execute_step("digit_i", {"index": "0"}, {"current": "7"})


def test_pass25_program_step_accepts_extended_primitives() -> None:
    assert ProgramStep(ProgramStepKind.PARSE, "parse_binary_int_expr", output_key="expr").primitive == "parse_binary_int_expr"
    assert ProgramStep(ProgramStepKind.TRANSFORM, "mul_const", {"const": 2}, "out").primitive == "mul_const"
    assert ProgramStep(ProgramStepKind.TRANSFORM, "affine_small", {"a": 2, "b": 1}, "out").primitive == "affine_small"
    assert ProgramStep(ProgramStepKind.TRANSFORM, "binary_op", {"op": "add"}, "out").primitive == "binary_op"


def test_parse_binary_int_expr_parses_arithmetic_and_symbolic_operator() -> None:
    assert _execute_step("parse_binary_int_expr", {}, {"current": "2 + 3"}) == {"lhs": 2, "op": "+", "rhs": 3}
    assert _execute_step("parse_binary_int_expr", {}, {"current": "9 @ 4"}) == {"lhs": 9, "op": "@", "rhs": 4}


def test_parse_binary_int_expr_rejects_malformed_float_and_extra_tokens() -> None:
    with pytest.raises(ProgramExecutionError):
        _execute_step("parse_binary_int_expr", {}, {"current": "2 +"})
    with pytest.raises(ProgramExecutionError):
        _execute_step("parse_binary_int_expr", {}, {"current": "2.0 + 3"})
    with pytest.raises(ProgramExecutionError):
        _execute_step("parse_binary_int_expr", {}, {"current": "2 + 3 + 4"})


def test_mul_const_and_affine_small_success_and_failures() -> None:
    assert _execute_step("mul_const", {"const": 4}, {"current": "3"}) == 12
    assert _execute_step("affine_small", {"a": 2, "b": -1}, {"current": "4"}) == 7

    with pytest.raises(ProgramExecutionError):
        _execute_step("mul_const", {"const": "4"}, {"current": "3"})
    with pytest.raises(ProgramExecutionError):
        _execute_step("mul_const", {"const": 4}, {"current": "3.5"})
    with pytest.raises(ProgramExecutionError):
        _execute_step("affine_small", {"a": "2", "b": 1}, {"current": "4"})
    with pytest.raises(ProgramExecutionError):
        _execute_step("affine_small", {"a": 2, "b": 1.5}, {"current": "4"})
    with pytest.raises(ProgramExecutionError):
        _execute_step("affine_small", {"a": 11, "b": 0}, {"current": "4"})
    with pytest.raises(ProgramExecutionError):
        _execute_step("affine_small", {"a": 1, "b": 1001}, {"current": "4"})


def test_binary_op_success_and_failures() -> None:
    parsed = {"lhs": 9, "op": "@", "rhs": 4}

    assert _execute_step("binary_op", {"op": "add"}, {"current": parsed}) == 13
    assert _execute_step("binary_op", {"op": "sub"}, {"current": parsed}) == 5
    assert _execute_step("binary_op", {"op": "abs_sub"}, {"current": parsed}) == 5
    assert _execute_step("binary_op", {"op": "mul_small"}, {"current": parsed}) == 36

    with pytest.raises(ProgramExecutionError):
        _execute_step("binary_op", {"op": "div"}, {"current": parsed})
    with pytest.raises(ProgramExecutionError):
        _execute_step("binary_op", {"op": "add"}, {"current": {"lhs": 1, "rhs": 2}})
    with pytest.raises(ProgramExecutionError):
        _execute_step("binary_op", {"op": "add"}, {"current": {"lhs": "1", "op": "+", "rhs": 2}})
    with pytest.raises(ProgramExecutionError):
        _execute_step("binary_op", {"op": "mul_small"}, {"current": {"lhs": 10001, "op": "*", "rhs": 2}})
    with pytest.raises(ProgramExecutionError):
        _execute_step("binary_op", {"op": "mul_small"}, {"current": {"lhs": 1000, "op": "*", "rhs": 1001}})


def test_pass25_executor_trace_error_for_invalid_extended_program() -> None:
    program = _program(
        ProgramStep(ProgramStepKind.PARSE, "parse_binary_int_expr", output_key="expr"),
        ProgramStep(ProgramStepKind.TRANSFORM, "binary_op", {"op": "mul_small"}, "out"),
        output_key="out",
    )
    trace = execute_program(program, "10001 * 2")

    assert trace.output_value is None
    assert trace.error is not None


def test_no_silent_none_output() -> None:
    program = _program(ProgramStep(ProgramStepKind.SELECT, "digit_i", {"index": 5}, "out"), output_key="out")
    trace = execute_program(program, "12")

    assert trace.output_value is None
    assert trace.error is not None
