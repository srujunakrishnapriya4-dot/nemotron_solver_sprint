"""Deterministic safe executor for the Pass 2 DSL."""

from __future__ import annotations

from typing import Any, Mapping

from nemotron_engine.core.schemas import ExecutionTrace, Program, ProgramExecutionError


def execute_program(program: Program, raw_input: str) -> ExecutionTrace:
    context: dict[str, Any] = {"raw": str(raw_input), "current": str(raw_input)}
    try:
        for step in program.steps:
            value = _execute_step(step.primitive, step.args, context)
            if value is None:
                raise ProgramExecutionError(f"Primitive {step.primitive} returned None.")
            context[step.output_key] = value
            context["current"] = value
        if program.output_key not in context:
            raise ProgramExecutionError(f"Missing program output key: {program.output_key}")
        output = context[program.output_key]
        if output is None:
            raise ProgramExecutionError("Program output cannot be None.")
        return ExecutionTrace(input_value=str(raw_input), output_value=str(output), intermediate_values=context)
    except ProgramExecutionError as exc:
        return ExecutionTrace(input_value=str(raw_input), output_value=None, intermediate_values=context, error=str(exc))


def _execute_step(primitive: str, args: Mapping[str, Any], context: dict[str, Any]) -> Any:
    try:
        return _execute_step_checked(primitive, args, context)
    except ProgramExecutionError:
        raise
    except (TypeError, ValueError, IndexError, KeyError) as exc:
        raise ProgramExecutionError(f"Primitive {primitive} failed with invalid input or args.") from exc


def _execute_step_checked(primitive: str, args: Mapping[str, Any], context: dict[str, Any]) -> Any:
    value = _input_value(args, context)
    if primitive == "parse_int":
        return _parse_int(value)
    if primitive == "parse_digits":
        text = str(value)
        if not text.isdigit():
            raise ProgramExecutionError("parse_digits requires only digits.")
        return [int(char) for char in text]
    if primitive == "parse_bitstring":
        text = str(value)
        if not text or any(char not in "01" for char in text):
            raise ProgramExecutionError("parse_bitstring requires a non-empty 0/1 string.")
        return text
    if primitive == "parse_symbols":
        text = str(value)
        if any(char.isspace() for char in text):
            raise ProgramExecutionError("parse_symbols does not allow whitespace.")
        return list(text)
    if primitive == "parse_token_sequence":
        tokens = str(value).split()
        if not tokens:
            raise ProgramExecutionError("parse_token_sequence produced no tokens.")
        return tokens
    if primitive == "parse_binary_int_expr":
        return _parse_binary_int_expr(value)
    if primitive == "identity":
        return value
    if primitive in {"lhs", "rhs"}:
        parts = str(value).split("->", 1)
        if len(parts) != 2:
            raise ProgramExecutionError(f"{primitive} requires an arrow expression.")
        return parts[0].strip() if primitive == "lhs" else parts[1].strip()
    if primitive in {"operand_i", "digit_i", "bit_i", "symbol_i"}:
        index = _required_int(args, "index")
        if not isinstance(value, (str, list, tuple)):
            raise ProgramExecutionError(f"{primitive} requires a sequence input.")
        sequence = list(value)
        try:
            return sequence[index]
        except IndexError as exc:
            raise ProgramExecutionError(f"{primitive} index out of range.") from exc
    if primitive == "add_const":
        return _parse_int(value) + _required_int(args, "const")
    if primitive == "sub_const":
        return _parse_int(value) - _required_int(args, "const")
    if primitive == "mul_const":
        return _parse_int(value) * _required_int(args, "const")
    if primitive == "affine_small":
        x_value = _parse_int(value)
        a_value = _required_int(args, "a")
        b_value = _required_int(args, "b")
        if abs(a_value) > 10:
            raise ProgramExecutionError("affine_small requires abs(a) <= 10.")
        if abs(b_value) > 1000:
            raise ProgramExecutionError("affine_small requires abs(b) <= 1000.")
        return a_value * x_value + b_value
    if primitive == "reverse":
        if isinstance(value, str):
            return value[::-1]
        if isinstance(value, (list, tuple)):
            return list(reversed(value))
        raise ProgramExecutionError("reverse requires a string or sequence input.")
    if primitive == "digit_sum":
        digits = value if isinstance(value, list) else _execute_step("parse_digits", {}, {"current": value})
        return sum(int(item) for item in digits)
    if primitive == "concat":
        values = args.get("values")
        if values is None:
            keys = args.get("keys")
            if not isinstance(keys, list):
                raise ProgramExecutionError("concat requires values or keys list.")
            values = [_lookup(context, str(key)) for key in keys]
        if not isinstance(values, list):
            raise ProgramExecutionError("concat values must be a list.")
        return "".join(str(item) for item in values)
    if primitive == "bit_not":
        text = str(value)
        if not text or any(char not in "01" for char in text):
            raise ProgramExecutionError("bit_not requires a bitstring.")
        return "".join("1" if char == "0" else "0" for char in text)
    if primitive == "binary_op":
        return _apply_binary_op(value, args)
    if primitive == "symbol_bijection":
        mapping = _required_mapping(args, "mapping")
        if not isinstance(value, (str, list, tuple)):
            raise ProgramExecutionError("symbol_bijection requires a string or sequence input.")
        return [_map_symbol(str(item), mapping) for item in list(value)]
    if primitive == "raw":
        return "".join(str(item) for item in value) if isinstance(value, list) else str(value)
    if primitive == "zero_pad":
        width = _required_int(args, "width")
        if width < 0:
            raise ProgramExecutionError("zero_pad width must be non-negative.")
        return str(value).zfill(width)
    if primitive == "bitstring":
        text = str(value)
        if not text or any(char not in "01" for char in text):
            raise ProgramExecutionError("bitstring format requires only 0/1.")
        return text
    if primitive == "encode_symbols":
        mapping = _required_mapping(args, "mapping")
        return "".join(_map_symbol(char, mapping) for char in str(value))
    raise ProgramExecutionError(f"Unsupported primitive: {primitive}")


def _input_value(args: Mapping[str, Any], context: dict[str, Any]) -> Any:
    if "value" in args:
        return args["value"]
    if "input_key" in args:
        return _lookup(context, str(args["input_key"]))
    return context["current"]


def _lookup(context: dict[str, Any], key: str) -> Any:
    if key not in context:
        raise ProgramExecutionError(f"Missing input key: {key}")
    return context[key]


def _parse_int(value: Any) -> int:
    text = str(value)
    if not text or (text[0] in "+-" and not text[1:].isdigit()) or (text[0] not in "+-" and not text.isdigit()):
        raise ProgramExecutionError(f"Expected integer value, got {value!r}.")
    return int(text)


def _parse_binary_int_expr(value: Any) -> dict[str, int | str]:
    text = str(value).strip()
    parts = text.split()
    if len(parts) != 3:
        raise ProgramExecutionError("parse_binary_int_expr requires exactly '<int> <op> <int>'.")
    lhs_text, operator, rhs_text = parts
    if not operator:
        raise ProgramExecutionError("parse_binary_int_expr requires an operator token.")
    return {"lhs": _parse_int(lhs_text), "op": operator, "rhs": _parse_int(rhs_text)}


def _apply_binary_op(value: Any, args: Mapping[str, Any]) -> int:
    lhs, _operator, rhs = _binary_expr_parts(value)
    op = args.get("op")
    if not isinstance(op, str):
        raise ProgramExecutionError("binary_op requires string arg: op")
    if op == "add":
        return lhs + rhs
    if op == "sub":
        return lhs - rhs
    if op == "abs_sub":
        return abs(lhs - rhs)
    if op == "mul_small":
        product = lhs * rhs
        if abs(lhs) > 10000 or abs(rhs) > 10000 or abs(product) > 1000000:
            raise ProgramExecutionError("binary_op mul_small exceeded safe integer limits.")
        return product
    raise ProgramExecutionError(f"Unsupported binary_op operation: {op!r}.")


def _binary_expr_parts(value: Any) -> tuple[int, str, int]:
    if not isinstance(value, Mapping):
        raise ProgramExecutionError("binary_op requires parsed binary expression mapping.")
    if set(value.keys()) != {"lhs", "op", "rhs"}:
        raise ProgramExecutionError("binary_op parsed context must contain lhs, op, rhs.")
    lhs = value["lhs"]
    operator = value["op"]
    rhs = value["rhs"]
    if isinstance(lhs, bool) or not isinstance(lhs, int):
        raise ProgramExecutionError("binary_op lhs must be an integer.")
    if not isinstance(operator, str) or not operator:
        raise ProgramExecutionError("binary_op op token must be a non-empty string.")
    if isinstance(rhs, bool) or not isinstance(rhs, int):
        raise ProgramExecutionError("binary_op rhs must be an integer.")
    return lhs, operator, rhs


def _required_int(args: Mapping[str, Any], key: str) -> int:
    if key not in args or isinstance(args[key], bool) or not isinstance(args[key], int):
        raise ProgramExecutionError(f"Missing integer arg: {key}")
    return args[key]


def _required_mapping(args: Mapping[str, Any], key: str) -> Mapping[str, str]:
    value = args.get(key)
    if not isinstance(value, Mapping):
        raise ProgramExecutionError(f"Missing mapping arg: {key}")
    return {str(k): str(v) for k, v in value.items()}


def _map_symbol(symbol: str, mapping: Mapping[str, str]) -> str:
    if symbol not in mapping:
        raise ProgramExecutionError(f"No mapping for symbol {symbol!r}.")
    return mapping[symbol]


__all__ = ["execute_program"]
