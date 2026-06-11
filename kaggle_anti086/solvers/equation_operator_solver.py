from __future__ import annotations

from dataclasses import dataclass
import itertools
import re
from typing import Callable

from kaggle_anti086.solvers.base import BaseSolver
from kaggle_anti086.solvers.types import SolverCandidate, SolverResult


INT_RE = r"[-+]?\d+"
PREFIX_RE = re.compile(r"^\[[^\]]+\]\s*")
BINARY_RE = re.compile(
    rf"(?P<a>{INT_RE})\s*(?P<op>[^\w\s\d;,.?=<>-]+|[A-Za-z@#$%^&*/+|~]+)\s*(?P<b>{INT_RE})\s*(?:->|=>|=)\s*(?P<y>{INT_RE})"
)
FUNC_RE = re.compile(rf"\b\w+\s*\(\s*(?P<a>{INT_RE})\s*,\s*(?P<b>{INT_RE})\s*\)\s*(?:->|=>|=)\s*(?P<y>{INT_RE})")
UNARY_RE = re.compile(rf"(?<![\w)])(?P<x>{INT_RE})\s*(?:->|=>|=)\s*(?P<y>{INT_RE})(?!\s*[,)]|\s*[^\w\s\d;,.?=<>-]+\s*{INT_RE})")
BINARY_QUERY_RE = re.compile(
    rf"(?P<a>{INT_RE})\s*(?P<op>[^\w\s\d;,.?=<>-]+|[A-Za-z@#$%^&*/+|~]+)\s*(?P<b>{INT_RE})"
)
FUNC_QUERY_RE = re.compile(rf"\b\w+\s*\(\s*(?P<a>{INT_RE})\s*,\s*(?P<b>{INT_RE})\s*\)")
UNARY_QUERY_RE = re.compile(rf"(?<![\w])(?P<x>{INT_RE})(?![\w])")


@dataclass(frozen=True)
class Example:
    inputs: tuple[int, ...]
    output: int
    op: str | None = None


@dataclass(frozen=True)
class ParsedTask:
    mode: str
    examples: tuple[Example, ...]
    query: tuple[int, ...]
    query_op: str | None = None


@dataclass(frozen=True)
class Operation:
    name: str
    fn: Callable[[tuple[int, ...]], int | None]
    family: str = "basic"


class EquationOperatorSolver(BaseSolver):
    @property
    def name(self) -> str:
        return "equation_operator_solver"

    @property
    def supported_families(self) -> tuple[str, ...]:
        return ("equation_operator",)

    def solve(self, row: dict) -> SolverResult:
        family = str(row.get("family", "unknown"))
        prompt = str(row.get("prompt", "") or "")
        if family != "equation_operator" and not _looks_equation_operator(prompt):
            return self.abstain("not_equation_operator")
        parsed = parse_equation_operator_prompt(prompt)
        if parsed is None:
            return self.abstain("equation_operator_parse_failed")
        candidate_ops = _candidate_operations(parsed, prompt)
        passing: list[tuple[Operation, int]] = []
        for operation in candidate_ops:
            if _verifies(operation, parsed.examples):
                query_value = operation.fn(parsed.query)
                if query_value is not None:
                    passing.append((operation, query_value))
        if not passing:
            return self.abstain("equation_operator_no_verified_candidate")
        answers = sorted({value for _operation, value in passing})
        ambiguity_count = max(0, len(passing) - 1)
        if len(answers) > 1:
            return self.abstain("equation_operator_ambiguous_different_outputs")
        answer = str(answers[0])
        risk = "medium" if ambiguity_count else "low"
        confidence = 0.72 if ambiguity_count else 0.91
        return SolverResult(
            solver_name=self.name,
            family="equation_operator",
            candidates=[
                SolverCandidate(
                    answer=answer,
                    source=self.name,
                    family="equation_operator",
                    subfamily=parsed.mode,
                    confidence=confidence,
                    example_consistency=1.0,
                    verified=True,
                    risk=risk,
                    metadata={
                        "verification_status": "PASS",
                        "ambiguity_count": ambiguity_count,
                        "operation_names": sorted(operation.name for operation, _value in passing),
                        "route_reason": "verified_equation_operator",
                        "examples_verified": len(parsed.examples),
                    },
                )
            ],
            abstained=False,
            reason="",
            metadata={"parsed_mode": parsed.mode, "ambiguity_count": ambiguity_count},
        )


def parse_equation_operator_prompt(prompt: str) -> ParsedTask | None:
    body = PREFIX_RE.sub("", str(prompt or "").strip())
    if not body:
        return None
    binary_examples = _binary_examples(body)
    if binary_examples:
        query = _binary_query(body, binary_examples)
        if query is None:
            return None
        query_values, query_op = query
        example_ops = {example.op for example in binary_examples if example.op is not None}
        if len(example_ops) > 1:
            return None
        if example_ops and query_op not in example_ops:
            return None
        return ParsedTask("binary", tuple(binary_examples), query_values, query_op)
    unary_examples = _unary_examples(body)
    if unary_examples:
        query_unary = _unary_query(body)
        if query_unary is None:
            return None
        return ParsedTask("unary", tuple(unary_examples), (query_unary,), None)
    return None


def _binary_examples(body: str) -> list[Example]:
    examples: list[Example] = []
    occupied: list[tuple[int, int]] = []
    for match in FUNC_RE.finditer(body):
        examples.append(Example((int(match.group("a")), int(match.group("b"))), int(match.group("y")), "FUNC"))
        occupied.append(match.span())
    for match in BINARY_RE.finditer(body):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        op = match.group("op").strip()
        if op in {"->", "=>", "="}:
            continue
        examples.append(Example((int(match.group("a")), int(match.group("b"))), int(match.group("y")), op))
    return examples


def _unary_examples(body: str) -> list[Example]:
    if BINARY_RE.search(body) or FUNC_RE.search(body):
        return []
    return [Example((int(match.group("x")),), int(match.group("y")), None) for match in UNARY_RE.finditer(body)]


def _query_segment(body: str) -> str:
    markers = list(re.finditer(r"\b(?:query|input|target|solve|infer target|now solve)\b\s*:?", body, flags=re.IGNORECASE))
    if markers:
        return body[markers[-1].end() :]
    return body


def _binary_query(body: str, examples: list[Example]) -> tuple[tuple[int, int], str | None] | None:
    segment = _query_segment(body)
    func_match = FUNC_QUERY_RE.search(segment)
    if func_match:
        return (int(func_match.group("a")), int(func_match.group("b"))), "FUNC"
    matches = list(BINARY_QUERY_RE.finditer(segment))
    if not matches:
        return None
    match = matches[-1]
    return (int(match.group("a")), int(match.group("b"))), match.group("op").strip()


def _unary_query(body: str) -> int | None:
    segment = _query_segment(body)
    values = [int(match.group("x")) for match in UNARY_QUERY_RE.finditer(segment)]
    if not values:
        return None
    example_values = {example.inputs[0] for example in _unary_examples(body)} | {example.output for example in _unary_examples(body)}
    query_values = [value for value in values if value not in example_values]
    return query_values[-1] if query_values else values[-1]


def _candidate_operations(parsed: ParsedTask, prompt: str) -> list[Operation]:
    if parsed.mode == "binary":
        return _binary_operations(prompt)
    return _unary_operations(parsed.examples)


def _binary_operations(prompt: str) -> list[Operation]:
    ops = [
        Operation("a_plus_b", lambda args: args[0] + args[1]),
        Operation("a_minus_b", lambda args: args[0] - args[1]),
        Operation("b_minus_a", lambda args: args[1] - args[0]),
        Operation("a_times_b", lambda args: args[0] * args[1]),
        Operation("a_div_b_exact", lambda args: _exact_div(args[0], args[1])),
        Operation("b_div_a_exact", lambda args: _exact_div(args[1], args[0])),
        Operation("a_mod_b", lambda args: None if args[1] == 0 else args[0] % args[1]),
        Operation("b_mod_a", lambda args: None if args[0] == 0 else args[1] % args[0]),
        Operation("min", lambda args: min(args[0], args[1])),
        Operation("max", lambda args: max(args[0], args[1])),
        Operation("abs_diff", lambda args: abs(args[0] - args[1])),
    ]
    if re.search(r"\b(?:xor|bitwise|\^)\b", prompt, flags=re.IGNORECASE):
        ops.append(Operation("a_xor_b", lambda args: args[0] ^ args[1]))
    ops.extend(_binary_affine_operations())
    return ops


def _unary_operations(examples: tuple[Example, ...]) -> list[Operation]:
    constants = sorted({abs(value) for example in examples for value in (*example.inputs, example.output) if abs(value) <= 20})
    constants.extend(value for value in range(-10, 11) if value not in constants)
    ops = [
        Operation("x_square", lambda args: args[0] * args[0]),
        Operation("abs_x", lambda args: abs(args[0])),
        Operation("digit_sum", lambda args: _digit_sum(args[0])),
        Operation("reverse_digits", lambda args: _reverse_digits(args[0])),
    ]
    for c in constants:
        ops.extend(
            [
                Operation(f"x_plus_{c}", lambda args, c=c: args[0] + c, "unary_affine"),
                Operation(f"x_minus_{c}", lambda args, c=c: args[0] - c, "unary_affine"),
                Operation(f"{c}_minus_x", lambda args, c=c: c - args[0], "unary_affine"),
                Operation(f"x_times_{c}", lambda args, c=c: args[0] * c, "unary_affine"),
                Operation(f"x_div_{c}_exact", lambda args, c=c: _exact_div(args[0], c), "unary_affine"),
                Operation(f"x_mod_{c}", lambda args, c=c: None if c == 0 else args[0] % c, "unary_affine"),
            ]
        )
    return ops


def _binary_affine_operations() -> list[Operation]:
    ops: list[Operation] = []
    for c0, c1, c2 in itertools.product(range(-5, 6), repeat=3):
        if c1 == 0 and c2 == 0:
            continue
        ops.append(Operation(f"affine_{c0}_{c1}_{c2}", lambda args, c0=c0, c1=c1, c2=c2: c0 + c1 * args[0] + c2 * args[1], "binary_affine"))
    return ops


def _verifies(operation: Operation, examples: tuple[Example, ...]) -> bool:
    return all(operation.fn(example.inputs) == example.output for example in examples)


def _exact_div(a: int, b: int) -> int | None:
    if b == 0 or a % b != 0:
        return None
    return a // b


def _digit_sum(x: int) -> int:
    return sum(int(char) for char in str(abs(x)))


def _reverse_digits(x: int) -> int:
    sign = -1 if x < 0 else 1
    return sign * int(str(abs(x))[::-1])


def _looks_equation_operator(prompt: str) -> bool:
    return bool(re.search(r"\b(?:equation|operator|modulo|exact division)\b", prompt, flags=re.IGNORECASE))
