"""Conservative deterministic family router for Sprint 1 solvers."""

from __future__ import annotations

import re

from .problem_parser import parse_problem
from .solver_base import ParsedProblem, RouterDecision


DSL_SOLVER = "dsl_synthesizer"

_FAMILY_PATTERNS: tuple[tuple[str, str, tuple[re.Pattern[str], ...]], ...] = (
    (
        "modular",
        "modular_solver",
        (
            re.compile(r"\b(modulo|mod|remainder|residue|congru|base\s+\d+|parity|even|odd)\b", re.IGNORECASE),
        ),
    ),
    (
        "bit",
        "bit_solver",
        (
            re.compile(r"\b(bit|binary|bitwise|xor|nand|nor|and|or|shift|0b[01]+)\b", re.IGNORECASE),
        ),
    ),
    (
        "mapping",
        "mapping_solver",
        (
            re.compile(r"\b(map|mapping|translate|encode|decode|cipher|symbol|bijection|substitution)\b", re.IGNORECASE),
        ),
    ),
    (
        "arithmetic",
        "arithmetic_solver",
        (
            re.compile(r"\b(integer|arithmetic|operator|plus|minus|add|subtract|multiply|divide|sum|product)\b", re.IGNORECASE),
            re.compile(r"\d+\s*[+\-*/]\s*\d+"),
        ),
    ),
    (
        "sequence",
        "sequence_solver",
        (
            re.compile(r"\b(sequence|series|next|term|progression|fibonacci|recurrence)\b", re.IGNORECASE),
        ),
    ),
    (
        "string",
        "string_solver",
        (
            re.compile(r"\b(string|word|letter|reverse|uppercase|lowercase|prefix|suffix|concat|palindrome)\b", re.IGNORECASE),
        ),
    ),
)


def route_family(problem: ParsedProblem | str, *, problem_id: str | None = None) -> RouterDecision:
    """Return a conservative ordered solver list with the DSL synthesizer last."""

    parsed = parse_problem(problem, problem_id=problem_id) if isinstance(problem, str) else problem
    text = _route_text(parsed)
    solvers: list[str] = []
    families: list[str] = []
    for family, solver, patterns in _FAMILY_PATTERNS:
        if any(pattern.search(text) for pattern in patterns):
            solvers.append(solver)
            families.append(family)
    solvers.append(DSL_SOLVER)
    return RouterDecision(problem_id=parsed.problem_id, ordered_solvers=tuple(solvers), matched_families=tuple(families))


def _route_text(problem: ParsedProblem) -> str:
    parts = [problem.raw_prompt, problem.target_input]
    for example in problem.examples:
        parts.extend((example.input_value, example.output_value))
    return "\n".join(parts)


__all__ = ["DSL_SOLVER", "route_family"]
