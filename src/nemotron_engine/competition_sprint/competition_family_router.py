"""Deterministic router for parsed competition problems."""

from __future__ import annotations

from dataclasses import dataclass, fields

from nemotron_engine.core.schemas import stable_hash

from .competition_prompt_adapter import CompetitionProblem


class CompetitionRouterError(ValueError):
    """Raised when a competition routing decision is invalid."""


_ROUTES = {
    "bit_manipulation": ("competition_bit_adapter",),
    "cipher_text": ("competition_cipher_solver",),
    "roman_numeral": ("competition_roman_solver",),
    "unit_conversion": ("competition_numeric_solver",),
    "gravity_numeric": ("competition_gravity_solver",),
    "equation_symbolic": ("competition_equation_solver",),
    "unknown": ("abstain",),
}


@dataclass(frozen=True)
class CompetitionRouterDecision:
    problem_id: str
    family: str
    ordered_handlers: tuple[str, ...]
    decision_hash: str = ""

    def __post_init__(self) -> None:
        if not str(self.problem_id).strip() or not str(self.family).strip():
            raise CompetitionRouterError("router decision requires identifiers.")
        handlers = tuple(str(item) for item in self.ordered_handlers)
        if not handlers:
            raise CompetitionRouterError("router decision requires handlers.")
        object.__setattr__(self, "problem_id", str(self.problem_id))
        object.__setattr__(self, "family", str(self.family))
        object.__setattr__(self, "ordered_handlers", handlers)
        _set_or_check_hash(self, "decision_hash")


def route_competition_problem(problem: CompetitionProblem) -> CompetitionRouterDecision:
    """Route a parsed competition problem to conservative local handlers."""

    family = problem.family if problem.family in _ROUTES else "unknown"
    return CompetitionRouterDecision(problem_id=problem.problem_id, family=family, ordered_handlers=_ROUTES[family])


def _set_or_check_hash(instance: object, hash_field: str) -> None:
    expected = _payload_hash(instance, hash_field)
    current = getattr(instance, hash_field)
    if not current:
        object.__setattr__(instance, hash_field, expected)
    elif current != expected:
        raise CompetitionRouterError(f"{hash_field} does not match payload.")


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


__all__ = ["CompetitionRouterDecision", "CompetitionRouterError", "route_competition_problem"]
