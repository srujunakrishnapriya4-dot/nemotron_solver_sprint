from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.competition_sprint import (  # noqa: E402
    CompetitionExample,
    CompetitionProblem,
    CompetitionRouterError,
    route_competition_problem,
)


def problem(family: str) -> CompetitionProblem:
    return CompetitionProblem("p", "raw", family, (CompetitionExample("a", "b"),), "a")


@pytest.mark.parametrize(
    ("family", "handler"),
    [
        ("bit_manipulation", "competition_bit_adapter"),
        ("cipher_text", "competition_cipher_solver"),
        ("roman_numeral", "competition_roman_solver"),
        ("unit_conversion", "competition_numeric_solver"),
        ("gravity_numeric", "competition_gravity_solver"),
        ("equation_symbolic", "competition_equation_solver"),
    ],
)
def test_each_family_routes_correctly(family: str, handler: str) -> None:
    decision = route_competition_problem(problem(family))

    assert decision.family == family
    assert decision.ordered_handlers == (handler,)


def test_unknown_routes_safely() -> None:
    decision = route_competition_problem(problem("other"))

    assert decision.family == "unknown"
    assert decision.ordered_handlers == ("abstain",)


def test_router_hash_forgery_rejected() -> None:
    decision = route_competition_problem(problem("bit_manipulation"))

    with pytest.raises(CompetitionRouterError):
        replace(decision, decision_hash="forged")
