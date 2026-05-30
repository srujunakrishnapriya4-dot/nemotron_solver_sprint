from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.sprint_solvers import SprintSolverError, parse_problem, route_family  # noqa: E402


def _route(prompt: str):
    return route_family(parse_problem(prompt))


def test_bit_prompts_route_bit_solver_before_dsl() -> None:
    decision = _route("binary bit xor\n1 -> 0\nTarget: 0 -> ?")
    assert decision.ordered_solvers[0] == "bit_solver"
    assert decision.ordered_solvers[-1] == "dsl_synthesizer"


def test_mapping_prompts_route_mapping_solver_before_dsl() -> None:
    decision = _route("cipher mapping\nA -> B\nTarget: C -> ?")
    assert decision.ordered_solvers[0] == "mapping_solver"
    assert decision.ordered_solvers[-1] == "dsl_synthesizer"


def test_arithmetic_prompts_route_arithmetic_solver_before_dsl() -> None:
    decision = _route("integer operator add\n1 -> 2\nTarget: 3 -> ?")
    assert decision.ordered_solvers[0] == "arithmetic_solver"
    assert decision.ordered_solvers[-1] == "dsl_synthesizer"


def test_sequence_prompts_route_sequence_solver_before_dsl() -> None:
    decision = _route("next term in sequence\n1,2,3 -> 4\nTarget: 2,4,6 -> ?")
    assert decision.ordered_solvers[0] == "sequence_solver"
    assert decision.ordered_solvers[-1] == "dsl_synthesizer"


def test_string_prompts_route_string_solver_before_dsl() -> None:
    decision = _route("reverse word string\nabc -> cba\nTarget: def -> ?")
    assert decision.ordered_solvers[0] == "string_solver"
    assert decision.ordered_solvers[-1] == "dsl_synthesizer"


def test_modular_parity_prompts_route_modular_solver_before_dsl() -> None:
    decision = _route("parity modulo\n2 -> even\nTarget: 3 -> ?")
    assert decision.ordered_solvers[0] == "modular_solver"
    assert decision.ordered_solvers[-1] == "dsl_synthesizer"


def test_unknown_prompts_still_include_dsl_synthesizer_last() -> None:
    decision = _route("foo -> bar\nTarget: baz -> ?")
    assert decision.ordered_solvers == ("dsl_synthesizer",)


def test_router_decision_hash_forgery_rejected() -> None:
    decision = _route("foo -> bar\nTarget: baz -> ?")

    with pytest.raises(SprintSolverError):
        replace(decision, decision_hash="forged")
