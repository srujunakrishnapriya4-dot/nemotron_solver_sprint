from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.core.schemas import CanonicalProblem, DomainKind, DomainSignature, ExamplePair, SymbolTable, TargetQuery  # noqa: E402
from nemotron_engine.parsing import canonicalize_prompt, compute_round_trip_score, reconstruct_prompt_skeleton, verify_round_trip  # noqa: E402


def _unsafe_problem(problem: CanonicalProblem, **updates: object) -> CanonicalProblem:
    forged = object.__new__(CanonicalProblem)
    for name in problem.__dataclass_fields__:
        object.__setattr__(forged, name, getattr(problem, name))
    for key, value in updates.items():
        object.__setattr__(forged, key, value)
    return forged


def test_reconstruct_prompt_skeleton_and_score_high() -> None:
    problem = canonicalize_prompt("p", "1 -> 2\n3 -> ?")
    skeleton = reconstruct_prompt_skeleton(problem)

    assert "Examples:" in skeleton
    assert compute_round_trip_score(problem.raw_prompt, skeleton) >= 0.95
    assert verify_round_trip(problem) is True


def test_verify_round_trip_fails_low_score_structure() -> None:
    problem = CanonicalProblem(
        problem_id="p",
        raw_prompt="completely different words",
        examples=(ExamplePair("1", "2"),),
        target=TargetQuery("3"),
        input_domain=DomainSignature(DomainKind.INTEGER, "1", "1"),
        output_domain=DomainSignature(DomainKind.INTEGER, "2", "2"),
        symbol_table=SymbolTable(),
        parse_confidence=0.5,
        round_trip_score=0.1,
    )

    assert verify_round_trip(problem) is False


def test_verify_round_trip_fails_uncertain_missing_target_or_missing_examples() -> None:
    uncertain = CanonicalProblem(
        problem_id="p",
        raw_prompt="1 -> 2\n3 -> ?",
        examples=(ExamplePair("1", "2"),),
        target=TargetQuery("3", certain=False),
        input_domain=DomainSignature(DomainKind.INTEGER, "1", "1"),
        output_domain=DomainSignature(DomainKind.INTEGER, "2", "2"),
        symbol_table=SymbolTable(),
        parse_confidence=0.5,
        round_trip_score=0.98,
    )
    valid = canonicalize_prompt("p", "1 -> 2\n3 -> ?")

    assert verify_round_trip(uncertain) is False
    assert verify_round_trip(_unsafe_problem(valid, target=None)) is False
    assert verify_round_trip(_unsafe_problem(valid, examples=())) is False
