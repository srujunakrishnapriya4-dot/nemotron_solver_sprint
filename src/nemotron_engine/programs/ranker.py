"""Deterministic ranking for verified Pass 3 solver attempts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from nemotron_engine.core.schemas import Program, ProgramStepKind, VerificationStatus


@dataclass(frozen=True)
class ProgramRank:
    attempt_index: int
    training_safe: bool
    leakage_free: bool
    ambiguity_free: bool
    step_count: int
    constant_cost: int
    transform_cost: int
    score: tuple[int, int, int, int, int, int]


def rank_candidate_programs(attempts: Sequence[object]) -> list[ProgramRank]:
    ranks = [_rank_attempt(index, attempt) for index, attempt in enumerate(attempts)]
    return sorted(ranks, key=lambda rank: rank.score)


def choose_unique_best(attempts: Sequence[object]) -> object | None:
    accepted = [attempt for attempt in attempts if _is_training_safe(attempt)]
    if not accepted:
        return None
    outputs = {_target_output(attempt) for attempt in accepted}
    if len(outputs) != 1 or None in outputs:
        return None
    ranks = rank_candidate_programs(accepted)
    return accepted[ranks[0].attempt_index] if ranks else None


def _rank_attempt(index: int, attempt: object) -> ProgramRank:
    program = getattr(attempt, "program", None)
    proof = getattr(attempt, "proof", None)
    training_safe = _is_training_safe(attempt)
    leakage_free = bool(proof is not None and proof.leakage_report.has_leakage is False)
    ambiguity_free = bool(proof is not None and proof.ambiguity_report.ambiguous is False)
    step_count = len(program.steps) if isinstance(program, Program) else 999
    constant_cost = _constant_cost(program)
    transform_cost = _transform_cost(program)
    score = (
        0 if training_safe else 1,
        0 if leakage_free else 1,
        0 if ambiguity_free else 1,
        step_count,
        constant_cost,
        transform_cost,
    )
    return ProgramRank(index, training_safe, leakage_free, ambiguity_free, step_count, constant_cost, transform_cost, score)


def _is_training_safe(attempt: object) -> bool:
    proof = getattr(attempt, "proof", None)
    return bool(
        getattr(attempt, "status", None) is VerificationStatus.PASS
        and getattr(attempt, "program", None) is not None
        and proof is not None
        and proof.primary_verifier_pass is True
        and proof.shadow_verifier_pass is True
        and proof.is_training_safe is True
        and proof.leakage_report.has_leakage is False
        and proof.ambiguity_report.ambiguous is False
        and proof.format_report.valid is True
    )


def _target_output(attempt: object) -> str | None:
    proof = getattr(attempt, "proof", None)
    if proof is None or proof.target_execution is None or proof.target_execution.output_value in {None, ""}:
        return None
    return str(proof.target_execution.output_value)


def _constant_cost(program: Program | None) -> int:
    if program is None:
        return 999
    cost = 0
    for step in program.steps:
        for value in step.args.values():
            if isinstance(value, int) and not isinstance(value, bool):
                cost += abs(value)
    return cost


def _transform_cost(program: Program | None) -> int:
    if program is None:
        return 999
    weights = {
        "identity": 0,
        "raw": 0,
        "add_const": 1,
        "sub_const": 1,
        "mul_const": 2,
        "affine_small": 3,
        "reverse": 1,
        "bit_not": 1,
        "digit_sum": 2,
        "binary_op": 2,
        "symbol_bijection": 2,
    }
    return sum(weights.get(step.primitive, 1) for step in program.steps if step.kind is ProgramStepKind.TRANSFORM)


__all__ = ["ProgramRank", "choose_unique_best", "rank_candidate_programs"]
