"""Round-trip structural verification for canonical prompts."""

from __future__ import annotations

import re

from nemotron_engine.core.schemas import CanonicalProblem, TargetQuery


def reconstruct_prompt_skeleton(problem: CanonicalProblem) -> str:
    lines = ["Examples:"]
    for example in problem.examples:
        lines.append(f"{example.input_value} -> {example.output_value}")
    lines.append("Target:")
    lines.append(f"{problem.target.input_value} -> ?")
    return "\n".join(lines)


def compute_round_trip_score(raw_prompt: str, reconstructed: str) -> float:
    raw_tokens = _structural_tokens(raw_prompt)
    reconstructed_tokens = _structural_tokens(reconstructed)
    if not raw_tokens or not reconstructed_tokens:
        return 0.0
    overlap = sum(1 for token in reconstructed_tokens if token in raw_tokens)
    return overlap / len(reconstructed_tokens)


def verify_round_trip(problem: CanonicalProblem, *, min_score: float = 0.95) -> bool:
    if not problem.examples or not isinstance(problem.target, TargetQuery):
        return False
    if not _target_is_structurally_valid(problem.target):
        return False
    reconstructed = reconstruct_prompt_skeleton(problem)
    score = compute_round_trip_score(problem.raw_prompt, reconstructed)
    return score >= min_score


def _structural_tokens(text: str) -> list[str]:
    lowered = str(text).lower()
    tokens = re.findall(r"[a-z0-9_./+-]+|->|\?", lowered)
    return [token for token in tokens if token not in {"examples", "target", "input", "output"}]


def _target_is_structurally_valid(target: TargetQuery) -> bool:
    value = target.input_value.strip()
    return target.certain is True and bool(value) and value.upper() not in {"UNKNOWN", "?", "TODO", "MISSING", "NONE", "NULL"}


__all__ = [
    "compute_round_trip_score",
    "reconstruct_prompt_skeleton",
    "verify_round_trip",
]
