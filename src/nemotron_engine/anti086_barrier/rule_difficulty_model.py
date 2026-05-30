from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from nemotron_engine.core.schemas import stable_hash


class RuleDifficultyError(ValueError):
    pass


@dataclass(frozen=True)
class RuleDifficultyScore:
    family: str
    difficulty_score: float
    novelty_score: float
    ambiguity_score: float
    solver_verifiability: bool
    private_like_score: float
    reasons: tuple[str, ...]
    score_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("difficulty_score", "novelty_score", "ambiguity_score", "private_like_score"):
            value = float(getattr(self, name))
            if value < 0.0 or value > 1.0:
                raise RuleDifficultyError(f"{name} must be in [0,1]")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "reasons", tuple(str(reason) for reason in self.reasons))
        _set_or_check_hash(self, "score_hash")


def score_rule(row: dict[str, Any]) -> RuleDifficultyScore:
    family = str(row.get("family", "unknown"))
    rule_id = str(row.get("rule_id", ""))
    prompt = str(row.get("prompt", ""))
    params = row.get("parameters") if isinstance(row.get("parameters"), dict) else {}
    reasons: list[str] = []
    difficulty = 0.25
    novelty = 0.45
    ambiguity = 0.15

    if family == "bit_manipulation":
        depth = int(params.get("composition_depth", row.get("composition_depth", 1)))
        difficulty += min(0.35, depth * 0.1)
        if any(token in rule_id for token in ("choice", "majority", "mask", "endian")):
            difficulty += 0.2
            novelty += 0.15
            reasons.append("mixed_boolean_composition")
        if "ambiguous" in rule_id:
            ambiguity += 0.4
    elif family in {"cipher_text", "equation_symbolic"}:
        if any(token in rule_id for token in ("composition", "symbol", "operator", "completion")):
            difficulty += 0.3
            novelty += 0.2
            reasons.append("symbolic_composition")
    elif family in {"unit_conversion", "gravity_numeric"}:
        if any(token in rule_id for token in ("rounding", "affine", "extrapolation")):
            difficulty += 0.3
            novelty += 0.15
            reasons.append("numeric_rounding_or_extrapolation")
    if len(prompt) > 300:
        novelty += 0.05
    verifier = bool(row.get("verification_trace") or row.get("target_prediction") == row.get("answer") or row.get("verified_status") == "verified_correct")
    private_like = min(1.0, (difficulty * 0.55) + (novelty * 0.35) + ((1.0 - ambiguity) * 0.10))
    return RuleDifficultyScore(
        family=family,
        difficulty_score=min(1.0, difficulty),
        novelty_score=min(1.0, novelty),
        ambiguity_score=min(1.0, ambiguity),
        solver_verifiability=verifier,
        private_like_score=private_like,
        reasons=tuple(reasons or ("baseline_rule_features",)),
    )


def reject_reason(score: RuleDifficultyScore) -> str | None:
    if score.ambiguity_score >= 0.55:
        return "ambiguity_too_high"
    if score.novelty_score < 0.35:
        return "novelty_too_low"
    if not score.solver_verifiability:
        return "solver_not_verified"
    if score.difficulty_score < 0.35:
        return "too_easy"
    return None


def _set_or_check_hash(instance: object, hash_field: str) -> None:
    expected = stable_hash({field.name: getattr(instance, field.name) for field in fields(instance) if field.name != hash_field})
    current = getattr(instance, hash_field)
    if not current:
        object.__setattr__(instance, hash_field, expected)
    elif current != expected:
        raise RuleDifficultyError(f"{hash_field} does not match payload")
