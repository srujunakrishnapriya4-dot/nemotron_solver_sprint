from __future__ import annotations

from dataclasses import dataclass, fields

from nemotron_engine.core.schemas import stable_hash


class FailedTrainingAuditError(ValueError):
    pass


@dataclass(frozen=True)
class FailedTrainingPath:
    name: str
    public_score: float | None
    verdict: str
    failure_reason: str
    why_gpu_was_wasted: str
    banned_going_forward: tuple[str, ...]
    proof_required_before_submit: tuple[str, ...]
    audit_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "banned_going_forward", tuple(self.banned_going_forward))
        object.__setattr__(self, "proof_required_before_submit", tuple(self.proof_required_before_submit))
        _set_or_check_hash(self, "audit_hash")


def known_failed_training_paths() -> tuple[FailedTrainingPath, ...]:
    return (
        FailedTrainingPath(
            name="SPRINT-4 full-prompt loss",
            public_score=0.49,
            verdict="DO_NOT_USE",
            failure_reason="Trainer supervised the entire rendered chat with labels=input_ids, including user prompts and examples.",
            why_gpu_was_wasted="The update optimized prompt reconstruction instead of answer behavior, corrupting a known-good parent adapter.",
            banned_going_forward=("labels=input_ids", "full_prompt_loss_without_override", "blind_30b_continuation"),
            proof_required_before_submit=("assistant_only_loss_masking", "fast_probe_sane_outputs", "manual_scorebook_not_rejected"),
        ),
        FailedTrainingPath(
            name="SPRINT-4.1 huge assistant-only child",
            public_score=None,
            verdict="DO_NOT_SUBMIT_WITHOUT_STRONG_EVIDENCE",
            failure_reason="Assistant-only masking was repaired, but loss stayed around 30, adapter was about 4GB, and behavioral eval was too slow to prove improvement.",
            why_gpu_was_wasted="It consumed scarce GPU without a cheap non-regression gate or adapter-size guard.",
            banned_going_forward=("huge_custom_child_without_known_good_structure", "package_behavior_unknown", "slow_large_eval_as_first_gate"),
            proof_required_before_submit=("adapter_size_not_suspicious", "train_loss_below_20_or_override", "fast_probe_sane_outputs", "operator_selected"),
        ),
    )


def audit_failed_training_paths() -> dict:
    paths = known_failed_training_paths()
    return {
        "failed_paths": [path.__dict__ for path in paths],
        "global_bans": sorted({ban for path in paths for ban in path.banned_going_forward}),
        "required_proof": sorted({proof for path in paths for proof in path.proof_required_before_submit}),
        "audit_hash": stable_hash([path.audit_hash for path in paths]),
    }


def _set_or_check_hash(instance: object, hash_field: str) -> None:
    expected = stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})
    current = getattr(instance, hash_field)
    if not current:
        object.__setattr__(instance, hash_field, expected)
    elif current != expected:
        raise FailedTrainingAuditError(f"{hash_field} does not match payload.")
