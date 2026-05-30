"""Manual failure recovery plan contracts for Pass 15."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import re
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash

from .runbook_config import RunbookConfig, reject_forbidden_runbook_claims, validate_runbook_config


class FailureRecoveryError(ValueError):
    """Raised when recovery actions are unsafe."""


REQUIRED_RECOVERY_TRIGGERS = (
    "test suite fails",
    "artifact audit fails",
    "package rehearsal fails",
    "runtime rehearsal fails",
    "preflight fails",
    "final submission rejected manually",
    "accidental file artifact created",
)
_FORBIDDEN_ACTION_PATTERNS = (
    "rm -rf",
    "git reset --hard",
    "git clean",
    "remove-item -recurse -force",
    "kaggle submit",
    "kaggle competitions submit",
    "kaggle.api",
    "submit to kaggle",
    "auto submit",
)
_SECRET_EXPOSURE_PATTERNS = (
    "display token",
    "print token",
    "cat token",
    "type token",
    "show token",
    "echo token",
    "display credentials",
    "print credentials",
    "cat credentials",
    "type credentials",
    "show credentials",
    "kaggle.json",
    ".env",
    "api_key",
    "secret",
)


@dataclass(frozen=True)
class RecoveryAction:
    action_id: str
    trigger: str
    action: str
    manual_only: bool
    destructive: bool
    allowed: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    action_hash: str = ""

    def __post_init__(self) -> None:
        action_id = _require_non_empty(self.action_id, "action_id")
        trigger = _require_non_empty(self.trigger, "trigger")
        action = _require_non_empty(self.action, "action")
        for name in ("manual_only", "destructive", "allowed"):
            if not isinstance(getattr(self, name), bool):
                raise FailureRecoveryError(f"{name} must be boolean.")
        metadata = dict(self.metadata)
        reject_forbidden_runbook_claims(metadata, error_cls=FailureRecoveryError)
        _validate_action_text(action)
        if self.destructive and self.allowed:
            raise FailureRecoveryError("destructive recovery actions cannot be allowed.")
        if self.destructive and not self.manual_only:
            raise FailureRecoveryError("destructive recovery actions must be manual-only and disallowed.")
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "trigger", trigger)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "metadata", metadata)
        expected = compute_recovery_action_hash(self)
        if not self.action_hash:
            object.__setattr__(self, "action_hash", expected)
        elif self.action_hash != expected:
            raise FailureRecoveryError("action_hash does not match recovery action payload.")


@dataclass(frozen=True)
class FailureRecoveryPlan:
    actions: tuple[RecoveryAction, ...]
    runbook_config_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    plan_hash: str = ""

    def __post_init__(self) -> None:
        actions = tuple(_action(item) for item in self.actions)
        if not actions:
            raise FailureRecoveryError("failure recovery plan requires actions.")
        ids = tuple(item.action_id for item in actions)
        if len(ids) != len(set(ids)):
            raise FailureRecoveryError("recovery action ids must be unique.")
        _require_non_empty(self.runbook_config_hash, "runbook_config_hash")
        triggers = {item.trigger for item in actions}
        missing = tuple(trigger for trigger in REQUIRED_RECOVERY_TRIGGERS if trigger not in triggers)
        if missing:
            raise FailureRecoveryError(f"missing required recovery triggers: {missing}")
        for action in actions:
            if action.action_hash != compute_recovery_action_hash(action):
                raise FailureRecoveryError("action_hash does not match recovery action payload.")
        metadata = dict(self.metadata)
        reject_forbidden_runbook_claims(metadata, error_cls=FailureRecoveryError)
        object.__setattr__(self, "actions", tuple(sorted(actions, key=lambda item: item.action_id)))
        object.__setattr__(self, "metadata", metadata)
        expected = compute_failure_recovery_hash(self)
        if not self.plan_hash:
            object.__setattr__(self, "plan_hash", expected)
        elif self.plan_hash != expected:
            raise FailureRecoveryError("plan_hash does not match recovery plan payload.")


def build_default_failure_recovery_plan(config: RunbookConfig | Mapping[str, Any], metadata: Mapping[str, Any] | None = None) -> FailureRecoveryPlan:
    cfg = validate_runbook_config(config)
    actions = tuple(
        RecoveryAction(
            action_id=f"recovery-{index:02d}",
            trigger=trigger,
            action=_default_action_for_trigger(trigger),
            manual_only=True,
            destructive=False,
            allowed=True,
            metadata={},
        )
        for index, trigger in enumerate(REQUIRED_RECOVERY_TRIGGERS, start=1)
    )
    return FailureRecoveryPlan(actions=actions, runbook_config_hash=cfg.config_hash, metadata=dict(metadata or {}))


def validate_failure_recovery_plan(plan: FailureRecoveryPlan | Mapping[str, Any], config: RunbookConfig | Mapping[str, Any] | None = None) -> FailureRecoveryPlan:
    cfg = validate_runbook_config(config) if config is not None else None
    normalized = plan if isinstance(plan, FailureRecoveryPlan) else FailureRecoveryPlan(**dict(plan))
    if normalized.plan_hash != compute_failure_recovery_hash(normalized):
        raise FailureRecoveryError("plan_hash does not match recovery plan payload.")
    if cfg is not None and normalized.runbook_config_hash != cfg.config_hash:
        raise FailureRecoveryError("failure recovery runbook_config_hash mismatch.")
    return normalized


def compute_failure_recovery_hash(plan: FailureRecoveryPlan | Mapping[str, Any]) -> str:
    payload = dict(plan) if isinstance(plan, Mapping) else {item.name: getattr(plan, item.name) for item in fields(FailureRecoveryPlan)}
    payload.pop("plan_hash", None)
    return stable_hash(payload)


def compute_recovery_action_hash(action: RecoveryAction | Mapping[str, Any]) -> str:
    payload = dict(action) if isinstance(action, Mapping) else {item.name: getattr(action, item.name) for item in fields(RecoveryAction)}
    payload.pop("action_hash", None)
    return stable_hash(payload)


def _default_action_for_trigger(trigger: str) -> str:
    return {
        "test suite fails": "Stop external action, record failing test output, fix only allowed source changes, and rerun scoped tests.",
        "artifact audit fails": "Stop external action, inspect forbidden artifact report, remove only explicitly confirmed accidental artifacts outside this runbook contract.",
        "package rehearsal fails": "Stop external action, inspect package rehearsal blockers, and rebuild evidence through the locked rehearsal flow.",
        "runtime rehearsal fails": "Stop external action, inspect runtime rehearsal blockers, and record whether unloaded dry-run evidence is allowed.",
        "preflight fails": "Stop external action, inspect preflight blockers, and rerun the locked preflight process after fixes.",
        "final submission rejected manually": "Record the human-visible rejection reference and return to package and rehearsal evidence review.",
        "accidental file artifact created": "Stop external action, record the path, and remove it only after separate human confirmation.",
    }[trigger]


def _action(value: RecoveryAction | Mapping[str, Any]) -> RecoveryAction:
    return value if isinstance(value, RecoveryAction) else RecoveryAction(**dict(value))


def _validate_action_text(action: str) -> None:
    text = re.sub(r"\s+", " ", action.lower())
    if any(pattern in text for pattern in _FORBIDDEN_ACTION_PATTERNS):
        raise FailureRecoveryError("recovery action contains forbidden command or submit text.")
    if any(pattern in text for pattern in _SECRET_EXPOSURE_PATTERNS):
        raise FailureRecoveryError("recovery action cannot expose secrets or tokens.")


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FailureRecoveryError(f"{field_name} must be non-empty.")
    return value.strip()


__all__ = [
    "FailureRecoveryError",
    "FailureRecoveryPlan",
    "REQUIRED_RECOVERY_TRIGGERS",
    "RecoveryAction",
    "build_default_failure_recovery_plan",
    "compute_failure_recovery_hash",
    "compute_recovery_action_hash",
    "validate_failure_recovery_plan",
]
