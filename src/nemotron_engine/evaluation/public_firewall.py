"""Public leaderboard firewall policy for Pass 8."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash


PUBLIC_FIREWALL_DECISIONS = ("allow", "reject", "investigate")


class PublicFirewallError(ValueError):
    """Raised when public firewall inputs or reports are inconsistent."""


@dataclass(frozen=True)
class PublicFirewallConfig:
    public_score_delta: float
    private_like_score_delta: float
    internal_score_delta: float
    stability_score_delta: float
    public_driven_change_count: int = 0
    max_public_driven_changes: int = 2
    public_gain_epsilon: float = 0.0
    private_regression_tolerance: float = 0.0
    stability_drop_tolerance: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        for name in (
            "public_score_delta",
            "private_like_score_delta",
            "internal_score_delta",
            "stability_score_delta",
            "public_gain_epsilon",
            "private_regression_tolerance",
            "stability_drop_tolerance",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PublicFirewallError(f"{name} must be numeric.")
            if math.isnan(float(value)) or math.isinf(float(value)):
                raise PublicFirewallError(f"{name} must be finite.")
            object.__setattr__(self, name, float(value))
        for name in ("public_driven_change_count", "max_public_driven_changes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise PublicFirewallError(f"{name} must be a non-negative integer.")
        for name in ("public_gain_epsilon", "private_regression_tolerance", "stability_drop_tolerance"):
            value = getattr(self, name)
            if float(value) < 0.0:
                raise PublicFirewallError(f"{name} must be non-negative.")
        object.__setattr__(self, "metadata", dict(self.metadata))
        expected = _payload_hash(self, "config_hash")
        if not self.config_hash:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise PublicFirewallError("config_hash does not match public firewall config payload.")


@dataclass(frozen=True)
class PublicFirewallReport:
    decision: str
    public_score_delta: float
    private_like_score_delta: float
    internal_score_delta: float
    stability_score_delta: float
    public_driven_change_count: int
    max_public_driven_changes: int = 2
    public_gain_epsilon: float = 0.0
    private_regression_tolerance: float = 0.0
    stability_drop_tolerance: float = 0.0
    reasons: tuple[str, ...] = ()
    report_hash: str = ""

    def __post_init__(self) -> None:
        if self.decision not in PUBLIC_FIREWALL_DECISIONS:
            raise PublicFirewallError("decision must be allow, reject, or investigate.")
        for name in ("public_score_delta", "private_like_score_delta", "internal_score_delta", "stability_score_delta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PublicFirewallError(f"{name} must be numeric.")
            if math.isnan(float(value)) or math.isinf(float(value)):
                raise PublicFirewallError(f"{name} must be finite.")
            object.__setattr__(self, name, float(value))
        for name in ("max_public_driven_changes", "public_driven_change_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise PublicFirewallError(f"{name} must be a non-negative integer.")
        for name in ("public_gain_epsilon", "private_regression_tolerance", "stability_drop_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0.0 or math.isnan(float(value)) or math.isinf(float(value)):
                raise PublicFirewallError(f"{name} must be a finite non-negative number.")
            object.__setattr__(self, name, float(value))
        expected_decision, expected_reasons = _policy_decision(
            self.public_score_delta,
            self.private_like_score_delta,
            self.internal_score_delta,
            self.stability_score_delta,
            self.public_driven_change_count,
            self.max_public_driven_changes,
            self.public_gain_epsilon,
            self.private_regression_tolerance,
            self.stability_drop_tolerance,
        )
        if self.decision == "allow" and expected_decision != "allow":
            raise PublicFirewallError("decision=allow violates public firewall policy.")
        object.__setattr__(self, "reasons", tuple(str(item) for item in self.reasons))
        expected = _payload_hash(self, "report_hash")
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected)
        elif self.report_hash != expected:
            raise PublicFirewallError("report_hash does not match public firewall report payload.")


def evaluate_public_firewall(config: PublicFirewallConfig) -> PublicFirewallReport:
    """Reject or investigate leaderboard-driven changes that lack private support."""

    if not isinstance(config, PublicFirewallConfig):
        raise PublicFirewallError("config must be a PublicFirewallConfig.")
    decision, reasons = _policy_decision(
        config.public_score_delta,
        config.private_like_score_delta,
        config.internal_score_delta,
        config.stability_score_delta,
        config.public_driven_change_count,
        config.max_public_driven_changes,
        config.public_gain_epsilon,
        config.private_regression_tolerance,
        config.stability_drop_tolerance,
    )
    return PublicFirewallReport(
        decision=decision,
        public_score_delta=config.public_score_delta,
        private_like_score_delta=config.private_like_score_delta,
        internal_score_delta=config.internal_score_delta,
        stability_score_delta=config.stability_score_delta,
        public_driven_change_count=config.public_driven_change_count,
        max_public_driven_changes=config.max_public_driven_changes,
        public_gain_epsilon=config.public_gain_epsilon,
        private_regression_tolerance=config.private_regression_tolerance,
        stability_drop_tolerance=config.stability_drop_tolerance,
        reasons=tuple(reasons),
    )


def _policy_decision(
    public_score_delta: float,
    private_like_score_delta: float,
    internal_score_delta: float,
    stability_score_delta: float,
    public_driven_change_count: int,
    max_public_driven_changes: int,
    public_gain_epsilon: float,
    private_regression_tolerance: float,
    stability_drop_tolerance: float,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    decision = "allow"
    public_gain = public_score_delta > public_gain_epsilon
    private_regression = private_like_score_delta < -private_regression_tolerance
    stability_drop = stability_score_delta < -stability_drop_tolerance
    private_gain = private_like_score_delta > 0
    internal_gain = internal_score_delta > 0
    if public_driven_change_count > max_public_driven_changes:
        decision = "reject"
        reasons.append("too_many_public_driven_changes")
    if public_gain and private_regression:
        decision = "reject"
        reasons.append("public_gain_private_regression")
    if public_gain and stability_drop:
        decision = "reject"
        reasons.append("public_gain_stability_drop")
    if decision != "reject" and public_gain and not private_gain and not internal_gain:
        decision = "investigate"
        reasons.append("public_only_improvement")
    if decision != "reject" and public_score_delta < 0 and (private_gain or internal_gain):
        decision = "investigate"
        reasons.append("public_drop_internal_gain")
    return decision, reasons


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


__all__ = [
    "PUBLIC_FIREWALL_DECISIONS",
    "PublicFirewallConfig",
    "PublicFirewallError",
    "PublicFirewallReport",
    "evaluate_public_firewall",
]
