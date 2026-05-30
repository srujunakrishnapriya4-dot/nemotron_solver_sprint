"""Symbolic reward-audit gates for future GRPO admission."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from nemotron_engine.core.schemas import stable_hash


class RewardAuditError(ValueError):
    """Raised when reward-audit inputs or reports are invalid."""


@dataclass(frozen=True)
class RewardAuditConfig:
    min_samples: int = 100
    length_gaming_threshold: float = 0.01
    heldout_regression_threshold: float = 0.01

    def __post_init__(self) -> None:
        if isinstance(self.min_samples, bool) or not isinstance(self.min_samples, int) or self.min_samples < 0:
            raise RewardAuditError("min_samples must be a non-negative integer.")
        for name in ("length_gaming_threshold", "heldout_regression_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise RewardAuditError(f"{name} must be non-negative.")


@dataclass(frozen=True)
class RewardAuditReport:
    sample_count: int
    false_positive_rate: float
    false_negative_rate: float
    format_failure_rate: float
    leakage_reward_rate: float
    length_gaming_rate: float
    heldout_regression_rate: float
    accepted: bool
    rejection_reasons: tuple[str, ...] = ()
    report_hash: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int) or self.sample_count < 0:
            raise RewardAuditError("sample_count must be a non-negative integer.")
        for name in (
            "false_positive_rate",
            "false_negative_rate",
            "format_failure_rate",
            "leakage_reward_rate",
            "length_gaming_rate",
            "heldout_regression_rate",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                raise RewardAuditError(f"{name} must be in [0, 1].")
            object.__setattr__(self, name, float(value))
        if not isinstance(self.accepted, bool):
            raise RewardAuditError("accepted must be boolean.")
        object.__setattr__(self, "rejection_reasons", tuple(str(item) for item in self.rejection_reasons))
        hard_reasons = _hard_threshold_rejection_reasons(
            sample_count=self.sample_count,
            false_positive_rate=self.false_positive_rate,
            format_failure_rate=self.format_failure_rate,
            leakage_reward_rate=self.leakage_reward_rate,
            length_gaming_rate=self.length_gaming_rate,
            heldout_regression_rate=self.heldout_regression_rate,
        )
        if self.accepted and hard_reasons:
            raise RewardAuditError("accepted reward audit violates hard thresholds.")
        if self.accepted and self.rejection_reasons:
            raise RewardAuditError("accepted reward audits cannot include rejection reasons.")
        if not self.accepted and not self.rejection_reasons:
            raise RewardAuditError("rejected reward audits require reasons.")
        expected = _payload_hash(self, "report_hash")
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected)
        elif self.report_hash != expected:
            raise RewardAuditError("report_hash does not match reward audit payload.")


def audit_reward_model_signals(
    *,
    sample_count: int,
    false_positive_rate: float,
    false_negative_rate: float = 0.0,
    format_failure_rate: float,
    leakage_reward_rate: float,
    length_gaming_rate: float,
    heldout_regression_rate: float,
    config: RewardAuditConfig | None = None,
) -> RewardAuditReport:
    cfg = config or RewardAuditConfig()
    reasons = _reward_rejection_reasons(
        sample_count=sample_count,
        false_positive_rate=false_positive_rate,
        format_failure_rate=format_failure_rate,
        leakage_reward_rate=leakage_reward_rate,
        length_gaming_rate=length_gaming_rate,
        heldout_regression_rate=heldout_regression_rate,
        config=cfg,
    )
    return RewardAuditReport(
        sample_count=sample_count,
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
        format_failure_rate=format_failure_rate,
        leakage_reward_rate=leakage_reward_rate,
        length_gaming_rate=length_gaming_rate,
        heldout_regression_rate=heldout_regression_rate,
        accepted=not reasons,
        rejection_reasons=tuple(reasons),
    )


def grpo_admission_decision(report: RewardAuditReport) -> bool:
    if not isinstance(report, RewardAuditReport):
        raise RewardAuditError("report must be a RewardAuditReport.")
    hard_reasons = _hard_threshold_rejection_reasons(
        sample_count=report.sample_count,
        false_positive_rate=report.false_positive_rate,
        format_failure_rate=report.format_failure_rate,
        leakage_reward_rate=report.leakage_reward_rate,
        length_gaming_rate=report.length_gaming_rate,
        heldout_regression_rate=report.heldout_regression_rate,
    )
    return report.accepted is True and not hard_reasons


def _reward_rejection_reasons(
    *,
    sample_count: int,
    false_positive_rate: float,
    format_failure_rate: float,
    leakage_reward_rate: float,
    length_gaming_rate: float,
    heldout_regression_rate: float,
    config: RewardAuditConfig,
) -> list[str]:
    reasons: list[str] = []
    if sample_count < config.min_samples:
        reasons.append("insufficient_samples")
    if false_positive_rate > 0.01:
        reasons.append("false_positive_rate")
    if format_failure_rate > 0.005:
        reasons.append("format_failure_rate")
    if leakage_reward_rate > 0:
        reasons.append("leakage_reward_rate")
    if length_gaming_rate > config.length_gaming_threshold:
        reasons.append("length_gaming_rate")
    if heldout_regression_rate > config.heldout_regression_threshold:
        reasons.append("heldout_regression_rate")
    return reasons


def _hard_threshold_rejection_reasons(
    *,
    sample_count: int,
    false_positive_rate: float,
    format_failure_rate: float,
    leakage_reward_rate: float,
    length_gaming_rate: float,
    heldout_regression_rate: float,
) -> list[str]:
    return _reward_rejection_reasons(
        sample_count=sample_count,
        false_positive_rate=false_positive_rate,
        format_failure_rate=format_failure_rate,
        leakage_reward_rate=leakage_reward_rate,
        length_gaming_rate=length_gaming_rate,
        heldout_regression_rate=heldout_regression_rate,
        config=RewardAuditConfig(),
    )


def _payload_hash(instance: Any, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


__all__ = [
    "RewardAuditConfig",
    "RewardAuditError",
    "RewardAuditReport",
    "audit_reward_model_signals",
    "grpo_admission_decision",
]
