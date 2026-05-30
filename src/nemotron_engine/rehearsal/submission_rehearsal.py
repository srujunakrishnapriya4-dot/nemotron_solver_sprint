"""Submission rehearsal gate for Pass 14.

This module never submits, never calls Kaggle, and never writes a package.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.packaging import PreflightGateReport
from nemotron_engine.release import ReleaseCandidateReport

from .package_rehearsal import PackageRehearsalReport, validate_package_rehearsal_report
from .promotion_bridge import PromotionBridgeReport, validate_promotion_rehearsal_report
from .rehearsal_config import RehearsalConfig, reject_unsafe_metadata_claims, validate_rehearsal_config
from .runtime_rehearsal import RuntimeRehearsalReport, validate_runtime_rehearsal_report


class SubmissionRehearsalError(ValueError):
    """Raised when submission rehearsal evidence is unsafe."""


@dataclass(frozen=True)
class SubmissionRehearsalConfig:
    require_preflight: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.require_preflight, bool):
            raise SubmissionRehearsalError("require_preflight must be boolean.")
        metadata = dict(self.metadata)
        try:
            reject_unsafe_metadata_claims(metadata, allow_adapter_evidence=False)
        except Exception as exc:
            raise SubmissionRehearsalError(str(exc)) from exc
        object.__setattr__(self, "metadata", metadata)
        expected = _payload_hash(self, "config_hash")
        if not self.config_hash:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise SubmissionRehearsalError("config_hash does not match submission rehearsal config payload.")


@dataclass(frozen=True)
class SubmissionRehearsalReport:
    rehearsal_config_hash: str
    submission_rehearsal_config_hash: str
    promotion_bridge_hash: str
    package_rehearsal_hash: str
    runtime_rehearsal_hash: str
    preflight_gate_hash: str | None
    release_candidate_hash: str | None
    dry_run: bool
    submitted: bool
    passed: bool
    promotion_allowed: bool = False
    package_passed: bool = False
    runtime_passed: bool = False
    preflight_required: bool = False
    preflight_passed: bool | None = None
    release_candidate_required: bool = False
    release_candidate_ready: bool | None = None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    report_hash: str = ""

    def __post_init__(self) -> None:
        for name in (
            "rehearsal_config_hash",
            "submission_rehearsal_config_hash",
            "promotion_bridge_hash",
            "package_rehearsal_hash",
            "runtime_rehearsal_hash",
        ):
            _require_non_empty(getattr(self, name), name)
        for name in ("preflight_gate_hash", "release_candidate_hash"):
            value = getattr(self, name)
            if value is not None:
                _require_non_empty(value, name)
        bool_values = (
            self.dry_run,
            self.submitted,
            self.passed,
            self.promotion_allowed,
            self.package_passed,
            self.runtime_passed,
            self.preflight_required,
            self.release_candidate_required,
        )
        if not all(isinstance(value, bool) for value in bool_values):
            raise SubmissionRehearsalError("submission rehearsal booleans must be bool values.")
        for name in ("preflight_passed", "release_candidate_ready"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise SubmissionRehearsalError(f"{name} must be boolean when supplied.")
        if self.submitted:
            raise SubmissionRehearsalError("submitted=True is impossible in Pass 14 rehearsal.")
        errors = tuple(str(item) for item in self.errors)
        warnings = tuple(str(item) for item in self.warnings)
        if self.passed and errors:
            raise SubmissionRehearsalError("passed=True cannot include errors.")
        if self.passed:
            if not (self.promotion_allowed and self.package_passed and self.runtime_passed):
                raise SubmissionRehearsalError("passed=True requires passing promotion/package/runtime component evidence.")
            if self.preflight_required and (not self.preflight_gate_hash or self.preflight_passed is not True):
                raise SubmissionRehearsalError("passed=True requires passing preflight evidence when preflight is required.")
            if self.release_candidate_required and (not self.release_candidate_hash or self.release_candidate_ready is not True):
                raise SubmissionRehearsalError("passed=True requires ready release candidate evidence when release candidate is required.")
        metadata = dict(self.metadata)
        try:
            reject_unsafe_metadata_claims(metadata, allow_adapter_evidence=False)
        except Exception as exc:
            raise SubmissionRehearsalError(str(exc)) from exc
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "metadata", metadata)
        expected = _payload_hash(self, "report_hash")
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected)
        elif self.report_hash != expected:
            raise SubmissionRehearsalError("report_hash does not match submission rehearsal payload.")


def run_submission_rehearsal(
    *,
    rehearsal_config: RehearsalConfig,
    promotion_bridge_report: PromotionBridgeReport,
    package_rehearsal_report: PackageRehearsalReport,
    runtime_rehearsal_report: RuntimeRehearsalReport,
    preflight_gate_report: PreflightGateReport | None = None,
    release_candidate_report: ReleaseCandidateReport | None = None,
    metadata: Mapping[str, Any] | None = None,
    config: SubmissionRehearsalConfig | None = None,
) -> SubmissionRehearsalReport:
    rehearsal = validate_rehearsal_config(rehearsal_config)
    bridge = validate_promotion_rehearsal_report(promotion_bridge_report)
    package = validate_package_rehearsal_report(package_rehearsal_report)
    runtime = validate_runtime_rehearsal_report(runtime_rehearsal_report)
    cfg = config or SubmissionRehearsalConfig()
    errors: list[str] = []
    warnings: list[str] = []
    preflight_hash = None
    release_hash = None
    preflight_passed = None
    release_ready = None
    preflight_required = bool(cfg.require_preflight)
    release_required = bool(rehearsal.require_release_candidate)
    if bridge.allowed is not True:
        errors.append("promotion_bridge_failed")
    if package.passed is not True:
        errors.append("package_rehearsal_failed")
    if runtime.passed is not True:
        errors.append("runtime_rehearsal_failed")
    warnings.extend(package.warnings)
    warnings.extend(runtime.warnings)
    if preflight_gate_report is not None:
        _validate_report_hash(preflight_gate_report, "gate_hash")
        preflight_hash = preflight_gate_report.gate_hash
        preflight_passed = preflight_gate_report.passed
        if preflight_gate_report.passed is not True:
            errors.append("preflight_gate_failed")
    elif preflight_required:
        errors.append("preflight_gate_missing")
    if release_candidate_report is not None:
        _validate_report_hash(release_candidate_report, "report_hash")
        release_hash = release_candidate_report.report_hash
        release_ready = release_candidate_report.ready
        if rehearsal.require_release_candidate and release_candidate_report.ready is not True:
            errors.append("release_candidate_not_ready")
    elif rehearsal.require_release_candidate:
        errors.append("release_candidate_missing")
    if warnings and not rehearsal.allow_warnings:
        errors.append("warnings_not_allowed")
    return SubmissionRehearsalReport(
        rehearsal_config_hash=rehearsal.config_hash,
        submission_rehearsal_config_hash=cfg.config_hash,
        promotion_bridge_hash=bridge.report_hash,
        package_rehearsal_hash=package.report_hash,
        runtime_rehearsal_hash=runtime.report_hash,
        preflight_gate_hash=preflight_hash,
        release_candidate_hash=release_hash,
        dry_run=rehearsal.dry_run,
        submitted=False,
        passed=not errors,
        promotion_allowed=bridge.allowed,
        package_passed=package.passed,
        runtime_passed=runtime.passed,
        preflight_required=preflight_required,
        preflight_passed=preflight_passed,
        release_candidate_required=release_required,
        release_candidate_ready=release_ready,
        errors=tuple(sorted(set(errors))),
        warnings=tuple(sorted(set(warnings))),
        metadata=dict(metadata or {}),
    )


def validate_submission_rehearsal_report(report: SubmissionRehearsalReport | Mapping[str, Any]) -> SubmissionRehearsalReport:
    normalized = report if isinstance(report, SubmissionRehearsalReport) else SubmissionRehearsalReport(**dict(report))
    if normalized.report_hash != _payload_hash(normalized, "report_hash"):
        raise SubmissionRehearsalError("report_hash does not match submission rehearsal payload.")
    if normalized.submitted:
        raise SubmissionRehearsalError("submitted=True is impossible in Pass 14 rehearsal.")
    if normalized.passed and normalized.errors:
        raise SubmissionRehearsalError("passed=True cannot include errors.")
    return normalized


def _validate_report_hash(report: object, hash_field: str) -> None:
    try:
        observed = getattr(report, hash_field)
        expected = stable_hash({item.name: getattr(report, item.name) for item in fields(report) if item.name != hash_field})
    except Exception as exc:
        raise SubmissionRehearsalError("could not validate input report hash.") from exc
    if observed != expected:
        raise SubmissionRehearsalError("input report hash mismatch.")


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SubmissionRehearsalError(f"{field_name} must be non-empty.")
    return value


__all__ = [
    "SubmissionRehearsalConfig",
    "SubmissionRehearsalError",
    "SubmissionRehearsalReport",
    "run_submission_rehearsal",
    "validate_submission_rehearsal_report",
]
