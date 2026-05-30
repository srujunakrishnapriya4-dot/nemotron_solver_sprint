"""Deterministic smoke-training reports for Pass 12."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash

from .smoke_config import SmokeTrainingConfigError, _reject_unsafe_metadata


class SmokeTrainingReportError(ValueError):
    """Raised when a smoke-training report is inconsistent."""


@dataclass(frozen=True)
class SmokeTrainingReport:
    smoke_config_hash: str
    smoke_dataset_hash: str
    backend_invocation_hash: str
    backend_result_hash: str | None
    adapter_validation_hash: str | None
    training_log_hash: str | None
    training_run_manifest_hash: str | None
    trained: bool
    passed: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    report_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("smoke_config_hash", "smoke_dataset_hash", "backend_invocation_hash"):
            _require_non_empty(getattr(self, name), name)
        for name in ("backend_result_hash", "adapter_validation_hash", "training_log_hash", "training_run_manifest_hash"):
            value = getattr(self, name)
            if value is not None:
                _require_non_empty(value, name)
        if not isinstance(self.trained, bool) or not isinstance(self.passed, bool):
            raise SmokeTrainingReportError("trained and passed must be booleans.")
        errors = tuple(str(item) for item in self.errors)
        warnings = tuple(str(item) for item in self.warnings)
        metadata = dict(self.metadata)
        _reject_metadata(metadata)
        require_log_validation = metadata.get("require_log_validation") is not False
        dry_run = metadata.get("dry_run") is True
        if self.passed and errors:
            raise SmokeTrainingReportError("passed=True cannot include errors.")
        if self.trained:
            if not self.backend_result_hash:
                raise SmokeTrainingReportError("trained=True requires backend_result_hash.")
            if not self.adapter_validation_hash:
                raise SmokeTrainingReportError("trained=True requires adapter_validation_hash.")
            if not self.training_run_manifest_hash:
                raise SmokeTrainingReportError("trained=True requires training_run_manifest_hash.")
            if require_log_validation and not self.training_log_hash:
                raise SmokeTrainingReportError("require_log_validation=True requires training_log_hash.")
        if self.passed and not self.trained:
            if dry_run is not True:
                raise SmokeTrainingReportError("passed=True requires trained=True unless metadata dry_run=True.")
            if self.backend_result_hash:
                raise SmokeTrainingReportError("dry-run pass cannot contain backend result evidence.")
            if self.adapter_validation_hash or self.training_run_manifest_hash:
                raise SmokeTrainingReportError("dry-run pass cannot contain adapter or training manifest evidence.")
        if dry_run and not self.trained and (self.backend_result_hash or self.adapter_validation_hash or self.training_run_manifest_hash):
            raise SmokeTrainingReportError("dry-run pass cannot contain adapter or training manifest evidence.")
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "metadata", metadata)
        expected = compute_smoke_training_report_hash(self)
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected)
        elif self.report_hash != expected:
            raise SmokeTrainingReportError("report_hash does not match smoke training report payload.")


def build_smoke_training_report(
    *,
    smoke_config_hash: str,
    smoke_dataset_hash: str,
    backend_invocation_hash: str,
    backend_result_hash: str | None,
    adapter_validation_hash: str | None,
    training_log_hash: str | None,
    training_run_manifest_hash: str | None,
    trained: bool,
    passed: bool,
    errors: tuple[str, ...] = (),
    warnings: tuple[str, ...] = (),
    metadata: Mapping[str, Any] | None = None,
) -> SmokeTrainingReport:
    return SmokeTrainingReport(
        smoke_config_hash=smoke_config_hash,
        smoke_dataset_hash=smoke_dataset_hash,
        backend_invocation_hash=backend_invocation_hash,
        backend_result_hash=backend_result_hash,
        adapter_validation_hash=adapter_validation_hash,
        training_log_hash=training_log_hash,
        training_run_manifest_hash=training_run_manifest_hash,
        trained=trained,
        passed=passed,
        errors=errors,
        warnings=warnings,
        metadata=dict(metadata or {}),
    )


def validate_smoke_training_report(report: SmokeTrainingReport | Mapping[str, Any]) -> SmokeTrainingReport:
    normalized = report if isinstance(report, SmokeTrainingReport) else SmokeTrainingReport(**dict(report))
    if normalized.report_hash != compute_smoke_training_report_hash(normalized):
        raise SmokeTrainingReportError("report_hash does not match smoke training report payload.")
    return normalized


def compute_smoke_training_report_hash(report: SmokeTrainingReport | Mapping[str, Any]) -> str:
    payload = dict(report) if isinstance(report, Mapping) else {item.name: getattr(report, item.name) for item in fields(SmokeTrainingReport)}
    payload.pop("report_hash", None)
    return stable_hash(payload)


def _reject_metadata(metadata: Mapping[str, Any]) -> None:
    try:
        _reject_unsafe_metadata(metadata, reject_split_claims=True)
    except SmokeTrainingConfigError as exc:
        raise SmokeTrainingReportError(str(exc)) from exc


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SmokeTrainingReportError(f"{field_name} must be non-empty.")
    return value


__all__ = [
    "SmokeTrainingReport",
    "SmokeTrainingReportError",
    "build_smoke_training_report",
    "compute_smoke_training_report_hash",
    "validate_smoke_training_report",
]
