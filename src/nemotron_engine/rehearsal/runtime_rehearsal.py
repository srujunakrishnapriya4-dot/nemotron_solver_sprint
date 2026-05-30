"""Runtime rehearsal over Pass 9 runtime evidence."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.packaging import RuntimeValidationReport

from .adapter_evidence import AdapterEvidenceBundle, validate_adapter_evidence_bundle
from .package_rehearsal import PackageRehearsalReport, validate_package_rehearsal_report
from .rehearsal_config import RehearsalConfig, reject_unsafe_metadata_claims, validate_rehearsal_config


class RuntimeRehearsalError(ValueError):
    """Raised when runtime rehearsal evidence is unsafe."""


@dataclass(frozen=True)
class RuntimeRehearsalConfig:
    expected_adapter_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        if self.expected_adapter_hash is not None and (not isinstance(self.expected_adapter_hash, str) or not self.expected_adapter_hash.strip()):
            raise RuntimeRehearsalError("expected_adapter_hash must be non-empty when supplied.")
        metadata = dict(self.metadata)
        try:
            reject_unsafe_metadata_claims(metadata, allow_adapter_evidence=True)
        except Exception as exc:
            raise RuntimeRehearsalError(str(exc)) from exc
        object.__setattr__(self, "metadata", metadata)
        expected = _payload_hash(self, "config_hash")
        if not self.config_hash:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise RuntimeRehearsalError("config_hash does not match runtime rehearsal config payload.")


@dataclass(frozen=True)
class RuntimeRehearsalReport:
    rehearsal_config_hash: str
    runtime_rehearsal_config_hash: str
    adapter_hash: str
    adapter_evidence_hash: str
    package_rehearsal_hash: str
    runtime_validation_hash: str | None
    runtime_loaded: bool
    model_loaded: bool
    adapter_loaded: bool
    backend_validated: bool
    unloaded_dry_run: bool
    passed: bool
    allow_runtime_unloaded: bool = False
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    report_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("rehearsal_config_hash", "runtime_rehearsal_config_hash", "adapter_hash", "adapter_evidence_hash", "package_rehearsal_hash"):
            _require_non_empty(getattr(self, name), name)
        if self.runtime_validation_hash is not None:
            _require_non_empty(self.runtime_validation_hash, "runtime_validation_hash")
        if not all(
            isinstance(value, bool)
            for value in (
                self.runtime_loaded,
                self.model_loaded,
                self.adapter_loaded,
                self.backend_validated,
                self.unloaded_dry_run,
                self.passed,
                self.allow_runtime_unloaded,
            )
        ):
            raise RuntimeRehearsalError("runtime rehearsal boolean fields must be bool values.")
        if (self.runtime_loaded or self.model_loaded or self.adapter_loaded) and (
            not self.runtime_validation_hash or not self.backend_validated or self.unloaded_dry_run
        ):
            raise RuntimeRehearsalError("loaded runtime flags require validated runtime evidence.")
        errors = tuple(str(item) for item in self.errors)
        warnings = tuple(str(item) for item in self.warnings)
        if self.passed and errors:
            raise RuntimeRehearsalError("passed=True cannot include errors.")
        if self.passed:
            validated_runtime = bool(self.runtime_validation_hash and self.backend_validated and not self.unloaded_dry_run)
            allowed_unloaded = bool(self.unloaded_dry_run and self.allow_runtime_unloaded and self.runtime_validation_hash is None)
            if not (validated_runtime or allowed_unloaded):
                raise RuntimeRehearsalError("passed=True requires validated runtime evidence or explicitly allowed unloaded dry-run.")
        metadata = dict(self.metadata)
        _reject_runtime_success_metadata(metadata)
        try:
            reject_unsafe_metadata_claims(metadata, allow_adapter_evidence=True)
        except Exception as exc:
            raise RuntimeRehearsalError(str(exc)) from exc
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "metadata", metadata)
        expected = _payload_hash(self, "report_hash")
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected)
        elif self.report_hash != expected:
            raise RuntimeRehearsalError("report_hash does not match runtime rehearsal payload.")


def run_runtime_rehearsal(
    *,
    rehearsal_config: RehearsalConfig,
    adapter_evidence_bundle: AdapterEvidenceBundle,
    package_rehearsal_report: PackageRehearsalReport,
    runtime_validation_report: RuntimeValidationReport | None = None,
    metadata: Mapping[str, Any] | None = None,
    config: RuntimeRehearsalConfig | None = None,
) -> RuntimeRehearsalReport:
    rehearsal = validate_rehearsal_config(rehearsal_config)
    evidence = validate_adapter_evidence_bundle(adapter_evidence_bundle)
    package = validate_package_rehearsal_report(package_rehearsal_report)
    cfg = config or RuntimeRehearsalConfig(expected_adapter_hash=evidence.adapter_hash)
    errors: list[str] = []
    warnings: list[str] = []
    runtime_hash = None
    runtime_loaded = model_loaded = adapter_loaded = backend_validated = False
    unloaded = False
    if package.passed is not True:
        errors.append("package_rehearsal_failed")
    if package.adapter_hash != evidence.adapter_hash:
        errors.append("package_adapter_hash_mismatch")
    expected_adapter = cfg.expected_adapter_hash or evidence.adapter_hash
    if expected_adapter != evidence.adapter_hash:
        errors.append("adapter_hash_mismatch")
    meta = dict(metadata or {})
    runtime_adapter_hash = meta.get("runtime_adapter_hash") or meta.get("adapter_hash")
    if runtime_adapter_hash is not None and runtime_adapter_hash != evidence.adapter_hash:
        errors.append("runtime_adapter_hash_mismatch")
    if runtime_validation_report is None:
        unloaded = True
        warnings.append("runtime_unloaded_dry_run")
        if rehearsal.allow_runtime_unloaded is not True:
            errors.append("runtime_validation_missing")
    else:
        _validate_report_hash(runtime_validation_report, "report_hash")
        runtime_hash = runtime_validation_report.report_hash
        runtime_loaded = runtime_validation_report.runtime_loaded
        model_loaded = runtime_validation_report.model_loaded
        adapter_loaded = runtime_validation_report.adapter_loaded
        backend_validated = runtime_validation_report.backend_validated
        warnings.extend(runtime_validation_report.warnings)
        if runtime_validation_report.passed is not True:
            errors.append("runtime_validation_failed")
        if (runtime_loaded or model_loaded or adapter_loaded) and not backend_validated:
            errors.append("runtime_fake_loaded_success")
        if runtime_validation_report.dry_run and (runtime_loaded or model_loaded or adapter_loaded):
            errors.append("runtime_fake_loaded_success")
    if warnings and not rehearsal.allow_warnings:
        errors.append("warnings_not_allowed")
    passed = not errors and (runtime_validation_report is not None or rehearsal.allow_runtime_unloaded)
    return RuntimeRehearsalReport(
        rehearsal_config_hash=rehearsal.config_hash,
        runtime_rehearsal_config_hash=cfg.config_hash,
        adapter_hash=evidence.adapter_hash,
        adapter_evidence_hash=evidence.evidence_hash,
        package_rehearsal_hash=package.report_hash,
        runtime_validation_hash=runtime_hash,
        runtime_loaded=runtime_loaded,
        model_loaded=model_loaded,
        adapter_loaded=adapter_loaded,
        backend_validated=backend_validated,
        unloaded_dry_run=unloaded,
        passed=passed,
        allow_runtime_unloaded=rehearsal.allow_runtime_unloaded,
        errors=tuple(sorted(set(errors))),
        warnings=tuple(sorted(set(warnings))),
        metadata=meta,
    )


def validate_runtime_rehearsal_report(report: RuntimeRehearsalReport | Mapping[str, Any]) -> RuntimeRehearsalReport:
    normalized = report if isinstance(report, RuntimeRehearsalReport) else RuntimeRehearsalReport(**dict(report))
    if normalized.report_hash != _payload_hash(normalized, "report_hash"):
        raise RuntimeRehearsalError("report_hash does not match runtime rehearsal payload.")
    if normalized.passed and normalized.errors:
        raise RuntimeRehearsalError("passed=True cannot include errors.")
    return normalized


def _validate_report_hash(report: object, hash_field: str) -> None:
    try:
        observed = getattr(report, hash_field)
        expected = stable_hash({item.name: getattr(report, item.name) for item in fields(report) if item.name != hash_field})
    except Exception as exc:
        raise RuntimeRehearsalError("could not validate input report hash.") from exc
    if observed != expected:
        raise RuntimeRehearsalError("input report hash mismatch.")


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeRehearsalError(f"{field_name} must be non-empty.")
    return value


def _reject_runtime_success_metadata(metadata: Mapping[str, Any]) -> None:
    forbidden_phrases = (
        "runtime_success",
        "runtime loaded",
        "model loaded",
        "adapter loaded",
        "successful runtime",
    )
    for key, value in metadata.items():
        key_text = str(key).lower().replace("-", "_")
        value_text = str(value).lower().replace("-", "_") if isinstance(value, str) else ""
        spaced_key = key_text.replace("_", " ")
        spaced_value = value_text.replace("_", " ")
        if (key_text == "runtime_success" or any(phrase in spaced_key or phrase in spaced_value for phrase in forbidden_phrases)) and value not in (False, None, ""):
            raise RuntimeRehearsalError(f"forbidden runtime success metadata claim: {key}.")


__all__ = [
    "RuntimeRehearsalConfig",
    "RuntimeRehearsalError",
    "RuntimeRehearsalReport",
    "run_runtime_rehearsal",
    "validate_runtime_rehearsal_report",
]
