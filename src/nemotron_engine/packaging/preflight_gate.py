"""Final offline preflight gate for Pass 9 packages."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.evaluation.promotion_gates import PromotionDecision, PromotionGateReport
from nemotron_engine.evaluation.submission_exact import SubmissionExactReport

from .artifact_audit import ArtifactAuditReport
from .package_builder import PackageBuildReport
from .runtime_validator import RuntimeValidationReport
from .submission_manifest import SubmissionManifest, validate_submission_manifest


class PreflightGateError(ValueError):
    """Raised when preflight reports are inconsistent or unsafe."""


@dataclass(frozen=True)
class PreflightGateConfig:
    allow_package_dry_run: bool = False
    allow_runtime_dry_run_unloaded: bool = False
    allow_warnings: bool = False
    required_hash_fields: tuple[str, ...] = (
        "promotion_decision_hash",
        "serving_config_hash",
        "training_plan_hash",
        "dataset_manifest_hash",
        "trace_manifest_hash",
        "lora_config_hash",
        "adapter_hash",
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        if not all(isinstance(value, bool) for value in (self.allow_package_dry_run, self.allow_runtime_dry_run_unloaded, self.allow_warnings)):
            raise PreflightGateError("preflight allow flags must be booleans.")
        if any(not isinstance(item, str) or not item.strip() for item in self.required_hash_fields):
            raise PreflightGateError("required_hash_fields must contain non-empty strings.")
        object.__setattr__(self, "required_hash_fields", tuple(self.required_hash_fields))
        object.__setattr__(self, "metadata", dict(self.metadata))
        expected = _payload_hash(self, "config_hash")
        if not self.config_hash:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise PreflightGateError("config_hash does not match preflight config payload.")


@dataclass(frozen=True)
class PreflightGateReport:
    passed: bool
    failure_reasons: tuple[str, ...]
    input_report_hashes: Mapping[str, str]
    required_hashes: Mapping[str, str | None]
    warnings: tuple[str, ...] = ()
    warnings_allowed: bool = False
    runtime_fake_success: bool = False
    package_dry_run_claimed_built: bool = False
    gate_hash: str = ""

    def __post_init__(self) -> None:
        if not all(isinstance(value, bool) for value in (self.passed, self.warnings_allowed, self.runtime_fake_success, self.package_dry_run_claimed_built)):
            raise PreflightGateError("preflight report booleans must be bool values.")
        failures = tuple(str(item) for item in self.failure_reasons)
        warnings = tuple(str(item) for item in self.warnings)
        input_hashes = {str(key): str(value) for key, value in sorted(self.input_report_hashes.items())}
        required_hashes = {str(key): (None if value is None else str(value)) for key, value in sorted(self.required_hashes.items())}
        if self.passed and failures:
            raise PreflightGateError("passed=True cannot include failure_reasons.")
        if not self.passed and not failures:
            raise PreflightGateError("passed=False requires failure_reasons.")
        if self.passed and any(not _non_empty(value) for value in required_hashes.values()):
            raise PreflightGateError("passed=True requires all required_hashes to be non-empty.")
        if self.passed and warnings and not self.warnings_allowed:
            raise PreflightGateError("passed=True cannot include warnings unless warnings are explicitly allowed.")
        if self.passed and self.runtime_fake_success:
            raise PreflightGateError("passed=True cannot include runtime fake success.")
        if self.passed and self.package_dry_run_claimed_built:
            raise PreflightGateError("passed=True cannot include package dry-run built claim.")
        if any(not value for value in input_hashes.values()):
            raise PreflightGateError("input_report_hashes must be non-empty.")
        object.__setattr__(self, "failure_reasons", failures)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "input_report_hashes", input_hashes)
        object.__setattr__(self, "required_hashes", required_hashes)
        expected = _payload_hash(self, "gate_hash")
        if not self.gate_hash:
            object.__setattr__(self, "gate_hash", expected)
        elif self.gate_hash != expected:
            raise PreflightGateError("gate_hash does not match preflight report payload.")


def evaluate_preflight_gate(
    *,
    submission_manifest: SubmissionManifest,
    package_build_report: PackageBuildReport,
    runtime_validation_report: RuntimeValidationReport,
    artifact_audit_report: ArtifactAuditReport,
    promotion_gate_report: PromotionGateReport,
    submission_exact_report: SubmissionExactReport,
    metadata: Mapping[str, Any] | None = None,
    config: PreflightGateConfig | None = None,
) -> PreflightGateReport:
    cfg = config or PreflightGateConfig()
    _validate_report_hash(package_build_report, "report_hash")
    _validate_report_hash(runtime_validation_report, "report_hash")
    _validate_report_hash(artifact_audit_report, "report_hash")
    _validate_report_hash(promotion_gate_report, "decision_hash")
    _validate_report_hash(submission_exact_report, "report_hash")
    try:
        validate_submission_manifest(submission_manifest)
    except Exception as exc:
        raise PreflightGateError("submission manifest failed validation.") from exc
    failures: list[str] = []
    warnings: list[str] = []
    if promotion_gate_report.decision != PromotionDecision.ALLOW or promotion_gate_report.passed is not True:
        failures.append("promotion_not_allow")
    if submission_exact_report.passed is not True:
        failures.append("submission_exact_failed")
    if package_build_report.dry_run and package_build_report.built:
        failures.append("package_dry_run_claimed_built")
    if package_build_report.dry_run and not cfg.allow_package_dry_run:
        failures.append("package_dry_run_not_allowed")
    if package_build_report.passed is not True:
        failures.append("package_build_failed")
    if runtime_validation_report.dry_run and (
        runtime_validation_report.runtime_loaded or runtime_validation_report.model_loaded or runtime_validation_report.adapter_loaded
    ):
        failures.append("runtime_fake_success")
    if runtime_validation_report.dry_run and not runtime_validation_report.backend_validated and not cfg.allow_runtime_dry_run_unloaded:
        failures.append("runtime_unloaded_not_allowed")
    if runtime_validation_report.passed is not True:
        failures.append("runtime_validation_failed")
    if artifact_audit_report.passed is not True:
        failures.append("artifact_audit_failed")
    if artifact_audit_report.forbidden_paths:
        failures.append("forbidden_artifacts")
    if package_build_report.manifest_hash != submission_manifest.manifest_hash:
        failures.append("manifest_hash_mismatch")
    if package_build_report.package_hash != submission_manifest.package_hash:
        failures.append("package_hash_mismatch")
    if submission_manifest.promotion_decision_hash != promotion_gate_report.decision_hash:
        failures.append("promotion_decision_hash_mismatch")
    if submission_manifest.serving_config_hash != submission_exact_report.serving_config_hash:
        failures.append("submission_serving_hash_mismatch")
    if runtime_validation_report.serving_config_hash and runtime_validation_report.serving_config_hash != submission_exact_report.serving_config_hash:
        failures.append("runtime_serving_hash_mismatch")
    if runtime_validation_report.manifest_hash is not None and runtime_validation_report.manifest_hash != submission_manifest.manifest_hash:
        failures.append("runtime_manifest_hash_mismatch")
    required_hashes = {
        "promotion_decision_hash": promotion_gate_report.decision_hash,
        "serving_config_hash": submission_manifest.serving_config_hash,
        "training_plan_hash": submission_manifest.training_plan_hash,
        "dataset_manifest_hash": submission_manifest.dataset_manifest_hash,
        "trace_manifest_hash": submission_manifest.trace_manifest_hash,
        "lora_config_hash": submission_manifest.lora_config_hash,
        "adapter_hash": submission_manifest.adapter_hash,
    }
    for field_name in cfg.required_hash_fields:
        if not _non_empty(required_hashes.get(field_name)):
            failures.append(f"missing_hash:{field_name}")
    if runtime_validation_report.adapter_rank is not None and runtime_validation_report.adapter_rank > 32:
        failures.append("adapter_rank_exceeds_32")
    all_warnings = (
        tuple(package_build_report.warnings)
        + tuple(runtime_validation_report.warnings)
        + tuple(artifact_audit_report.warnings)
        + tuple(str(item) for item in (metadata or {}).get("warnings", ()))
    )
    warnings.extend(all_warnings)
    if warnings and not cfg.allow_warnings:
        failures.append("warnings_not_allowed")
    runtime_fake_success = bool(
        runtime_validation_report.dry_run
        and (runtime_validation_report.runtime_loaded or runtime_validation_report.model_loaded or runtime_validation_report.adapter_loaded)
    )
    package_dry_run_claimed_built = bool(package_build_report.dry_run and package_build_report.built)
    unique_failures = tuple(sorted(set(failures)))
    return PreflightGateReport(
        passed=not unique_failures,
        failure_reasons=unique_failures,
        input_report_hashes={
            "submission_manifest": submission_manifest.manifest_hash,
            "package_build_report": package_build_report.report_hash,
            "runtime_validation_report": runtime_validation_report.report_hash,
            "artifact_audit_report": artifact_audit_report.report_hash,
            "promotion_gate_report": promotion_gate_report.decision_hash,
            "submission_exact_report": submission_exact_report.report_hash,
        },
        required_hashes=required_hashes,
        warnings=tuple(sorted(set(warnings))),
        warnings_allowed=cfg.allow_warnings,
        runtime_fake_success=runtime_fake_success,
        package_dry_run_claimed_built=package_dry_run_claimed_built,
    )


def _validate_report_hash(report: object, hash_field: str) -> None:
    try:
        observed = getattr(report, hash_field)
        expected = stable_hash({item.name: getattr(report, item.name) for item in fields(report) if item.name != hash_field})
    except Exception as exc:
        raise PreflightGateError("could not validate input report hash.") from exc
    if observed != expected:
        raise PreflightGateError("input report hash mismatch.")


def _non_empty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


__all__ = [
    "PreflightGateConfig",
    "PreflightGateError",
    "PreflightGateReport",
    "evaluate_preflight_gate",
]
