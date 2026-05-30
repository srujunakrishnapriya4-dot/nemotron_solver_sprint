"""Final system audit composition for Pass 10."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.evaluation.promotion_gates import PromotionDecision, PromotionGateReport
from nemotron_engine.packaging.preflight_gate import PreflightGateReport

from .artifact_safety import ArtifactSafetyReport, validate_release_artifact_safety
from .import_safety import ImportSafetyReport
from .locked_pass_registry import LockedPassRegistry, validate_locked_pass_registry


class SystemAuditError(ValueError):
    """Raised when system audit reports are inconsistent."""


@dataclass(frozen=True)
class SystemAuditConfig:
    require_promotion_report: bool = False
    require_preflight_report: bool = False
    allow_warnings: bool = False
    accepted_suite_summaries: tuple[str, ...] = ("597 passed, 1 skipped", "595 passed, 1 skipped", "559 passed, 1 skipped", "494 passed, 1 skipped")
    metadata: Mapping[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        if not all(isinstance(value, bool) for value in (self.require_promotion_report, self.require_preflight_report, self.allow_warnings)):
            raise SystemAuditError("system audit flags must be booleans.")
        if not isinstance(self.accepted_suite_summaries, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.accepted_suite_summaries
        ):
            raise SystemAuditError("accepted_suite_summaries must be a tuple of non-empty strings.")
        object.__setattr__(self, "metadata", dict(self.metadata))
        expected = _payload_hash(self, "config_hash")
        if not self.config_hash:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise SystemAuditError("config_hash does not match system audit config payload.")


@dataclass(frozen=True)
class SystemAuditReport:
    locked_registry_hash: str
    import_safety_hash: str
    artifact_safety_hash: str
    promotion_decision_hash: str | None
    preflight_gate_hash: str | None
    scoped_suite_result_summary: str | None
    passed: bool
    failure_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    component_status: Mapping[str, bool] = field(default_factory=dict)
    warnings_allowed: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    report_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("locked_registry_hash", "import_safety_hash", "artifact_safety_hash"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise SystemAuditError(f"{name} must be non-empty.")
        for name in ("promotion_decision_hash", "preflight_gate_hash", "scoped_suite_result_summary"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise SystemAuditError(f"{name} must be non-empty when supplied.")
        failures = tuple(str(item) for item in self.failure_reasons)
        warnings = tuple(str(item) for item in self.warnings)
        status = {str(key): bool(value) for key, value in sorted(self.component_status.items())}
        if self.passed and failures:
            raise SystemAuditError("passed=True cannot include failure_reasons.")
        if self.passed and any(value is False for value in status.values()):
            raise SystemAuditError("passed=True cannot include failed component status.")
        if self.passed and warnings and not self.warnings_allowed:
            raise SystemAuditError("passed=True cannot include warnings unless warnings are allowed.")
        object.__setattr__(self, "failure_reasons", failures)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "component_status", status)
        object.__setattr__(self, "metadata", dict(self.metadata))
        expected = _payload_hash(self, "report_hash")
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected)
        elif self.report_hash != expected:
            raise SystemAuditError("report_hash does not match system audit report payload.")


def run_system_audit(
    *,
    locked_pass_registry: LockedPassRegistry,
    import_safety_report: ImportSafetyReport,
    artifact_safety_report: ArtifactSafetyReport,
    pass8_promotion_gate_report: PromotionGateReport | None = None,
    pass9_preflight_gate_report: PreflightGateReport | None = None,
    scoped_suite_result_summary: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    config: SystemAuditConfig | None = None,
) -> SystemAuditReport:
    cfg = config or SystemAuditConfig()
    failures: list[str] = []
    warnings: list[str] = []
    component_status: dict[str, bool] = {}
    try:
        validate_locked_pass_registry(locked_pass_registry)
        component_status["locked_pass_registry"] = True
    except Exception as exc:
        component_status["locked_pass_registry"] = False
        failures.append(f"locked_pass_registry_failed:{exc}")
    _validate_report_hash(import_safety_report, "report_hash")
    _validate_report_hash(artifact_safety_report, "report_hash")
    try:
        validate_release_artifact_safety(artifact_safety_report)
    except Exception as exc:
        failures.append(f"artifact_safety_invalid:{exc}")
    component_status["import_safety"] = bool(import_safety_report.passed)
    component_status["artifact_safety"] = bool(artifact_safety_report.passed)
    if not import_safety_report.passed:
        failures.append("import_safety_failed")
    if not artifact_safety_report.passed:
        failures.append("artifact_safety_failed")
    promotion_hash = None
    if pass8_promotion_gate_report is None:
        component_status["promotion_gate"] = not cfg.require_promotion_report
        if cfg.require_promotion_report:
            failures.append("promotion_report_missing")
    else:
        _validate_report_hash(pass8_promotion_gate_report, "decision_hash")
        promotion_hash = pass8_promotion_gate_report.decision_hash
        component_status["promotion_gate"] = pass8_promotion_gate_report.decision == PromotionDecision.ALLOW and pass8_promotion_gate_report.passed
        if not component_status["promotion_gate"]:
            failures.append("promotion_not_allow")
    preflight_hash = None
    if pass9_preflight_gate_report is None:
        component_status["preflight_gate"] = not cfg.require_preflight_report
        if cfg.require_preflight_report:
            failures.append("preflight_report_missing")
    else:
        _validate_report_hash(pass9_preflight_gate_report, "gate_hash")
        preflight_hash = pass9_preflight_gate_report.gate_hash
        component_status["preflight_gate"] = bool(pass9_preflight_gate_report.passed)
        if not pass9_preflight_gate_report.passed:
            failures.append("preflight_failed")
    suite_ok = scoped_suite_result_summary is not None and scoped_suite_result_summary in cfg.accepted_suite_summaries
    component_status["scoped_suite"] = suite_ok
    if scoped_suite_result_summary is None:
        failures.append("scoped_suite_summary_missing")
    elif not suite_ok:
        failures.append("scoped_suite_summary_not_accepted")
    warnings.extend(str(item) for item in import_safety_report.warnings)
    warnings.extend(str(item) for item in artifact_safety_report.warnings)
    warnings.extend(str(item) for item in (metadata or {}).get("warnings", ()))
    if warnings and not cfg.allow_warnings:
        failures.append("warnings_not_allowed")
    unique_failures = tuple(sorted(set(failures)))
    return SystemAuditReport(
        locked_registry_hash=locked_pass_registry.registry_hash,
        import_safety_hash=import_safety_report.report_hash,
        artifact_safety_hash=artifact_safety_report.report_hash,
        promotion_decision_hash=promotion_hash,
        preflight_gate_hash=preflight_hash,
        scoped_suite_result_summary=scoped_suite_result_summary,
        passed=not unique_failures,
        failure_reasons=unique_failures,
        warnings=tuple(sorted(set(warnings))),
        component_status=component_status,
        warnings_allowed=cfg.allow_warnings,
        metadata=dict(metadata or {}),
    )


def validate_system_audit_report(report: SystemAuditReport) -> SystemAuditReport:
    if not isinstance(report, SystemAuditReport):
        raise SystemAuditError("report must be a SystemAuditReport.")
    expected = _payload_hash(report, "report_hash")
    if report.report_hash != expected:
        raise SystemAuditError("report_hash does not match system audit report payload.")
    if report.passed and report.failure_reasons:
        raise SystemAuditError("passed report contains failure_reasons.")
    if report.passed and any(value is False for value in report.component_status.values()):
        raise SystemAuditError("passed report contains failed component status.")
    return report


def _validate_report_hash(report: object, hash_field: str) -> None:
    try:
        observed = getattr(report, hash_field)
        expected = stable_hash({item.name: getattr(report, item.name) for item in fields(report) if item.name != hash_field})
    except Exception as exc:
        raise SystemAuditError("could not validate input report hash.") from exc
    if observed != expected:
        raise SystemAuditError("input report hash mismatch.")


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


__all__ = [
    "SystemAuditConfig",
    "SystemAuditError",
    "SystemAuditReport",
    "run_system_audit",
    "validate_system_audit_report",
]
