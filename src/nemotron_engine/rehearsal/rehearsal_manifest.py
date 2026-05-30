"""Final deterministic Pass 14 rehearsal manifest."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash

from .adapter_evidence import AdapterEvidenceBundle, validate_adapter_evidence_bundle
from .package_rehearsal import PackageRehearsalReport, validate_package_rehearsal_report
from .promotion_bridge import PromotionBridgeReport, validate_promotion_rehearsal_report
from .rehearsal_config import RehearsalConfig, reject_unsafe_metadata_claims, validate_rehearsal_config
from .runtime_rehearsal import RuntimeRehearsalReport, validate_runtime_rehearsal_report
from .submission_rehearsal import SubmissionRehearsalReport, validate_submission_rehearsal_report


class FinalRehearsalManifestError(ValueError):
    """Raised when the final rehearsal manifest is inconsistent."""


@dataclass(frozen=True)
class FinalRehearsalManifest:
    manifest_id: str
    rehearsal_config_hash: str
    adapter_evidence_hash: str
    promotion_bridge_hash: str
    package_rehearsal_hash: str
    runtime_rehearsal_hash: str
    submission_rehearsal_hash: str
    inference_evaluation_hash: str | None
    smoke_handoff_hash: str | None
    release_candidate_hash: str | None
    dry_run: bool
    package_built: bool
    runtime_loaded: bool
    submitted: bool
    ready_for_external_submission: bool
    package_passed: bool = False
    runtime_passed: bool = False
    runtime_unloaded_dry_run: bool = False
    unloaded_runtime_dry_run_allowed: bool = False
    submission_passed: bool = False
    allow_warnings: bool = False
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    manifest_hash: str = ""

    def __post_init__(self) -> None:
        for name in (
            "manifest_id",
            "rehearsal_config_hash",
            "adapter_evidence_hash",
            "promotion_bridge_hash",
            "package_rehearsal_hash",
            "runtime_rehearsal_hash",
            "submission_rehearsal_hash",
        ):
            _require_non_empty(getattr(self, name), name)
        for name in ("inference_evaluation_hash", "smoke_handoff_hash", "release_candidate_hash"):
            value = getattr(self, name)
            if value is not None:
                _require_non_empty(value, name)
        for name in (
            "dry_run",
            "package_built",
            "runtime_loaded",
            "submitted",
            "ready_for_external_submission",
            "package_passed",
            "runtime_passed",
            "runtime_unloaded_dry_run",
            "unloaded_runtime_dry_run_allowed",
            "submission_passed",
            "allow_warnings",
        ):
            if not isinstance(getattr(self, name), bool):
                raise FinalRehearsalManifestError(f"{name} must be boolean.")
        if self.submitted:
            raise FinalRehearsalManifestError("submitted=True is impossible in Pass 14.")
        errors = tuple(str(item) for item in self.errors)
        warnings = tuple(str(item) for item in self.warnings)
        if self.ready_for_external_submission:
            if not self.dry_run:
                raise FinalRehearsalManifestError("ready_for_external_submission requires dry_run=True.")
            if self.submitted:
                raise FinalRehearsalManifestError("ready_for_external_submission requires submitted=False.")
            if errors:
                raise FinalRehearsalManifestError("ready_for_external_submission=True cannot include errors.")
            if warnings and self.allow_warnings is not True:
                raise FinalRehearsalManifestError("ready_for_external_submission=True cannot include warnings unless allowed.")
            if not self.package_passed:
                raise FinalRehearsalManifestError("ready_for_external_submission=True requires package_passed=True.")
            runtime_ok = self.runtime_passed or (self.runtime_unloaded_dry_run and self.unloaded_runtime_dry_run_allowed)
            if not runtime_ok:
                raise FinalRehearsalManifestError("ready_for_external_submission=True requires runtime pass evidence or allowed unloaded runtime dry-run.")
            if not self.submission_passed:
                raise FinalRehearsalManifestError("ready_for_external_submission=True requires submission_passed=True.")
        metadata = dict(self.metadata)
        try:
            reject_unsafe_metadata_claims(metadata, allow_adapter_evidence=True)
        except Exception as exc:
            raise FinalRehearsalManifestError(str(exc)) from exc
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "metadata", metadata)
        expected_id = compute_final_rehearsal_manifest_id(self)
        if self.manifest_id != expected_id:
            raise FinalRehearsalManifestError("manifest_id does not match final rehearsal identity payload.")
        expected_hash = compute_final_rehearsal_manifest_hash(self)
        if not self.manifest_hash:
            object.__setattr__(self, "manifest_hash", expected_hash)
        elif self.manifest_hash != expected_hash:
            raise FinalRehearsalManifestError("manifest_hash does not match final rehearsal manifest payload.")


def build_final_rehearsal_manifest(
    *,
    rehearsal_config: RehearsalConfig,
    adapter_evidence_bundle: AdapterEvidenceBundle,
    promotion_bridge_report: PromotionBridgeReport,
    package_rehearsal_report: PackageRehearsalReport,
    runtime_rehearsal_report: RuntimeRehearsalReport,
    submission_rehearsal_report: SubmissionRehearsalReport,
    inference_evaluation_hash: str | None = None,
    release_candidate_hash: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FinalRehearsalManifest:
    rehearsal = validate_rehearsal_config(rehearsal_config)
    evidence = validate_adapter_evidence_bundle(adapter_evidence_bundle)
    bridge = validate_promotion_rehearsal_report(promotion_bridge_report)
    package = validate_package_rehearsal_report(package_rehearsal_report)
    runtime = validate_runtime_rehearsal_report(runtime_rehearsal_report)
    submission = validate_submission_rehearsal_report(submission_rehearsal_report)
    errors: list[str] = []
    warnings: list[str] = []
    if package.passed is not True:
        errors.append("package_rehearsal_failed")
    if runtime.passed is not True:
        errors.append("runtime_rehearsal_failed")
    if submission.passed is not True:
        errors.append("submission_rehearsal_failed")
    if bridge.allowed is not True:
        errors.append("promotion_bridge_failed")
    if evidence.evidence_complete is not True:
        errors.append("adapter_evidence_incomplete")
    warnings.extend(evidence.warnings)
    warnings.extend(package.warnings)
    warnings.extend(runtime.warnings)
    warnings.extend(submission.warnings)
    if warnings and not rehearsal.allow_warnings:
        errors.append("warnings_not_allowed")
    ready = bool(
        rehearsal.dry_run
        and not errors
        and submission.passed
        and package.passed
        and runtime.passed
        and submission.submitted is False
        and (not warnings or rehearsal.allow_warnings)
    )
    payload = {
        "rehearsal_config_hash": rehearsal.config_hash,
        "adapter_evidence_hash": evidence.evidence_hash,
        "promotion_bridge_hash": bridge.report_hash,
        "package_rehearsal_hash": package.report_hash,
        "runtime_rehearsal_hash": runtime.report_hash,
        "submission_rehearsal_hash": submission.report_hash,
        "dry_run": rehearsal.dry_run,
    }
    manifest_id = stable_hash(payload)
    return FinalRehearsalManifest(
        manifest_id=manifest_id,
        rehearsal_config_hash=rehearsal.config_hash,
        adapter_evidence_hash=evidence.evidence_hash,
        promotion_bridge_hash=bridge.report_hash,
        package_rehearsal_hash=package.report_hash,
        runtime_rehearsal_hash=runtime.report_hash,
        submission_rehearsal_hash=submission.report_hash,
        inference_evaluation_hash=inference_evaluation_hash or bridge.inference_evaluation_hash,
        smoke_handoff_hash=evidence.smoke_handoff_hash,
        release_candidate_hash=release_candidate_hash or submission.release_candidate_hash,
        dry_run=rehearsal.dry_run,
        package_built=package.built,
        runtime_loaded=runtime.runtime_loaded,
        submitted=False,
        ready_for_external_submission=ready,
        package_passed=package.passed,
        runtime_passed=runtime.passed,
        runtime_unloaded_dry_run=runtime.unloaded_dry_run,
        unloaded_runtime_dry_run_allowed=rehearsal.allow_runtime_unloaded,
        submission_passed=submission.passed,
        allow_warnings=rehearsal.allow_warnings,
        errors=tuple(sorted(set(errors))),
        warnings=tuple(sorted(set(warnings))),
        metadata={**dict(metadata or {}), "allow_warnings": rehearsal.allow_warnings, "meaning": "local dry-run rehearsal evidence for later human external-submission consideration"},
    )


def validate_final_rehearsal_manifest(manifest: FinalRehearsalManifest | Mapping[str, Any]) -> FinalRehearsalManifest:
    normalized = manifest if isinstance(manifest, FinalRehearsalManifest) else FinalRehearsalManifest(**dict(manifest))
    if normalized.manifest_id != compute_final_rehearsal_manifest_id(normalized):
        raise FinalRehearsalManifestError("manifest_id does not match final rehearsal identity payload.")
    if normalized.manifest_hash != compute_final_rehearsal_manifest_hash(normalized):
        raise FinalRehearsalManifestError("manifest_hash does not match final rehearsal manifest payload.")
    if normalized.submitted:
        raise FinalRehearsalManifestError("submitted=True is impossible in Pass 14.")
    return normalized


def compute_final_rehearsal_manifest_hash(manifest: FinalRehearsalManifest | Mapping[str, Any]) -> str:
    payload = dict(manifest) if isinstance(manifest, Mapping) else {item.name: getattr(manifest, item.name) for item in fields(FinalRehearsalManifest)}
    payload.pop("manifest_hash", None)
    return stable_hash(payload)


def compute_final_rehearsal_manifest_id(manifest: FinalRehearsalManifest | Mapping[str, Any]) -> str:
    payload = dict(manifest) if isinstance(manifest, Mapping) else {item.name: getattr(manifest, item.name) for item in fields(FinalRehearsalManifest)}
    return stable_hash(
        {
            "rehearsal_config_hash": payload.get("rehearsal_config_hash"),
            "adapter_evidence_hash": payload.get("adapter_evidence_hash"),
            "promotion_bridge_hash": payload.get("promotion_bridge_hash"),
            "package_rehearsal_hash": payload.get("package_rehearsal_hash"),
            "runtime_rehearsal_hash": payload.get("runtime_rehearsal_hash"),
            "submission_rehearsal_hash": payload.get("submission_rehearsal_hash"),
            "dry_run": payload.get("dry_run"),
        }
    )


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FinalRehearsalManifestError(f"{field_name} must be non-empty.")
    return value


__all__ = [
    "FinalRehearsalManifest",
    "FinalRehearsalManifestError",
    "build_final_rehearsal_manifest",
    "compute_final_rehearsal_manifest_hash",
    "validate_final_rehearsal_manifest",
]
