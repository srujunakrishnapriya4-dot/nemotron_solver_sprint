"""Package dry-run rehearsal for Pass 14."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.packaging import (
    PackageBuildConfig,
    PackageBuildReport,
    SubmissionManifest,
    build_submission_package,
    compute_manifest_hash,
    validate_submission_manifest,
)

from .adapter_evidence import AdapterEvidenceBundle, validate_adapter_evidence_bundle
from .promotion_bridge import PromotionBridgeReport, validate_promotion_rehearsal_report
from .rehearsal_config import RehearsalConfig, _is_under_directory, reject_unsafe_metadata_claims, validate_rehearsal_config


class PackageRehearsalError(ValueError):
    """Raised when package rehearsal evidence is unsafe."""


@dataclass(frozen=True)
class PackageRehearsalConfig:
    root_path: str | None = None
    output_path: str | None = None
    created_by: str = "pass14_rehearsal"
    require_package_build_report: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.require_package_build_report, bool):
            raise PackageRehearsalError("require_package_build_report must be boolean.")
        if self.root_path is not None and (not isinstance(self.root_path, str) or not self.root_path.strip()):
            raise PackageRehearsalError("root_path must be non-empty when supplied.")
        if self.output_path is not None and (not isinstance(self.output_path, str) or not self.output_path.strip()):
            raise PackageRehearsalError("output_path must be non-empty when supplied.")
        if not isinstance(self.created_by, str) or not self.created_by.strip():
            raise PackageRehearsalError("created_by must be non-empty.")
        metadata = dict(self.metadata)
        try:
            reject_unsafe_metadata_claims(metadata, allow_adapter_evidence=False)
        except Exception as exc:
            raise PackageRehearsalError(str(exc)) from exc
        object.__setattr__(self, "metadata", metadata)
        expected = _payload_hash(self, "config_hash")
        if not self.config_hash:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise PackageRehearsalError("config_hash does not match package rehearsal config payload.")


@dataclass(frozen=True)
class PackageRehearsalReport:
    rehearsal_config_hash: str
    package_rehearsal_config_hash: str
    adapter_hash: str
    adapter_evidence_hash: str
    promotion_bridge_hash: str
    promotion_decision_hash: str
    submission_manifest_hash: str | None
    package_build_report_hash: str | None
    package_hash: str | None
    package_path: str | None
    built: bool
    dry_run: bool
    allow_zip_build: bool
    output_dir: str | None
    passed: bool
    package_build_evidence_required: bool = False
    required_hashes: dict[str, str | None] = field(default_factory=dict)
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    report_hash: str = ""

    def __post_init__(self) -> None:
        for name in (
            "rehearsal_config_hash",
            "package_rehearsal_config_hash",
            "adapter_hash",
            "adapter_evidence_hash",
            "promotion_bridge_hash",
            "promotion_decision_hash",
        ):
            _require_non_empty(getattr(self, name), name)
        for name in ("submission_manifest_hash", "package_build_report_hash", "package_hash", "package_path", "output_dir"):
            value = getattr(self, name)
            if value is not None:
                _require_non_empty(value, name)
        if not all(isinstance(value, bool) for value in (self.built, self.dry_run, self.allow_zip_build, self.passed, self.package_build_evidence_required)):
            raise PackageRehearsalError("package rehearsal boolean fields must be bool values.")
        errors = tuple(str(item) for item in self.errors)
        warnings = tuple(str(item) for item in self.warnings)
        if self.passed and errors:
            raise PackageRehearsalError("passed=True cannot include errors.")
        if self.built and self.dry_run and not self.allow_zip_build:
            raise PackageRehearsalError("built=True is forbidden for dry-run when allow_zip_build=False.")
        if self.allow_zip_build is False and self.built:
            raise PackageRehearsalError("built=True is forbidden when allow_zip_build=False.")
        if self.allow_zip_build is False and self.package_path is not None:
            raise PackageRehearsalError("package_path is forbidden when allow_zip_build=False.")
        if self.built and not self.package_path:
            raise PackageRehearsalError("built=True requires package_path.")
        if self.package_path is not None and self.output_dir is not None and not _is_under_directory(self.package_path, self.output_dir):
            raise PackageRehearsalError("package_path must be under output_dir.")
        if self.package_path is not None and Path(self.package_path).name.lower() == "submission.zip":
            if not (self.allow_zip_build and self.output_dir and _is_under_directory(self.package_path, self.output_dir)):
                raise PackageRehearsalError("submission.zip package path requires explicit safe output_dir.")
        metadata = dict(self.metadata)
        try:
            reject_unsafe_metadata_claims(metadata, allow_adapter_evidence=False)
        except Exception as exc:
            raise PackageRehearsalError(str(exc)) from exc
        object.__setattr__(self, "required_hashes", {str(k): (None if v is None else str(v)) for k, v in sorted(self.required_hashes.items())})
        if self.passed:
            _require_non_empty(self.submission_manifest_hash, "submission_manifest_hash")
            required_names = (
                "adapter_hash",
                "promotion_decision_hash",
                "training_plan_hash",
                "dataset_manifest_hash",
                "trace_manifest_hash",
                "lora_config_hash",
            )
            missing = tuple(name for name in required_names if not self.required_hashes.get(name))
            if missing:
                raise PackageRehearsalError(f"passed=True requires required_hashes: {', '.join(missing)}.")
            if self.required_hashes["adapter_hash"] != self.adapter_hash:
                raise PackageRehearsalError("required_hashes.adapter_hash must match adapter_hash.")
            if self.required_hashes["promotion_decision_hash"] != self.promotion_decision_hash:
                raise PackageRehearsalError("required_hashes.promotion_decision_hash must match promotion_decision_hash.")
            if self.package_build_evidence_required and not self.package_build_report_hash:
                raise PackageRehearsalError("passed=True requires package_build_report_hash when build evidence is required.")
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "metadata", metadata)
        expected = _payload_hash(self, "report_hash")
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected)
        elif self.report_hash != expected:
            raise PackageRehearsalError("report_hash does not match package rehearsal payload.")


def run_package_rehearsal(
    *,
    rehearsal_config: RehearsalConfig,
    adapter_evidence_bundle: AdapterEvidenceBundle,
    promotion_bridge_report: PromotionBridgeReport,
    submission_manifest: SubmissionManifest | None = None,
    package_build_report: PackageBuildReport | None = None,
    metadata: Mapping[str, Any] | None = None,
    config: PackageRehearsalConfig | None = None,
) -> PackageRehearsalReport:
    rehearsal = validate_rehearsal_config(rehearsal_config)
    evidence = validate_adapter_evidence_bundle(adapter_evidence_bundle)
    bridge = validate_promotion_rehearsal_report(promotion_bridge_report)
    cfg = config or PackageRehearsalConfig()
    errors: list[str] = []
    warnings: list[str] = []
    manifest_hash = None
    package_hash = None
    build_hash = None
    package_path = None
    built = False
    required_hashes = dict(getattr(bridge, "required_hashes", {}) or {})
    package_build_required = bool(cfg.require_package_build_report or rehearsal.allow_zip_build)

    if bridge.allowed is not True:
        errors.append("promotion_bridge_not_allowed")
    if evidence.evidence_complete is not True:
        errors.append("adapter_evidence_incomplete")
    if submission_manifest is not None:
        try:
            validate_submission_manifest(submission_manifest, cfg.root_path)
        except Exception as exc:
            errors.append(f"submission_manifest_invalid:{exc}")
        else:
            if submission_manifest.manifest_hash != compute_manifest_hash(submission_manifest):
                errors.append("submission_manifest_hash_mismatch")
            manifest_hash = submission_manifest.manifest_hash
            package_hash = submission_manifest.package_hash
            required_hashes = {
                "adapter_hash": submission_manifest.adapter_hash,
                "promotion_decision_hash": submission_manifest.promotion_decision_hash,
                "training_plan_hash": submission_manifest.training_plan_hash,
                "dataset_manifest_hash": submission_manifest.dataset_manifest_hash,
                "trace_manifest_hash": submission_manifest.trace_manifest_hash,
                "lora_config_hash": submission_manifest.lora_config_hash,
            }
            if submission_manifest.adapter_hash != evidence.adapter_hash:
                errors.append("submission_manifest_adapter_hash_mismatch")
            if submission_manifest.promotion_decision_hash != bridge.promotion_decision_hash:
                errors.append("submission_manifest_promotion_hash_mismatch")
    else:
        errors.append("submission_manifest_missing")
    if package_build_report is not None:
        _validate_report_hash(package_build_report, "report_hash")
        build_hash = package_build_report.report_hash
        package_path = package_build_report.package_path
        built = package_build_report.built
        if package_build_report.passed is not True:
            errors.append("package_build_failed")
        if package_build_report.dry_run and package_build_report.built:
            errors.append("package_dry_run_claimed_built")
        if manifest_hash is not None and package_build_report.manifest_hash != manifest_hash:
            errors.append("package_manifest_hash_mismatch")
        if package_hash is not None and package_build_report.package_hash != package_hash:
            errors.append("package_hash_mismatch")
    elif rehearsal.allow_zip_build:
        if not rehearsal.output_dir:
            errors.append("output_dir_required")
        elif submission_manifest is not None and cfg.root_path and cfg.output_path:
            output_path = Path(cfg.output_path)
            if not output_path.is_absolute():
                output_path = Path(rehearsal.output_dir) / output_path
            if not _is_safe_output_path(output_path, rehearsal.output_dir, cfg.root_path):
                errors.append("package_output_path_unsafe")
            else:
                report = build_submission_package(
                    PackageBuildConfig(
                        root_path=cfg.root_path,
                        manifest=submission_manifest,
                        output_path=str(output_path),
                        allowed_output_parent=rehearsal.output_dir,
                        build_zip=True,
                        dry_run=bool(rehearsal.dry_run),
                        metadata={"pass14_rehearsal": True},
                    )
                )
                build_hash = report.report_hash
                package_path = report.package_path
                built = report.built
                if not report.passed:
                    errors.extend(report.errors)
    else:
        built = False
    if package_path is not None:
        if rehearsal.output_dir is None or not _is_under_directory(package_path, rehearsal.output_dir):
            errors.append("package_path_outside_output_dir")
        root_path = Path(cfg.root_path).resolve() if cfg.root_path else None
        if root_path is not None and Path(package_path).resolve().parent == root_path:
            errors.append("package_path_in_repo_root")
        if Path(package_path).name.lower() == "submission.zip" and not (rehearsal.allow_zip_build and rehearsal.output_dir and _is_under_directory(package_path, rehearsal.output_dir)):
            errors.append("unsafe_submission_zip_name")
    if built and not rehearsal.allow_zip_build:
        errors.append("zip_build_not_allowed")
    required_names = (
        "adapter_hash",
        "promotion_decision_hash",
        "training_plan_hash",
        "dataset_manifest_hash",
        "trace_manifest_hash",
        "lora_config_hash",
    )
    for name in required_names:
        if not required_hashes.get(name):
            errors.append(f"required_hash_missing:{name}")
    if manifest_hash is None:
        errors.append("submission_manifest_hash_missing")
    if package_build_required and build_hash is None:
        errors.append("package_build_report_missing")
    if errors and (
        "zip_build_not_allowed" in errors
        or "package_path_outside_output_dir" in errors
        or "package_path_in_repo_root" in errors
        or "unsafe_submission_zip_name" in errors
    ):
        built = False
        package_path = None
    return PackageRehearsalReport(
        rehearsal_config_hash=rehearsal.config_hash,
        package_rehearsal_config_hash=cfg.config_hash,
        adapter_hash=evidence.adapter_hash,
        adapter_evidence_hash=evidence.evidence_hash,
        promotion_bridge_hash=bridge.report_hash,
        promotion_decision_hash=bridge.promotion_decision_hash,
        submission_manifest_hash=manifest_hash,
        package_build_report_hash=build_hash,
        package_hash=package_hash,
        package_path=package_path,
        built=built,
        dry_run=rehearsal.dry_run,
        allow_zip_build=rehearsal.allow_zip_build,
        output_dir=rehearsal.output_dir,
        passed=not errors,
        package_build_evidence_required=package_build_required,
        required_hashes=required_hashes,
        errors=tuple(sorted(set(errors))),
        warnings=tuple(sorted(set(warnings))),
        metadata=dict(metadata or {}),
    )


def validate_package_rehearsal_report(report: PackageRehearsalReport | Mapping[str, Any]) -> PackageRehearsalReport:
    normalized = report if isinstance(report, PackageRehearsalReport) else PackageRehearsalReport(**dict(report))
    if normalized.report_hash != _payload_hash(normalized, "report_hash"):
        raise PackageRehearsalError("report_hash does not match package rehearsal payload.")
    if normalized.passed and normalized.errors:
        raise PackageRehearsalError("passed=True cannot include errors.")
    return normalized


def _is_safe_output_path(output_path: str | Path, output_dir: str | Path, root_path: str | Path) -> bool:
    output = Path(output_path).resolve()
    out_dir = Path(output_dir).resolve()
    root = Path(root_path).resolve()
    return _is_under_directory(output, out_dir) and output.parent != root


def _validate_report_hash(report: object, hash_field: str) -> None:
    try:
        observed = getattr(report, hash_field)
        expected = stable_hash({item.name: getattr(report, item.name) for item in fields(report) if item.name != hash_field})
    except Exception as exc:
        raise PackageRehearsalError("could not validate input report hash.") from exc
    if observed != expected:
        raise PackageRehearsalError("input report hash mismatch.")


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PackageRehearsalError(f"{field_name} must be non-empty.")
    return value


__all__ = [
    "PackageRehearsalConfig",
    "PackageRehearsalError",
    "PackageRehearsalReport",
    "run_package_rehearsal",
    "validate_package_rehearsal_report",
]
