"""Adapter evidence bundle for Pass 14 rehearsal."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.smoke_training import (
    SmokeHandoffReport,
    SmokeTrainingReport,
    validate_smoke_handoff_report,
    validate_smoke_training_report,
)
from nemotron_engine.training_backend import AdapterValidationReport, TrainingRunManifest, validate_training_run_manifest

from .rehearsal_config import RehearsalConfig, reject_unsafe_metadata_claims, validate_rehearsal_config


class AdapterEvidenceError(ValueError):
    """Raised when adapter evidence is incomplete or forged."""


@dataclass(frozen=True)
class AdapterEvidenceBundle:
    adapter_hash: str
    adapter_validation_hash: str
    training_run_manifest_hash: str | None
    smoke_report_hash: str | None
    smoke_handoff_hash: str | None
    lora_config_hash: str | None
    adapter_dir: str | None
    artifact_hashes: dict[str, str]
    rank: int | None
    evidence_complete: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("adapter_hash", "adapter_validation_hash"):
            _require_non_empty(getattr(self, name), name)
        for name in ("training_run_manifest_hash", "smoke_report_hash", "smoke_handoff_hash", "lora_config_hash", "adapter_dir"):
            value = getattr(self, name)
            if value is not None:
                _require_non_empty(value, name)
        hashes = {str(key): str(value) for key, value in sorted(self.artifact_hashes.items())}
        if not hashes:
            raise AdapterEvidenceError("artifact_hashes must be non-empty.")
        if any(not _is_sha256(value) for value in hashes.values()):
            raise AdapterEvidenceError("artifact_hashes values must be sha256 digests.")
        if self.rank is not None and (isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1 or self.rank > 32):
            raise AdapterEvidenceError("rank must be in [1, 32].")
        if not isinstance(self.evidence_complete, bool):
            raise AdapterEvidenceError("evidence_complete must be boolean.")
        errors = tuple(str(item) for item in self.errors)
        warnings = tuple(str(item) for item in self.warnings)
        if self.evidence_complete and errors:
            raise AdapterEvidenceError("evidence_complete=True cannot include errors.")
        if self.evidence_complete:
            expected_adapter_hash = stable_hash({"artifact_hashes": hashes, "rank": self.rank})
            if self.adapter_hash != expected_adapter_hash:
                raise AdapterEvidenceError("adapter_hash does not match adapter artifact evidence payload.")
        metadata = dict(self.metadata)
        try:
            reject_unsafe_metadata_claims(metadata, allow_adapter_evidence=True)
        except Exception as exc:
            raise AdapterEvidenceError(str(exc)) from exc
        object.__setattr__(self, "artifact_hashes", hashes)
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "metadata", metadata)
        expected = compute_adapter_evidence_hash(self)
        if not self.evidence_hash:
            object.__setattr__(self, "evidence_hash", expected)
        elif self.evidence_hash != expected:
            raise AdapterEvidenceError("evidence_hash does not match adapter evidence payload.")


def build_adapter_evidence_bundle(
    *,
    adapter_validation_report: AdapterValidationReport,
    rehearsal_config: RehearsalConfig | None = None,
    training_run_manifest: TrainingRunManifest | None = None,
    smoke_training_report: SmokeTrainingReport | None = None,
    smoke_handoff_report: SmokeHandoffReport | None = None,
    lora_config_hash: str | None = None,
    adapter_dir: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AdapterEvidenceBundle:
    if not isinstance(adapter_validation_report, AdapterValidationReport):
        raise AdapterEvidenceError("adapter_validation_report must be an AdapterValidationReport.")
    if rehearsal_config is not None:
        rehearsal_config = validate_rehearsal_config(rehearsal_config)
    _validate_report_hash(adapter_validation_report, "report_hash")
    if adapter_validation_report.passed is not True:
        raise AdapterEvidenceError("adapter validation report did not pass.")
    if not adapter_validation_report.adapter_hash:
        raise AdapterEvidenceError("adapter validation report is missing adapter_hash.")
    if not adapter_validation_report.artifact_hashes:
        raise AdapterEvidenceError("adapter validation report is missing artifact_hashes.")
    errors: list[str] = []
    warnings: list[str] = []
    if adapter_validation_report.rank is not None and adapter_validation_report.rank > 32:
        errors.append("adapter_rank_exceeds_32")

    manifest_hash = None
    if training_run_manifest is not None:
        try:
            validate_training_run_manifest(training_run_manifest, adapter_validation_report=adapter_validation_report if training_run_manifest.trained else None)
        except Exception as exc:
            errors.append(f"training_run_manifest_invalid:{exc}")
        manifest_hash = training_run_manifest.manifest_hash
        if training_run_manifest.trained and training_run_manifest.adapter_hash != adapter_validation_report.adapter_hash:
            errors.append("training_manifest_adapter_hash_mismatch")
        if training_run_manifest.adapter_validation_hash and training_run_manifest.adapter_validation_hash != adapter_validation_report.report_hash:
            errors.append("training_manifest_adapter_validation_hash_mismatch")

    smoke_hash = None
    if smoke_training_report is not None:
        try:
            smoke_training_report = validate_smoke_training_report(smoke_training_report)
            smoke_hash = smoke_training_report.report_hash
        except Exception as exc:
            errors.append(f"smoke_training_report_invalid:{exc}")
        else:
            if smoke_training_report.passed is not True or smoke_training_report.trained is not True:
                errors.append("smoke_training_not_passed_trained")
            if smoke_training_report.adapter_validation_hash != adapter_validation_report.report_hash:
                errors.append("smoke_adapter_validation_hash_mismatch")
            if manifest_hash is not None and smoke_training_report.training_run_manifest_hash != manifest_hash:
                errors.append("smoke_training_manifest_hash_mismatch")

    handoff_hash = None
    if smoke_handoff_report is not None:
        try:
            smoke_handoff_report = validate_smoke_handoff_report(smoke_handoff_report)
            handoff_hash = smoke_handoff_report.report_hash
        except Exception as exc:
            errors.append(f"smoke_handoff_invalid:{exc}")
        else:
            if smoke_handoff_report.adapter_hash != adapter_validation_report.adapter_hash:
                errors.append("smoke_handoff_adapter_hash_mismatch")
            if smoke_handoff_report.adapter_validation_hash != adapter_validation_report.report_hash:
                errors.append("smoke_handoff_adapter_validation_hash_mismatch")
            if smoke_training_report is not None:
                if smoke_handoff_report.training_run_manifest_hash != smoke_training_report.training_run_manifest_hash:
                    errors.append("smoke_handoff_training_manifest_hash_mismatch")
                if smoke_handoff_report.backend_result_hash != smoke_training_report.backend_result_hash:
                    errors.append("smoke_handoff_backend_hash_mismatch")

    require_smoke = bool(rehearsal_config.require_smoke_evidence) if rehearsal_config is not None else False
    if require_smoke and (smoke_training_report is None or smoke_handoff_report is None):
        errors.append("smoke_evidence_missing")
    if require_smoke and manifest_hash is None:
        errors.append("training_run_manifest_missing")
    evidence_complete = not errors and bool(adapter_validation_report.passed and adapter_validation_report.adapter_hash and adapter_validation_report.artifact_hashes)
    return AdapterEvidenceBundle(
        adapter_hash=str(adapter_validation_report.adapter_hash or ""),
        adapter_validation_hash=adapter_validation_report.report_hash,
        training_run_manifest_hash=manifest_hash,
        smoke_report_hash=smoke_hash,
        smoke_handoff_hash=handoff_hash,
        lora_config_hash=lora_config_hash,
        adapter_dir=adapter_dir or adapter_validation_report.adapter_dir,
        artifact_hashes=dict(adapter_validation_report.artifact_hashes),
        rank=adapter_validation_report.rank,
        evidence_complete=evidence_complete,
        errors=tuple(sorted(set(errors))),
        warnings=tuple(sorted(set(warnings))),
        metadata=dict(metadata or {}),
    )


def validate_adapter_evidence_bundle(bundle: AdapterEvidenceBundle | Mapping[str, Any]) -> AdapterEvidenceBundle:
    normalized = bundle if isinstance(bundle, AdapterEvidenceBundle) else AdapterEvidenceBundle(**dict(bundle))
    if normalized.evidence_hash != compute_adapter_evidence_hash(normalized):
        raise AdapterEvidenceError("evidence_hash does not match adapter evidence payload.")
    if normalized.evidence_complete and normalized.errors:
        raise AdapterEvidenceError("evidence_complete=True cannot include errors.")
    return normalized


def compute_adapter_evidence_hash(bundle: AdapterEvidenceBundle | Mapping[str, Any]) -> str:
    payload = dict(bundle) if isinstance(bundle, Mapping) else {item.name: getattr(bundle, item.name) for item in fields(AdapterEvidenceBundle)}
    payload.pop("evidence_hash", None)
    return stable_hash(payload)


def _validate_report_hash(report: object, hash_field: str) -> None:
    try:
        observed = getattr(report, hash_field)
        expected = stable_hash({item.name: getattr(report, item.name) for item in fields(report) if item.name != hash_field})
    except Exception as exc:
        raise AdapterEvidenceError("could not validate input report hash.") from exc
    if observed != expected:
        raise AdapterEvidenceError("input report hash mismatch.")


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdapterEvidenceError(f"{field_name} must be non-empty.")
    return value


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


__all__ = [
    "AdapterEvidenceBundle",
    "AdapterEvidenceError",
    "build_adapter_evidence_bundle",
    "compute_adapter_evidence_hash",
    "validate_adapter_evidence_bundle",
]
