"""Deterministic training run manifests for validated external backends."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash

from .adapter_validator import AdapterValidationReport
from .backend_contracts import BackendResult, TrainingStage, _reject_forbidden_claims, validate_backend_result
from .training_log_validator import TrainingLogValidationReport


class TrainingRunManifestError(ValueError):
    """Raised when a training run manifest is invalid."""


@dataclass(frozen=True)
class TrainingRunManifest:
    manifest_id: str
    invocation_hash: str
    backend_result_hash: str
    adapter_validation_hash: str | None
    training_log_hash: str | None
    training_plan_hash: str
    lora_config_hash: str
    dataset_manifest_hash: str | None
    stage: str
    trained: bool
    adapter_hash: str | None
    artifact_hashes: Mapping[str, str]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    manifest_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("manifest_id", "invocation_hash", "backend_result_hash", "training_plan_hash", "lora_config_hash", "stage"):
            _require_non_empty(getattr(self, name), name)
        for name in ("adapter_validation_hash", "training_log_hash", "dataset_manifest_hash", "adapter_hash"):
            value = getattr(self, name)
            if value is not None:
                _require_non_empty(value, name)
        if not isinstance(self.trained, bool):
            raise TrainingRunManifestError("trained must be boolean.")
        if self.stage not in {item.value for item in TrainingStage}:
            raise TrainingRunManifestError("stage must be a Pass 11 TrainingStage value.")
        hashes = {str(key): str(value) for key, value in sorted(self.artifact_hashes.items())}
        if any(not key.strip() or not value.strip() for key, value in hashes.items()):
            raise TrainingRunManifestError("artifact_hashes must be non-empty.")
        metadata = dict(self.metadata)
        try:
            _reject_forbidden_claims(metadata)
        except Exception as exc:
            raise TrainingRunManifestError(str(exc)) from exc
        if self.trained:
            if not self.adapter_validation_hash:
                raise TrainingRunManifestError("trained=True requires adapter_validation_hash.")
            if not self.adapter_hash:
                raise TrainingRunManifestError("trained=True requires adapter_hash.")
        else:
            if self.adapter_hash is not None:
                raise TrainingRunManifestError("trained=False cannot claim adapter_hash.")
            for key, value in _flatten(metadata):
                text = f"{key} {value}".lower()
                if "promoted" in text and "adapter" in text:
                    raise TrainingRunManifestError("trained=False cannot claim promoted adapter.")
        object.__setattr__(self, "artifact_hashes", hashes)
        object.__setattr__(self, "metadata", metadata)
        expected_manifest_id = compute_training_run_manifest_id(self)
        if self.manifest_id != expected_manifest_id:
            raise TrainingRunManifestError("manifest_id does not match training run manifest identity payload.")
        expected = compute_training_run_manifest_hash(self)
        if not self.manifest_hash:
            object.__setattr__(self, "manifest_hash", expected)
        elif self.manifest_hash != expected:
            raise TrainingRunManifestError("manifest_hash does not match training run manifest payload.")


def build_training_run_manifest(
    *,
    invocation_hash: str,
    backend_result: BackendResult,
    training_plan_hash: str,
    lora_config_hash: str,
    dataset_manifest_hash: str | None,
    adapter_validation_report: AdapterValidationReport | None = None,
    training_log_report: TrainingLogValidationReport | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TrainingRunManifest:
    result = validate_backend_result(backend_result, require_artifact_hashes=backend_result.trained)
    if invocation_hash != result.invocation_hash:
        raise TrainingRunManifestError("invocation_hash must match backend_result.invocation_hash.")
    adapter_hash = None
    artifact_hashes: Mapping[str, str] = {}
    adapter_validation_hash = None
    if adapter_validation_report is not None:
        if adapter_validation_report.report_hash != _payload_hash(adapter_validation_report, "report_hash"):
            raise TrainingRunManifestError("adapter validation report hash mismatch.")
        adapter_validation_hash = adapter_validation_report.report_hash
        adapter_hash = adapter_validation_report.adapter_hash
        artifact_hashes = adapter_validation_report.artifact_hashes
    training_log_hash = None
    if training_log_report is not None:
        if training_log_report.report_hash != _payload_hash(training_log_report, "report_hash"):
            raise TrainingRunManifestError("training log report hash mismatch.")
        training_log_hash = training_log_report.report_hash
    if result.trained:
        if adapter_validation_report is None or adapter_validation_report.passed is not True:
            raise TrainingRunManifestError("trained backend result requires passed adapter validation.")
        if not adapter_hash:
            raise TrainingRunManifestError("trained backend result requires adapter_hash.")
    manifest_id = stable_hash(
        {
            "invocation_hash": invocation_hash,
            "backend_result_hash": result.result_hash,
            "adapter_validation_hash": adapter_validation_hash,
            "training_log_hash": training_log_hash,
            "training_plan_hash": training_plan_hash,
            "lora_config_hash": lora_config_hash,
            "dataset_manifest_hash": dataset_manifest_hash,
            "stage": result.stage,
            "trained": result.trained,
            "adapter_hash": adapter_hash,
        }
    )
    return TrainingRunManifest(
        manifest_id=manifest_id,
        invocation_hash=invocation_hash,
        backend_result_hash=result.result_hash,
        adapter_validation_hash=adapter_validation_hash,
        training_log_hash=training_log_hash,
        training_plan_hash=training_plan_hash,
        lora_config_hash=lora_config_hash,
        dataset_manifest_hash=dataset_manifest_hash,
        stage=result.stage,
        trained=result.trained,
        adapter_hash=adapter_hash if result.trained else None,
        artifact_hashes=artifact_hashes,
        metadata=dict(metadata or {}),
    )


def validate_training_run_manifest(
    manifest: TrainingRunManifest,
    *,
    backend_result: BackendResult | None = None,
    adapter_validation_report: AdapterValidationReport | None = None,
) -> TrainingRunManifest:
    if not isinstance(manifest, TrainingRunManifest):
        raise TrainingRunManifestError("manifest must be a TrainingRunManifest.")
    if manifest.manifest_hash != compute_training_run_manifest_hash(manifest):
        raise TrainingRunManifestError("manifest_hash does not match training run manifest payload.")
    if backend_result is not None:
        result = validate_backend_result(backend_result, require_artifact_hashes=backend_result.trained)
        if manifest.stage != result.stage:
            raise TrainingRunManifestError("stage mismatch.")
        if manifest.backend_result_hash != result.result_hash:
            raise TrainingRunManifestError("backend_result_hash mismatch.")
        if manifest.trained != result.trained:
            raise TrainingRunManifestError("trained flag mismatch.")
    if manifest.trained:
        if adapter_validation_report is None:
            raise TrainingRunManifestError("trained manifest requires adapter validation report.")
        if adapter_validation_report.passed is not True:
            raise TrainingRunManifestError("adapter validation did not pass.")
        if manifest.adapter_validation_hash != adapter_validation_report.report_hash:
            raise TrainingRunManifestError("adapter_validation_hash mismatch.")
        if manifest.adapter_hash != adapter_validation_report.adapter_hash:
            raise TrainingRunManifestError("adapter_hash mismatch.")
    return manifest


def compute_training_run_manifest_hash(manifest: TrainingRunManifest | Mapping[str, Any]) -> str:
    if isinstance(manifest, Mapping):
        payload = dict(manifest)
    else:
        payload = {item.name: getattr(manifest, item.name) for item in fields(TrainingRunManifest)}
    payload.pop("manifest_hash", None)
    return stable_hash(payload)


def compute_training_run_manifest_id(manifest: TrainingRunManifest | Mapping[str, Any]) -> str:
    if isinstance(manifest, Mapping):
        payload = dict(manifest)
    else:
        payload = {item.name: getattr(manifest, item.name) for item in fields(TrainingRunManifest)}
    return stable_hash(
        {
            "invocation_hash": payload.get("invocation_hash"),
            "backend_result_hash": payload.get("backend_result_hash"),
            "adapter_validation_hash": payload.get("adapter_validation_hash"),
            "training_log_hash": payload.get("training_log_hash"),
            "training_plan_hash": payload.get("training_plan_hash"),
            "lora_config_hash": payload.get("lora_config_hash"),
            "dataset_manifest_hash": payload.get("dataset_manifest_hash"),
            "stage": payload.get("stage"),
            "trained": payload.get("trained"),
            "adapter_hash": payload.get("adapter_hash"),
        }
    )


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingRunManifestError(f"{field_name} must be non-empty.")
    return value


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


def _flatten(value: Mapping[str, Any], prefix: str = "") -> tuple[tuple[str, Any], ...]:
    items: list[tuple[str, Any]] = []
    for key, child in sorted(value.items(), key=lambda item: str(item[0])):
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, Mapping):
            items.extend(_flatten(child, name))
        else:
            items.append((name, child))
    return tuple(items)


__all__ = [
    "TrainingRunManifest",
    "TrainingRunManifestError",
    "build_training_run_manifest",
    "compute_training_run_manifest_id",
    "compute_training_run_manifest_hash",
    "validate_training_run_manifest",
]
