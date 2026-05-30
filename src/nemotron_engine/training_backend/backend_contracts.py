"""Strict external training backend contracts for Pass 11."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import math
import re
from pathlib import Path
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash


class BackendContractError(ValueError):
    """Raised when an external backend contract is invalid."""


class BackendKind(str, Enum):
    EXTERNAL_SFT = "external_sft"
    EXTERNAL_DPO = "external_dpo"
    EXTERNAL_GRPO = "external_grpo"
    EXTERNAL_FINAL_SFT = "external_final_sft"


class TrainingStage(str, Enum):
    SFT = "sft"
    DPO = "dpo"
    GRPO = "grpo"
    FINAL_SFT_REFRESH = "final_sft_refresh"


class BackendStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    DRY_RUN = "dry_run"
    REJECTED = "rejected"


KIND_STAGE = {
    BackendKind.EXTERNAL_SFT.value: TrainingStage.SFT.value,
    BackendKind.EXTERNAL_DPO.value: TrainingStage.DPO.value,
    BackendKind.EXTERNAL_GRPO.value: TrainingStage.GRPO.value,
    BackendKind.EXTERNAL_FINAL_SFT.value: TrainingStage.FINAL_SFT_REFRESH.value,
}


@dataclass(frozen=True)
class BackendInvocation:
    invocation_id: str
    backend_kind: str
    stage: str
    training_plan_hash: str
    lora_config_hash: str
    dataset_manifest_hash: str | None
    input_row_count: int
    input_pair_count: int
    seed: int
    dry_run: bool
    output_dir: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    invocation_hash: str = ""

    def __post_init__(self) -> None:
        _require_non_empty(self.invocation_id, "invocation_id")
        backend_kind = _enum_value(BackendKind, self.backend_kind, "backend_kind")
        stage = _enum_value(TrainingStage, self.stage, "stage")
        _require_kind_stage(backend_kind, stage)
        for name in ("training_plan_hash", "lora_config_hash"):
            _require_non_empty(getattr(self, name), name)
        if self.dataset_manifest_hash is not None:
            _require_non_empty(self.dataset_manifest_hash, "dataset_manifest_hash")
        for name in ("input_row_count", "input_pair_count", "seed"):
            _require_int(getattr(self, name), name, minimum=0)
        _require_bool(self.dry_run, "dry_run")
        if self.output_dir is not None:
            _require_non_empty(self.output_dir, "output_dir")
            _validate_ref_path(self.output_dir, "output_dir")
        metadata = _metadata(self.metadata)
        _reject_forbidden_claims(metadata)
        object.__setattr__(self, "backend_kind", backend_kind)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "metadata", metadata)
        expected = _payload_hash(self, "invocation_hash")
        if not self.invocation_hash:
            object.__setattr__(self, "invocation_hash", expected)
        elif self.invocation_hash != expected:
            raise BackendContractError("invocation_hash does not match invocation payload.")


@dataclass(frozen=True)
class BackendArtifactRef:
    path: str
    role: str
    sha256: str | None = None
    size_bytes: int | None = None
    required: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.path, "artifact.path")
        _validate_ref_path(self.path, "artifact.path")
        _require_non_empty(self.role, "artifact.role")
        if self.sha256 is not None and not _is_sha256(self.sha256):
            raise BackendContractError("artifact.sha256 must be a lowercase 64-character hex digest.")
        if self.size_bytes is not None:
            _require_int(self.size_bytes, "artifact.size_bytes", minimum=0)
        _require_bool(self.required, "artifact.required")
        metadata = _metadata(self.metadata)
        _reject_forbidden_claims(metadata)
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True)
class BackendMetric:
    name: str
    value: float
    step: int | None = None
    split: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "metric.name")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)) or not math.isfinite(float(self.value)):
            raise BackendContractError("metric.value must be finite.")
        if self.step is not None:
            _require_int(self.step, "metric.step", minimum=0)
        if self.split is not None:
            _require_non_empty(self.split, "metric.split")
        metadata = _metadata(self.metadata)
        _reject_forbidden_claims(metadata)
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True)
class BackendResult:
    result_id: str
    invocation_hash: str
    backend_kind: str
    stage: str
    status: str
    trained: bool
    adapter_path: str | None
    checkpoint_path: str | None
    artifacts: tuple[BackendArtifactRef, ...] = ()
    metrics: tuple[BackendMetric, ...] = ()
    started: bool = False
    finished: bool = False
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    result_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("result_id", "invocation_hash"):
            _require_non_empty(getattr(self, name), name)
        backend_kind = _enum_value(BackendKind, self.backend_kind, "backend_kind")
        stage = _enum_value(TrainingStage, self.stage, "stage")
        status = _enum_value(BackendStatus, self.status, "status")
        _require_kind_stage(backend_kind, stage)
        for name in ("trained", "started", "finished"):
            _require_bool(getattr(self, name), name)
        if self.adapter_path is not None:
            _require_non_empty(self.adapter_path, "adapter_path")
            _validate_ref_path(self.adapter_path, "adapter_path")
        if self.checkpoint_path is not None:
            _require_non_empty(self.checkpoint_path, "checkpoint_path")
            _validate_ref_path(self.checkpoint_path, "checkpoint_path")
        if self.error is not None and not isinstance(self.error, str):
            raise BackendContractError("error must be a string or None.")
        artifacts = tuple(_artifact(item) for item in self.artifacts)
        metrics = tuple(_metric(item) for item in self.metrics)
        metadata = _metadata(self.metadata)
        _reject_forbidden_claims(metadata)
        required_adapter_artifacts = tuple(item for item in artifacts if item.required and _is_adapter_role(item.role))
        if status != BackendStatus.SUCCESS.value and self.trained is not False:
            raise BackendContractError("trained=True requires status == success.")
        if self.trained:
            if status != BackendStatus.SUCCESS.value:
                raise BackendContractError("trained=True requires status == success.")
            if self.started is not True or self.finished is not True:
                raise BackendContractError("trained=True requires started=True and finished=True.")
            if self.error is not None:
                raise BackendContractError("trained=True requires error is None.")
            if self.adapter_path is None:
                raise BackendContractError("trained=True requires adapter_path.")
            if not required_adapter_artifacts:
                raise BackendContractError("trained=True requires at least one required adapter artifact.")
            if self.checkpoint_path is not None:
                raise BackendContractError("trained=True cannot include checkpoint_path for LoRA-only results.")
            if any(item.sha256 is None or item.size_bytes is None for item in required_adapter_artifacts):
                raise BackendContractError("trained=True required adapter artifacts need sha256 and size_bytes.")
        if status == BackendStatus.DRY_RUN.value and self.trained:
            raise BackendContractError("dry-run backend result cannot claim trained=True.")
        if self.adapter_path is None and any(_is_adapter_role(item.role) for item in artifacts):
            raise BackendContractError("adapter artifacts require adapter_path.")
        object.__setattr__(self, "backend_kind", backend_kind)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "metadata", metadata)
        expected = compute_backend_result_hash(self)
        if not self.result_hash:
            object.__setattr__(self, "result_hash", expected)
        elif self.result_hash != expected:
            raise BackendContractError("result_hash does not match result payload.")


def validate_backend_invocation(invocation: BackendInvocation | Mapping[str, Any]) -> BackendInvocation:
    if isinstance(invocation, BackendInvocation):
        expected = _payload_hash(invocation, "invocation_hash")
        if invocation.invocation_hash != expected:
            raise BackendContractError("invocation_hash does not match invocation payload.")
        return invocation
    if isinstance(invocation, Mapping):
        return BackendInvocation(**dict(invocation))
    raise BackendContractError("invocation must be a BackendInvocation.")


def validate_backend_result(
    result: BackendResult | Mapping[str, Any],
    *,
    invocation: BackendInvocation | None = None,
    allow_checkpoint_path: bool = False,
    require_artifact_hashes: bool = False,
) -> BackendResult:
    normalized = _result(result)
    if invocation is not None:
        invocation = validate_backend_invocation(invocation)
        if normalized.invocation_hash != invocation.invocation_hash:
            raise BackendContractError("backend result invocation_hash mismatch.")
        if normalized.backend_kind != invocation.backend_kind or normalized.stage != invocation.stage:
            raise BackendContractError("backend result kind/stage mismatch.")
        if invocation.dry_run and normalized.trained:
            raise BackendContractError("dry_run invocation cannot produce trained=True result.")
    if normalized.checkpoint_path is not None and not allow_checkpoint_path:
        raise BackendContractError("checkpoint_path is not allowed for LoRA-only backend results.")
    if normalized.trained and require_artifact_hashes:
        required = tuple(item for item in normalized.artifacts if item.required and _is_adapter_role(item.role))
        if any(item.sha256 is None or item.size_bytes is None for item in required):
            raise BackendContractError("required adapter artifacts need sha256 and size_bytes.")
    return normalized


def compute_backend_result_hash(result: BackendResult | Mapping[str, Any]) -> str:
    if isinstance(result, Mapping):
        payload = dict(result)
    else:
        payload = {item.name: getattr(result, item.name) for item in fields(BackendResult)}
    payload.pop("result_hash", None)
    return stable_hash(payload)


def _result(value: BackendResult | Mapping[str, Any]) -> BackendResult:
    if isinstance(value, BackendResult):
        return value
    if not isinstance(value, Mapping):
        raise BackendContractError("backend result must be a BackendResult or mapping.")
    payload = dict(value)
    payload["artifacts"] = tuple(_artifact(item) for item in payload.get("artifacts", ()))
    payload["metrics"] = tuple(_metric(item) for item in payload.get("metrics", ()))
    return BackendResult(**payload)


def _artifact(value: BackendArtifactRef | Mapping[str, Any]) -> BackendArtifactRef:
    if isinstance(value, BackendArtifactRef):
        return value
    if isinstance(value, Mapping):
        return BackendArtifactRef(**dict(value))
    raise BackendContractError("artifacts must contain BackendArtifactRef values.")


def _metric(value: BackendMetric | Mapping[str, Any]) -> BackendMetric:
    if isinstance(value, BackendMetric):
        return value
    if isinstance(value, Mapping):
        return BackendMetric(**dict(value))
    raise BackendContractError("metrics must contain BackendMetric values.")


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


def _enum_value(enum_cls: type[Enum], value: Any, field_name: str) -> str:
    if isinstance(value, enum_cls):
        return str(value.value)
    if not isinstance(value, str):
        raise BackendContractError(f"{field_name} must be a string enum value.")
    try:
        return str(enum_cls(value).value)
    except ValueError as exc:
        raise BackendContractError(f"{field_name} has invalid value: {value!r}.") from exc


def _require_kind_stage(backend_kind: str, stage: str) -> None:
    if KIND_STAGE.get(backend_kind) != stage:
        raise BackendContractError("backend_kind and stage are incompatible.")


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BackendContractError(f"{field_name} must be a non-empty string.")
    return value


def _require_int(value: Any, field_name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BackendContractError(f"{field_name} must be an integer.")
    if value < minimum:
        raise BackendContractError(f"{field_name} must be >= {minimum}.")
    return value


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise BackendContractError(f"{field_name} must be boolean.")
    return value


def _metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BackendContractError("metadata must be a mapping.")
    return dict(value)


def _validate_ref_path(value: str, field_name: str) -> None:
    raw = value.replace("\\", "/")
    if "\x00" in raw:
        raise BackendContractError(f"{field_name} must not contain NUL.")
    if re.match(r"^[A-Za-z]:", raw):
        raise BackendContractError(f"{field_name} must not contain a drive letter.")
    if raw.startswith("/") and not raw.startswith(("external://", "s3://", "gs://", "hf://")):
        raise BackendContractError(f"{field_name} absolute paths require an explicit external ref scheme.")
    parts = tuple(part for part in raw.split("/") if part)
    if any(part == ".." for part in parts):
        raise BackendContractError(f"{field_name} must not contain path traversal.")


def _is_adapter_role(role: str) -> bool:
    text = role.lower()
    return "adapter" in text or text in {"lora", "adapter_model", "adapter_config"}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _reject_forbidden_claims(metadata: Mapping[str, Any]) -> None:
    for key, value in _flatten(metadata):
        text = _normalized_text(f"{key} {value}")
        truthy = value is True or (isinstance(value, str) and value.strip().lower() in {"true", "yes", "success", "ready", "guaranteed"})
        if ("kaggle" in text or "leaderboard" in text or "submission" in text) and any(
            word in text for word in ("success", "ready", "submitted", "passed", "winner")
        ):
            raise BackendContractError("metadata cannot claim Kaggle/leaderboard/submission success.")
        if ("95+" in str(value).lower() or "95 plus" in text or "95" in text) and "guarante" in text:
            raise BackendContractError("metadata cannot claim score guarantees.")
        if "score" in text and "guarante" in text:
            raise BackendContractError("metadata cannot claim score guarantees.")
        if "guaranteed" in text:
            raise BackendContractError("metadata cannot claim guarantees.")
        if truthy and "trained" in text and "adapter" in text and not ("validated" in text or "hash" in text):
            raise BackendContractError("metadata cannot claim trained adapter without validated evidence.")


def _flatten(value: Any, prefix: str = "") -> tuple[tuple[str, Any], ...]:
    if is_dataclass(value):
        value = {item.name: getattr(value, item.name) for item in fields(value)}
    if isinstance(value, Mapping):
        items: list[tuple[str, Any]] = []
        for key, child in sorted(value.items(), key=lambda item: str(item[0])):
            name = f"{prefix}.{key}" if prefix else str(key)
            items.extend(_flatten(child, name))
        return tuple(items)
    if isinstance(value, (tuple, list, set)):
        items = []
        for index, child in enumerate(value):
            items.extend(_flatten(child, f"{prefix}.{index}"))
        return tuple(items)
    return ((prefix, value),)


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+]+", " ", value.lower())).strip()


__all__ = [
    "BackendArtifactRef",
    "BackendContractError",
    "BackendInvocation",
    "BackendKind",
    "BackendMetric",
    "BackendResult",
    "BackendStatus",
    "TrainingStage",
    "compute_backend_result_hash",
    "validate_backend_invocation",
    "validate_backend_result",
]
