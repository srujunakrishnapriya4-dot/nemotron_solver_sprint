"""Caller-supplied smoke backend result wrapper for Pass 12."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.training_backend import BackendInvocation, BackendResult, validate_backend_result

from .smoke_config import SmokeTrainingConfigError, _reject_unsafe_metadata


class SmokeBackendError(ValueError):
    """Raised when a smoke backend result is invalid."""


class SmokeBackendProtocol(Protocol):
    def __call__(self, invocation: BackendInvocation) -> Any:
        """Return a SmokeBackendResult, mapping, dataclass, or Pass 11 BackendResult payload."""


@dataclass(frozen=True)
class SmokeBackendResult:
    backend_result: BackendResult | Mapping[str, Any]
    adapter_dir: str | Path | None = None
    training_log: Sequence[Mapping[str, Any]] | Sequence[str] | str | Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    result_hash: str = ""

    def __post_init__(self) -> None:
        try:
            result = validate_backend_result(self.backend_result)
        except Exception as exc:
            raise SmokeBackendError("backend_result must be a valid Pass 11 BackendResult.") from exc
        metadata = dict(self.metadata)
        _reject_metadata(metadata)
        if self.adapter_dir is not None:
            object.__setattr__(self, "adapter_dir", str(self.adapter_dir))
        if result.trained and self.adapter_dir is None:
            raise SmokeBackendError("trained=True requires adapter_dir.")
        object.__setattr__(self, "backend_result", result)
        object.__setattr__(self, "metadata", metadata)
        expected = _payload_hash(self)
        if not self.result_hash:
            object.__setattr__(self, "result_hash", expected)
        elif self.result_hash != expected:
            raise SmokeBackendError("result_hash does not match smoke backend result payload.")


def normalize_smoke_backend_result(
    raw: Any,
    *,
    invocation: BackendInvocation | None = None,
    require_log_validation: bool = False,
) -> SmokeBackendResult:
    if isinstance(raw, SmokeBackendResult):
        result = raw
    elif isinstance(raw, BackendResult):
        result = SmokeBackendResult(raw, adapter_dir=raw.adapter_path)
    elif is_dataclass(raw):
        result = _from_mapping(asdict(raw))
    elif isinstance(raw, Mapping):
        result = _from_mapping(raw)
    else:
        raise SmokeBackendError("smoke backend result must be a mapping, dataclass, SmokeBackendResult, or BackendResult.")
    return validate_smoke_backend_result(result, invocation=invocation, require_log_validation=require_log_validation)


def validate_smoke_backend_result(
    result: SmokeBackendResult | Mapping[str, Any],
    *,
    invocation: BackendInvocation | None = None,
    require_log_validation: bool = False,
) -> SmokeBackendResult:
    normalized = result if isinstance(result, SmokeBackendResult) else normalize_smoke_backend_result(result)
    if normalized.result_hash != _payload_hash(normalized):
        raise SmokeBackendError("result_hash does not match smoke backend result payload.")
    try:
        backend_result = validate_backend_result(normalized.backend_result, invocation=invocation, require_artifact_hashes=normalized.backend_result.trained)
    except Exception as exc:
        raise SmokeBackendError("backend_result failed Pass 11 validation.") from exc
    if backend_result.trained and normalized.adapter_dir is None:
        raise SmokeBackendError("trained=True requires adapter_dir.")
    if require_log_validation and normalized.training_log is None:
        raise SmokeBackendError("require_log_validation=True requires training_log.")
    return normalized


def _from_mapping(payload: Mapping[str, Any]) -> SmokeBackendResult:
    data = dict(payload)
    if "backend_result" not in data:
        data = {"backend_result": data}
    return SmokeBackendResult(**data)


def _payload_hash(result: SmokeBackendResult) -> str:
    return stable_hash(
        {
            "backend_result_hash": result.backend_result.result_hash,
            "adapter_dir": str(result.adapter_dir) if result.adapter_dir is not None else None,
            "training_log": _log_payload(result.training_log),
            "metadata": dict(result.metadata),
        }
    )


def _log_payload(log: Any) -> Any:
    if log is None:
        return None
    if isinstance(log, Path):
        return {"path": str(log)}
    if isinstance(log, str):
        return {"path": log}
    if isinstance(log, Sequence) and not isinstance(log, (str, bytes, bytearray)):
        return tuple(dict(item) if isinstance(item, Mapping) else str(item) for item in log)
    return str(log)


def _reject_metadata(metadata: Mapping[str, Any]) -> None:
    try:
        _reject_unsafe_metadata(metadata, reject_split_claims=True)
    except SmokeTrainingConfigError as exc:
        raise SmokeBackendError(str(exc)) from exc


__all__ = [
    "SmokeBackendError",
    "SmokeBackendProtocol",
    "SmokeBackendResult",
    "normalize_smoke_backend_result",
    "validate_smoke_backend_result",
]
