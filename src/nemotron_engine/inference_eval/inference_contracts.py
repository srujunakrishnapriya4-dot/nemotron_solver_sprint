"""Safe external inference backend contracts for Pass 13."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
import math
import re
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash


class InferenceContractError(ValueError):
    """Raised when inference backend evidence is invalid."""


class InferenceBackendKind(str, Enum):
    EXTERNAL_TEXT_GENERATION = "external_text_generation"
    EXTERNAL_CHAT_COMPLETION = "external_chat_completion"
    EXTERNAL_LOCAL_RUNTIME = "external_local_runtime"
    EXTERNAL_MOCK_FOR_TESTS = "external_mock_for_tests"


class InferenceStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    DRY_RUN = "dry_run"
    REJECTED = "rejected"


@dataclass(frozen=True)
class InferenceBackendInvocation:
    invocation_id: str
    backend_kind: str
    serving_config_hash: str
    prompt_batch_hash: str
    model_hash: str | None
    adapter_hash: str | None
    dry_run: bool
    expected_completion_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    invocation_hash: str = ""

    def __post_init__(self) -> None:
        _require_non_empty(self.invocation_id, "invocation_id")
        backend_kind = _enum_value(InferenceBackendKind, self.backend_kind, "backend_kind")
        for name in ("serving_config_hash", "prompt_batch_hash"):
            _require_non_empty(getattr(self, name), name)
        for name in ("model_hash", "adapter_hash"):
            value = getattr(self, name)
            if value is not None:
                _require_non_empty(value, name)
        _require_bool(self.dry_run, "dry_run")
        _require_int(self.expected_completion_count, "expected_completion_count", minimum=0)
        metadata = _safe_metadata(self.metadata, context="invocation")
        object.__setattr__(self, "backend_kind", backend_kind)
        object.__setattr__(self, "metadata", metadata)
        expected = _payload_hash(self, "invocation_hash")
        if not self.invocation_hash:
            object.__setattr__(self, "invocation_hash", expected)
        elif self.invocation_hash != expected:
            raise InferenceContractError("invocation_hash does not match invocation payload.")


@dataclass(frozen=True)
class InferenceCompletion:
    problem_id: str
    prompt_hash: str
    completion_text: str
    finish_reason: str | None = None
    token_count: int | None = None
    latency_ms: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    completion_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("problem_id", "prompt_hash"):
            _require_non_empty(getattr(self, name), name)
        if not isinstance(self.completion_text, str) or self.completion_text == "":
            raise InferenceContractError("completion_text must be a non-empty raw string.")
        if self.finish_reason is not None and not isinstance(self.finish_reason, str):
            raise InferenceContractError("finish_reason must be a string or None.")
        if self.token_count is not None:
            _require_int(self.token_count, "token_count", minimum=0)
        if self.latency_ms is not None:
            object.__setattr__(self, "latency_ms", _require_finite_non_negative(self.latency_ms, "latency_ms"))
        metadata = _safe_metadata(self.metadata, context="completion")
        object.__setattr__(self, "metadata", metadata)
        expected = _payload_hash(self, "completion_hash")
        if not self.completion_hash:
            object.__setattr__(self, "completion_hash", expected)
        elif self.completion_hash != expected:
            raise InferenceContractError("completion_hash does not match completion payload.")


@dataclass(frozen=True)
class InferenceMetric:
    name: str
    value: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "metric.name")
        object.__setattr__(self, "value", _require_finite(self.value, "metric.value"))
        object.__setattr__(self, "metadata", _safe_metadata(self.metadata, context="metric"))


@dataclass(frozen=True)
class InferenceBackendResult:
    result_id: str
    invocation_hash: str
    backend_kind: str
    status: str
    completions: tuple[InferenceCompletion, ...]
    metrics: tuple[InferenceMetric, ...] = ()
    started: bool = False
    finished: bool = False
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    result_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("result_id", "invocation_hash"):
            _require_non_empty(getattr(self, name), name)
        backend_kind = _enum_value(InferenceBackendKind, self.backend_kind, "backend_kind")
        status = _enum_value(InferenceStatus, self.status, "status")
        completions = tuple(_completion(item) for item in self.completions)
        metrics = tuple(_metric(item) for item in self.metrics)
        for name in ("started", "finished"):
            _require_bool(getattr(self, name), name)
        if self.error is not None and not isinstance(self.error, str):
            raise InferenceContractError("error must be a string or None.")
        metadata = _safe_metadata(self.metadata, context="result")
        if status == InferenceStatus.SUCCESS.value:
            if self.started is not True or self.finished is not True:
                raise InferenceContractError("status=success requires started=True and finished=True.")
            if self.error is not None:
                raise InferenceContractError("status=success requires error is None.")
            if not completions:
                raise InferenceContractError("status=success requires at least one completion.")
        else:
            if self.error is None and completions:
                raise InferenceContractError("non-success result requires error or empty completions.")
        object.__setattr__(self, "backend_kind", backend_kind)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "completions", completions)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "metadata", metadata)
        expected = compute_inference_result_hash(self)
        if not self.result_hash:
            object.__setattr__(self, "result_hash", expected)
        elif self.result_hash != expected:
            raise InferenceContractError("result_hash does not match result payload.")


def validate_inference_invocation(invocation: InferenceBackendInvocation | Mapping[str, Any]) -> InferenceBackendInvocation:
    if isinstance(invocation, InferenceBackendInvocation):
        expected = _payload_hash(invocation, "invocation_hash")
        if invocation.invocation_hash != expected:
            raise InferenceContractError("invocation_hash does not match invocation payload.")
        return invocation
    if isinstance(invocation, Mapping):
        return InferenceBackendInvocation(**dict(invocation))
    raise InferenceContractError("invocation must be an InferenceBackendInvocation or mapping.")


def validate_inference_result(
    result: InferenceBackendResult | Mapping[str, Any],
    *,
    invocation: InferenceBackendInvocation | Mapping[str, Any] | None = None,
) -> InferenceBackendResult:
    normalized = _result(result)
    if invocation is not None:
        inv = validate_inference_invocation(invocation)
        if normalized.invocation_hash != inv.invocation_hash:
            raise InferenceContractError("backend result invocation_hash mismatch.")
        if normalized.backend_kind != inv.backend_kind:
            raise InferenceContractError("backend result backend_kind mismatch.")
        if inv.dry_run and normalized.status == InferenceStatus.SUCCESS.value:
            raise InferenceContractError("dry_run invocation cannot produce success completions.")
        if inv.dry_run and normalized.completions:
            raise InferenceContractError("dry_run invocation cannot produce completions.")
        if normalized.status == InferenceStatus.SUCCESS.value and len(normalized.completions) != inv.expected_completion_count:
            raise InferenceContractError("success completion count does not match expected_completion_count.")
    if normalized.result_hash != compute_inference_result_hash(normalized):
        raise InferenceContractError("result_hash does not match result payload.")
    return normalized


def compute_inference_result_hash(result: InferenceBackendResult | Mapping[str, Any]) -> str:
    payload = _mapping_payload(result)
    payload.pop("result_hash", None)
    return stable_hash(payload)


def _result(value: InferenceBackendResult | Mapping[str, Any]) -> InferenceBackendResult:
    if isinstance(value, InferenceBackendResult):
        return value
    if not isinstance(value, Mapping):
        raise InferenceContractError("backend result must be an InferenceBackendResult or mapping.")
    payload = dict(value)
    payload["completions"] = tuple(_completion(item) for item in payload.get("completions", ()))
    payload["metrics"] = tuple(_metric(item) for item in payload.get("metrics", ()))
    return InferenceBackendResult(**payload)


def _completion(value: InferenceCompletion | Mapping[str, Any]) -> InferenceCompletion:
    if isinstance(value, InferenceCompletion):
        return value
    if isinstance(value, Mapping):
        return InferenceCompletion(**dict(value))
    raise InferenceContractError("completions must contain InferenceCompletion values.")


def _metric(value: InferenceMetric | Mapping[str, Any]) -> InferenceMetric:
    if isinstance(value, InferenceMetric):
        return value
    if isinstance(value, Mapping):
        return InferenceMetric(**dict(value))
    raise InferenceContractError("metrics must contain InferenceMetric values.")


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


def _mapping_payload(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return {item.name: getattr(value, item.name) for item in fields(value)}
    raise InferenceContractError("value must be a dataclass or mapping.")


def _enum_value(enum_cls: type[Enum], value: Any, field_name: str) -> str:
    if isinstance(value, enum_cls):
        return str(value.value)
    if not isinstance(value, str):
        raise InferenceContractError(f"{field_name} must be a string enum value.")
    try:
        return str(enum_cls(value).value)
    except ValueError as exc:
        raise InferenceContractError(f"{field_name} has invalid value: {value!r}.") from exc


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InferenceContractError(f"{field_name} must be a non-empty string.")
    return value


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise InferenceContractError(f"{field_name} must be boolean.")
    return value


def _require_int(value: Any, field_name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InferenceContractError(f"{field_name} must be an integer.")
    if value < minimum:
        raise InferenceContractError(f"{field_name} must be >= {minimum}.")
    return value


def _require_finite(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise InferenceContractError(f"{field_name} must be finite.")
    return float(value)


def _require_finite_non_negative(value: Any, field_name: str) -> float:
    number = _require_finite(value, field_name)
    if number < 0.0:
        raise InferenceContractError(f"{field_name} must be non-negative.")
    return number


FORBIDDEN_METADATA_KEYS = frozenset(
    {
        "expected_answer",
        "gold_answer",
        "target_answer",
        "label",
        "correct",
        "is_correct",
        "correctness",
        "normalized_answer",
        "final_answer",
        "leakage",
        "split_leakage",
        "contaminated",
        "contamination",
        "score",
        "public_score",
        "private_score",
        "score_claim",
        "score_delta",
        "accuracy_claim",
        "leaderboard_score",
        "public_leaderboard_score",
        "private_leaderboard_score",
    }
)


def _safe_metadata(value: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InferenceContractError(f"{context} metadata must be a mapping.")
    metadata = dict(value)
    _reject_forbidden_metadata(metadata)
    return metadata


def _reject_forbidden_metadata(metadata: Mapping[str, Any]) -> None:
    for key, value in _flatten(metadata):
        key_text = _normalized_text(key)
        combined = _normalized_text(f"{key} {value}")
        if any(_normalized_text(forbidden) in key_text for forbidden in FORBIDDEN_METADATA_KEYS):
            raise InferenceContractError("metadata contains forbidden answer/correctness/leakage fields.")
        if "kaggle" in combined or "leaderboard" in combined:
            raise InferenceContractError("metadata cannot mention Kaggle or leaderboard claims.")
        if "submission" in combined and any(word in combined for word in ("success", "ready", "submitted", "passed", "win")):
            raise InferenceContractError("metadata cannot claim submission success.")
        if "promotion" in combined and any(word in combined for word in ("success", "ready", "allow", "passed", "promoted")):
            raise InferenceContractError("metadata cannot claim promotion success.")
        if "package" in combined and any(word in combined for word in ("success", "ready", "built", "passed")):
            raise InferenceContractError("metadata cannot claim package success.")
        if "score" in combined and any(word in combined for word in ("guarantee", "guaranteed", "95", "95+")):
            raise InferenceContractError("metadata cannot claim score guarantees.")
        if "95+" in str(value).lower() or "95 plus" in combined:
            raise InferenceContractError("metadata cannot claim 95+ score.")
        if "guaranteed" in combined or "guarantee" in combined:
            raise InferenceContractError("metadata cannot claim guarantees.")


def _flatten(value: Any, prefix: str = "") -> tuple[tuple[str, Any], ...]:
    if is_dataclass(value):
        value = asdict(value)
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


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+_]+", " ", str(value).lower())).strip()


__all__ = [
    "InferenceBackendInvocation",
    "InferenceBackendKind",
    "InferenceBackendResult",
    "InferenceCompletion",
    "InferenceContractError",
    "InferenceMetric",
    "InferenceStatus",
    "compute_inference_result_hash",
    "validate_inference_invocation",
    "validate_inference_result",
]
