"""Raw completion capture records for Pass 13."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash

from .inference_contracts import (
    InferenceBackendInvocation,
    InferenceBackendResult,
    InferenceCompletion,
    _reject_forbidden_metadata,
    validate_inference_invocation,
    validate_inference_result,
)
from .prompt_batch import PromptBatch, validate_prompt_batch


class CompletionCaptureError(ValueError):
    """Raised when completion capture evidence is invalid."""


@dataclass(frozen=True)
class CompletionRecord:
    problem_id: str
    prompt_hash: str
    completion_text: str
    serving_config_hash: str
    backend_invocation_hash: str
    backend_result_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    completion_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("problem_id", "prompt_hash", "serving_config_hash", "backend_invocation_hash", "backend_result_hash"):
            _require_non_empty(getattr(self, name), name)
        if not isinstance(self.completion_text, str) or self.completion_text == "":
            raise CompletionCaptureError("completion_text must be a non-empty raw string.")
        metadata = _safe_metadata(self.metadata)
        object.__setattr__(self, "metadata", metadata)
        expected = _payload_hash(self, "completion_hash")
        if not self.completion_hash:
            object.__setattr__(self, "completion_hash", expected)
        elif self.completion_hash != expected:
            raise CompletionCaptureError("completion_hash does not match completion record payload.")


@dataclass(frozen=True)
class CompletionCaptureReport:
    records: tuple[CompletionRecord, ...]
    prompt_batch_hash: str
    serving_config_hash: str
    backend_invocation_hash: str
    backend_result_hash: str
    captured_count: int
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    report_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("prompt_batch_hash", "serving_config_hash", "backend_invocation_hash", "backend_result_hash"):
            _require_non_empty(getattr(self, name), name)
        records = tuple(_record(item) for item in self.records)
        if not isinstance(self.captured_count, int) or isinstance(self.captured_count, bool) or self.captured_count < 0:
            raise CompletionCaptureError("captured_count must be a non-negative integer.")
        if self.captured_count != len(records):
            raise CompletionCaptureError("captured_count must equal len(records).")
        if len({item.problem_id for item in records}) != len(records):
            raise CompletionCaptureError("records contain duplicate problem_id values.")
        for record in records:
            if record.serving_config_hash != self.serving_config_hash:
                raise CompletionCaptureError("record serving_config_hash mismatch.")
            if record.backend_invocation_hash != self.backend_invocation_hash:
                raise CompletionCaptureError("record backend_invocation_hash mismatch.")
            if record.backend_result_hash != self.backend_result_hash:
                raise CompletionCaptureError("record backend_result_hash mismatch.")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "errors", tuple(str(item) for item in self.errors))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        expected = compute_completion_capture_hash(self)
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected)
        elif self.report_hash != expected:
            raise CompletionCaptureError("report_hash does not match completion capture report payload.")


def capture_completions(
    prompt_batch: PromptBatch,
    backend_invocation: InferenceBackendInvocation,
    backend_result: InferenceBackendResult,
) -> CompletionCaptureReport:
    batch = validate_prompt_batch(prompt_batch)
    invocation = validate_inference_invocation(backend_invocation)
    result = validate_inference_result(backend_result, invocation=invocation)
    examples_by_id = {item.problem_id: item for item in batch.examples}
    completions_by_id = {item.problem_id: item for item in result.completions}
    if len(completions_by_id) != len(result.completions):
        raise CompletionCaptureError("backend result contains duplicate completion problem_id values.")
    missing = tuple(sorted(set(examples_by_id) - set(completions_by_id)))
    extra = tuple(sorted(set(completions_by_id) - set(examples_by_id)))
    if missing:
        raise CompletionCaptureError(f"missing completion for prompt(s): {missing}")
    if extra:
        raise CompletionCaptureError(f"extra completion for unknown prompt(s): {extra}")
    records: list[CompletionRecord] = []
    for problem_id in sorted(examples_by_id):
        example = examples_by_id[problem_id]
        completion = completions_by_id[problem_id]
        if completion.prompt_hash != example.prompt_hash:
            raise CompletionCaptureError("prompt_hash mismatch.")
        records.append(
            CompletionRecord(
                problem_id=problem_id,
                prompt_hash=completion.prompt_hash,
                completion_text=completion.completion_text,
                serving_config_hash=batch.serving_config_hash,
                backend_invocation_hash=invocation.invocation_hash,
                backend_result_hash=result.result_hash,
                metadata=dict(completion.metadata),
            )
        )
    report = CompletionCaptureReport(
        records=tuple(records),
        prompt_batch_hash=batch.batch_hash,
        serving_config_hash=batch.serving_config_hash,
        backend_invocation_hash=invocation.invocation_hash,
        backend_result_hash=result.result_hash,
        captured_count=len(records),
    )
    validate_completion_records(report, prompt_batch=batch, backend_result=result)
    return report


def validate_completion_records(
    report: CompletionCaptureReport | Mapping[str, Any],
    *,
    prompt_batch: PromptBatch | None = None,
    backend_result: InferenceBackendResult | None = None,
) -> CompletionCaptureReport:
    normalized = report if isinstance(report, CompletionCaptureReport) else CompletionCaptureReport(**dict(report))
    if normalized.report_hash != compute_completion_capture_hash(normalized):
        raise CompletionCaptureError("report_hash does not match completion capture report payload.")
    if prompt_batch is not None:
        batch = validate_prompt_batch(prompt_batch)
        if normalized.prompt_batch_hash != batch.batch_hash:
            raise CompletionCaptureError("prompt_batch_hash mismatch.")
        expected = {item.problem_id: item.prompt_hash for item in batch.examples}
        observed = {item.problem_id: item.prompt_hash for item in normalized.records}
        if set(expected) != set(observed):
            raise CompletionCaptureError("completion record IDs do not match prompt batch.")
        for problem_id, prompt_hash in expected.items():
            if observed[problem_id] != prompt_hash:
                raise CompletionCaptureError("prompt_hash mismatch.")
        if normalized.serving_config_hash != batch.serving_config_hash:
            raise CompletionCaptureError("serving_config_hash mismatch.")
    if backend_result is not None:
        result = validate_inference_result(backend_result)
        if normalized.backend_result_hash != result.result_hash:
            raise CompletionCaptureError("backend_result_hash mismatch.")
    return normalized


def compute_completion_capture_hash(report: CompletionCaptureReport | Mapping[str, Any]) -> str:
    payload = dict(report) if isinstance(report, Mapping) else {item.name: getattr(report, item.name) for item in fields(CompletionCaptureReport)}
    payload.pop("report_hash", None)
    return stable_hash(payload)


def _record(value: CompletionRecord | Mapping[str, Any]) -> CompletionRecord:
    if isinstance(value, CompletionRecord):
        return value
    if isinstance(value, Mapping):
        return CompletionRecord(**dict(value))
    raise CompletionCaptureError("records must contain CompletionRecord values.")


def _safe_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CompletionCaptureError("metadata must be a mapping.")
    metadata = dict(value)
    try:
        _reject_forbidden_metadata(metadata)
    except Exception as exc:
        raise CompletionCaptureError(str(exc)) from exc
    return metadata


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CompletionCaptureError(f"{field_name} must be a non-empty string.")
    return value


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


__all__ = [
    "CompletionCaptureError",
    "CompletionCaptureReport",
    "CompletionRecord",
    "capture_completions",
    "compute_completion_capture_hash",
    "validate_completion_records",
]
