"""Caller-supplied inference backend runner for Pass 13."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Callable, Mapping

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.evaluation.submission_exact import compare_serving_configs
from nemotron_engine.runtime.serving_config import ServingConfig

from .inference_contracts import (
    InferenceBackendInvocation,
    InferenceBackendResult,
    InferenceContractError,
    InferenceStatus,
    _reject_forbidden_metadata,
    validate_inference_invocation,
    validate_inference_result,
)
from .prompt_batch import PromptBatch, validate_prompt_batch


class InferenceRunnerError(ValueError):
    """Raised when an inference backend cannot be invoked safely."""


@dataclass(frozen=True)
class InferenceRunnerConfig:
    serving_config: ServingConfig
    allow_backend_on_dry_run: bool = False
    allow_non_submission_exact_eval: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.serving_config, ServingConfig):
            raise InferenceRunnerError("serving_config must be a ServingConfig.")
        for name in ("allow_backend_on_dry_run", "allow_non_submission_exact_eval"):
            if not isinstance(getattr(self, name), bool):
                raise InferenceRunnerError(f"{name} must be boolean.")
        metadata = _safe_metadata(self.metadata)
        object.__setattr__(self, "metadata", metadata)
        expected = _payload_hash(self, "config_hash")
        if not self.config_hash:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise InferenceRunnerError("config_hash does not match runner config payload.")


@dataclass(frozen=True)
class InferenceRunReport:
    invocation_hash: str
    prompt_batch_hash: str
    serving_config_hash: str
    called_backend: bool
    backend_result: InferenceBackendResult | None
    status: str
    completed: bool
    non_submission_exact: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    report_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("invocation_hash", "prompt_batch_hash", "serving_config_hash"):
            _require_non_empty(getattr(self, name), name)
        for name in ("called_backend", "completed", "non_submission_exact"):
            if not isinstance(getattr(self, name), bool):
                raise InferenceRunnerError(f"{name} must be boolean.")
        if self.status not in {item.value for item in InferenceStatus}:
            raise InferenceRunnerError("invalid inference run status.")
        result = self.backend_result
        if result is not None and not isinstance(result, InferenceBackendResult):
            result = InferenceBackendResult(**dict(result))  # type: ignore[arg-type]
        errors = tuple(str(item) for item in self.errors)
        warnings = tuple(str(item) for item in self.warnings)
        metadata = _safe_metadata(self.metadata)
        dry_run = metadata.get("dry_run") is True or self.status == InferenceStatus.DRY_RUN.value
        success_like = self.status == InferenceStatus.SUCCESS.value or self.completed
        if self.status == InferenceStatus.SUCCESS.value and self.completed is not True:
            raise InferenceRunnerError("status=success requires completed=True.")
        if self.completed and self.status != InferenceStatus.SUCCESS.value:
            raise InferenceRunnerError("completed=True requires status=success.")
        if self.status != InferenceStatus.SUCCESS.value and self.completed:
            raise InferenceRunnerError("status other than success requires completed=False.")
        if dry_run and success_like:
            raise InferenceRunnerError("dry-run report cannot be success-like.")
        if self.non_submission_exact and success_like:
            raise InferenceRunnerError("non-submission-exact report cannot be success-like.")
        if self.completed and errors:
            raise InferenceRunnerError("completed=True cannot include errors.")
        if self.completed and result is not None and result.status != InferenceStatus.SUCCESS.value:
            raise InferenceRunnerError("completed=True requires successful backend_result.")
        if result is not None and result.status != InferenceStatus.SUCCESS.value and not errors and not dry_run:
            raise InferenceRunnerError("errors cannot be empty while backend_result failed.")
        if self.status == InferenceStatus.SUCCESS.value:
            if result is None or result.status != InferenceStatus.SUCCESS.value:
                raise InferenceRunnerError("success report requires successful backend_result.")
            if errors:
                raise InferenceRunnerError("success report cannot include errors.")
        object.__setattr__(self, "backend_result", result)
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "metadata", metadata)
        expected = _payload_hash(self, "report_hash")
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected)
        elif self.report_hash != expected:
            raise InferenceRunnerError("report_hash does not match inference run report payload.")


def invoke_inference_backend(
    invocation: InferenceBackendInvocation,
    prompt_batch: PromptBatch,
    backend: Callable[[InferenceBackendInvocation, PromptBatch], Any] | object | None = None,
    *,
    config: InferenceRunnerConfig | None = None,
) -> InferenceRunReport:
    if config is None:
        config = InferenceRunnerConfig(serving_config=ServingConfig(prompt_template_hash="prompt", tokenizer_hash="tokenizer", model_hash="model"))
    cfg = config
    invocation = validate_inference_invocation(invocation)
    batch = validate_prompt_batch(prompt_batch, serving_config=cfg.serving_config)
    if batch.batch_hash != invocation.prompt_batch_hash:
        raise InferenceRunnerError("invocation prompt_batch_hash mismatch.")
    serving_hash = stable_hash(cfg.serving_config.to_dict())
    if invocation.serving_config_hash != serving_hash or batch.serving_config_hash != serving_hash:
        raise InferenceRunnerError("serving_config_hash mismatch.")
    exact_report = compare_serving_configs(cfg.serving_config, cfg.serving_config)
    non_exact = exact_report.passed is not True
    if non_exact and not cfg.allow_non_submission_exact_eval:
        raise InferenceRunnerError("serving config is not submission-exact.")
    if invocation.dry_run and not cfg.allow_backend_on_dry_run:
        result = InferenceBackendResult(
            result_id="dry-run-" + stable_hash({"invocation_hash": invocation.invocation_hash})[:16],
            invocation_hash=invocation.invocation_hash,
            backend_kind=invocation.backend_kind,
            status=InferenceStatus.DRY_RUN.value,
            completions=(),
            metrics=(),
            started=False,
            finished=False,
            error=None,
            metadata={},
        )
        return InferenceRunReport(
            invocation_hash=invocation.invocation_hash,
            prompt_batch_hash=batch.batch_hash,
            serving_config_hash=serving_hash,
            called_backend=False,
            backend_result=result,
            status=InferenceStatus.DRY_RUN.value,
            completed=False,
            non_submission_exact=non_exact,
            errors=(),
            warnings=tuple(exact_report.errors),
            metadata={"dry_run": True},
        )
    if backend is None:
        raise InferenceRunnerError("dry_run=False requires an explicit backend.")
    try:
        raw = _call_backend(backend, invocation, batch)
    except Exception as exc:
        result = InferenceBackendResult(
            result_id="failed-" + stable_hash({"invocation_hash": invocation.invocation_hash, "error": str(exc)})[:16],
            invocation_hash=invocation.invocation_hash,
            backend_kind=invocation.backend_kind,
            status=InferenceStatus.FAILED.value,
            completions=(),
            metrics=(),
            started=True,
            finished=True,
            error=str(exc) or type(exc).__name__,
            metadata={},
        )
        return InferenceRunReport(
            invocation_hash=invocation.invocation_hash,
            prompt_batch_hash=batch.batch_hash,
            serving_config_hash=serving_hash,
            called_backend=True,
            backend_result=result,
            status=InferenceStatus.FAILED.value,
            completed=False,
            non_submission_exact=non_exact,
            errors=("backend exception",),
            warnings=tuple(exact_report.errors),
            metadata={"dry_run": invocation.dry_run},
        )
    try:
        result = validate_inference_result(_mapping_from_raw(raw), invocation=invocation)
    except Exception as exc:
        failed = InferenceBackendResult(
            result_id="rejected-" + stable_hash({"invocation_hash": invocation.invocation_hash, "error": str(exc)})[:16],
            invocation_hash=invocation.invocation_hash,
            backend_kind=invocation.backend_kind,
            status=InferenceStatus.REJECTED.value,
            completions=(),
            metrics=(),
            started=True,
            finished=True,
            error=str(exc) or type(exc).__name__,
            metadata={},
        )
        return InferenceRunReport(
            invocation_hash=invocation.invocation_hash,
            prompt_batch_hash=batch.batch_hash,
            serving_config_hash=serving_hash,
            called_backend=True,
            backend_result=failed,
            status=InferenceStatus.REJECTED.value,
            completed=False,
            non_submission_exact=non_exact,
            errors=("malformed backend result", str(exc)),
            warnings=tuple(exact_report.errors),
            metadata={"dry_run": invocation.dry_run},
        )
    completed = result.status == InferenceStatus.SUCCESS.value and not non_exact and not invocation.dry_run
    status = InferenceStatus.SUCCESS.value if completed else result.status
    errors: tuple[str, ...] = ()
    if result.status != InferenceStatus.SUCCESS.value:
        errors = (result.error or "backend result failed",)
    if non_exact and result.status == InferenceStatus.SUCCESS.value:
        status = InferenceStatus.REJECTED.value
        errors = ("non_submission_exact",)
    return InferenceRunReport(
        invocation_hash=invocation.invocation_hash,
        prompt_batch_hash=batch.batch_hash,
        serving_config_hash=serving_hash,
        called_backend=True,
        backend_result=result,
        status=status,
        completed=completed,
        non_submission_exact=non_exact,
        errors=errors,
        warnings=tuple(exact_report.errors),
        metadata={"dry_run": invocation.dry_run},
    )


def validate_inference_run_report(report: InferenceRunReport | Mapping[str, Any]) -> InferenceRunReport:
    normalized = report if isinstance(report, InferenceRunReport) else InferenceRunReport(**dict(report))
    if normalized.report_hash != _payload_hash(normalized, "report_hash"):
        raise InferenceRunnerError("report_hash does not match inference run report payload.")
    return normalized


def _call_backend(backend: Callable[[InferenceBackendInvocation, PromptBatch], Any] | object, invocation: InferenceBackendInvocation, batch: PromptBatch) -> Any:
    if callable(backend):
        return backend(invocation, batch)
    run = getattr(backend, "run", None)
    if callable(run):
        return run(invocation, batch)
    raise InferenceRunnerError("backend must be a callable or expose run(invocation, prompt_batch).")


def _mapping_from_raw(raw: Any) -> Mapping[str, Any] | InferenceBackendResult:
    if isinstance(raw, InferenceBackendResult):
        return raw
    if isinstance(raw, Mapping):
        return raw
    if is_dataclass(raw):
        return asdict(raw)
    raise InferenceRunnerError("backend result must be a mapping, dataclass, or InferenceBackendResult.")


def _safe_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InferenceRunnerError("metadata must be a mapping.")
    metadata = dict(value)
    try:
        _reject_forbidden_metadata(metadata)
    except Exception as exc:
        raise InferenceRunnerError(str(exc)) from exc
    return metadata


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InferenceRunnerError(f"{field_name} must be non-empty.")
    return value


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


__all__ = [
    "InferenceRunReport",
    "InferenceRunnerConfig",
    "InferenceRunnerError",
    "invoke_inference_backend",
    "validate_inference_run_report",
]
