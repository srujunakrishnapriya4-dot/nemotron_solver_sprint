"""Safe invocation wrapper for explicit external training backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from nemotron_engine.core.schemas import stable_hash

from .adapter_validator import AdapterValidationConfig, AdapterValidationReport, validate_adapter_output
from .backend_contracts import (
    BackendArtifactRef,
    BackendContractError,
    BackendInvocation,
    BackendResult,
    BackendStatus,
    _reject_forbidden_claims,
    validate_backend_invocation,
    validate_backend_result,
)
from .training_log_validator import TrainingLogValidationConfig, TrainingLogValidationReport, validate_training_log


class BackendRunnerError(ValueError):
    """Raised when backend invocation cannot be completed safely."""


@dataclass(frozen=True)
class BackendRunnerConfig:
    require_adapter_validation: bool = True
    require_log_validation: bool = False
    allow_backend_on_dry_run: bool = False
    allow_checkpoint_path: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("require_adapter_validation", "require_log_validation", "allow_backend_on_dry_run", "allow_checkpoint_path"):
            if not isinstance(getattr(self, name), bool):
                raise BackendRunnerError(f"{name} must be boolean.")
        metadata = dict(self.metadata)
        _reject_forbidden_claims(metadata)
        object.__setattr__(self, "metadata", metadata)
        expected = _payload_hash(self, "config_hash")
        if not self.config_hash:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise BackendRunnerError("config_hash does not match backend runner config payload.")


@dataclass(frozen=True)
class BackendRunReport:
    invocation_hash: str
    called_backend: bool
    backend_result: BackendResult | None
    adapter_validation_report: AdapterValidationReport | None
    training_log_report: TrainingLogValidationReport | None
    trained: bool
    status: str
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    report_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.invocation_hash, str) or not self.invocation_hash.strip():
            raise BackendRunnerError("invocation_hash must be non-empty.")
        if not isinstance(self.called_backend, bool) or not isinstance(self.trained, bool):
            raise BackendRunnerError("called_backend and trained must be booleans.")
        if self.status not in {item.value for item in BackendStatus}:
            raise BackendRunnerError("invalid backend run status.")
        errors = tuple(str(item) for item in self.errors)
        warnings = tuple(str(item) for item in self.warnings)
        if self.trained:
            if self.backend_result is None or self.backend_result.trained is not True:
                raise BackendRunnerError("trained=True requires trained backend_result.")
            if self.adapter_validation_report is None or self.adapter_validation_report.passed is not True:
                raise BackendRunnerError("trained=True requires passed adapter validation report.")
            if errors:
                raise BackendRunnerError("trained=True cannot include errors.")
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", warnings)
        expected = _payload_hash(self, "report_hash")
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected)
        elif self.report_hash != expected:
            raise BackendRunnerError("report_hash does not match backend run report payload.")


def invoke_training_backend(
    invocation: BackendInvocation,
    backend: Callable[[BackendInvocation], Any] | object | None = None,
    *,
    config: BackendRunnerConfig | None = None,
    adapter_validation_config: AdapterValidationConfig | None = None,
    training_log: Any | None = None,
    training_log_config: TrainingLogValidationConfig | None = None,
) -> BackendRunReport:
    cfg = config or BackendRunnerConfig()
    invocation = validate_backend_invocation(invocation)
    if invocation.dry_run and not cfg.allow_backend_on_dry_run:
        result = BackendResult(
            result_id="dry-run-" + stable_hash({"invocation_hash": invocation.invocation_hash})[:16],
            invocation_hash=invocation.invocation_hash,
            backend_kind=invocation.backend_kind,
            stage=invocation.stage,
            status=BackendStatus.DRY_RUN.value,
            trained=False,
            adapter_path=None,
            checkpoint_path=None,
            artifacts=(),
            metrics=(),
            started=False,
            finished=False,
            error=None,
        )
        return BackendRunReport(invocation.invocation_hash, False, result, None, None, False, BackendStatus.DRY_RUN.value)
    if backend is None:
        raise BackendRunnerError("dry_run=False requires an explicit backend.")
    try:
        raw = _call_backend(backend, invocation)
    except Exception as exc:
        result = BackendResult(
            result_id="failed-" + stable_hash({"invocation_hash": invocation.invocation_hash, "error": str(exc)})[:16],
            invocation_hash=invocation.invocation_hash,
            backend_kind=invocation.backend_kind,
            stage=invocation.stage,
            status=BackendStatus.FAILED.value,
            trained=False,
            adapter_path=None,
            checkpoint_path=None,
            artifacts=(),
            metrics=(),
            started=True,
            finished=True,
            error=str(exc) or type(exc).__name__,
        )
        return BackendRunReport(invocation.invocation_hash, True, result, None, None, False, BackendStatus.FAILED.value, errors=("backend exception",))
    result = validate_backend_callable_result(raw, invocation=invocation, config=cfg)
    errors: list[str] = []
    adapter_report = None
    log_report = None
    if result.trained and cfg.require_adapter_validation:
        adapter_cfg = adapter_validation_config or AdapterValidationConfig(result.adapter_path or "")
        adapter_report = validate_adapter_output(adapter_cfg)
        if adapter_report.passed is not True:
            errors.extend(adapter_report.errors or ("adapter validation failed",))
    if training_log is not None:
        log_report = validate_training_log(training_log, training_log_config)
        if log_report.passed is not True:
            errors.extend(log_report.errors or ("training log validation failed",))
    elif result.trained and cfg.require_log_validation:
        errors.append("training log validation required")
    trained = result.trained and not errors and (not cfg.require_adapter_validation or (adapter_report is not None and adapter_report.passed))
    if result.trained and not trained:
        errors.append("validated trained result requirements not satisfied")
    return BackendRunReport(
        invocation_hash=invocation.invocation_hash,
        called_backend=True,
        backend_result=result,
        adapter_validation_report=adapter_report,
        training_log_report=log_report,
        trained=trained,
        status=result.status if not errors else BackendStatus.REJECTED.value,
        errors=tuple(sorted(set(errors))),
        warnings=(),
    )


def validate_backend_callable_result(raw: Any, *, invocation: BackendInvocation, config: BackendRunnerConfig | None = None) -> BackendResult:
    cfg = config or BackendRunnerConfig()
    payload = _mapping_from_raw(raw)
    result = validate_backend_result(
        payload,
        invocation=invocation,
        allow_checkpoint_path=cfg.allow_checkpoint_path,
        require_artifact_hashes=cfg.require_adapter_validation,
    )
    if result.trained and cfg.require_adapter_validation:
        required = tuple(item for item in result.artifacts if item.required and "adapter" in item.role.lower())
        if not required:
            raise BackendRunnerError("trained backend result requires required adapter artifact.")
        for artifact in required:
            if artifact.sha256 is None or artifact.size_bytes is None:
                raise BackendRunnerError("required adapter artifacts require sha256 and size_bytes.")
    return result


def _call_backend(backend: Callable[[BackendInvocation], Any] | object, invocation: BackendInvocation) -> Any:
    if callable(backend):
        return backend(invocation)
    run = getattr(backend, "run", None)
    if callable(run):
        return run(invocation)
    raise BackendRunnerError("backend must be a callable or expose run(invocation).")


def _mapping_from_raw(raw: Any) -> Mapping[str, Any] | BackendResult:
    if isinstance(raw, BackendResult):
        return raw
    if isinstance(raw, Mapping):
        return raw
    if is_dataclass(raw):
        return asdict(raw)
    raise BackendRunnerError("backend result must be a mapping, dataclass, or BackendResult.")


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


__all__ = [
    "BackendRunReport",
    "BackendRunnerConfig",
    "BackendRunnerError",
    "invoke_training_backend",
    "validate_backend_callable_result",
]
