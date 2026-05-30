"""Tiny smoke-training execution boundary for caller-supplied backends."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Mapping

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.training_backend import (
    AdapterValidationConfig,
    BackendInvocation,
    BackendResult,
    BackendRunReport,
    BackendRunnerConfig,
    TrainingLogValidationConfig,
    TrainingLogValidationReport,
    TrainingRunManifest,
    build_run_manifest_from_backend_outputs,
    invoke_training_backend,
    validate_backend_invocation,
    validate_training_log,
    validate_training_run_manifest,
)

from .smoke_backend import SmokeBackendResult, normalize_smoke_backend_result
from .smoke_config import SmokeTrainingConfig, SmokeTrainingConfigError, _reject_unsafe_metadata, validate_smoke_training_config
from .smoke_dataset import SmokeDataset, SmokeDatasetError, validate_smoke_dataset


class SmokeRunnerError(ValueError):
    """Raised when smoke training cannot be run safely."""


@dataclass(frozen=True)
class SmokeRunConfig:
    smoke_config: SmokeTrainingConfig
    smoke_dataset: SmokeDataset
    invocation: BackendInvocation
    adapter_validation_config: AdapterValidationConfig | None = None
    training_log_config: TrainingLogValidationConfig | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        cfg = _config(self.smoke_config)
        dataset = _dataset(self.smoke_dataset, cfg)
        try:
            invocation = validate_backend_invocation(self.invocation)
        except Exception as exc:
            raise SmokeRunnerError("invocation must be a valid Pass 11 BackendInvocation.") from exc
        if invocation.stage != cfg.stage:
            raise SmokeRunnerError("invocation stage must match smoke config stage.")
        if invocation.dry_run != cfg.dry_run:
            raise SmokeRunnerError("invocation dry_run must match smoke config dry_run.")
        if not cfg.dry_run and not invocation.output_dir:
            raise SmokeRunnerError("non-dry smoke invocation requires output_dir.")
        if cfg.stage in {"sft", "final_sft_refresh"} and invocation.input_row_count > cfg.max_examples:
            raise SmokeRunnerError("invocation input_row_count exceeds max_examples.")
        if cfg.stage == "dpo" and invocation.input_pair_count > cfg.max_examples:
            raise SmokeRunnerError("invocation input_pair_count exceeds max_examples.")
        metadata = dict(self.metadata)
        _reject_metadata(metadata)
        object.__setattr__(self, "smoke_config", cfg)
        object.__setattr__(self, "smoke_dataset", dataset)
        object.__setattr__(self, "invocation", invocation)
        object.__setattr__(self, "metadata", metadata)
        expected = _payload_hash(self, "config_hash")
        if not self.config_hash:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise SmokeRunnerError("config_hash does not match smoke run config payload.")


@dataclass(frozen=True)
class SmokeRunReport:
    smoke_config_hash: str
    smoke_dataset_hash: str
    backend_invocation_hash: str
    backend_run_report_hash: str | None
    backend_result_hash: str | None
    adapter_validation_hash: str | None
    training_log_hash: str | None
    training_run_manifest_hash: str | None
    called_backend: bool
    trained: bool
    passed: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    report_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("smoke_config_hash", "smoke_dataset_hash", "backend_invocation_hash"):
            _require_non_empty(getattr(self, name), name)
        for name in ("backend_run_report_hash", "backend_result_hash", "adapter_validation_hash", "training_log_hash", "training_run_manifest_hash"):
            value = getattr(self, name)
            if value is not None:
                _require_non_empty(value, name)
        if not isinstance(self.called_backend, bool) or not isinstance(self.trained, bool) or not isinstance(self.passed, bool):
            raise SmokeRunnerError("called_backend, trained, and passed must be booleans.")
        errors = tuple(str(item) for item in self.errors)
        warnings = tuple(str(item) for item in self.warnings)
        metadata = dict(self.metadata)
        _reject_metadata(metadata)
        require_log_validation = metadata.get("require_log_validation") is not False
        dry_run = metadata.get("dry_run") is True
        if self.passed and errors:
            raise SmokeRunnerError("passed=True cannot include errors.")
        if self.passed and not self.backend_run_report_hash:
            raise SmokeRunnerError("passed=True requires backend_run_report_hash.")
        if self.passed and not self.backend_result_hash and not (dry_run and self.trained is False):
            raise SmokeRunnerError("passed=True requires backend_result_hash unless this is an explicit dry-run pass.")
        if self.passed and self.trained is False:
            if dry_run is not True:
                raise SmokeRunnerError("passed=True with trained=False requires metadata dry_run=True.")
            if self.adapter_validation_hash or self.training_run_manifest_hash:
                raise SmokeRunnerError("dry-run pass cannot contain adapter or training manifest evidence.")
        if dry_run and self.trained:
            raise SmokeRunnerError("dry-run smoke report cannot claim trained=True.")
        if self.trained:
            if not self.backend_run_report_hash:
                raise SmokeRunnerError("trained=True requires backend_run_report_hash.")
            if not self.backend_result_hash:
                raise SmokeRunnerError("trained=True requires backend_result_hash.")
            if not self.adapter_validation_hash:
                raise SmokeRunnerError("trained=True requires adapter_validation_hash.")
            if not self.training_run_manifest_hash:
                raise SmokeRunnerError("trained=True requires training_run_manifest_hash.")
            if require_log_validation and not self.training_log_hash:
                raise SmokeRunnerError("require_log_validation=True requires training_log_hash.")
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "metadata", metadata)
        expected = _payload_hash(self, "report_hash")
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected)
        elif self.report_hash != expected:
            raise SmokeRunnerError("report_hash does not match smoke run report payload.")


def run_smoke_training(
    run_config: SmokeRunConfig,
    backend: Callable[[BackendInvocation], Any] | object | None = None,
) -> SmokeRunReport:
    cfg = run_config if isinstance(run_config, SmokeRunConfig) else SmokeRunConfig(**dict(run_config))  # type: ignore[arg-type]
    smoke_cfg = cfg.smoke_config
    invocation = cfg.invocation
    if smoke_cfg.dry_run is False and backend is None:
        raise SmokeRunnerError("dry_run=False requires an explicit backend.")
    if smoke_cfg.dry_run and not smoke_cfg.allow_backend_on_dry_run:
        backend_report = invoke_training_backend(
            invocation,
            None,
            config=BackendRunnerConfig(
                require_adapter_validation=smoke_cfg.require_adapter_validation,
                require_log_validation=False,
                allow_backend_on_dry_run=False,
            ),
        )
        return _build_report(cfg, backend_report, None, None, None, errors=())

    captured: dict[str, SmokeBackendResult] = {}

    def wrapped_backend(inv: BackendInvocation) -> BackendResult:
        raw = _call_backend(backend, inv)
        smoke_result = normalize_smoke_backend_result(raw, invocation=inv, require_log_validation=smoke_cfg.require_log_validation)
        if smoke_cfg.require_adapter_validation:
            _require_adapter_config_compatible(smoke_result, cfg.adapter_validation_config)
        captured["smoke_result"] = smoke_result
        return smoke_result.backend_result

    backend_report = invoke_training_backend(
        invocation,
        wrapped_backend,
        config=BackendRunnerConfig(
            require_adapter_validation=smoke_cfg.require_adapter_validation,
            require_log_validation=False,
            allow_backend_on_dry_run=smoke_cfg.allow_backend_on_dry_run,
            allow_checkpoint_path=False,
        ),
        adapter_validation_config=cfg.adapter_validation_config,
    )
    smoke_result = captured.get("smoke_result")
    log_report = None
    errors = list(backend_report.errors)
    manifest = None
    if smoke_result is not None and smoke_cfg.require_log_validation:
        try:
            log_report = validate_training_log(smoke_result.training_log, cfg.training_log_config)
            if log_report.passed is not True:
                errors.extend(log_report.errors or ("training log validation failed",))
        except Exception as exc:
            errors.append(f"training log validation failed: {exc}")
    if backend_report.trained and not errors:
        try:
            if backend_report.backend_result is None:
                raise SmokeRunnerError("missing backend result")
            if smoke_cfg.require_adapter_validation and backend_report.adapter_validation_report is None:
                raise SmokeRunnerError("missing adapter validation report")
            manifest = build_run_manifest_from_backend_outputs(
                invocation=invocation,
                backend_result=backend_report.backend_result,
                adapter_report=backend_report.adapter_validation_report,
                log_report=log_report,
                metadata={"pass": 12, "smoke_training": True},
            )
            validate_training_run_manifest(
                manifest,
                backend_result=backend_report.backend_result,
                adapter_validation_report=backend_report.adapter_validation_report,
            )
        except Exception as exc:
            errors.append(f"training run manifest validation failed: {exc}")
    return _build_report(cfg, backend_report, smoke_result, log_report, manifest, errors=tuple(errors))


def validate_smoke_run_report(report: SmokeRunReport | Mapping[str, Any]) -> SmokeRunReport:
    normalized = report if isinstance(report, SmokeRunReport) else SmokeRunReport(**dict(report))
    if normalized.report_hash != _payload_hash(normalized, "report_hash"):
        raise SmokeRunnerError("report_hash does not match smoke run report payload.")
    return normalized


def _build_report(
    cfg: SmokeRunConfig,
    backend_report: BackendRunReport,
    smoke_result: SmokeBackendResult | None,
    log_report: TrainingLogValidationReport | None,
    manifest: TrainingRunManifest | None,
    *,
    errors: tuple[str, ...],
) -> SmokeRunReport:
    backend_result = backend_report.backend_result
    adapter_report = backend_report.adapter_validation_report
    all_errors = tuple(sorted(set(str(item) for item in errors if str(item))))
    trained = bool(
        backend_report.trained
        and not all_errors
        and backend_result is not None
        and (not cfg.smoke_config.require_adapter_validation or (adapter_report is not None and adapter_report.passed))
        and (not cfg.smoke_config.require_log_validation or (log_report is not None and log_report.passed))
        and manifest is not None
        and manifest.trained
    )
    passed = trained
    if cfg.smoke_config.dry_run and not cfg.smoke_config.allow_backend_on_dry_run:
        trained = False
        passed = not all_errors and backend_report.trained is False and backend_report.called_backend is False
    return SmokeRunReport(
        smoke_config_hash=cfg.smoke_config.config_hash,
        smoke_dataset_hash=cfg.smoke_dataset.dataset_hash,
        backend_invocation_hash=cfg.invocation.invocation_hash,
        backend_run_report_hash=backend_report.report_hash,
        backend_result_hash=backend_result.result_hash if backend_result is not None else None,
        adapter_validation_hash=adapter_report.report_hash if adapter_report is not None else None,
        training_log_hash=log_report.report_hash if log_report is not None else None,
        training_run_manifest_hash=manifest.manifest_hash if manifest is not None else None,
        called_backend=backend_report.called_backend,
        trained=trained,
        passed=passed,
        errors=all_errors,
        warnings=backend_report.warnings,
        metadata={
            **dict(cfg.metadata),
            "dry_run": cfg.smoke_config.dry_run,
            "require_log_validation": cfg.smoke_config.require_log_validation,
            "smoke_backend_result_hash": smoke_result.result_hash if smoke_result is not None else None,
        },
    )


def _call_backend(backend: Callable[[BackendInvocation], Any] | object | None, invocation: BackendInvocation) -> Any:
    if backend is None:
        raise SmokeRunnerError("backend is required.")
    if callable(backend):
        return backend(invocation)
    run = getattr(backend, "run", None)
    if callable(run):
        return run(invocation)
    raise SmokeRunnerError("backend must be a callable or expose run(invocation).")


def _require_adapter_config_compatible(result: SmokeBackendResult, config: AdapterValidationConfig | None) -> None:
    if result.backend_result.trained and config is None:
        raise SmokeRunnerError("trained smoke result requires adapter_validation_config.")
    if result.adapter_dir is None or config is None:
        return
    left = Path(str(result.adapter_dir))
    right = Path(str(config.adapter_dir))
    try:
        if left.resolve() == right.resolve():
            return
    except OSError:
        pass
    if left.as_posix().rstrip("/") == right.as_posix().rstrip("/"):
        return
    raise SmokeRunnerError("adapter_dir must match adapter validation config.")


def _config(value: SmokeTrainingConfig) -> SmokeTrainingConfig:
    try:
        return validate_smoke_training_config(value)
    except SmokeTrainingConfigError as exc:
        raise SmokeRunnerError(str(exc)) from exc


def _dataset(value: SmokeDataset, config: SmokeTrainingConfig) -> SmokeDataset:
    try:
        return validate_smoke_dataset(value, config)
    except SmokeDatasetError as exc:
        raise SmokeRunnerError(str(exc)) from exc


def _reject_metadata(metadata: Mapping[str, Any]) -> None:
    try:
        _reject_unsafe_metadata(metadata, reject_split_claims=True)
    except SmokeTrainingConfigError as exc:
        raise SmokeRunnerError(str(exc)) from exc


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SmokeRunnerError(f"{field_name} must be non-empty.")
    return value


__all__ = [
    "SmokeRunConfig",
    "SmokeRunReport",
    "SmokeRunnerError",
    "run_smoke_training",
    "validate_smoke_run_report",
]
