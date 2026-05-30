"""Offline runtime-readiness validator for Pass 9."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import json
from pathlib import Path
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.runtime.serving_config import ServingConfig, ServingConfigError, validate_serving_config
from nemotron_engine.training.lora_config import LoRAConfigError, validate_adapter_config_json

from .submission_manifest import SubmissionManifest, validate_submission_manifest


class RuntimeValidationError(ValueError):
    """Raised when runtime validation reports are unsafe or inconsistent."""


@dataclass(frozen=True)
class RuntimeValidationConfig:
    serving_config: ServingConfig
    manifest: SubmissionManifest | None = None
    package_root: str | Path | None = None
    python_entrypoint: str | Path | None = None
    adapter_dir: str | Path | None = None
    adapter_config: Mapping[str, Any] | None = None
    required_environment: tuple[str, ...] = ()
    environment_metadata: Mapping[str, Any] = field(default_factory=dict)
    backend_result: Mapping[str, Any] | None = None
    dry_run: bool = True
    dry_run_allow_unloaded: bool = False
    adapter_expected: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.serving_config, ServingConfig):
            raise RuntimeValidationError("serving_config must be a ServingConfig.")
        if self.manifest is not None and not isinstance(self.manifest, SubmissionManifest):
            raise RuntimeValidationError("manifest must be a SubmissionManifest when supplied.")
        for name in ("package_root", "python_entrypoint", "adapter_dir"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))
        if self.adapter_config is not None and not isinstance(self.adapter_config, Mapping):
            raise RuntimeValidationError("adapter_config must be a mapping when supplied.")
        if self.backend_result is not None and not isinstance(self.backend_result, Mapping):
            raise RuntimeValidationError("backend_result must be a mapping when supplied.")
        if not all(isinstance(value, bool) for value in (self.dry_run, self.dry_run_allow_unloaded, self.adapter_expected)):
            raise RuntimeValidationError("runtime dry-run flags must be booleans.")
        object.__setattr__(self, "required_environment", tuple(str(item) for item in self.required_environment))
        object.__setattr__(self, "environment_metadata", dict(self.environment_metadata))
        object.__setattr__(self, "metadata", dict(self.metadata))
        expected = _payload_hash(self, "config_hash")
        if not self.config_hash:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise RuntimeValidationError("config_hash does not match runtime validation config payload.")


@dataclass(frozen=True)
class RuntimeValidationReport:
    passed: bool
    dry_run: bool
    runtime_loaded: bool
    model_loaded: bool
    adapter_loaded: bool
    backend_validated: bool
    backend_name: str | None
    manifest_hash: str | None
    serving_config_hash: str
    adapter_checked: bool
    adapter_rank: int | None = None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    report_hash: str = ""

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, bool)
            for value in (
                self.passed,
                self.dry_run,
                self.runtime_loaded,
                self.model_loaded,
                self.adapter_loaded,
                self.backend_validated,
                self.adapter_checked,
            )
        ):
            raise RuntimeValidationError("runtime report boolean fields must be bool values.")
        if not isinstance(self.serving_config_hash, str) or not self.serving_config_hash.strip():
            raise RuntimeValidationError("serving_config_hash must be non-empty.")
        if self.manifest_hash is not None and (not isinstance(self.manifest_hash, str) or not self.manifest_hash.strip()):
            raise RuntimeValidationError("manifest_hash must be non-empty when supplied.")
        if self.backend_name is not None and (not isinstance(self.backend_name, str) or not self.backend_name.strip()):
            raise RuntimeValidationError("backend_name must be non-empty when supplied.")
        if (self.runtime_loaded or self.model_loaded or self.adapter_loaded) and not self.backend_validated:
            raise RuntimeValidationError("loaded runtime flags require validated backend proof.")
        if self.dry_run and (self.runtime_loaded or self.model_loaded or self.adapter_loaded):
            raise RuntimeValidationError("dry-run runtime report cannot claim loaded components.")
        if self.adapter_rank is not None and (not isinstance(self.adapter_rank, int) or isinstance(self.adapter_rank, bool) or self.adapter_rank < 1 or self.adapter_rank > 32):
            raise RuntimeValidationError("adapter_rank must be in [1, 32].")
        errors = tuple(str(item) for item in self.errors)
        warnings = tuple(str(item) for item in self.warnings)
        if self.passed and errors:
            raise RuntimeValidationError("passed=True cannot include runtime errors.")
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", warnings)
        expected = _payload_hash(self, "report_hash")
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected)
        elif self.report_hash != expected:
            raise RuntimeValidationError("report_hash does not match runtime validation report payload.")


def validate_python_entrypoint(path: str | Path | None) -> tuple[str, ...]:
    if path is None:
        return ()
    target = Path(path)
    if not target.exists():
        return (f"python entrypoint missing: {target}",)
    if not target.is_file():
        return (f"python entrypoint is not a file: {target}",)
    return ()


def validate_serving_config_runtime(config: ServingConfig, *, adapter_config: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        validate_serving_config(config)
    except ServingConfigError as exc:
        errors.append(str(exc))
    if config.temperature != 0.0:
        errors.append("temperature must be 0.0.")
    if config.top_p != 1.0:
        errors.append("top_p must be 1.0.")
    if config.num_samples != 1:
        errors.append("num_samples must be 1.")
    if config.majority_vote:
        errors.append("majority_vote must be False.")
    if config.max_tokens <= 0:
        errors.append("max_tokens must be > 0.")
    for name in ("prompt_template_hash", "tokenizer_hash", "model_hash"):
        if not isinstance(getattr(config, name), str) or not getattr(config, name).strip():
            errors.append(f"{name} must be non-empty.")
    if adapter_config is not None and (not isinstance(config.adapter_hash, str) or not config.adapter_hash.strip()):
        errors.append("adapter_hash must be non-empty when adapter_config is supplied.")
    return tuple(errors)


def validate_adapter_runtime_refs(
    *,
    adapter_dir: str | Path | None = None,
    adapter_config: Mapping[str, Any] | None = None,
) -> tuple[tuple[str, ...], bool, int | None]:
    errors: list[str] = []
    checked = False
    rank = None
    payload = adapter_config
    if adapter_dir is not None:
        config_path = Path(adapter_dir) / "adapter_config.json"
        if not config_path.exists():
            errors.append(f"adapter_config.json missing in adapter_dir: {adapter_dir}")
        else:
            try:
                payload = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"invalid adapter_config.json: {exc}")
    if payload is not None:
        checked = True
        try:
            validate_adapter_config_json(payload)
            rank = _adapter_rank(payload)
        except LoRAConfigError as exc:
            errors.append(str(exc))
    return tuple(errors), checked, rank


def validate_offline_runtime(config: RuntimeValidationConfig) -> RuntimeValidationReport:
    if not isinstance(config, RuntimeValidationConfig):
        raise RuntimeValidationError("config must be a RuntimeValidationConfig.")
    errors: list[str] = []
    warnings: list[str] = []
    manifest_hash = None
    if config.manifest is not None:
        try:
            validate_submission_manifest(config.manifest, config.package_root)
            manifest_hash = config.manifest.manifest_hash
        except Exception as exc:
            errors.append(f"manifest validation failed: {exc}")
    errors.extend(validate_python_entrypoint(config.python_entrypoint))
    adapter_errors, adapter_checked, adapter_rank = validate_adapter_runtime_refs(
        adapter_dir=config.adapter_dir,
        adapter_config=config.adapter_config,
    )
    errors.extend(adapter_errors)
    errors.extend(validate_serving_config_runtime(config.serving_config, adapter_config=config.adapter_config if adapter_checked else None))
    for name in config.required_environment:
        if name not in config.environment_metadata or config.environment_metadata.get(name) in (None, ""):
            errors.append(f"missing environment metadata: {name}")
    backend_validated = False
    runtime_loaded = False
    model_loaded = False
    adapter_loaded = False
    backend_name = None
    if config.backend_result is None:
        warnings.append("runtime_unloaded_dry_run")
        if not config.dry_run_allow_unloaded:
            errors.append("runtime backend absent; loaded state unverified")
    else:
        _reject_forbidden_success_claims(config.backend_result)
        _reject_forbidden_success_claims(config.metadata)
        backend_validated = True
        backend_name, runtime_loaded, model_loaded, adapter_loaded = _validate_backend_result(config.backend_result, config.adapter_expected or adapter_checked)
        backend_errors = config.backend_result.get("errors", ())
        if isinstance(backend_errors, str):
            backend_errors = (backend_errors,)
        errors.extend(str(item) for item in backend_errors)
    passed = not errors and (backend_validated or config.dry_run_allow_unloaded)
    return RuntimeValidationReport(
        passed=passed,
        dry_run=config.dry_run,
        runtime_loaded=runtime_loaded,
        model_loaded=model_loaded,
        adapter_loaded=adapter_loaded,
        backend_validated=backend_validated,
        backend_name=backend_name,
        manifest_hash=manifest_hash,
        serving_config_hash=stable_hash(config.serving_config.to_dict()),
        adapter_checked=adapter_checked,
        adapter_rank=adapter_rank,
        errors=tuple(sorted(set(errors))),
        warnings=tuple(sorted(set(warnings))),
    )


def _validate_backend_result(payload: Mapping[str, Any], adapter_expected: bool) -> tuple[str, bool, bool, bool]:
    name = payload.get("backend_name", "backend")
    if not isinstance(name, str) or not name.strip():
        raise RuntimeValidationError("backend_result.backend_name must be a non-empty string.")
    if "kaggle" in name.lower():
        raise RuntimeValidationError("backend_result.backend_name must not claim Kaggle runtime.")
    for key in ("runtime_loaded", "model_loaded", "adapter_loaded"):
        if key not in payload or not isinstance(payload[key], bool):
            raise RuntimeValidationError(f"backend_result.{key} must be a boolean.")
    runtime_loaded = bool(payload["runtime_loaded"])
    model_loaded = bool(payload["model_loaded"])
    adapter_loaded = bool(payload["adapter_loaded"])
    if not runtime_loaded or not model_loaded:
        raise RuntimeValidationError("backend result did not prove runtime/model load.")
    if adapter_expected and not adapter_loaded:
        raise RuntimeValidationError("backend result did not prove adapter load.")
    return name, runtime_loaded, model_loaded, adapter_loaded


def _reject_forbidden_success_claims(payload: Mapping[str, Any]) -> None:
    forbidden_keys = {
        "kaggle_runtime_success",
        "kaggle_success",
        "leaderboard_success",
        "submission_success",
    }
    for key, value in payload.items():
        key_text = str(key).lower()
        if key_text in forbidden_keys and value is True:
            raise RuntimeValidationError(f"forbidden runtime success claim: {key}")
        if ("kaggle" in key_text or "leaderboard" in key_text or "submission" in key_text) and "success" in key_text and value is True:
            raise RuntimeValidationError(f"forbidden runtime success claim: {key}")


def _adapter_rank(config: Mapping[str, Any]) -> int | None:
    ranks: list[int] = []
    for key in ("r", "rank", "lora_rank"):
        if key in config:
            ranks.append(int(config[key]))
    peft = config.get("peft_config")
    if isinstance(peft, Mapping) and "r" in peft:
        ranks.append(int(peft["r"]))
    return max(ranks) if ranks else None


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


__all__ = [
    "RuntimeValidationConfig",
    "RuntimeValidationError",
    "RuntimeValidationReport",
    "validate_adapter_runtime_refs",
    "validate_offline_runtime",
    "validate_python_entrypoint",
    "validate_serving_config_runtime",
]
