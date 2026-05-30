"""Submission-exact decoding checks for Pass 8 promotion."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
from pathlib import Path
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.runtime.serving_config import ServingConfig, ServingConfigError, validate_serving_config
from nemotron_engine.submission.submission_validator import (
    MAX_LORA_RANK,
    SubmissionValidationResult,
    validate_adapter_dir,
    validate_submission_zip,
)
from nemotron_engine.training.lora_config import LoRAConfigError, validate_adapter_config_json


SUBMISSION_CRITICAL_FIELDS = (
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "num_samples",
    "majority_vote",
    "batch_size",
    "prompt_template_hash",
    "tokenizer_hash",
    "model_hash",
    "adapter_hash",
)


class SubmissionExactError(ValueError):
    """Raised when submission-exact validation cannot be completed safely."""


@dataclass(frozen=True)
class SubmissionExactConfig:
    serving_config: ServingConfig
    strict: bool = True
    dry_run: bool = True
    required_hash_fields: tuple[str, ...] = ("prompt_template_hash", "tokenizer_hash", "model_hash")
    adapter_config: Mapping[str, Any] | None = None
    adapter_dir: str | None = None
    submission_zip: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.serving_config, ServingConfig):
            raise SubmissionExactError("serving_config must be a ServingConfig.")
        if not isinstance(self.strict, bool) or not isinstance(self.dry_run, bool):
            raise SubmissionExactError("strict and dry_run must be booleans.")
        if not isinstance(self.required_hash_fields, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.required_hash_fields
        ):
            raise SubmissionExactError("required_hash_fields must be a tuple of non-empty strings.")
        if self.adapter_config is not None and not isinstance(self.adapter_config, Mapping):
            raise SubmissionExactError("adapter_config must be a mapping when provided.")
        object.__setattr__(self, "metadata", dict(self.metadata))
        expected = _payload_hash(self, "config_hash")
        if not self.config_hash:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise SubmissionExactError("config_hash does not match submission exact config payload.")


@dataclass(frozen=True)
class SubmissionExactReport:
    config_hash: str
    serving_config_hash: str
    passed: bool
    dry_run: bool
    runtime_success: bool
    required_hashes_present: bool
    adapter_checked: bool
    adapter_rank: int | None = None
    critical_serving_fields: Mapping[str, Any] = field(default_factory=dict)
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    report_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("config_hash", "serving_config_hash"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise SubmissionExactError(f"{name} must be non-empty.")
        if not all(isinstance(value, bool) for value in (self.passed, self.dry_run, self.runtime_success, self.required_hashes_present, self.adapter_checked)):
            raise SubmissionExactError("report booleans must be bool values.")
        if self.dry_run and self.runtime_success:
            raise SubmissionExactError("dry-run report cannot claim runtime success.")
        if self.adapter_rank is not None:
            _validate_rank(self.adapter_rank)
        critical = _normalize_critical_fields(self.critical_serving_fields)
        if self.passed:
            _validate_submission_exact_critical_fields(
                critical,
                adapter_checked=self.adapter_checked,
                adapter_rank=self.adapter_rank,
            )
        errors = tuple(str(item) for item in self.errors)
        warnings = tuple(str(item) for item in self.warnings)
        if self.passed and errors:
            raise SubmissionExactError("passed=True cannot include errors.")
        object.__setattr__(self, "critical_serving_fields", critical)
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", warnings)
        expected = _payload_hash(self, "report_hash")
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected)
        elif self.report_hash != expected:
            raise SubmissionExactError("report_hash does not match submission exact report payload.")


def validate_submission_exact_config(config: SubmissionExactConfig) -> SubmissionExactReport:
    """Validate deterministic decoding settings and artifact references."""

    if not isinstance(config, SubmissionExactConfig):
        raise SubmissionExactError("config must be a SubmissionExactConfig.")
    errors: list[str] = []
    warnings: list[str] = []
    serving = config.serving_config
    try:
        validate_serving_config(serving)
    except ServingConfigError as exc:
        errors.append(str(exc))
    if serving.temperature != 0.0:
        errors.append("temperature must be 0.0.")
    if serving.top_p != 1.0:
        errors.append("top_p must be 1.0.")
    if serving.num_samples != 1:
        errors.append("num_samples must be 1.")
    if serving.majority_vote:
        errors.append("majority_vote must be False.")
    if serving.max_tokens <= 0:
        errors.append("max_tokens must be > 0.")
    if config.strict:
        missing = tuple(field for field in config.required_hash_fields if not _non_empty(getattr(serving, field, None)))
        if missing:
            errors.append(f"missing required hashes: {missing}")
    else:
        missing = ()
    artifact_result = validate_submission_artifact_refs(
        adapter_config=config.adapter_config,
        adapter_dir=config.adapter_dir,
        submission_zip=config.submission_zip,
    )
    adapter_checked = bool(artifact_result.adapter_config)
    adapter_rank = _adapter_rank(artifact_result.adapter_config) if artifact_result.valid and artifact_result.adapter_config else None
    if artifact_result.errors:
        errors.extend(artifact_result.errors)
    if adapter_checked and not _non_empty(serving.adapter_hash):
        errors.append("adapter_hash is required when adapter config is supplied.")
    warnings.extend(artifact_result.warnings)
    passed = not errors
    return SubmissionExactReport(
        config_hash=config.config_hash,
        serving_config_hash=stable_hash(serving.to_dict()),
        passed=passed,
        dry_run=config.dry_run,
        runtime_success=False,
        required_hashes_present=not missing,
        adapter_checked=adapter_checked,
        adapter_rank=adapter_rank,
        critical_serving_fields=_critical_serving_fields(serving),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def compare_serving_configs(
    reference: ServingConfig,
    candidate: ServingConfig,
    *,
    adapter_config: Mapping[str, Any] | None = None,
    adapter_dir: str | Path | None = None,
    submission_zip: str | Path | bytes | Any | None = None,
) -> SubmissionExactReport:
    """Fail if any submission-critical decoding field differs."""

    errors: list[str] = []
    for field_name in SUBMISSION_CRITICAL_FIELDS:
        if getattr(reference, field_name) != getattr(candidate, field_name):
            errors.append(f"submission critical field differs: {field_name}")
    try:
        validate_serving_config(candidate)
    except ServingConfigError as exc:
        errors.append(str(exc))
    critical = _critical_serving_fields(candidate)
    try:
        _validate_submission_exact_critical_fields(critical, adapter_checked=False, adapter_rank=None)
    except SubmissionExactError as exc:
        errors.append(str(exc))
    artifact_sources = [adapter_config is not None, adapter_dir is not None, submission_zip is not None]
    adapter_checked = False
    adapter_rank = None
    if any(artifact_sources):
        artifact = validate_submission_artifact_refs(
            adapter_config=adapter_config,
            adapter_dir=adapter_dir,
            submission_zip=submission_zip,
        )
        adapter_checked = bool(artifact.adapter_config)
        adapter_rank = _adapter_rank(artifact.adapter_config) if artifact.valid and artifact.adapter_config else None
        errors.extend(artifact.errors)
        if adapter_checked and not _non_empty(candidate.adapter_hash):
            errors.append("adapter_hash is required when adapter config is supplied.")
    combined_errors = tuple(errors)
    return SubmissionExactReport(
        config_hash=stable_hash({"comparison": "serving", "reference": reference.to_dict(), "candidate": candidate.to_dict()}),
        serving_config_hash=stable_hash(candidate.to_dict()),
        passed=not combined_errors,
        dry_run=True,
        runtime_success=False,
        required_hashes_present=all(_non_empty(critical.get(name)) for name in ("prompt_template_hash", "tokenizer_hash", "model_hash")),
        adapter_checked=adapter_checked,
        adapter_rank=adapter_rank,
        critical_serving_fields=critical,
        errors=combined_errors,
        warnings=(),
    )


def validate_submission_artifact_refs(
    *,
    adapter_config: Mapping[str, Any] | None = None,
    adapter_dir: str | Path | None = None,
    submission_zip: str | Path | bytes | Any | None = None,
) -> SubmissionValidationResult:
    """Validate exactly one adapter artifact source without creating artifacts."""

    sources = [adapter_config is not None, adapter_dir is not None, submission_zip is not None]
    if sum(1 for item in sources if item) != 1:
        return SubmissionValidationResult(valid=False, errors=("exactly one artifact source is required.",))
    if adapter_config is not None:
        if not isinstance(adapter_config, Mapping):
            return SubmissionValidationResult(valid=False, errors=("adapter_config must be a mapping.",))
        try:
            validate_adapter_config_json(adapter_config)
        except LoRAConfigError as exc:
            return SubmissionValidationResult(valid=False, errors=(str(exc),), adapter_config=dict(adapter_config))
        return SubmissionValidationResult(valid=True, adapter_config=dict(adapter_config), adapter_config_path="<mapping>")
    if adapter_dir is not None:
        return validate_adapter_dir(adapter_dir)
    return validate_submission_zip(submission_zip)


def _adapter_rank(config: Mapping[str, Any] | None) -> int | None:
    if not config:
        return None
    ranks: list[int] = []
    for key in ("r", "rank", "lora_rank"):
        if key in config:
            ranks.append(_validate_rank(config[key]))
    peft = config.get("peft_config")
    if isinstance(peft, Mapping) and "r" in peft:
        ranks.append(_validate_rank(peft["r"]))
    return max(ranks) if ranks else None


def _critical_serving_fields(config: ServingConfig) -> dict[str, Any]:
    return {name: getattr(config, name) for name in SUBMISSION_CRITICAL_FIELDS}


def _normalize_critical_fields(fields_payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(fields_payload, Mapping):
        raise SubmissionExactError("critical_serving_fields must be a mapping.")
    normalized = {str(key): value for key, value in sorted(fields_payload.items())}
    if "stop" in normalized:
        stop = normalized["stop"]
        if isinstance(stop, str):
            normalized["stop"] = (stop,)
        elif isinstance(stop, (tuple, list)):
            normalized["stop"] = tuple(str(item) for item in stop)
        else:
            raise SubmissionExactError("critical stop field must be a string sequence.")
    return normalized


def _validate_submission_exact_critical_fields(
    critical: Mapping[str, Any],
    *,
    adapter_checked: bool,
    adapter_rank: int | None,
) -> None:
    missing_fields = tuple(name for name in SUBMISSION_CRITICAL_FIELDS if name not in critical)
    if missing_fields:
        raise SubmissionExactError(f"missing critical serving fields: {missing_fields}")
    if critical["temperature"] != 0.0:
        raise SubmissionExactError("critical temperature must be 0.0.")
    if critical["top_p"] != 1.0:
        raise SubmissionExactError("critical top_p must be 1.0.")
    if critical["num_samples"] != 1:
        raise SubmissionExactError("critical num_samples must be 1.")
    if critical["majority_vote"] is not False:
        raise SubmissionExactError("critical majority_vote must be False.")
    if not isinstance(critical["max_tokens"], int) or critical["max_tokens"] <= 0:
        raise SubmissionExactError("critical max_tokens must be > 0.")
    if not isinstance(critical["batch_size"], int) or critical["batch_size"] <= 0:
        raise SubmissionExactError("critical batch_size must be > 0.")
    for hash_name in ("prompt_template_hash", "tokenizer_hash", "model_hash"):
        if not _non_empty(critical.get(hash_name)):
            raise SubmissionExactError(f"critical {hash_name} must be non-empty.")
    if (adapter_checked or adapter_rank is not None) and not _non_empty(critical.get("adapter_hash")):
        raise SubmissionExactError("critical adapter_hash must be non-empty when adapter is checked.")
    if adapter_rank is not None:
        _validate_rank(adapter_rank)
    for name in ("temperature", "top_p"):
        value = critical[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(float(value)) or math.isinf(float(value)):
            raise SubmissionExactError(f"critical {name} must be finite.")


def _validate_rank(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SubmissionExactError("adapter rank must be an integer.")
    if value < 1 or value > MAX_LORA_RANK:
        raise SubmissionExactError(f"adapter rank must be in [1, {MAX_LORA_RANK}].")
    return value


def _non_empty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


__all__ = [
    "SUBMISSION_CRITICAL_FIELDS",
    "SubmissionExactConfig",
    "SubmissionExactError",
    "SubmissionExactReport",
    "compare_serving_configs",
    "validate_submission_artifact_refs",
    "validate_submission_exact_config",
]
