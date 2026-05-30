"""Training log validation for external backend outputs."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from nemotron_engine.core.schemas import stable_hash

from .backend_contracts import _reject_forbidden_claims


class TrainingLogValidationError(ValueError):
    """Raised when training logs or log reports are invalid."""


ALLOWED_METRICS = (
    "loss",
    "eval_loss",
    "grad_norm",
    "learning_rate",
    "accuracy",
    "format_error_rate",
    "extraction_error_rate",
    "reward",
    "reward_margin",
)


@dataclass(frozen=True)
class TrainingLogValidationConfig:
    max_loss: float = 100.0
    max_grad_norm: float = 1000.0
    allowed_metrics: tuple[str, ...] = ALLOWED_METRICS
    warnings_are_fatal: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("max_loss", "max_grad_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
                raise TrainingLogValidationError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, float(value))
        if not isinstance(self.warnings_are_fatal, bool):
            raise TrainingLogValidationError("warnings_are_fatal must be boolean.")
        allowed = tuple(str(item) for item in self.allowed_metrics)
        if any(item not in ALLOWED_METRICS for item in allowed):
            raise TrainingLogValidationError("allowed_metrics contains unsupported metric.")
        metadata = dict(self.metadata)
        _reject_forbidden_claims(metadata)
        object.__setattr__(self, "allowed_metrics", allowed)
        object.__setattr__(self, "metadata", metadata)
        expected = _payload_hash(self, "config_hash")
        if not self.config_hash:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise TrainingLogValidationError("config_hash does not match training log config payload.")


@dataclass(frozen=True)
class TrainingLogValidationReport:
    passed: bool
    entry_count: int
    metric_names: tuple[str, ...]
    last_step: int | None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    series_hash: str = ""
    report_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise TrainingLogValidationError("passed must be boolean.")
        if isinstance(self.entry_count, bool) or not isinstance(self.entry_count, int) or self.entry_count < 0:
            raise TrainingLogValidationError("entry_count must be a non-negative integer.")
        if self.last_step is not None and (isinstance(self.last_step, bool) or not isinstance(self.last_step, int) or self.last_step < 0):
            raise TrainingLogValidationError("last_step must be a non-negative integer or None.")
        names = tuple(str(item) for item in self.metric_names)
        errors = tuple(str(item) for item in self.errors)
        warnings = tuple(str(item) for item in self.warnings)
        if self.passed and (errors or warnings):
            raise TrainingLogValidationError("passed=True cannot include errors or warnings.")
        if not isinstance(self.series_hash, str) or not self.series_hash.strip():
            raise TrainingLogValidationError("series_hash must be non-empty.")
        object.__setattr__(self, "metric_names", names)
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", warnings)
        expected = _payload_hash(self, "report_hash")
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected)
        elif self.report_hash != expected:
            raise TrainingLogValidationError("report_hash does not match training log report payload.")


def parse_training_log_lines(source: Sequence[Mapping[str, Any]] | Sequence[str] | str | Path) -> tuple[dict[str, Any], ...]:
    if isinstance(source, (str, Path)) and not isinstance(source, list):
        path = Path(source)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise TrainingLogValidationError("could not read training log JSONL.") from exc
        return parse_training_log_lines(lines)
    if not isinstance(source, Sequence):
        raise TrainingLogValidationError("training log source must be a sequence or path.")
    entries: list[dict[str, Any]] = []
    for item in source:
        if isinstance(item, Mapping):
            entries.append(dict(item))
        elif isinstance(item, str):
            if not item.strip():
                continue
            try:
                payload = json.loads(item)
            except json.JSONDecodeError as exc:
                raise TrainingLogValidationError("training log JSONL line is invalid.") from exc
            if not isinstance(payload, Mapping):
                raise TrainingLogValidationError("training log JSONL lines must contain objects.")
            entries.append(dict(payload))
        else:
            raise TrainingLogValidationError("training log entries must be mappings or JSON strings.")
    return tuple(entries)


def validate_metric_series(entries: Sequence[Mapping[str, Any]], config: TrainingLogValidationConfig | None = None) -> tuple[str, ...]:
    cfg = config or TrainingLogValidationConfig()
    errors: list[str] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            errors.append(f"entry {index} is not a mapping")
            continue
        try:
            _reject_forbidden_claims(entry)
        except Exception as exc:
            errors.append(f"entry {index} fake success claim: {exc}")
        if entry.get("trained") is True:
            errors.append(f"entry {index} trained=True claim requires backend result")
        step = entry.get("step")
        if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
            errors.append(f"entry {index} step must be a non-negative integer")
        for key, value in entry.items():
            if key == "step":
                continue
            if key == "metadata":
                if not isinstance(value, Mapping):
                    errors.append(f"entry {index} metadata must be a mapping")
                else:
                    try:
                        _reject_forbidden_claims(value)
                    except Exception as exc:
                        errors.append(f"entry {index} metadata fake success claim: {exc}")
                continue
            if key not in cfg.allowed_metrics:
                errors.append(f"entry {index} unsupported log key: {key}")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                errors.append(f"entry {index} {key} must be finite")
                continue
            number = float(value)
            if key in {"loss", "eval_loss"} and number > cfg.max_loss:
                errors.append(f"entry {index} {key} exceeds max_loss")
            if key == "grad_norm" and number > cfg.max_grad_norm:
                errors.append(f"entry {index} grad_norm exceeds max_grad_norm")
            if key == "learning_rate" and number < 0:
                errors.append(f"entry {index} learning_rate must be non-negative")
            if key in {"accuracy", "format_error_rate", "extraction_error_rate"} and not 0.0 <= number <= 1.0:
                errors.append(f"entry {index} {key} must be in [0, 1]")
    return tuple(errors)


def validate_training_log(
    source: Sequence[Mapping[str, Any]] | Sequence[str] | str | Path,
    config: TrainingLogValidationConfig | None = None,
) -> TrainingLogValidationReport:
    cfg = config or TrainingLogValidationConfig()
    entries = parse_training_log_lines(source)
    errors = list(validate_metric_series(entries, cfg))
    warnings: list[str] = []
    metric_names = tuple(sorted({str(key) for entry in entries for key in entry if key in cfg.allowed_metrics}))
    steps = tuple(entry.get("step") for entry in entries if isinstance(entry.get("step"), int) and not isinstance(entry.get("step"), bool))
    series_hash = stable_hash({"entries": entries})
    passed = not errors and (not warnings or not cfg.warnings_are_fatal)
    return TrainingLogValidationReport(
        passed=passed,
        entry_count=len(entries),
        metric_names=metric_names,
        last_step=max(steps) if steps else None,
        errors=tuple(sorted(set(errors))),
        warnings=tuple(sorted(set(warnings))),
        series_hash=series_hash,
    )


def _looks_metric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


__all__ = [
    "ALLOWED_METRICS",
    "TrainingLogValidationConfig",
    "TrainingLogValidationError",
    "TrainingLogValidationReport",
    "parse_training_log_lines",
    "validate_metric_series",
    "validate_training_log",
]
