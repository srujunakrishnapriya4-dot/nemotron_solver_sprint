"""Tiny smoke-training configuration contracts for Pass 12."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import re
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash


class SmokeTrainingConfigError(ValueError):
    """Raised when a smoke-training configuration is unsafe."""


ALLOWED_SMOKE_STAGES = frozenset({"sft", "dpo", "final_sft_refresh"})
FORBIDDEN_SPLIT_WORDS = frozenset({"eval", "holdout", "private", "private_like", "stress", "contaminated", "contamination"})
SUCCESS_WORDS = frozenset({"success", "ready", "submitted", "passed", "winner", "won", "guaranteed", "guarantee"})


@dataclass(frozen=True)
class SmokeTrainingConfig:
    stage: str
    max_examples: int
    max_steps: int
    max_runtime_seconds: int
    require_adapter_validation: bool = True
    require_log_validation: bool = True
    allow_backend_on_dry_run: bool = False
    dry_run: bool = True
    output_dir: str | None = None
    seed: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        if self.stage not in ALLOWED_SMOKE_STAGES:
            raise SmokeTrainingConfigError("stage must be one of: sft, dpo, final_sft_refresh.")
        _require_int_range(self.max_examples, "max_examples", minimum=1, maximum=16)
        _require_int_range(self.max_steps, "max_steps", minimum=1, maximum=20)
        _require_int_range(self.max_runtime_seconds, "max_runtime_seconds", minimum=1, maximum=900)
        _require_int_range(self.seed, "seed", minimum=0, maximum=None)
        for name in ("require_adapter_validation", "require_log_validation", "allow_backend_on_dry_run", "dry_run"):
            _require_bool(getattr(self, name), name)
        if self.output_dir is not None:
            if not isinstance(self.output_dir, str) or not self.output_dir.strip():
                raise SmokeTrainingConfigError("output_dir must be a non-empty string or None.")
        elif self.dry_run is False:
            raise SmokeTrainingConfigError("output_dir is required when dry_run=False.")
        metadata = _metadata(self.metadata)
        _reject_unsafe_metadata(metadata, reject_split_claims=True)
        object.__setattr__(self, "metadata", metadata)
        expected = compute_smoke_config_hash(self)
        if not self.config_hash:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise SmokeTrainingConfigError("config_hash does not match smoke config payload.")


def validate_smoke_training_config(config: SmokeTrainingConfig | Mapping[str, Any]) -> SmokeTrainingConfig:
    if isinstance(config, SmokeTrainingConfig):
        if config.config_hash != compute_smoke_config_hash(config):
            raise SmokeTrainingConfigError("config_hash does not match smoke config payload.")
        return config
    if isinstance(config, Mapping):
        return SmokeTrainingConfig(**dict(config))
    raise SmokeTrainingConfigError("config must be a SmokeTrainingConfig or mapping.")


def compute_smoke_config_hash(config: SmokeTrainingConfig | Mapping[str, Any]) -> str:
    payload = dict(config) if isinstance(config, Mapping) else {item.name: getattr(config, item.name) for item in fields(SmokeTrainingConfig)}
    payload.pop("config_hash", None)
    return stable_hash(payload)


def _metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SmokeTrainingConfigError("metadata must be a mapping.")
    return dict(value)


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise SmokeTrainingConfigError(f"{field_name} must be boolean.")
    return value


def _require_int_range(value: Any, field_name: str, *, minimum: int, maximum: int | None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SmokeTrainingConfigError(f"{field_name} must be an integer.")
    if value < minimum:
        raise SmokeTrainingConfigError(f"{field_name} must be >= {minimum}.")
    if maximum is not None and value > maximum:
        raise SmokeTrainingConfigError(f"{field_name} must be <= {maximum}.")
    return value


def _reject_unsafe_metadata(metadata: Mapping[str, Any], *, reject_split_claims: bool) -> None:
    for key, value in _flatten(metadata):
        raw = f"{key} {value}"
        text = _normalized_text(raw)
        truthy = value is True or (isinstance(value, str) and value.strip().lower() in {"true", "yes", "success", "ready", "passed"})
        if reject_split_claims and any(word in text for word in FORBIDDEN_SPLIT_WORDS):
            raise SmokeTrainingConfigError("metadata cannot reference contaminated or eval/private/holdout/stress data.")
        if any(word in text for word in ("kaggle", "leaderboard", "submission", "submit")) and (
            truthy or any(word in text for word in SUCCESS_WORDS)
        ):
            raise SmokeTrainingConfigError("metadata cannot claim Kaggle/leaderboard/submission success.")
        if "95+" in raw.lower() or "95 plus" in text:
            raise SmokeTrainingConfigError("metadata cannot claim 95+ success.")
        if "score" in text and ("guarantee" in text or "guaranteed" in text):
            raise SmokeTrainingConfigError("metadata cannot claim score guarantees.")
        if "leaderboard" in text or "kaggle" in text:
            raise SmokeTrainingConfigError("metadata cannot reference leaderboard or Kaggle success evidence.")


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
    "SmokeTrainingConfig",
    "SmokeTrainingConfigError",
    "compute_smoke_config_hash",
    "validate_smoke_training_config",
]
