"""Strict LoRA config validation for Pass 7 dry-run training plans."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import json
from pathlib import Path
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash, stable_json_dumps


class LoRAConfigError(ValueError):
    """Raised when a LoRA config or adapter config is unsafe."""


@dataclass(frozen=True)
class LoRAConfig:
    base_model_name: str
    target_modules: tuple[str, ...]
    rank: int
    alpha: int
    dropout: float
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    use_rslora: bool = False
    modules_to_save: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        _require_non_empty(self.base_model_name, "base_model_name")
        object.__setattr__(self, "target_modules", _require_string_tuple(self.target_modules, "target_modules", non_empty=True))
        _require_int(self.rank, "rank", minimum=1, maximum=32)
        _require_int(self.alpha, "alpha", minimum=1)
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise LoRAConfigError("dropout must be numeric.")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise LoRAConfigError("dropout must be in [0, 1).")
        object.__setattr__(self, "dropout", float(self.dropout))
        _require_non_empty(self.bias, "bias")
        _require_non_empty(self.task_type, "task_type")
        if not isinstance(self.use_rslora, bool):
            raise LoRAConfigError("use_rslora must be boolean.")
        object.__setattr__(self, "modules_to_save", _require_string_tuple(self.modules_to_save, "modules_to_save"))
        object.__setattr__(self, "metadata", dict(self.metadata))
        expected = compute_lora_config_hash(self)
        if not self.config_hash:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise LoRAConfigError("config_hash does not match LoRA config payload.")


def validate_lora_config(config: LoRAConfig) -> LoRAConfig:
    if not isinstance(config, LoRAConfig):
        raise LoRAConfigError("config must be a LoRAConfig.")
    return config


def compute_lora_config_hash(config: LoRAConfig) -> str:
    payload = {item.name: getattr(config, item.name) for item in fields(LoRAConfig) if item.name != "config_hash"}
    return stable_hash(payload)


def save_lora_config(config: LoRAConfig, path: Path) -> None:
    validate_lora_config(config)
    path.write_text(stable_json_dumps(config) + "\n", encoding="utf-8")


def load_lora_config(path: Path) -> LoRAConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LoRAConfigError("could not read LoRA config JSON.") from exc
    if not isinstance(payload, dict):
        raise LoRAConfigError("LoRA config JSON must be an object.")
    return LoRAConfig(**payload)


def validate_adapter_config_json(payload_or_path: Mapping[str, Any] | Path) -> bool:
    if isinstance(payload_or_path, Path):
        try:
            payload = json.loads(payload_or_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LoRAConfigError("could not read adapter config JSON.") from exc
    else:
        payload = payload_or_path
    if not isinstance(payload, Mapping):
        raise LoRAConfigError("adapter config must be an object.")
    for key in ("r", "rank", "lora_rank"):
        if key in payload:
            _validate_adapter_rank(payload[key], key)
    peft = payload.get("peft_config")
    if peft is not None:
        if not isinstance(peft, Mapping):
            raise LoRAConfigError("peft_config must be an object.")
        if "r" in peft:
            _validate_adapter_rank(peft["r"], "peft_config.r")
    return True


def _validate_adapter_rank(value: Any, field_name: str) -> None:
    _require_int(value, field_name, minimum=1, maximum=32)


def _require_int(value: Any, field_name: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LoRAConfigError(f"{field_name} must be an integer.")
    if value < minimum:
        raise LoRAConfigError(f"{field_name} must be >= {minimum}.")
    if maximum is not None and value > maximum:
        raise LoRAConfigError(f"{field_name} must be <= {maximum}.")
    return value


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LoRAConfigError(f"{field_name} must be a non-empty string.")
    return value


def _require_string_tuple(value: Any, field_name: str, *, non_empty: bool = False) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (tuple, list)):
        raise LoRAConfigError(f"{field_name} must be a tuple/list of strings.")
    result = tuple(value)
    if non_empty and not result:
        raise LoRAConfigError(f"{field_name} must be non-empty.")
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise LoRAConfigError(f"{field_name} entries must be non-empty strings.")
    return result


__all__ = [
    "LoRAConfig",
    "LoRAConfigError",
    "compute_lora_config_hash",
    "load_lora_config",
    "save_lora_config",
    "validate_adapter_config_json",
    "validate_lora_config",
]
