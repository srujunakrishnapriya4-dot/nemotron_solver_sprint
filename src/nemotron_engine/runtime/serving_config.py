"""Deterministic strict serving configuration for Pass 1 submission runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping


class ServingConfigError(ValueError):
    """Raised when a serving configuration violates strict submission constraints."""


class ServingRuntime(str, Enum):
    VLLM = "vllm"
    OPENAI_COMPATIBLE = "openai_compatible"
    OTHER = "other"


@dataclass(frozen=True)
class ServingConfig:
    model_path: str | None = None
    served_model_name: str | None = None
    runtime: ServingRuntime = ServingRuntime.VLLM
    host: str = "0.0.0.0"
    port: int = 8000
    tensor_parallel_size: int = 1
    max_model_len: int | None = None
    dtype: str = "auto"
    gpu_memory_utilization: float = 0.90
    quantization: str | None = None
    enable_prefix_caching: bool = True
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    stop: tuple[str, ...] = ()
    num_samples: int = 1
    majority_vote: bool = False
    batch_size: int = 1
    strict_submission_mode: bool = True
    prompt_template_hash: str | None = None
    tokenizer_hash: str | None = None
    model_hash: str | None = None
    adapter_hash: str | None = None
    extra_args: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime", _coerce_runtime(self.runtime))
        object.__setattr__(self, "stop", tuple(self.stop))
        if self.model_path is not None and not str(self.model_path).strip():
            raise ServingConfigError("model_path must not be empty when provided.")
        if self.served_model_name is not None and not self.served_model_name.strip():
            raise ServingConfigError("served_model_name must not be empty when provided.")
        if not (1 <= int(self.port) <= 65535):
            raise ServingConfigError("port must be in [1, 65535].")
        if int(self.tensor_parallel_size) < 1:
            raise ServingConfigError("tensor_parallel_size must be >= 1.")
        if self.max_model_len is not None and int(self.max_model_len) < 1:
            raise ServingConfigError("max_model_len must be >= 1 when provided.")
        if not (0.0 < float(self.gpu_memory_utilization) <= 1.0):
            raise ServingConfigError("gpu_memory_utilization must be in (0, 1].")
        validate_serving_config(self)

    @property
    def model_name(self) -> str | None:
        return self.served_model_name or self.model_path

    def to_vllm_kwargs(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_path,
            "served_model_name": self.served_model_name,
            "dtype": self.dtype,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enable_prefix_caching": self.enable_prefix_caching,
        }
        if self.max_model_len is not None:
            payload["max_model_len"] = self.max_model_len
        if self.quantization is not None:
            payload["quantization"] = self.quantization
        payload.update(dict(self.extra_args))
        return {key: value for key, value in payload.items() if value is not None}

    def to_vllm_command(self) -> list[str]:
        if not self.model_path:
            raise ServingConfigError("model_path is required to build a vLLM command.")
        command = [
            "vllm",
            "serve",
            self.model_path,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--dtype",
            self.dtype,
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
        ]
        if self.served_model_name:
            command.extend(["--served-model-name", self.served_model_name])
        if self.max_model_len is not None:
            command.extend(["--max-model-len", str(self.max_model_len)])
        if self.quantization:
            command.extend(["--quantization", self.quantization])
        if self.enable_prefix_caching:
            command.append("--enable-prefix-caching")
        for key, value in sorted(dict(self.extra_args).items()):
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    command.append(flag)
            else:
                command.extend([flag, str(value)])
        return command

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["runtime"] = self.runtime.value
        payload["stop"] = list(self.stop)
        payload["extra_args"] = dict(self.extra_args)
        return payload


def validate_serving_config(config: ServingConfig) -> ServingConfig:
    """Reject non-deterministic strict-submission inference settings."""

    errors: list[str] = []
    if config.strict_submission_mode:
        if config.temperature != 0.0:
            errors.append("temperature must be 0.0 in strict submission mode.")
        if config.top_p <= 0 or config.top_p > 1:
            errors.append("top_p must be in (0, 1] in strict submission mode.")
        if config.max_tokens <= 0:
            errors.append("max_tokens must be > 0 in strict submission mode.")
        if config.num_samples != 1:
            errors.append("num_samples must be 1 in strict submission mode.")
        if config.majority_vote:
            errors.append("majority_vote must be false in strict submission mode.")
        if config.batch_size <= 0:
            errors.append("batch_size must be > 0 in strict submission mode.")
    if errors:
        raise ServingConfigError("; ".join(errors))
    return config


def save_serving_config(config: ServingConfig, path: str | Path) -> None:
    """Save deterministic JSON with stable key ordering."""

    validate_serving_config(config)
    target = Path(path)
    target.write_text(
        json.dumps(config.to_dict(), sort_keys=True, indent=2, separators=(",", ": ")) + "\n",
        encoding="utf-8",
    )


def load_serving_config(path: str | Path) -> ServingConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ServingConfigError("Serving config JSON must contain an object.")
    data = dict(payload)
    if "stop" in data:
        data["stop"] = tuple(data["stop"])
    return ServingConfig(**data)


def build_serving_config(**kwargs: Any) -> ServingConfig:
    return ServingConfig(**kwargs)


def _coerce_runtime(value: ServingRuntime | str) -> ServingRuntime:
    if isinstance(value, ServingRuntime):
        return value
    try:
        return ServingRuntime(str(value))
    except ValueError as exc:
        raise ServingConfigError(f"Unsupported serving runtime: {value!r}.") from exc


__all__ = [
    "ServingConfig",
    "ServingConfigError",
    "ServingRuntime",
    "build_serving_config",
    "load_serving_config",
    "save_serving_config",
    "validate_serving_config",
]
