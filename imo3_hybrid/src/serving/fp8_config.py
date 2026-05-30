from __future__ import annotations

from enum import Enum
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ServingRuntime(str, Enum):
    """Serving runtimes this config can reason about explicitly."""

    VLLM = "vllm"
    OTHER = "other"


class FP8SupportState(str, Enum):
    """Explicit state for FP8 availability/intent."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    UNSUPPORTED = "unsupported"


class FP8CheckpointDType(str, Enum):
    """Checkpoint storage / weight precision seen by the serving layer."""

    AUTO = "auto"
    BF16 = "bf16"
    FP16 = "fp16"
    FP8 = "fp8"


class FP8KVCacheDType(str, Enum):
    """KV-cache precision options that a runtime may expose."""

    AUTO = "auto"
    FP8 = "fp8"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    BF16 = "bf16"

    @property
    def is_fp8(self) -> bool:
        return self in {self.FP8, self.FP8_E4M3, self.FP8_E5M2}


class FP8WeightsConfig(BaseModel):
    """Typed weight-side FP8 configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)

    enabled: bool = False
    checkpoint_dtype: FP8CheckpointDType = FP8CheckpointDType.AUTO
    quantization: str | None = Field(
        default=None,
        description="Runtime quantization selector. For vLLM this is typically 'fp8'.",
    )
    calibration_path: str | None = None
    activation_scheme: str | None = Field(
        default=None,
        description="Optional runtime-specific activation scaling/calibration scheme.",
    )

    @field_validator("quantization")
    @classmethod
    def _normalize_quantization(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().lower()
        return cleaned or None

    @model_validator(mode="after")
    def _validate_enabled_block(self) -> "FP8WeightsConfig":
        if not self.enabled:
            if self.quantization is not None:
                raise ValueError("weights.quantization must be null when weights.enabled is false")
            if self.calibration_path is not None:
                raise ValueError("weights.calibration_path must be null when weights.enabled is false")
            if self.activation_scheme is not None:
                raise ValueError("weights.activation_scheme must be null when weights.enabled is false")
        else:
            if self.quantization is None:
                raise ValueError("weights.quantization is required when weights.enabled is true")
            if self.quantization != "fp8":
                raise ValueError("weights.quantization currently supports only 'fp8'")
        return self


class FP8KVCacheConfig(BaseModel):
    """Typed KV-cache FP8 configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)

    enabled: bool = False
    dtype: FP8KVCacheDType = FP8KVCacheDType.AUTO
    dynamic_scales: bool = True
    scale_path: str | None = None

    @model_validator(mode="after")
    def _validate_enabled_block(self) -> "FP8KVCacheConfig":
        if not self.enabled:
            if self.scale_path is not None:
                raise ValueError("kv_cache.scale_path must be null when kv_cache.enabled is false")
            if self.dtype.is_fp8:
                raise ValueError("kv_cache.dtype cannot be an FP8 dtype when kv_cache.enabled is false")
        else:
            if not self.dtype.is_fp8:
                raise ValueError("kv_cache.dtype must be an FP8 dtype when kv_cache.enabled is true")
        return self


class FP8CompatibilityReport(BaseModel):
    """Explicit compatibility result for serving/runtime consumers."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)

    runtime: ServingRuntime
    state: FP8SupportState
    exportable: bool
    runtime_supported: bool
    hardware_validated: bool | None = None
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def is_usable(self) -> bool:
        return self.exportable and self.state == FP8SupportState.ENABLED


class FP8ServingConfig(BaseModel):
    """Single typed contract for FP8-related serving settings.

    This config is intentionally conservative:
    - it models only boundaries we can state explicitly,
    - it does not claim hardware support unless evidence is supplied,
    - it exports deterministic runtime payloads.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)

    runtime: ServingRuntime = ServingRuntime.VLLM
    state: FP8SupportState = FP8SupportState.DISABLED
    weights: FP8WeightsConfig = Field(default_factory=FP8WeightsConfig)
    kv_cache: FP8KVCacheConfig = Field(default_factory=FP8KVCacheConfig)
    allow_fallback_to_bf16: bool = True
    disabled_reason: str | None = None
    unsupported_reason: str | None = None
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_consistency(self) -> "FP8ServingConfig":
        any_enabled = self.weights.enabled or self.kv_cache.enabled

        if self.state == FP8SupportState.DISABLED:
            if any_enabled:
                raise ValueError("disabled FP8 state cannot enable weights or kv_cache")
            if self.unsupported_reason is not None:
                raise ValueError("unsupported_reason must be null when state is disabled")
        elif self.state == FP8SupportState.UNSUPPORTED:
            if any_enabled:
                raise ValueError("unsupported FP8 state cannot enable weights or kv_cache")
            if not self.unsupported_reason:
                raise ValueError("unsupported_reason is required when state is unsupported")
        elif self.state == FP8SupportState.ENABLED:
            if not any_enabled:
                raise ValueError("enabled FP8 state requires weights or kv_cache to be enabled")
            if self.runtime != ServingRuntime.VLLM:
                raise ValueError("enabled FP8 configuration currently supports only runtime='vllm'")
            if self.unsupported_reason is not None:
                raise ValueError("unsupported_reason must be null when state is enabled")

        if self.state != FP8SupportState.DISABLED and self.disabled_reason is not None:
            raise ValueError("disabled_reason may only be set when state is disabled")

        return self

    @property
    def fp8_enabled(self) -> bool:
        return self.state == FP8SupportState.ENABLED

    @property
    def kv_cache_enabled(self) -> bool:
        return self.kv_cache.enabled

    @property
    def weights_enabled(self) -> bool:
        return self.weights.enabled

    def evaluate_compatibility(
        self,
        *,
        runtime: ServingRuntime | None = None,
        hardware_fp8_available: bool | None = None,
    ) -> FP8CompatibilityReport:
        resolved_runtime = runtime or self.runtime
        issues: list[str] = []
        warnings: list[str] = []
        runtime_supported = resolved_runtime == ServingRuntime.VLLM

        if self.state == FP8SupportState.DISABLED:
            return FP8CompatibilityReport(
                runtime=resolved_runtime,
                state=FP8SupportState.DISABLED,
                exportable=True,
                runtime_supported=runtime_supported,
                hardware_validated=hardware_fp8_available,
                warnings=[self.disabled_reason] if self.disabled_reason else [],
            )

        if self.state == FP8SupportState.UNSUPPORTED:
            issues.append(self.unsupported_reason or "FP8 configuration marked unsupported")
            return FP8CompatibilityReport(
                runtime=resolved_runtime,
                state=FP8SupportState.UNSUPPORTED,
                exportable=False,
                runtime_supported=runtime_supported,
                hardware_validated=hardware_fp8_available,
                issues=issues,
            )

        if not runtime_supported:
            issues.append("Only the vLLM runtime export path is supported for enabled FP8 configuration")

        if hardware_fp8_available is False:
            issues.append("Hardware/runtime probe reported FP8 unavailable")
        elif hardware_fp8_available is None:
            warnings.append("Hardware FP8 support is not validated by this config model")

        return FP8CompatibilityReport(
            runtime=resolved_runtime,
            state=FP8SupportState.ENABLED if not issues else FP8SupportState.UNSUPPORTED,
            exportable=not issues,
            runtime_supported=runtime_supported,
            hardware_validated=hardware_fp8_available,
            issues=issues,
            warnings=warnings,
        )

    def to_runtime_payload(self) -> dict[str, Any]:
        """Deterministic, typed export for serving adapters."""
        return {
            "runtime": self.runtime.value,
            "state": self.state.value,
            "allow_fallback_to_bf16": self.allow_fallback_to_bf16,
            "weights": {
                "enabled": self.weights.enabled,
                "checkpoint_dtype": self.weights.checkpoint_dtype.value,
                "quantization": self.weights.quantization,
                "calibration_path": self.weights.calibration_path,
                "activation_scheme": self.weights.activation_scheme,
            },
            "kv_cache": {
                "enabled": self.kv_cache.enabled,
                "dtype": self.kv_cache.dtype.value,
                "dynamic_scales": self.kv_cache.dynamic_scales,
                "scale_path": self.kv_cache.scale_path,
            },
            "disabled_reason": self.disabled_reason,
            "unsupported_reason": self.unsupported_reason,
            "notes": list(self.notes),
        }

    def to_vllm_kwargs(self) -> dict[str, Any]:
        """Export kwargs expected by a future vLLM adapter.

        This helper only emits keys when the FP8 state is enabled.
        Disabled/unsupported states export an explicit fallback payload.
        """
        if self.state != FP8SupportState.ENABLED:
            return {
                "quantization": None,
                "kv_cache_dtype": None,
                "fallback_dtype": "bf16" if self.allow_fallback_to_bf16 else None,
                "fp8_config": self.to_runtime_payload(),
            }

        return {
            "quantization": self.weights.quantization,
            "kv_cache_dtype": self.kv_cache.dtype.value if self.kv_cache.enabled else None,
            "fallback_dtype": "bf16" if self.allow_fallback_to_bf16 else None,
            "fp8_config": self.to_runtime_payload(),
        }

    def to_vllm_args(self) -> dict[str, Any]:
        """Backward-compatible alias retained for older serving helpers."""
        return self.to_vllm_kwargs()

    def to_benchmark_metadata(self) -> dict[str, Any]:
        """Compact export for throughput benchmarking metadata."""
        return {
            "runtime": self.runtime.value,
            "fp8_state": self.state.value,
            "weights_enabled": self.weights.enabled,
            "weights_quantization": self.weights.quantization,
            "weights_checkpoint_dtype": self.weights.checkpoint_dtype.value,
            "kv_cache_enabled": self.kv_cache.enabled,
            "kv_cache_dtype": self.kv_cache.dtype.value,
            "allow_fallback_to_bf16": self.allow_fallback_to_bf16,
        }

    def to_canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(by_alias=False, exclude_none=False, mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def memory_estimate_gib(
        param_count_billions: float,
        dtype: FP8CheckpointDType | str = FP8CheckpointDType.FP8,
    ) -> float:
        """Planning-only weight memory estimate.

        This is a deterministic size estimate, not a runtime fit guarantee.
        """
        dtype_value = dtype.value if isinstance(dtype, FP8CheckpointDType) else str(dtype).lower()
        bytes_per_param = {
            "fp8": 1.0,
            "bf16": 2.0,
            "fp16": 2.0,
            "auto": 1.0,
        }.get(dtype_value, 1.0)
        return float(param_count_billions) * float(bytes_per_param)

    @classmethod
    def disabled(
        cls,
        *,
        runtime: ServingRuntime = ServingRuntime.VLLM,
        reason: str = "FP8 explicitly disabled",
        allow_fallback_to_bf16: bool = True,
        notes: tuple[str, ...] = (),
    ) -> "FP8ServingConfig":
        return cls(
            runtime=runtime,
            state=FP8SupportState.DISABLED,
            weights=FP8WeightsConfig(enabled=False),
            kv_cache=FP8KVCacheConfig(enabled=False, dtype=FP8KVCacheDType.AUTO),
            allow_fallback_to_bf16=allow_fallback_to_bf16,
            disabled_reason=reason,
            notes=notes,
        )

    @classmethod
    def unsupported(
        cls,
        *,
        runtime: ServingRuntime,
        reason: str,
        allow_fallback_to_bf16: bool = True,
        notes: tuple[str, ...] = (),
    ) -> "FP8ServingConfig":
        return cls(
            runtime=runtime,
            state=FP8SupportState.UNSUPPORTED,
            weights=FP8WeightsConfig(enabled=False),
            kv_cache=FP8KVCacheConfig(enabled=False, dtype=FP8KVCacheDType.AUTO),
            allow_fallback_to_bf16=allow_fallback_to_bf16,
            unsupported_reason=reason,
            notes=notes,
        )

    @classmethod
    def vllm_fp8(
        cls,
        *,
        enable_kv_cache: bool = True,
        kv_cache_dtype: FP8KVCacheDType = FP8KVCacheDType.FP8_E4M3,
        checkpoint_dtype: FP8CheckpointDType = FP8CheckpointDType.AUTO,
        calibration_path: str | None = None,
        scale_path: str | None = None,
        activation_scheme: str | None = None,
        allow_fallback_to_bf16: bool = True,
        notes: tuple[str, ...] = (),
    ) -> "FP8ServingConfig":
        kv_cfg = FP8KVCacheConfig(
            enabled=enable_kv_cache,
            dtype=kv_cache_dtype if enable_kv_cache else FP8KVCacheDType.AUTO,
            scale_path=scale_path,
        )
        return cls(
            runtime=ServingRuntime.VLLM,
            state=FP8SupportState.ENABLED,
            weights=FP8WeightsConfig(
                enabled=True,
                checkpoint_dtype=checkpoint_dtype,
                quantization="fp8",
                calibration_path=calibration_path,
                activation_scheme=activation_scheme,
            ),
            kv_cache=kv_cfg,
            allow_fallback_to_bf16=allow_fallback_to_bf16,
            notes=notes,
        )


__all__ = [
    "FP8CheckpointDType",
    "FP8CompatibilityReport",
    "FP8KVCacheConfig",
    "FP8KVCacheDType",
    "FP8ServingConfig",
    "FP8SupportState",
    "FP8WeightsConfig",
    "ServingRuntime",
]