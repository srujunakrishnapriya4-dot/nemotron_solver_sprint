from __future__ import annotations

from enum import Enum
from math import floor
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


BYTES_PER_MIB = 1024 * 1024
BYTES_PER_GIB = 1024 * 1024 * 1024


class ServingRuntime(str, Enum):
    """Serving runtimes this policy can describe explicitly."""

    VLLM = "vllm"
    OTHER = "other"


class KVCacheState(str, Enum):
    """Explicit high-level KV-cache state."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    UNSUPPORTED = "unsupported"


class KVCacheDType(str, Enum):
    """KV-cache tensor precision options at the policy layer."""

    AUTO = "auto"
    FP8 = "fp8"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    BF16 = "bf16"
    FP16 = "fp16"

    @property
    def bytes_per_scalar(self) -> float | None:
        if self in {self.FP8, self.FP8_E4M3, self.FP8_E5M2}:
            return 1.0
        if self in {self.BF16, self.FP16}:
            return 2.0
        return None

    @property
    def is_fp8(self) -> bool:
        return self in {self.FP8, self.FP8_E4M3, self.FP8_E5M2}


class KVCacheLayout(str, Enum):
    """Logical layout style only; not a backend allocator implementation."""

    CONTIGUOUS = "contiguous"
    PAGED = "paged"


class KVCacheEvictionPolicy(str, Enum):
    """High-level policy name, not a backend eviction engine."""

    NONE = "none"
    LRU = "lru"
    OLDEST_FIRST = "oldest_first"
    RUNTIME_MANAGED = "runtime_managed"


class KVCacheBoundaryType(str, Enum):
    """Explicitly marks what this module does not implement."""

    RUNTIME_ALLOCATOR = "runtime_allocator"
    PAGE_TABLE = "page_table"
    PHYSICAL_MEMORY_MANAGER = "physical_memory_manager"
    CONCURRENCY_SCHEDULER = "concurrency_scheduler"
    BACKEND_KERNEL_BEHAVIOR = "backend_kernel_behavior"


class KVCacheSizingInputs(BaseModel):
    """Inputs required for deterministic footprint estimates.

    These are planning inputs only. They do not claim to reflect every backend's
    exact memory layout or overhead.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    num_layers: int = Field(gt=0)
    kv_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    max_context_tokens: int = Field(gt=0)
    max_concurrent_sequences: int = Field(gt=0)
    block_size_tokens: int | None = Field(default=None, gt=0)
    per_token_overhead_bytes: int = Field(default=0, ge=0)
    allocator_overhead_fraction: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def total_tokens_capacity(self) -> int:
        return self.max_context_tokens * self.max_concurrent_sequences


class KVCacheFootprintEstimate(BaseModel):
    """Deterministic estimate result from this policy layer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dtype: KVCacheDType
    layout: KVCacheLayout
    bytes_per_scalar: float
    bytes_per_token_per_layer: int
    total_token_capacity: int
    raw_cache_bytes: int
    estimated_total_bytes: int
    estimated_total_mib: float
    estimated_total_gib: float
    block_count: int | None = None
    block_size_tokens: int | None = None
    notes: tuple[str, ...] = ()

    def to_compact_dict(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype.value,
            "layout": self.layout.value,
            "bytes_per_scalar": self.bytes_per_scalar,
            "bytes_per_token_per_layer": self.bytes_per_token_per_layer,
            "total_token_capacity": self.total_token_capacity,
            "raw_cache_bytes": self.raw_cache_bytes,
            "estimated_total_bytes": self.estimated_total_bytes,
            "estimated_total_mib": self.estimated_total_mib,
            "estimated_total_gib": self.estimated_total_gib,
            "block_count": self.block_count,
            "block_size_tokens": self.block_size_tokens,
            "notes": list(self.notes),
        }


class KVCacheCompatibilityReport(BaseModel):
    """Explicit compatibility result for runtime consumers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime: ServingRuntime
    state: KVCacheState
    exportable: bool
    runtime_supported: bool
    hardware_validated: bool | None = None
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def is_usable(self) -> bool:
        return self.exportable and self.state == KVCacheState.ENABLED


class KVCachePolicy(BaseModel):
    """Typed KV-cache policy/config contract.

    This module intentionally stops at typed policy and deterministic sizing.
    It does not implement backend allocators, paging internals, or cache kernels.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime: ServingRuntime = ServingRuntime.VLLM
    state: KVCacheState = KVCacheState.ENABLED
    dtype: KVCacheDType = KVCacheDType.AUTO
    layout: KVCacheLayout = KVCacheLayout.PAGED
    eviction_policy: KVCacheEvictionPolicy = KVCacheEvictionPolicy.RUNTIME_MANAGED

    max_context_tokens: int = Field(default=4096, gt=0)
    max_concurrent_sequences: int = Field(default=32, gt=0)
    block_size_tokens: int | None = Field(default=16, gt=0)
    gpu_memory_utilization: float | None = Field(default=None, gt=0.0, le=1.0)
    max_cache_bytes: int | None = Field(default=None, ge=1)

    enable_prefix_caching: bool = False
    enable_sliding_window_reuse: bool = False
    allow_cpu_spill: bool = False

    disabled_reason: str | None = None
    unsupported_reason: str | None = None
    notes: tuple[str, ...] = ()

    @field_validator("block_size_tokens")
    @classmethod
    def _validate_block_size_multiple(cls, value: int | None) -> int | None:
        if value is None:
            return value
        if value <= 0:
            raise ValueError("block_size_tokens must be positive")
        return value

    @model_validator(mode="after")
    def _validate_consistency(self) -> "KVCachePolicy":
        if self.state == KVCacheState.DISABLED:
            if self.unsupported_reason is not None:
                raise ValueError("unsupported_reason must be null when state is disabled")
        elif self.state == KVCacheState.UNSUPPORTED:
            if not self.unsupported_reason:
                raise ValueError("unsupported_reason is required when state is unsupported")
        elif self.state == KVCacheState.ENABLED:
            if self.runtime != ServingRuntime.VLLM:
                raise ValueError("enabled KV-cache export is currently supported only for runtime='vllm'")

        if self.state != KVCacheState.DISABLED and self.disabled_reason is not None:
            raise ValueError("disabled_reason may only be set when state is disabled")

        if self.layout == KVCacheLayout.CONTIGUOUS and self.block_size_tokens is not None:
            raise ValueError("block_size_tokens must be null when layout is contiguous")

        if self.layout == KVCacheLayout.PAGED and self.block_size_tokens is None and self.state == KVCacheState.ENABLED:
            raise ValueError("block_size_tokens is required for enabled paged layout")

        if self.max_cache_bytes is not None and self.gpu_memory_utilization is not None:
            raise ValueError("max_cache_bytes and gpu_memory_utilization are mutually exclusive sizing controls")

        return self

    @property
    def caching_enabled(self) -> bool:
        return self.state == KVCacheState.ENABLED

    @property
    def explicit_boundaries(self) -> tuple[KVCacheBoundaryType, ...]:
        return (
            KVCacheBoundaryType.RUNTIME_ALLOCATOR,
            KVCacheBoundaryType.PAGE_TABLE,
            KVCacheBoundaryType.PHYSICAL_MEMORY_MANAGER,
            KVCacheBoundaryType.CONCURRENCY_SCHEDULER,
            KVCacheBoundaryType.BACKEND_KERNEL_BEHAVIOR,
        )

    def bytes_per_scalar(self, *, fallback_dtype: KVCacheDType | None = None) -> float | None:
        if self.dtype.bytes_per_scalar is not None:
            return self.dtype.bytes_per_scalar
        if fallback_dtype is not None:
            return fallback_dtype.bytes_per_scalar
        return None

    def estimate_footprint(
        self,
        *,
        inputs: KVCacheSizingInputs,
        fallback_dtype: KVCacheDType = KVCacheDType.BF16,
    ) -> KVCacheFootprintEstimate:
        """Estimate KV-cache memory footprint deterministically."""
        resolved_dtype = self.dtype if self.dtype != KVCacheDType.AUTO else fallback_dtype
        scalar_bytes = resolved_dtype.bytes_per_scalar
        if scalar_bytes is None:
            raise ValueError("Unable to estimate footprint without a resolved numeric KV dtype")

        bytes_per_token_per_layer = int(
            2 * inputs.kv_heads * inputs.head_dim * scalar_bytes + inputs.per_token_overhead_bytes
        )
        total_tokens = inputs.total_tokens_capacity
        raw_cache_bytes = bytes_per_token_per_layer * inputs.num_layers * total_tokens
        estimated_total = int(raw_cache_bytes * (1.0 + inputs.allocator_overhead_fraction))

        block_count: int | None = None
        block_size_tokens: int | None = None
        notes: list[str] = []

        if self.layout == KVCacheLayout.PAGED:
            block_size_tokens = inputs.block_size_tokens or self.block_size_tokens
            if block_size_tokens is None:
                raise ValueError("Paged layout footprint estimate requires block_size_tokens")
            block_count = (total_tokens + block_size_tokens - 1) // block_size_tokens
            notes.append("Paged-layout estimate excludes runtime page-table metadata")

        if self.dtype == KVCacheDType.AUTO:
            notes.append(f"Used fallback dtype '{fallback_dtype.value}' for deterministic sizing")

        return KVCacheFootprintEstimate(
            dtype=resolved_dtype,
            layout=self.layout,
            bytes_per_scalar=scalar_bytes,
            bytes_per_token_per_layer=bytes_per_token_per_layer,
            total_token_capacity=total_tokens,
            raw_cache_bytes=raw_cache_bytes,
            estimated_total_bytes=estimated_total,
            estimated_total_mib=estimated_total / BYTES_PER_MIB,
            estimated_total_gib=estimated_total / BYTES_PER_GIB,
            block_count=block_count,
            block_size_tokens=block_size_tokens,
            notes=tuple(notes),
        )

    def estimate_capacity_from_budget(
        self,
        *,
        num_layers: int,
        kv_heads: int,
        head_dim: int,
        available_bytes: int,
        fallback_dtype: KVCacheDType = KVCacheDType.BF16,
        per_token_overhead_bytes: int = 0,
        allocator_overhead_fraction: float = 0.0,
    ) -> dict[str, Any]:
        """Estimate capacity limits from a byte budget."""
        if available_bytes <= 0:
            raise ValueError("available_bytes must be positive")

        resolved_dtype = self.dtype if self.dtype != KVCacheDType.AUTO else fallback_dtype
        scalar_bytes = resolved_dtype.bytes_per_scalar
        if scalar_bytes is None:
            raise ValueError("Unable to estimate capacity without a resolved numeric KV dtype")

        bytes_per_token_all_layers = int(
            (2 * kv_heads * head_dim * scalar_bytes + per_token_overhead_bytes) * num_layers
        )
        effective_bytes = floor(available_bytes / (1.0 + allocator_overhead_fraction))
        total_tokens_capacity = effective_bytes // bytes_per_token_all_layers

        per_sequence_capacity = total_tokens_capacity // self.max_concurrent_sequences
        feasible_context_tokens = min(self.max_context_tokens, per_sequence_capacity)

        result: dict[str, Any] = {
            "dtype": resolved_dtype.value,
            "bytes_per_token_all_layers": bytes_per_token_all_layers,
            "available_bytes": available_bytes,
            "effective_bytes_after_overhead": effective_bytes,
            "total_tokens_capacity": total_tokens_capacity,
            "configured_max_concurrent_sequences": self.max_concurrent_sequences,
            "configured_max_context_tokens": self.max_context_tokens,
            "feasible_context_tokens_per_sequence": feasible_context_tokens,
            "supports_requested_shape": feasible_context_tokens >= self.max_context_tokens,
        }

        if self.layout == KVCacheLayout.PAGED and self.block_size_tokens is not None:
            result["estimated_block_count"] = (
                total_tokens_capacity + self.block_size_tokens - 1
            ) // self.block_size_tokens
            result["block_size_tokens"] = self.block_size_tokens

        return result

    def estimate_kv_memory_gib(
        self,
        *,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_context_tokens: int | None = None,
        max_concurrent_sequences: int | None = None,
        fallback_dtype: KVCacheDType = KVCacheDType.BF16,
    ) -> float:
        """Legacy-friendly planning helper derived from the typed estimator."""
        estimate = self.estimate_footprint(
            inputs=KVCacheSizingInputs(
                num_layers=num_layers,
                kv_heads=num_heads,
                head_dim=head_dim,
                max_context_tokens=max_context_tokens or self.max_context_tokens,
                max_concurrent_sequences=max_concurrent_sequences or self.max_concurrent_sequences,
                block_size_tokens=self.block_size_tokens,
            ),
            fallback_dtype=fallback_dtype,
        )
        return estimate.estimated_total_gib

    def recommended_max_seqs(
        self,
        *,
        available_gib: float,
        num_layers: int,
        kv_heads: int,
        head_dim: int,
        avg_sequence_length: int = 2048,
        fallback_dtype: KVCacheDType = KVCacheDType.BF16,
    ) -> int:
        """Legacy-friendly planning helper for concurrency sizing."""
        planning = self.estimate_capacity_from_budget(
            num_layers=num_layers,
            kv_heads=kv_heads,
            head_dim=head_dim,
            available_bytes=int(float(available_gib) * BYTES_PER_GIB),
            fallback_dtype=fallback_dtype,
        )
        total_tokens_capacity = int(planning["total_tokens_capacity"])
        return max(1, total_tokens_capacity // max(1, avg_sequence_length))

    def evaluate_compatibility(
        self,
        *,
        runtime: ServingRuntime | None = None,
        hardware_cache_supported: bool | None = None,
    ) -> KVCacheCompatibilityReport:
        resolved_runtime = runtime or self.runtime
        runtime_supported = resolved_runtime == ServingRuntime.VLLM
        issues: list[str] = []
        warnings: list[str] = []

        if self.state == KVCacheState.DISABLED:
            return KVCacheCompatibilityReport(
                runtime=resolved_runtime,
                state=KVCacheState.DISABLED,
                exportable=True,
                runtime_supported=runtime_supported,
                hardware_validated=hardware_cache_supported,
                warnings=[self.disabled_reason] if self.disabled_reason else [],
            )

        if self.state == KVCacheState.UNSUPPORTED:
            issues.append(self.unsupported_reason or "KV cache policy marked unsupported")
            return KVCacheCompatibilityReport(
                runtime=resolved_runtime,
                state=KVCacheState.UNSUPPORTED,
                exportable=False,
                runtime_supported=runtime_supported,
                hardware_validated=hardware_cache_supported,
                issues=issues,
            )

        if not runtime_supported:
            issues.append("Enabled KV-cache policy currently exports only to vLLM runtime adapters")

        if hardware_cache_supported is False:
            issues.append("Hardware/runtime probe reported cache mode unavailable")
        elif hardware_cache_supported is None:
            warnings.append("Hardware/runtime cache support is not validated by this policy model")

        if self.dtype == KVCacheDType.AUTO:
            warnings.append("KV-cache dtype is AUTO; runtime adapter must resolve the concrete dtype")

        if self.layout == KVCacheLayout.PAGED:
            warnings.append("Paged layout metadata excludes runtime page-table / allocator overhead")

        return KVCacheCompatibilityReport(
            runtime=resolved_runtime,
            state=KVCacheState.ENABLED if not issues else KVCacheState.UNSUPPORTED,
            exportable=not issues,
            runtime_supported=runtime_supported,
            hardware_validated=hardware_cache_supported,
            issues=issues,
            warnings=warnings,
        )

    def to_runtime_payload(self) -> dict[str, Any]:
        """Deterministic typed export for serving adapters."""
        return {
            "runtime": self.runtime.value,
            "state": self.state.value,
            "dtype": self.dtype.value,
            "layout": self.layout.value,
            "eviction_policy": self.eviction_policy.value,
            "max_context_tokens": self.max_context_tokens,
            "max_concurrent_sequences": self.max_concurrent_sequences,
            "block_size_tokens": self.block_size_tokens,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_cache_bytes": self.max_cache_bytes,
            "enable_prefix_caching": self.enable_prefix_caching,
            "enable_sliding_window_reuse": self.enable_sliding_window_reuse,
            "allow_cpu_spill": self.allow_cpu_spill,
            "disabled_reason": self.disabled_reason,
            "unsupported_reason": self.unsupported_reason,
            "notes": list(self.notes),
            "unsupported_boundaries": [boundary.value for boundary in self.explicit_boundaries],
        }

    def to_vllm_kwargs(self) -> dict[str, Any]:
        """Export kwargs for a future vLLM adapter."""
        if self.state != KVCacheState.ENABLED:
            return {
                "kv_cache_dtype": None,
                "block_size": None,
                "gpu_memory_utilization": None,
                "enable_prefix_caching": False,
                "kv_cache_policy": self.to_runtime_payload(),
            }

        return {
            "kv_cache_dtype": self.dtype.value if self.dtype != KVCacheDType.AUTO else "auto",
            "block_size": self.block_size_tokens if self.layout == KVCacheLayout.PAGED else None,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_context_tokens,
            "enable_prefix_caching": self.enable_prefix_caching,
            "kv_cache_policy": self.to_runtime_payload(),
        }

    def to_vllm_args(self) -> dict[str, Any]:
        """Backward-compatible alias retained for older serving helpers."""
        return self.to_vllm_kwargs()

    def to_benchmark_metadata(self) -> dict[str, Any]:
        """Compact export for throughput benchmarks and experiment logging."""
        return {
            "runtime": self.runtime.value,
            "kv_cache_state": self.state.value,
            "kv_cache_dtype": self.dtype.value,
            "kv_cache_layout": self.layout.value,
            "eviction_policy": self.eviction_policy.value,
            "max_context_tokens": self.max_context_tokens,
            "max_concurrent_sequences": self.max_concurrent_sequences,
            "block_size_tokens": self.block_size_tokens,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_cache_bytes": self.max_cache_bytes,
            "enable_prefix_caching": self.enable_prefix_caching,
            "enable_sliding_window_reuse": self.enable_sliding_window_reuse,
            "allow_cpu_spill": self.allow_cpu_spill,
        }

    def to_canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(by_alias=False, exclude_none=False, mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def disabled(
        cls,
        *,
        runtime: ServingRuntime = ServingRuntime.VLLM,
        reason: str = "KV cache explicitly disabled",
        notes: tuple[str, ...] = (),
    ) -> "KVCachePolicy":
        return cls(
            runtime=runtime,
            state=KVCacheState.DISABLED,
            dtype=KVCacheDType.AUTO,
            layout=KVCacheLayout.PAGED,
            eviction_policy=KVCacheEvictionPolicy.NONE,
            block_size_tokens=16,
            disabled_reason=reason,
            notes=notes,
        )

    @classmethod
    def unsupported(
        cls,
        *,
        runtime: ServingRuntime,
        reason: str,
        notes: tuple[str, ...] = (),
    ) -> "KVCachePolicy":
        return cls(
            runtime=runtime,
            state=KVCacheState.UNSUPPORTED,
            dtype=KVCacheDType.AUTO,
            layout=KVCacheLayout.PAGED,
            eviction_policy=KVCacheEvictionPolicy.NONE,
            block_size_tokens=16,
            unsupported_reason=reason,
            notes=notes,
        )

    @classmethod
    def vllm_paged(
        cls,
        *,
        dtype: KVCacheDType = KVCacheDType.AUTO,
        max_context_tokens: int = 4096,
        max_concurrent_sequences: int = 32,
        block_size_tokens: int = 16,
        gpu_memory_utilization: float | None = None,
        max_cache_bytes: int | None = None,
        enable_prefix_caching: bool = False,
        enable_sliding_window_reuse: bool = False,
        allow_cpu_spill: bool = False,
        notes: tuple[str, ...] = (),
    ) -> "KVCachePolicy":
        return cls(
            runtime=ServingRuntime.VLLM,
            state=KVCacheState.ENABLED,
            dtype=dtype,
            layout=KVCacheLayout.PAGED,
            eviction_policy=KVCacheEvictionPolicy.RUNTIME_MANAGED,
            max_context_tokens=max_context_tokens,
            max_concurrent_sequences=max_concurrent_sequences,
            block_size_tokens=block_size_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            max_cache_bytes=max_cache_bytes,
            enable_prefix_caching=enable_prefix_caching,
            enable_sliding_window_reuse=enable_sliding_window_reuse,
            allow_cpu_spill=allow_cpu_spill,
            notes=notes,
        )


__all__ = [
    "BYTES_PER_GIB",
    "BYTES_PER_MIB",
    "KVCacheBoundaryType",
    "KVCacheCompatibilityReport",
    "KVCacheDType",
    "KVCacheEvictionPolicy",
    "KVCacheFootprintEstimate",
    "KVCacheLayout",
    "KVCachePolicy",
    "KVCacheSizingInputs",
    "KVCacheState",
    "ServingRuntime",
]