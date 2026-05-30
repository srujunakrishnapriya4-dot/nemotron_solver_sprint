from __future__ import annotations

from enum import Enum
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ServingRuntime(str, Enum):
    """Serving runtimes this policy can reason about explicitly."""

    VLLM = "vllm"
    OTHER = "other"


class SpeculativeDecodeState(str, Enum):
    """Explicit top-level state for speculative decoding."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    UNSUPPORTED = "unsupported"


class DraftModelSource(str, Enum):
    """Where the draft path comes from."""

    EXTERNAL_MODEL = "external_model"
    RUNTIME_BUILTIN = "runtime_builtin"
    UNKNOWN = "unknown"


class VerificationMode(str, Enum):
    """High-level verification style only, not backend logic."""

    TARGET_MODEL = "target_model"
    RUNTIME_MANAGED = "runtime_managed"


class SchedulingMode(str, Enum):
    """Logical scheduling intent only."""

    SEQUENTIAL = "sequential"
    CONTINUOUS_BATCHING = "continuous_batching"
    RUNTIME_MANAGED = "runtime_managed"


class RuntimeBoundaryType(str, Enum):
    """Explicit unsupported/runtime-dependent boundaries."""

    TOKEN_ACCEPTANCE_ENGINE = "token_acceptance_engine"
    DRAFT_TARGET_SYNCHRONIZATION = "draft_target_synchronization"
    BACKEND_SAMPLER_INTEGRATION = "backend_sampler_integration"
    CONTINUOUS_BATCHING_SCHEDULER = "continuous_batching_scheduler"
    PAGED_KV_INTERACTION = "paged_kv_interaction"
    KERNEL_LEVEL_OPTIMIZATION = "kernel_level_optimization"


class DraftModelSpec(BaseModel):
    """Typed description of the draft path.

    This is intentionally policy-only. It does not load or validate a real model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: DraftModelSource = DraftModelSource.UNKNOWN
    model_name: str | None = None
    max_draft_tokens_per_step: int = Field(default=4, ge=1)
    expected_smaller_or_faster: bool = False
    notes: tuple[str, ...] = ()

    @field_validator("model_name")
    @classmethod
    def _normalize_model_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @model_validator(mode="after")
    def _validate_model_name(self) -> "DraftModelSpec":
        if self.source == DraftModelSource.EXTERNAL_MODEL and not self.model_name:
            raise ValueError("draft.model_name is required when source='external_model'")
        if self.source != DraftModelSource.EXTERNAL_MODEL and self.model_name is not None:
            raise ValueError("draft.model_name may only be set when source='external_model'")
        return self


class SpeculativeDecodeCompatibilityReport(BaseModel):
    """Explicit compatibility result for serving/runtime consumers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime: ServingRuntime
    state: SpeculativeDecodeState
    exportable: bool
    runtime_supported: bool
    backend_validated: bool | None = None
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def is_usable(self) -> bool:
        return self.exportable and self.state == SpeculativeDecodeState.ENABLED


class SpeculativeDecodePolicy(BaseModel):
    """Typed speculative decoding policy/config contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime: ServingRuntime = ServingRuntime.VLLM
    state: SpeculativeDecodeState = SpeculativeDecodeState.DISABLED

    draft: DraftModelSpec = Field(default_factory=DraftModelSpec)
    verification_mode: VerificationMode = VerificationMode.RUNTIME_MANAGED
    scheduling_mode: SchedulingMode = SchedulingMode.RUNTIME_MANAGED

    require_smaller_or_faster_draft: bool = True
    allow_disable_fallback: bool = True

    max_speculative_tokens_per_step: int = Field(default=4, ge=1)
    min_expected_acceptance_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    max_target_batch_size_for_speculation: int | None = Field(default=None, ge=1)
    target_only_when_batching_enabled: bool = True

    disabled_reason: str | None = None
    unsupported_reason: str | None = None
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_consistency(self) -> "SpeculativeDecodePolicy":
        if self.state == SpeculativeDecodeState.DISABLED:
            if self.unsupported_reason is not None:
                raise ValueError("unsupported_reason must be null when state is disabled")
        elif self.state == SpeculativeDecodeState.UNSUPPORTED:
            if not self.unsupported_reason:
                raise ValueError("unsupported_reason is required when state is unsupported")
        elif self.state == SpeculativeDecodeState.ENABLED:
            if self.runtime != ServingRuntime.VLLM:
                raise ValueError(
                    "enabled speculative decoding export is currently supported only for runtime='vllm'"
                )
            if self.unsupported_reason is not None:
                raise ValueError("unsupported_reason must be null when state is enabled")
            if self.max_speculative_tokens_per_step < self.draft.max_draft_tokens_per_step:
                raise ValueError(
                    "max_speculative_tokens_per_step must be >= draft.max_draft_tokens_per_step"
                )
            if self.require_smaller_or_faster_draft and not self.draft.expected_smaller_or_faster:
                raise ValueError(
                    "enabled speculative decoding requires a draft path marked expected_smaller_or_faster"
                )

        if self.state != SpeculativeDecodeState.DISABLED and self.disabled_reason is not None:
            raise ValueError("disabled_reason may only be set when state is disabled")

        if self.min_expected_acceptance_rate is not None and self.state != SpeculativeDecodeState.ENABLED:
            raise ValueError(
                "min_expected_acceptance_rate may only be set when speculative decoding is enabled"
            )

        if (
            self.target_only_when_batching_enabled
            and self.state == SpeculativeDecodeState.ENABLED
            and self.scheduling_mode == SchedulingMode.SEQUENTIAL
        ):
            raise ValueError(
                "target_only_when_batching_enabled is incompatible with sequential scheduling"
            )

        return self

    @property
    def enabled(self) -> bool:
        return self.state == SpeculativeDecodeState.ENABLED

    @property
    def explicit_boundaries(self) -> tuple[RuntimeBoundaryType, ...]:
        return (
            RuntimeBoundaryType.TOKEN_ACCEPTANCE_ENGINE,
            RuntimeBoundaryType.DRAFT_TARGET_SYNCHRONIZATION,
            RuntimeBoundaryType.BACKEND_SAMPLER_INTEGRATION,
            RuntimeBoundaryType.CONTINUOUS_BATCHING_SCHEDULER,
            RuntimeBoundaryType.PAGED_KV_INTERACTION,
            RuntimeBoundaryType.KERNEL_LEVEL_OPTIMIZATION,
        )

    def evaluate_compatibility(
        self,
        *,
        runtime: ServingRuntime | None = None,
        backend_speculation_supported: bool | None = None,
        batching_enabled: bool | None = None,
    ) -> SpeculativeDecodeCompatibilityReport:
        resolved_runtime = runtime or self.runtime
        runtime_supported = resolved_runtime == ServingRuntime.VLLM
        issues: list[str] = []
        warnings: list[str] = []

        if self.state == SpeculativeDecodeState.DISABLED:
            return SpeculativeDecodeCompatibilityReport(
                runtime=resolved_runtime,
                state=SpeculativeDecodeState.DISABLED,
                exportable=True,
                runtime_supported=runtime_supported,
                backend_validated=backend_speculation_supported,
                warnings=[self.disabled_reason] if self.disabled_reason else [],
            )

        if self.state == SpeculativeDecodeState.UNSUPPORTED:
            issues.append(self.unsupported_reason or "Speculative decoding policy marked unsupported")
            return SpeculativeDecodeCompatibilityReport(
                runtime=resolved_runtime,
                state=SpeculativeDecodeState.UNSUPPORTED,
                exportable=False,
                runtime_supported=runtime_supported,
                backend_validated=backend_speculation_supported,
                issues=issues,
            )

        if not runtime_supported:
            issues.append("Enabled speculative decoding policy currently exports only to vLLM runtime adapters")

        if backend_speculation_supported is False:
            issues.append("Backend/runtime probe reported speculative decoding unavailable")
        elif backend_speculation_supported is None:
            warnings.append("Backend speculative decoding support is not validated by this policy model")

        if self.target_only_when_batching_enabled:
            if batching_enabled is False:
                issues.append("Speculative decoding policy requires batching-enabled execution")
            elif batching_enabled is None:
                warnings.append("Batching mode is not validated; policy assumes batching-enabled execution")

        if self.draft.source == DraftModelSource.UNKNOWN:
            warnings.append("Draft model source is UNKNOWN; runtime must resolve the actual draft path")

        if self.min_expected_acceptance_rate is not None:
            warnings.append(
                "min_expected_acceptance_rate is a policy threshold only; this module does not measure acceptance"
            )

        return SpeculativeDecodeCompatibilityReport(
            runtime=resolved_runtime,
            state=SpeculativeDecodeState.ENABLED if not issues else SpeculativeDecodeState.UNSUPPORTED,
            exportable=not issues,
            runtime_supported=runtime_supported,
            backend_validated=backend_speculation_supported,
            issues=issues,
            warnings=warnings,
        )

    def should_enable(
        self,
        *,
        runtime: ServingRuntime | None = None,
        backend_speculation_supported: bool | None = None,
        batching_enabled: bool | None = None,
    ) -> bool:
        report = self.evaluate_compatibility(
            runtime=runtime,
            backend_speculation_supported=backend_speculation_supported,
            batching_enabled=batching_enabled,
        )
        return report.is_usable()

    def validate(self, num_gpus: int) -> bool:
        """Conservative legacy-style safety helper.

        This does not prove backend support. It only preserves an explicit,
        conservative enablement rule for external-draft configurations.
        """
        if self.state != SpeculativeDecodeState.ENABLED:
            return True
        if num_gpus < 2 and self.draft.source == DraftModelSource.EXTERNAL_MODEL:
            return False
        return True

    def expected_speedup(self) -> str:
        """Planning-only estimate string, never a performance guarantee."""
        if self.state != SpeculativeDecodeState.ENABLED:
            return "1.0x (disabled)"
        if self.draft.expected_smaller_or_faster:
            return "runtime-dependent (draft path expected smaller/faster)"
        return "runtime-dependent (no explicit draft speed advantage declared)"

    def to_runtime_payload(self) -> dict[str, Any]:
        """Deterministic typed export for serving adapters."""
        return {
            "runtime": self.runtime.value,
            "state": self.state.value,
            "draft": {
                "source": self.draft.source.value,
                "model_name": self.draft.model_name,
                "max_draft_tokens_per_step": self.draft.max_draft_tokens_per_step,
                "expected_smaller_or_faster": self.draft.expected_smaller_or_faster,
                "notes": list(self.draft.notes),
            },
            "verification_mode": self.verification_mode.value,
            "scheduling_mode": self.scheduling_mode.value,
            "require_smaller_or_faster_draft": self.require_smaller_or_faster_draft,
            "allow_disable_fallback": self.allow_disable_fallback,
            "max_speculative_tokens_per_step": self.max_speculative_tokens_per_step,
            "min_expected_acceptance_rate": self.min_expected_acceptance_rate,
            "max_target_batch_size_for_speculation": self.max_target_batch_size_for_speculation,
            "target_only_when_batching_enabled": self.target_only_when_batching_enabled,
            "disabled_reason": self.disabled_reason,
            "unsupported_reason": self.unsupported_reason,
            "notes": list(self.notes),
            "unsupported_boundaries": [boundary.value for boundary in self.explicit_boundaries],
        }

    def to_vllm_kwargs(self) -> dict[str, Any]:
        """Export kwargs-like settings for a future vLLM adapter."""
        if self.state != SpeculativeDecodeState.ENABLED:
            return {
                "speculative_config": self.to_runtime_payload(),
                "speculative_disabled": True,
                "num_speculative_tokens": None,
                "speculative_model": None,
            }

        return {
            "speculative_config": self.to_runtime_payload(),
            "speculative_disabled": False,
            "num_speculative_tokens": self.max_speculative_tokens_per_step,
            "speculative_model": self.draft.model_name
            if self.draft.source == DraftModelSource.EXTERNAL_MODEL
            else None,
        }

    def to_vllm_args(self) -> dict[str, Any]:
        """Backward-compatible alias retained for older serving helpers."""
        return self.to_vllm_kwargs()

    def to_benchmark_metadata(self) -> dict[str, Any]:
        """Compact export for throughput benchmarking and experiment tracking."""
        return {
            "runtime": self.runtime.value,
            "speculative_state": self.state.value,
            "draft_source": self.draft.source.value,
            "draft_model_name": self.draft.model_name,
            "draft_expected_smaller_or_faster": self.draft.expected_smaller_or_faster,
            "verification_mode": self.verification_mode.value,
            "scheduling_mode": self.scheduling_mode.value,
            "max_speculative_tokens_per_step": self.max_speculative_tokens_per_step,
            "min_expected_acceptance_rate": self.min_expected_acceptance_rate,
            "max_target_batch_size_for_speculation": self.max_target_batch_size_for_speculation,
            "target_only_when_batching_enabled": self.target_only_when_batching_enabled,
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
        reason: str = "Speculative decoding explicitly disabled",
        notes: tuple[str, ...] = (),
    ) -> "SpeculativeDecodePolicy":
        return cls(
            runtime=runtime,
            state=SpeculativeDecodeState.DISABLED,
            draft=DraftModelSpec(source=DraftModelSource.UNKNOWN, max_draft_tokens_per_step=4),
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
    ) -> "SpeculativeDecodePolicy":
        return cls(
            runtime=runtime,
            state=SpeculativeDecodeState.UNSUPPORTED,
            draft=DraftModelSpec(source=DraftModelSource.UNKNOWN, max_draft_tokens_per_step=4),
            unsupported_reason=reason,
            notes=notes,
        )

    @classmethod
    def vllm_external_draft(
        cls,
        *,
        draft_model_name: str,
        max_draft_tokens_per_step: int = 4,
        max_speculative_tokens_per_step: int | None = None,
        min_expected_acceptance_rate: float | None = None,
        max_target_batch_size_for_speculation: int | None = None,
        require_smaller_or_faster_draft: bool = True,
        expected_smaller_or_faster: bool = True,
        scheduling_mode: SchedulingMode = SchedulingMode.CONTINUOUS_BATCHING,
        verification_mode: VerificationMode = VerificationMode.RUNTIME_MANAGED,
        target_only_when_batching_enabled: bool = True,
        allow_disable_fallback: bool = True,
        notes: tuple[str, ...] = (),
    ) -> "SpeculativeDecodePolicy":
        resolved_max_speculative = (
            max_speculative_tokens_per_step
            if max_speculative_tokens_per_step is not None
            else max_draft_tokens_per_step
        )
        return cls(
            runtime=ServingRuntime.VLLM,
            state=SpeculativeDecodeState.ENABLED,
            draft=DraftModelSpec(
                source=DraftModelSource.EXTERNAL_MODEL,
                model_name=draft_model_name,
                max_draft_tokens_per_step=max_draft_tokens_per_step,
                expected_smaller_or_faster=expected_smaller_or_faster,
            ),
            verification_mode=verification_mode,
            scheduling_mode=scheduling_mode,
            require_smaller_or_faster_draft=require_smaller_or_faster_draft,
            allow_disable_fallback=allow_disable_fallback,
            max_speculative_tokens_per_step=resolved_max_speculative,
            min_expected_acceptance_rate=min_expected_acceptance_rate,
            max_target_batch_size_for_speculation=max_target_batch_size_for_speculation,
            target_only_when_batching_enabled=target_only_when_batching_enabled,
            notes=notes,
        )


__all__ = [
    "DraftModelSource",
    "DraftModelSpec",
    "RuntimeBoundaryType",
    "SchedulingMode",
    "ServingRuntime",
    "SpeculativeDecodeCompatibilityReport",
    "SpeculativeDecodePolicy",
    "SpeculativeDecodeState",
    "VerificationMode",
]