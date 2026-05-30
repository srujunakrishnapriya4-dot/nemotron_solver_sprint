from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import importlib
import importlib.util
import math
import time
import uuid
from typing import Any, Iterable, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.serving.fp8_config import FP8ServingConfig, FP8SupportState, ServingRuntime as FP8Runtime
from src.serving.kv_cache import KVCachePolicy, KVCacheState, ServingRuntime as KVRuntime
from src.serving.speculative_decode import (
    SchedulingMode,
    ServingRuntime as SpecRuntime,
    SpeculativeDecodePolicy,
    SpeculativeDecodeState,
)


class RuntimeState(str, Enum):
    UNINITIALIZED = "uninitialized"
    UNAVAILABLE = "unavailable"
    INITIALIZED = "initialized"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


class RequestStatus(str, Enum):
    OK = "ok"
    UNAVAILABLE = "unavailable"
    INVALID_REQUEST = "invalid_request"
    NOT_STARTED = "not_started"
    BACKEND_ERROR = "backend_error"


class RuntimeBoundaryType(str, Enum):
    BACKEND_IMPORT = "backend_import"
    ENGINE_CONSTRUCTOR = "engine_constructor"
    SAMPLING_PARAMS_SCHEMA = "sampling_params_schema"
    BACKEND_BATCHING = "backend_batching"
    GPU_RUNTIME = "gpu_runtime"
    NETWORK_SERVICE = "network_service"


class GenerationParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_tokens: int = Field(default=512, ge=1)
    max_new_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    repetition_penalty: float | None = Field(default=None, gt=0.0)
    stop: tuple[str, ...] = ()
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logprobs: int | None = Field(default=None, ge=0)
    n: int = Field(default=1, ge=1)
    seed: int | None = None

    @model_validator(mode="after")
    def _normalize_legacy_max_tokens(self) -> "GenerationParameters":
        if self.max_new_tokens is not None and self.max_new_tokens != self.max_tokens:
            object.__setattr__(self, "max_tokens", self.max_new_tokens)
        return self

    def to_backend_kwargs(self) -> dict[str, Any]:
        data = self.model_dump(exclude_none=True)
        data.pop("max_new_tokens", None)
        if "stop" in data:
            data["stop"] = list(self.stop)
        return data


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex[:16]}")
    prompt: str = Field(min_length=1)
    parameters: GenerationParameters = Field(default_factory=GenerationParameters)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _lift_legacy_flat_generation_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        legacy_keys = {
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "max_tokens",
            "max_new_tokens",
            "logprobs",
            "n",
            "seed",
        }
        if "parameters" not in data:
            params: dict[str, Any] = {}
            for key in list(legacy_keys):
                if key in data:
                    params[key] = data.pop(key)
            if params:
                data["parameters"] = params
        return data

    @field_validator("prompt")
    @classmethod
    def _validate_prompt(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("prompt must not be empty")
        return cleaned


class GenerationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    finish_reason: str | None = None
    token_count: int | None = None
    mean_token_entropy: float | None = None
    raw: dict[str, Any] | None = None


class GenerationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    status: RequestStatus
    model_name: str
    runtime_state: RuntimeState
    outputs: tuple[GenerationOutput, ...] = ()
    error_code: str | None = None
    error_message: str | None = None
    latency_ms: int | None = None
    backend_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == RequestStatus.OK and bool(self.outputs)


class StartResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool
    runtime_state: RuntimeState
    backend_name: str | None = None
    model_name: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class StopResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool
    runtime_state: RuntimeState
    message: str
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class RuntimeProbe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    vllm_importable: bool
    runtime_state: RuntimeState
    backend_name: str | None = None
    issues: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    boundaries: tuple[RuntimeBoundaryType, ...] = (
        RuntimeBoundaryType.BACKEND_IMPORT,
        RuntimeBoundaryType.ENGINE_CONSTRUCTOR,
        RuntimeBoundaryType.SAMPLING_PARAMS_SCHEMA,
        RuntimeBoundaryType.BACKEND_BATCHING,
        RuntimeBoundaryType.GPU_RUNTIME,
        RuntimeBoundaryType.NETWORK_SERVICE,
    )


class VLLMServerConfig(BaseModel):
    """Typed serving config for the local vLLM-backed generator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_path: str = Field(min_length=1)
    served_model_name: str | None = None

    tensor_parallel_size: int = Field(default=1, ge=1)
    gpu_memory_utilization: float = Field(default=0.9, gt=0.0, le=1.0)
    max_model_len: int | None = Field(default=None, ge=1)
    max_num_seqs: int | None = Field(default=None, ge=1)

    trust_remote_code: bool = False
    enforce_eager: bool = False
    dtype: str | None = None

    fp8: FP8ServingConfig = Field(default_factory=FP8ServingConfig.disabled)
    kv_cache: KVCachePolicy = Field(default_factory=KVCachePolicy.vllm_paged)
    speculative_decode: SpeculativeDecodePolicy = Field(default_factory=SpeculativeDecodePolicy.disabled)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model_path")
    @classmethod
    def _normalize_model_path(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("model_path must not be empty")
        return cleaned

    @field_validator("served_model_name")
    @classmethod
    def _normalize_served_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @model_validator(mode="after")
    def _validate_runtime_alignment(self) -> "VLLMServerConfig":
        if self.fp8.runtime != FP8Runtime.VLLM:
            raise ValueError("fp8 config must target runtime='vllm'")
        if self.kv_cache.runtime != KVRuntime.VLLM:
            raise ValueError("kv_cache policy must target runtime='vllm'")
        if self.speculative_decode.runtime != SpecRuntime.VLLM:
            raise ValueError("speculative_decode policy must target runtime='vllm'")
        return self

    @property
    def model_name(self) -> str:
        return self.served_model_name or self.model_path

    def to_engine_kwargs(self) -> dict[str, Any]:
        """Deterministic constructor kwargs candidate for the backend engine."""
        kwargs: dict[str, Any] = {
            "model": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": self.trust_remote_code,
            "enforce_eager": self.enforce_eager,
        }
        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len
        if self.max_num_seqs is not None:
            kwargs["max_num_seqs"] = self.max_num_seqs
        if self.dtype is not None:
            kwargs["dtype"] = self.dtype

        fp8_kwargs = self.fp8.to_vllm_kwargs()
        kv_kwargs = self.kv_cache.to_vllm_kwargs()
        spec_kwargs = self.speculative_decode.to_vllm_kwargs()

        for source in (fp8_kwargs, kv_kwargs, spec_kwargs):
            for key, value in source.items():
                if value is not None:
                    kwargs[key] = value
        return kwargs

    def to_runtime_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "trust_remote_code": self.trust_remote_code,
            "enforce_eager": self.enforce_eager,
            "dtype": self.dtype,
            "fp8": self.fp8.to_runtime_payload(),
            "kv_cache": self.kv_cache.to_runtime_payload(),
            "speculative_decode": self.speculative_decode.to_runtime_payload(),
            "metadata": self.metadata,
        }


@dataclass
class _BackendBundle:
    llm_cls: Any
    sampling_params_cls: Any
    backend_name: str = "vllm"


class _VLLMBackendAdapter:
    """Thin guarded adapter around a real vLLM backend."""

    def __init__(self, config: VLLMServerConfig) -> None:
        self._config = config
        self._bundle: _BackendBundle | None = None
        self._engine: Any | None = None

    @staticmethod
    def probe() -> RuntimeProbe:
        issues: list[str] = []
        warnings: list[str] = []

        importable = importlib.util.find_spec("vllm") is not None
        if not importable:
            issues.append("Python package 'vllm' is not importable in the current runtime")
            return RuntimeProbe(
                vllm_importable=False,
                runtime_state=RuntimeState.UNAVAILABLE,
                backend_name=None,
                issues=tuple(issues),
                warnings=tuple(warnings),
            )

        try:
            importlib.import_module("vllm")
        except Exception as exc:  # pragma: no cover - environment dependent
            issues.append(f"Importing 'vllm' failed: {exc}")
            return RuntimeProbe(
                vllm_importable=False,
                runtime_state=RuntimeState.UNAVAILABLE,
                backend_name=None,
                issues=tuple(issues),
                warnings=tuple(warnings),
            )

        return RuntimeProbe(
            vllm_importable=True,
            runtime_state=RuntimeState.INITIALIZED,
            backend_name="vllm",
            issues=tuple(issues),
            warnings=tuple(warnings),
        )

    def initialize(self) -> StartResult:
        probe = self.probe()
        diagnostics: dict[str, Any] = {
            "probe": probe.model_dump(mode="json"),
            "config": self._config.to_runtime_metadata(),
        }
        if not probe.vllm_importable:
            return StartResult(
                ok=False,
                runtime_state=RuntimeState.UNAVAILABLE,
                backend_name=None,
                model_name=self._config.model_name,
                error_code="runtime_unavailable",
                error_message="vLLM backend is not importable",
                diagnostics=diagnostics,
            )

        try:
            vllm_mod = importlib.import_module("vllm")
            llm_cls = getattr(vllm_mod, "LLM")
            sampling_params_cls = getattr(vllm_mod, "SamplingParams")
            self._bundle = _BackendBundle(
                llm_cls=llm_cls,
                sampling_params_cls=sampling_params_cls,
                backend_name="vllm",
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            return StartResult(
                ok=False,
                runtime_state=RuntimeState.ERROR,
                backend_name="vllm",
                model_name=self._config.model_name,
                error_code="backend_symbol_error",
                error_message=f"Failed to resolve vLLM symbols: {exc}",
                diagnostics=diagnostics,
            )

        engine_kwargs = self._config.to_engine_kwargs()
        diagnostics["engine_kwargs"] = engine_kwargs

        try:
            self._engine = self._bundle.llm_cls(**engine_kwargs)
        except Exception as exc:  # pragma: no cover - environment dependent
            return StartResult(
                ok=False,
                runtime_state=RuntimeState.ERROR,
                backend_name="vllm",
                model_name=self._config.model_name,
                error_code="engine_init_failed",
                error_message=f"vLLM engine initialization failed: {exc}",
                diagnostics=diagnostics,
            )

        return StartResult(
            ok=True,
            runtime_state=RuntimeState.RUNNING,
            backend_name="vllm",
            model_name=self._config.model_name,
            diagnostics=diagnostics,
        )

    def stop(self) -> StopResult:
        self._engine = None
        self._bundle = None
        return StopResult(
            ok=True,
            runtime_state=RuntimeState.STOPPED,
            message="Backend adapter released local engine references",
            diagnostics={},
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self._engine is None or self._bundle is None:
            return GenerationResponse(
                request_id=request.request_id,
                status=RequestStatus.NOT_STARTED,
                model_name=self._config.model_name,
                runtime_state=RuntimeState.UNINITIALIZED,
                error_code="not_started",
                error_message="Backend engine is not initialized",
                backend_name="vllm",
            )

        start_time = time.perf_counter()
        try:
            sampling_params = self._bundle.sampling_params_cls(**request.parameters.to_backend_kwargs())
        except Exception as exc:  # pragma: no cover - environment dependent
            return GenerationResponse(
                request_id=request.request_id,
                status=RequestStatus.INVALID_REQUEST,
                model_name=self._config.model_name,
                runtime_state=RuntimeState.RUNNING,
                error_code="sampling_params_error",
                error_message=f"Sampling parameter construction failed: {exc}",
                backend_name="vllm",
            )

        try:
            raw_outputs = self._engine.generate([request.prompt], sampling_params)
        except Exception as exc:  # pragma: no cover - environment dependent
            return GenerationResponse(
                request_id=request.request_id,
                status=RequestStatus.BACKEND_ERROR,
                model_name=self._config.model_name,
                runtime_state=RuntimeState.RUNNING,
                error_code="backend_generate_error",
                error_message=f"Backend generation failed: {exc}",
                backend_name="vllm",
            )

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        parsed_outputs: list[GenerationOutput] = []

        try:
            first = raw_outputs[0]
            for out in getattr(first, "outputs", []):
                mean_entropy = _extract_mean_token_entropy(getattr(out, "logprobs", None))
                parsed_outputs.append(
                    GenerationOutput(
                        text=getattr(out, "text", ""),
                        finish_reason=getattr(out, "finish_reason", None),
                        token_count=len(getattr(out, "token_ids", []) or []),
                        mean_token_entropy=mean_entropy,
                        raw={
                            "finish_reason": getattr(out, "finish_reason", None),
                            "token_ids": list(getattr(out, "token_ids", []) or []),
                            "logprobs_available": mean_entropy is not None,
                            "tool_calls": _extract_tool_calls(getattr(out, "tool_calls", None)),
                        },
                    )
                )
        except Exception as exc:  # pragma: no cover - environment dependent
            return GenerationResponse(
                request_id=request.request_id,
                status=RequestStatus.BACKEND_ERROR,
                model_name=self._config.model_name,
                runtime_state=RuntimeState.RUNNING,
                error_code="backend_parse_error",
                error_message=f"Backend output parsing failed: {exc}",
                latency_ms=latency_ms,
                backend_name="vllm",
            )

        return GenerationResponse(
            request_id=request.request_id,
            status=RequestStatus.OK,
            model_name=self._config.model_name,
            runtime_state=RuntimeState.RUNNING,
            outputs=tuple(parsed_outputs),
            latency_ms=latency_ms,
            backend_name="vllm",
            metadata={"num_outputs": len(parsed_outputs)},
        )


class VLLMClient:
    """Serving/runtime interface used by online orchestration."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        config: VLLMServerConfig | None = None,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        trust_remote_code: bool = False,
        enforce_eager: bool = False,
        dtype: str | None = None,
        fp8: FP8ServingConfig | None = None,
        kv_cache: KVCachePolicy | None = None,
        speculative_decode: SpeculativeDecodePolicy | None = None,
        served_model_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        base_url: str | None = None,
        timeout_sec: float | None = None,
        max_retries: int | None = None,
        model_name: str | None = None,
    ) -> None:
        normalized_metadata = dict(metadata or {})
        if base_url is not None:
            normalized_metadata["base_url"] = base_url
        if timeout_sec is not None:
            normalized_metadata["timeout_sec"] = timeout_sec
        if max_retries is not None:
            normalized_metadata["max_retries"] = max_retries

        effective_served_name = served_model_name or model_name

        if config is None:
            if model_path is None:
                raise ValueError("Either config or model_path must be provided")
            config = VLLMServerConfig(
                model_path=model_path,
                served_model_name=effective_served_name,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                trust_remote_code=trust_remote_code,
                enforce_eager=enforce_eager,
                dtype=dtype,
                fp8=fp8 or FP8ServingConfig.disabled(),
                kv_cache=kv_cache or KVCachePolicy.vllm_paged(),
                speculative_decode=speculative_decode or SpeculativeDecodePolicy.disabled(),
                metadata=normalized_metadata,
            )

        self._config = config
        self._runtime_state = RuntimeState.UNINITIALIZED
        self._adapter = _VLLMBackendAdapter(config)
        self._last_start_result: StartResult | None = None

    async def __aenter__(self) -> "VLLMClient":
        start_result = self.start()
        if not start_result.ok:
            raise RuntimeError(start_result.error_message or "Failed to start vLLM client")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.stop()

    @property
    def config(self) -> VLLMServerConfig:
        return self._config

    @property
    def runtime_state(self) -> RuntimeState:
        return self._runtime_state

    def probe_runtime(self) -> RuntimeProbe:
        return self._adapter.probe()

    def _config_compatibility(self, batching_enabled: bool | None = None) -> dict[str, Any]:
        fp8_report = self._config.fp8.evaluate_compatibility(runtime=FP8Runtime.VLLM)
        kv_report = self._config.kv_cache.evaluate_compatibility(runtime=KVRuntime.VLLM)
        spec_report = self._config.speculative_decode.evaluate_compatibility(
            runtime=SpecRuntime.VLLM,
            batching_enabled=batching_enabled
            if batching_enabled is not None
            else self._config.speculative_decode.scheduling_mode != SchedulingMode.SEQUENTIAL,
        )
        return {
            "fp8": fp8_report.model_dump(mode="json"),
            "kv_cache": kv_report.model_dump(mode="json"),
            "speculative_decode": spec_report.model_dump(mode="json"),
        }

    def initialize(self) -> StartResult:
        compatibility = self._config_compatibility()
        fp8_state = compatibility["fp8"]["state"]
        kv_state = compatibility["kv_cache"]["state"]
        spec_state = compatibility["speculative_decode"]["state"]

        blocking_issues: list[str] = []
        if fp8_state == FP8SupportState.UNSUPPORTED.value and not self._config.fp8.allow_fallback_to_bf16:
            blocking_issues.append("FP8 configuration is unsupported and fallback is disabled")
        if kv_state == KVCacheState.UNSUPPORTED.value:
            blocking_issues.append("KV-cache policy is unsupported for runtime export")
        if (
            spec_state == SpeculativeDecodeState.UNSUPPORTED.value
            and not self._config.speculative_decode.allow_disable_fallback
        ):
            blocking_issues.append("Speculative decoding policy is unsupported and disable-fallback is false")

        if blocking_issues:
            self._runtime_state = RuntimeState.ERROR
            result = StartResult(
                ok=False,
                runtime_state=self._runtime_state,
                backend_name=None,
                model_name=self._config.model_name,
                error_code="config_incompatible",
                error_message="; ".join(blocking_issues),
                diagnostics={
                    "compatibility": compatibility,
                    "config": self._config.to_runtime_metadata(),
                },
            )
            self._last_start_result = result
            return result

        result = self._adapter.initialize()
        self._runtime_state = result.runtime_state
        merged_diag = dict(result.diagnostics)
        merged_diag["compatibility"] = compatibility
        result = StartResult(
            ok=result.ok,
            runtime_state=result.runtime_state,
            backend_name=result.backend_name,
            model_name=result.model_name,
            error_code=result.error_code,
            error_message=result.error_message,
            diagnostics=merged_diag,
        )
        self._last_start_result = result
        return result

    def start(self) -> StartResult:
        return self.initialize()

    def stop(self) -> StopResult:
        stop_result = self._adapter.stop()
        self._runtime_state = stop_result.runtime_state
        return stop_result

    def health(self) -> RuntimeProbe:
        probe = self.probe_runtime()
        if self._runtime_state == RuntimeState.RUNNING and probe.vllm_importable:
            return RuntimeProbe(
                vllm_importable=True,
                runtime_state=RuntimeState.RUNNING,
                backend_name="vllm",
                issues=probe.issues,
                warnings=probe.warnings,
                boundaries=probe.boundaries,
            )
        if self._runtime_state == RuntimeState.UNAVAILABLE:
            return RuntimeProbe(
                vllm_importable=probe.vllm_importable,
                runtime_state=RuntimeState.UNAVAILABLE,
                backend_name=probe.backend_name,
                issues=probe.issues,
                warnings=probe.warnings,
                boundaries=probe.boundaries,
            )
        return RuntimeProbe(
            vllm_importable=probe.vllm_importable,
            runtime_state=self._runtime_state if self._runtime_state != RuntimeState.UNINITIALIZED else probe.runtime_state,
            backend_name=probe.backend_name,
            issues=probe.issues,
            warnings=probe.warnings,
            boundaries=probe.boundaries,
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self._runtime_state == RuntimeState.UNAVAILABLE:
            return GenerationResponse(
                request_id=request.request_id,
                status=RequestStatus.UNAVAILABLE,
                model_name=self._config.model_name,
                runtime_state=RuntimeState.UNAVAILABLE,
                error_code="runtime_unavailable",
                error_message="Serving runtime is unavailable in the current environment",
                backend_name=None,
                metadata={"compatibility": self._config_compatibility()},
            )
        if self._runtime_state not in {RuntimeState.RUNNING, RuntimeState.INITIALIZED}:
            return GenerationResponse(
                request_id=request.request_id,
                status=RequestStatus.NOT_STARTED,
                model_name=self._config.model_name,
                runtime_state=self._runtime_state,
                error_code="not_started",
                error_message="Serving runtime has not been started",
                backend_name=None,
            )
        response = self._adapter.generate(request)
        if response.status == RequestStatus.BACKEND_ERROR:
            self._runtime_state = RuntimeState.ERROR
        return response

    def generate_many(self, requests: Iterable[GenerationRequest]) -> tuple[GenerationResponse, ...]:
        return tuple(self.generate(request) for request in requests)

    async def batch_generate(
        self,
        requests: Iterable[GenerationRequest],
        max_concurrent: int = 32,
    ) -> tuple[GenerationResponse, ...]:
        # The local adapter path is synchronous; this async wrapper preserves
        # compatibility with older call sites without pretending backend batching.
        del max_concurrent
        return self.generate_many(requests)

    def unavailable_response(self, request: GenerationRequest, *, reason: str) -> GenerationResponse:
        return GenerationResponse(
            request_id=request.request_id,
            status=RequestStatus.UNAVAILABLE,
            model_name=self._config.model_name,
            runtime_state=RuntimeState.UNAVAILABLE,
            error_code="runtime_unavailable",
            error_message=reason,
            backend_name=None,
            metadata={"compatibility": self._config_compatibility()},
        )

    def describe_runtime(self) -> dict[str, Any]:
        return {
            "runtime_state": self._runtime_state.value,
            "probe": self.probe_runtime().model_dump(mode="json"),
            "config": self._config.to_runtime_metadata(),
            "last_start_result": self._last_start_result.model_dump(mode="json") if self._last_start_result else None,
        }


class VLLMServer:
    """Utility-only launch command builder for an external vLLM server process.

    This class does not start or manage the process itself. It only centralizes
    a deterministic command construction surface.
    """

    @staticmethod
    def build_launch_command(
        *,
        model_path: str,
        host: str = "127.0.0.1",
        port: int = 8000,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        trust_remote_code: bool = False,
        enforce_eager: bool = False,
        dtype: str | None = None,
        fp8: FP8ServingConfig | None = None,
        kv_cache: KVCachePolicy | None = None,
        speculative_decode: SpeculativeDecodePolicy | None = None,
    ) -> list[str]:
        config = VLLMServerConfig(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            trust_remote_code=trust_remote_code,
            enforce_eager=enforce_eager,
            dtype=dtype,
            fp8=fp8 or FP8ServingConfig.disabled(),
            kv_cache=kv_cache or KVCachePolicy.vllm_paged(),
            speculative_decode=speculative_decode or SpeculativeDecodePolicy.disabled(),
        )

        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            config.model_path,
            "--host",
            host,
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(config.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(config.gpu_memory_utilization),
        ]
        if config.max_model_len is not None:
            cmd.extend(["--max-model-len", str(config.max_model_len)])
        if config.max_num_seqs is not None:
            cmd.extend(["--max-num-seqs", str(config.max_num_seqs)])
        if config.trust_remote_code:
            cmd.append("--trust-remote-code")
        if config.enforce_eager:
            cmd.append("--enforce-eager")
        if config.dtype is not None:
            cmd.extend(["--dtype", config.dtype])

        engine_kwargs = config.to_engine_kwargs()
        if engine_kwargs.get("quantization") is not None:
            cmd.extend(["--quantization", str(engine_kwargs["quantization"])])
        if engine_kwargs.get("kv_cache_dtype") is not None:
            cmd.extend(["--kv-cache-dtype", str(engine_kwargs["kv_cache_dtype"])])
        if engine_kwargs.get("block_size") is not None:
            cmd.extend(["--block-size", str(engine_kwargs["block_size"])])
        if engine_kwargs.get("enable_prefix_caching"):
            cmd.append("--enable-prefix-caching")
        if engine_kwargs.get("num_speculative_tokens") is not None:
            cmd.extend(["--num-speculative-tokens", str(engine_kwargs["num_speculative_tokens"])])
        if engine_kwargs.get("speculative_model") is not None:
            cmd.extend(["--speculative-model", str(engine_kwargs["speculative_model"])])
        return cmd


def _extract_mean_token_entropy(logprobs_payload: Any) -> float | None:
    if logprobs_payload is None:
        return None
    entropies: list[float] = []
    entries = logprobs_payload if isinstance(logprobs_payload, (list, tuple)) else (logprobs_payload,)
    for entry in entries:
        logprob_values = _extract_logprob_values(entry)
        if not logprob_values:
            continue
        entropies.append(_entropy_from_logprobs(logprob_values[:5]))
    if not entropies:
        return None
    return sum(entropies) / len(entropies)


def _extract_logprob_values(entry: Any) -> list[float]:
    if entry is None:
        return []
    if isinstance(entry, Mapping):
        if "top_logprobs" in entry:
            return _extract_logprob_values(entry.get("top_logprobs"))
        values = list(entry.values())
        if values and not isinstance(values[0], (int, float, str)):
            extracted = [_coerce_logprob_value(value) for value in values]
            return [value for value in extracted if value is not None]
        extracted = [_coerce_logprob_value(value) for value in values]
        return [value for value in extracted if value is not None]
    if isinstance(entry, (list, tuple)):
        extracted = [_coerce_logprob_value(value) for value in entry]
        return [value for value in extracted if value is not None]
    top_logprobs = getattr(entry, "top_logprobs", None)
    if top_logprobs is not None:
        return _extract_logprob_values(top_logprobs)
    single = _coerce_logprob_value(entry)
    return [] if single is None else [single]


def _coerce_logprob_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Mapping):
        for key in ("logprob", "value"):
            if key in value:
                return _coerce_logprob_value(value.get(key))
        return None
    for attr in ("logprob", "value"):
        if hasattr(value, attr):
            try:
                raw = getattr(value, attr)
            except Exception:
                continue
            return _coerce_logprob_value(raw)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _entropy_from_logprobs(logprobs: list[float]) -> float:
    total = 0.0
    for logprob in logprobs:
        probability = math.exp(float(logprob))
        if probability > 0.0:
            total -= probability * math.log(probability, 2.0)
    return total


def _extract_tool_calls(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): _extract_tool_calls(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_extract_tool_calls(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except Exception:
            try:
                return value.model_dump()
            except Exception:
                return str(value)
    if hasattr(value, "__dict__"):
        try:
            return {str(key): _extract_tool_calls(item) for key, item in vars(value).items()}
        except Exception:
            return str(value)
    return value


__all__ = [
    "GenerationOutput",
    "GenerationParameters",
    "GenerationRequest",
    "GenerationResponse",
    "RequestStatus",
    "RuntimeBoundaryType",
    "RuntimeProbe",
    "RuntimeState",
    "StartResult",
    "StopResult",
    "VLLMClient",
    "VLLMServer",
    "VLLMServerConfig",
]
