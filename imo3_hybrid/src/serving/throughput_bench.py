from __future__ import annotations

from enum import Enum
import json
import statistics
import time
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.serving.fp8_config import FP8ServingConfig
from src.serving.kv_cache import KVCachePolicy
from src.serving.speculative_decode import SpeculativeDecodePolicy
from src.serving.vllm_server import (
    GenerationParameters,
    GenerationRequest,
    RequestStatus,
    RuntimeProbe,
    RuntimeState,
    StartResult,
    StopResult,
    VLLMClient,
    VLLMServerConfig,
)


class BenchmarkStatus(str, Enum):
    OK = "ok"
    UNAVAILABLE = "unavailable"
    START_FAILED = "start_failed"
    REQUEST_FAILED = "request_failed"
    INVALID_CONFIG = "invalid_config"
    INTERNAL_ERROR = "internal_error"


class PromptSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt: str = Field(min_length=1)
    label: str | None = None

    @field_validator("prompt")
    @classmethod
    def _validate_prompt(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("prompt must not be empty")
        return cleaned

    @field_validator("label")
    @classmethod
    def _normalize_label(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class ThroughputBenchConfig(BaseModel):
    """Typed benchmark configuration.

    This is orchestration-only: it does not assert backend support by itself.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    server: VLLMServerConfig
    prompts: tuple[PromptSpec, ...]
    generation: GenerationParameters = Field(default_factory=GenerationParameters)

    warmup_requests: int = Field(default=1, ge=0)
    benchmark_repeats: int = Field(default=1, ge=1)
    startup_per_run: bool = False
    stop_after_run: bool = True

    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_prompts(self) -> "ThroughputBenchConfig":
        if not self.prompts:
            raise ValueError("At least one prompt is required for benchmarking")
        return self

    def total_measured_requests(self) -> int:
        return len(self.prompts) * self.benchmark_repeats


class RequestBenchmarkRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    label: str | None = None
    status: RequestStatus
    latency_ms: int | None = None
    token_count: int | None = None
    tokens_per_second: float | None = None
    output_count: int = 0
    error_code: str | None = None
    error_message: str | None = None
    runtime_state: RuntimeState
    backend_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_count: int
    successful_request_count: int
    failed_request_count: int

    total_latency_ms: int | None = None
    mean_latency_ms: float | None = None
    median_latency_ms: float | None = None
    p95_latency_ms: float | None = None

    total_measured_tokens: int | None = None
    aggregate_tokens_per_second: float | None = None
    mean_request_tokens_per_second: float | None = None

    def has_real_token_throughput(self) -> bool:
        return self.total_measured_tokens is not None and self.aggregate_tokens_per_second is not None


class BenchmarkRunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: BenchmarkStatus
    runtime_probe: RuntimeProbe | None = None
    start_result: StartResult | None = None
    stop_result: StopResult | None = None

    request_records: tuple[RequestBenchmarkRecord, ...] = ()
    summary: BenchmarkSummary | None = None

    error_code: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ThroughputBenchReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config: ThroughputBenchConfig
    runs: tuple[BenchmarkRunRecord, ...]
    overall_status: BenchmarkStatus
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json", exclude_none=False),
            sort_keys=True,
            separators=(",", ":"),
        )


class ThroughputBenchRunner:
    """Structured benchmark orchestrator for the serving runtime.

    This runner does not fabricate measurements:
    - latencies come from wall-clock timing around real generate calls,
    - token throughput is only reported when token_count is present in runtime outputs,
    - unavailable runtime paths are explicit.
    """

    def __init__(self, config: ThroughputBenchConfig) -> None:
        self._config = config

    @property
    def config(self) -> ThroughputBenchConfig:
        return self._config

    def create_client(self) -> VLLMClient:
        return VLLMClient(config=self._config.server)

    def run(self) -> ThroughputBenchReport:
        if self._config.startup_per_run:
            runs = tuple(self._run_single_with_fresh_client() for _ in range(self._config.benchmark_repeats))
        else:
            runs = (self._run_single_shared_client(),)

        overall_status = self._derive_overall_status(runs)
        return ThroughputBenchReport(
            config=self._config,
            runs=runs,
            overall_status=overall_status,
            metadata={"startup_per_run": self._config.startup_per_run},
        )

    def _derive_overall_status(self, runs: Iterable[BenchmarkRunRecord]) -> BenchmarkStatus:
        runs = tuple(runs)
        if not runs:
            return BenchmarkStatus.INTERNAL_ERROR
        if any(run.status == BenchmarkStatus.OK for run in runs):
            return BenchmarkStatus.OK
        # return the first non-ok status deterministically
        return runs[0].status

    def _run_single_shared_client(self) -> BenchmarkRunRecord:
        client = self.create_client()
        probe = client.probe_runtime()
        if probe.runtime_state == RuntimeState.UNAVAILABLE:
            return BenchmarkRunRecord(
                status=BenchmarkStatus.UNAVAILABLE,
                runtime_probe=probe,
                error_code="runtime_unavailable",
                error_message="Serving runtime is unavailable for benchmarking",
                metadata={"config": self._config.server.to_runtime_metadata()},
            )

        start_result = client.start()
        if not start_result.ok:
            return BenchmarkRunRecord(
                status=BenchmarkStatus.START_FAILED
                if start_result.error_code != "runtime_unavailable"
                else BenchmarkStatus.UNAVAILABLE,
                runtime_probe=probe,
                start_result=start_result,
                error_code=start_result.error_code,
                error_message=start_result.error_message,
                metadata={"config": self._config.server.to_runtime_metadata()},
            )

        try:
            request_records = self._execute_benchmark_requests(client)
            summary = self._summarize_requests(request_records)
            status = BenchmarkStatus.OK if summary.successful_request_count > 0 else BenchmarkStatus.REQUEST_FAILED
            stop_result = client.stop() if self._config.stop_after_run else None
            return BenchmarkRunRecord(
                status=status,
                runtime_probe=probe,
                start_result=start_result,
                stop_result=stop_result,
                request_records=request_records,
                summary=summary,
                metadata={"config": self._config.server.to_runtime_metadata()},
            )
        except Exception as exc:
            stop_result = None
            if self._config.stop_after_run:
                try:
                    stop_result = client.stop()
                except Exception:
                    stop_result = None
            return BenchmarkRunRecord(
                status=BenchmarkStatus.INTERNAL_ERROR,
                runtime_probe=probe,
                start_result=start_result,
                stop_result=stop_result,
                error_code="bench_internal_error",
                error_message=str(exc),
                metadata={"config": self._config.server.to_runtime_metadata()},
            )

    def _run_single_with_fresh_client(self) -> BenchmarkRunRecord:
        # One run = one full client lifecycle, one pass over prompts
        single_run_config = ThroughputBenchConfig(
            server=self._config.server,
            prompts=self._config.prompts,
            generation=self._config.generation,
            warmup_requests=self._config.warmup_requests,
            benchmark_repeats=1,
            startup_per_run=False,
            stop_after_run=self._config.stop_after_run,
            metadata=self._config.metadata,
        )
        return ThroughputBenchRunner(single_run_config)._run_single_shared_client()

    def _execute_benchmark_requests(self, client: VLLMClient) -> tuple[RequestBenchmarkRecord, ...]:
        request_records: list[RequestBenchmarkRecord] = []

        warmup_requests = self._build_requests(repeats=self._config.warmup_requests)
        for warmup in warmup_requests:
            # Warmups are executed but intentionally excluded from benchmark records.
            client.generate(warmup)

        measured_requests = self._build_requests(repeats=1 if self._config.startup_per_run else self._config.benchmark_repeats)
        for request in measured_requests:
            wall_start = time.perf_counter()
            response = client.generate(request)
            wall_latency_ms = int((time.perf_counter() - wall_start) * 1000)

            token_count = self._extract_token_count(response)
            tokens_per_second = None
            latency_ms = response.latency_ms if response.latency_ms is not None else wall_latency_ms
            if token_count is not None and latency_ms > 0:
                tokens_per_second = token_count / (latency_ms / 1000.0)

            request_records.append(
                RequestBenchmarkRecord(
                    request_id=request.request_id,
                    label=request.metadata.get("label"),
                    status=response.status,
                    latency_ms=latency_ms,
                    token_count=token_count,
                    tokens_per_second=tokens_per_second,
                    output_count=len(response.outputs),
                    error_code=response.error_code,
                    error_message=response.error_message,
                    runtime_state=response.runtime_state,
                    backend_name=response.backend_name,
                    metadata={
                        "request_metadata": request.metadata,
                        "response_metadata": response.metadata,
                    },
                )
            )

        return tuple(request_records)

    def _build_requests(self, *, repeats: int) -> tuple[GenerationRequest, ...]:
        requests: list[GenerationRequest] = []
        for repeat_idx in range(repeats):
            for prompt_idx, prompt_spec in enumerate(self._config.prompts):
                label = prompt_spec.label or f"prompt_{prompt_idx}"
                requests.append(
                    GenerationRequest(
                        prompt=prompt_spec.prompt,
                        parameters=self._config.generation,
                        metadata={
                            "label": label,
                            "repeat_index": repeat_idx,
                            "prompt_index": prompt_idx,
                        },
                    )
                )
        return tuple(requests)

    def _extract_token_count(self, response) -> int | None:
        if response.status != RequestStatus.OK:
            return None
        counts: list[int] = []
        for output in response.outputs:
            if output.token_count is not None:
                counts.append(int(output.token_count))
        if not counts:
            return None
        return sum(counts)

    def _summarize_requests(self, records: tuple[RequestBenchmarkRecord, ...]) -> BenchmarkSummary:
        success_records = [record for record in records if record.status == RequestStatus.OK]
        failed_records = [record for record in records if record.status != RequestStatus.OK]

        latencies = [record.latency_ms for record in success_records if record.latency_ms is not None]
        total_latency_ms = sum(latencies) if latencies else None
        mean_latency_ms = statistics.fmean(latencies) if latencies else None
        median_latency_ms = statistics.median(latencies) if latencies else None
        p95_latency_ms = self._percentile(latencies, 95.0) if latencies else None

        token_counts = [record.token_count for record in success_records if record.token_count is not None]
        total_measured_tokens = sum(token_counts) if token_counts else None

        aggregate_tokens_per_second = None
        if total_measured_tokens is not None and total_latency_ms is not None and total_latency_ms > 0:
            aggregate_tokens_per_second = total_measured_tokens / (total_latency_ms / 1000.0)

        per_request_tps = [record.tokens_per_second for record in success_records if record.tokens_per_second is not None]
        mean_request_tps = statistics.fmean(per_request_tps) if per_request_tps else None

        return BenchmarkSummary(
            request_count=len(records),
            successful_request_count=len(success_records),
            failed_request_count=len(failed_records),
            total_latency_ms=total_latency_ms,
            mean_latency_ms=mean_latency_ms,
            median_latency_ms=median_latency_ms,
            p95_latency_ms=p95_latency_ms,
            total_measured_tokens=total_measured_tokens,
            aggregate_tokens_per_second=aggregate_tokens_per_second,
            mean_request_tokens_per_second=mean_request_tps,
        )

    def _percentile(self, values: list[int], p: float) -> float:
        if not values:
            raise ValueError("values must not be empty")
        ordered = sorted(values)
        if len(ordered) == 1:
            return float(ordered[0])

        position = (len(ordered) - 1) * (p / 100.0)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def build_default_bench_config(
    *,
    model_path: str,
    prompts: list[str],
    generation: GenerationParameters | None = None,
    served_model_name: str | None = None,
    fp8: FP8ServingConfig | None = None,
    kv_cache: KVCachePolicy | None = None,
    speculative_decode: SpeculativeDecodePolicy | None = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
    trust_remote_code: bool = False,
    enforce_eager: bool = False,
    dtype: str | None = None,
    warmup_requests: int = 1,
    benchmark_repeats: int = 1,
    startup_per_run: bool = False,
    stop_after_run: bool = True,
    metadata: dict[str, Any] | None = None,
) -> ThroughputBenchConfig:
    return ThroughputBenchConfig(
        server=VLLMServerConfig(
            model_path=model_path,
            served_model_name=served_model_name,
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
            metadata=metadata or {},
        ),
        prompts=tuple(PromptSpec(prompt=prompt, label=f"prompt_{idx}") for idx, prompt in enumerate(prompts)),
        generation=generation or GenerationParameters(),
        warmup_requests=warmup_requests,
        benchmark_repeats=benchmark_repeats,
        startup_per_run=startup_per_run,
        stop_after_run=stop_after_run,
        metadata=metadata or {},
    )


__all__ = [
    "BenchmarkRunRecord",
    "BenchmarkStatus",
    "BenchmarkSummary",
    "PromptSpec",
    "RequestBenchmarkRecord",
    "ThroughputBenchConfig",
    "ThroughputBenchReport",
    "ThroughputBenchRunner",
    "build_default_bench_config",
]