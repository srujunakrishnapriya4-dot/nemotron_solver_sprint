from __future__ import annotations

"""
Kaggle-safe execution runner for the online solve path.

This module is the deterministic competition harness around the canonical
inference engine. It is intentionally strict about:
- Kaggle CSV input/output shape
- deterministic ordering
- structured per-problem failure capture
- separation between competition submission rows and debug sidecars
"""

import csv
import ast
from dataclasses import asdict, dataclass, field, is_dataclass
import json
import random
import re
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.aggregation.canonicalize import canonicalize_competition_answer
from src.branches.branch_controller import GenerationResult
from src.common.constants import ANSWER_MAX, ANSWER_MIN
from src.common.schemas import AttemptBatchResult, AttemptEntropySource, AttemptRecord, ParsedProblem, RetrievedTrace, RouteDecision
from src.online.inference_engine import (
    InferenceEngine,
    InferenceEngineConfig,
    SolveResultBundle,
)
from src.online.python_tool_runtime import (
    PythonToolConfig,
    PythonToolSession,
    detect_python_tool_request,
    render_python_tool_feedback,
)
from src.serving.vllm_server import GenerationParameters, GenerationRequest, RequestStatus, VLLMClient

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

try:
    from src.online.adaptive_budget import GlobalBudgetTracker
except Exception:  # pragma: no cover
    GlobalBudgetTracker = None

try:
    from src.online.submission_formatter import (
        SubmissionRow,
        format_submission_row,
        render_debug_sidecar_csv,
        render_submission_csv,
        sort_submission_rows,
    )
except Exception:  # pragma: no cover
    SubmissionRow = None
    format_submission_row = None
    render_debug_sidecar_csv = None
    render_submission_csv = None
    sort_submission_rows = None


class KaggleProblemRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    problem: str


class KaggleRunFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problem_id: str
    code: str
    message: str
    recoverable: bool = True
    details: dict[str, Any] = Field(default_factory=dict)


@dataclass
class KaggleRunRecord:
    problem_id: str
    ordinal: int
    elapsed_sec: float = 0.0
    status: str = ""
    answer: int | None = None
    used_fallback_submission: bool = False
    solve_result: SolveResultBundle | None = None
    failure: KaggleRunFailure | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.problem_id = str(self.problem_id)
        self.ordinal = max(0, int(self.ordinal))
        self.elapsed_sec = max(0.0, float(self.elapsed_sec))
        self.status = str(self.status)
        self.answer = None if self.answer is None else int(self.answer)
        self.used_fallback_submission = bool(self.used_fallback_submission)
        self.metadata = dict(self.metadata or {})

    def model_dump(self, mode: str = "python", **_: Any) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "ordinal": self.ordinal,
            "elapsed_sec": self.elapsed_sec,
            "status": self.status,
            "answer": self.answer,
            "used_fallback_submission": self.used_fallback_submission,
            "solve_result": _safe_serialize(self.solve_result, mode=mode),
            "failure": _safe_serialize(self.failure, mode=mode),
            "metadata": _safe_serialize(self.metadata, mode=mode),
        }

    def dict(self) -> dict[str, Any]:
        return self.model_dump()


@dataclass
class KaggleSubmissionBundle:
    submission_rows: list[Any] = field(default_factory=list)
    debug_rows: list[dict[str, Any]] = field(default_factory=list)
    run_records: list[KaggleRunRecord] = field(default_factory=list)
    failures: list[KaggleRunFailure] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.submission_rows = list(self.submission_rows or [])
        self.debug_rows = [dict(row) for row in (self.debug_rows or [])]
        self.run_records = list(self.run_records or [])
        self.failures = list(self.failures or [])
        self.metadata = dict(self.metadata or {})

    def model_dump(self, mode: str = "python", **_: Any) -> dict[str, Any]:
        return {
            "submission_rows": _safe_serialize(self.submission_rows, mode=mode),
            "debug_rows": _safe_serialize(self.debug_rows, mode=mode),
            "run_records": _safe_serialize(self.run_records, mode=mode),
            "failures": _safe_serialize(self.failures, mode=mode),
            "metadata": _safe_serialize(self.metadata, mode=mode),
        }

    def dict(self) -> dict[str, Any]:
        return self.model_dump()


class KaggleRunnerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deterministic_seed: int = 0
    continue_on_problem_failure: bool = True
    emit_fallback_rows_on_failure: bool = True
    fallback_answer: int = Field(default=0, ge=0)
    include_debug_sidecar: bool = True
    total_time_budget_s: float | None = Field(default=None, gt=0.0)
    reserve_time_s: float = Field(default=300.0, ge=0.0)
    stop_when_out_of_budget: bool = True
    sort_input_records: bool = True
    validate_input_columns: bool = True
    prompt_config_path: str = "configs/prompts.yaml"
    minimal_prompt_profile: str = "minimal_default"
    attempts_per_problem: int = Field(default=8, ge=1)
    early_stop_threshold: int = Field(default=4, ge=2)
    max_attempt_tokens: int = Field(default=4096, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    min_p: float = Field(default=0.02, ge=0.0, le=1.0)
    request_logprobs: int = Field(default=5, ge=0)
    entropy_epsilon: float = Field(default=0.05, gt=0.0)
    allow_proxy_entropy: bool = True
    max_turns_per_attempt: int = Field(default=8, ge=1)
    max_tool_calls_per_attempt: int = Field(default=6, ge=0)
    max_tool_execution_time_sec: float = Field(default=5.0, gt=0.0)
    max_tool_stdout_chars: int = Field(default=4000, ge=128)
    max_tool_stderr_chars: int = Field(default=2000, ge=128)
    minimal_mode_max_route_uncertainty: float = Field(default=0.34, ge=0.0, le=1.0)
    minimal_mode_max_difficulty_score: float = Field(default=0.48, ge=0.0, le=1.0)
    minimal_mode_max_proof_burden: float = Field(default=0.30, ge=0.0, le=1.0)
    minimal_mode_max_retrieval_need: float = Field(default=0.28, ge=0.0, le=1.0)
    minimal_mode_max_repair_need: float = Field(default=0.28, ge=0.0, le=1.0)
    minimal_mode_high_entropy_threshold: float = Field(default=1.10, ge=0.0)
    minimal_mode_high_proxy_uncertainty_threshold: float = Field(default=0.42, ge=0.0, le=1.0)
    minimal_mode_fragmentation_threshold: float = Field(default=0.34, ge=0.0, le=1.0)
    minimal_mode_allow_post_attempt_escalation: bool = True
    escalation_min_confidence: float = Field(default=0.62, ge=0.0, le=1.0)
    escalation_min_answer_agreement: float = Field(default=0.50, ge=0.0, le=1.0)
    escalation_low_margin_threshold: float = Field(default=0.08, ge=0.0, le=1.0)
    mixed_domain_gap_threshold: float = Field(default=0.18, ge=0.0, le=1.0)
    hard_mode_min_route_uncertainty: float = Field(default=0.68, ge=0.0, le=1.0)
    hard_mode_min_difficulty_score: float = Field(default=0.80, ge=0.0, le=1.0)
    hard_mode_min_proof_burden: float = Field(default=0.58, ge=0.0, le=1.0)
    hard_mode_min_retrieval_need: float = Field(default=0.45, ge=0.0, le=1.0)
    hard_mode_min_repair_need: float = Field(default=0.45, ge=0.0, le=1.0)
    escalated_mode_min_retrieval_need: float = Field(default=0.34, ge=0.0, le=1.0)
    escalated_mode_min_verifier_uncertainty: float = Field(default=0.28, ge=0.0, le=1.0)
    escalated_mode_min_symbolic_burden: float = Field(default=0.26, ge=0.0, le=1.0)
    escalated_mode_min_critique_uncertainty: float = Field(default=0.40, ge=0.0, le=1.0)
    escalated_mode_min_repair_need: float = Field(default=0.38, ge=0.0, le=1.0)
    escalated_mode_min_search_uncertainty: float = Field(default=0.30, ge=0.0, le=1.0)
    escalated_mode_min_search_proof_burden: float = Field(default=0.26, ge=0.0, le=1.0)


@dataclass(frozen=True)
class MinimalPromptProfile:
    name: str
    system_prompt: str
    tool_prompt: str
    preference_prompt: str


_DEFAULT_MINIMAL_PROMPT_PROFILE = MinimalPromptProfile(
    name="minimal_default",
    system_prompt=(
        "Solve the following math problem step by step.\n"
        "Use Python code for calculations, verification, or brute-force search.\n"
        "The answer is a non-negative integer in [0, 99999].\n"
        "Put your final answer in \\boxed{}."
    ),
    tool_prompt=(
        "Execute Python code in a stateful Jupyter notebook. Available: math, numpy, sympy, itertools, "
        "collections, mpmath (64-digit precision). Use print() to see results. When you need Python, emit "
        "exactly <python_tool>...</python_tool> containing only the code, then stop."
    ),
    preference_prompt=(
        "Use sympy for exact symbolic computation. For number theory: use sympy.ntheory "
        "(factorint, divisors, totient, isprime). Verify analytical solutions with numerical checks or brute-force when feasible."
    ),
)


_BOXED_ANSWER_RE = re.compile(r"\\boxed\s*\{\s*([0-9,]+)\s*\}")
_FALLBACK_ANSWER_RE = re.compile(r"final\s+answer\s*(?:is|:)\s*([0-9,]+)", re.IGNORECASE)


def _parse_simple_yaml_scalar(raw: str) -> Any:
    text = str(raw).strip()
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        if text.startswith(("\"", "'")) and text.endswith(("\"", "'")):
            return ast.literal_eval(text)
    except Exception:
        pass
    try:
        return int(text)
    except Exception:
        pass
    try:
        return float(text)
    except Exception:
        pass
    return text


def _load_simple_yaml_mapping(target: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        content = raw_line.split("#", 1)[0].rstrip()
        if not content.strip():
            continue
        indent = len(content) - len(content.lstrip(" "))
        stripped = content.strip()
        if stripped.startswith("- "):
            continue
        if ":" not in stripped:
            continue
        key, remainder = stripped.split(":", 1)
        key = key.strip()
        remainder = remainder.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if not remainder:
            node: dict[str, Any] = {}
            parent[key] = node
            stack.append((indent, node))
            continue
        parent[key] = _parse_simple_yaml_scalar(remainder)
    return root


def load_kaggle_runner_config(
    path: str | Path = "configs/online.yaml",
) -> KaggleRunnerConfig:
    target = Path(path)
    if yaml is None:
        payload = _load_simple_yaml_mapping(target)
    else:
        payload = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Online config at {target} must be a mapping")

    runtime = payload.get("runtime", {}) if isinstance(payload.get("runtime"), Mapping) else {}
    minimal = runtime.get("minimal_mode", {}) if isinstance(runtime.get("minimal_mode"), Mapping) else {}
    selective = runtime.get("selective_gates", {}) if isinstance(runtime.get("selective_gates"), Mapping) else {}
    competition = runtime.get("competition", {}) if isinstance(runtime.get("competition"), Mapping) else {}
    debug = payload.get("output", {}).get("debug_sidecar", {}) if isinstance(payload.get("output"), Mapping) else {}

    config_payload = {
        "deterministic_seed": runtime.get("deterministic_seed", 0),
        "continue_on_problem_failure": competition.get("continue_on_problem_failure", True),
        "emit_fallback_rows_on_failure": competition.get("emit_fallback_rows_on_failure", True),
        "fallback_answer": competition.get("default_fallback_answer", 0),
        "include_debug_sidecar": debug.get("enabled", True),
        "total_time_budget_s": competition.get("total_time_budget_s"),
        "reserve_time_s": competition.get("reserve_time_s", 300.0),
        "stop_when_out_of_budget": competition.get("stop_when_out_of_budget", True),
        "prompt_config_path": runtime.get("prompt_config_path", "configs/prompts.yaml"),
        "minimal_prompt_profile": runtime.get("minimal_prompt_profile", "minimal_default"),
        "attempts_per_problem": minimal.get("attempts_per_problem", 8),
        "early_stop_threshold": minimal.get("early_stop_threshold", 4),
        "max_attempt_tokens": minimal.get("max_tokens", 4096),
        "temperature": minimal.get("temperature", 1.0),
        "min_p": minimal.get("min_p", 0.02),
        "request_logprobs": minimal.get("request_logprobs", 5),
        "allow_proxy_entropy": minimal.get("allow_proxy_entropy", True),
        "max_turns_per_attempt": minimal.get("max_turns_per_attempt", 8),
        "max_tool_calls_per_attempt": minimal.get("max_tool_calls_per_attempt", 6),
        "max_tool_execution_time_sec": minimal.get("tool_timeout_sec", 5.0),
        "max_tool_stdout_chars": minimal.get("max_tool_stdout_chars", 4000),
        "minimal_mode_max_route_uncertainty": selective.get("minimal_mode_max_route_uncertainty", 0.34),
        "minimal_mode_max_difficulty_score": selective.get("minimal_mode_max_difficulty_score", 0.48),
        "minimal_mode_max_proof_burden": selective.get("minimal_mode_max_proof_burden", 0.30),
        "minimal_mode_max_retrieval_need": selective.get("minimal_mode_max_retrieval_need", 0.28),
        "minimal_mode_max_repair_need": selective.get("minimal_mode_max_repair_need", 0.28),
        "minimal_mode_high_entropy_threshold": selective.get("minimal_mode_high_entropy_threshold", 1.10),
        "minimal_mode_high_proxy_uncertainty_threshold": selective.get("minimal_mode_high_proxy_uncertainty_threshold", 0.42),
        "minimal_mode_fragmentation_threshold": selective.get("minimal_mode_fragmentation_threshold", 0.34),
        "minimal_mode_allow_post_attempt_escalation": selective.get("minimal_mode_allow_post_attempt_escalation", True),
        "escalation_min_confidence": selective.get("escalation_min_confidence", 0.62),
        "escalation_min_answer_agreement": selective.get("escalation_min_answer_agreement", 0.50),
        "escalation_low_margin_threshold": selective.get("escalation_low_margin_threshold", 0.08),
        "mixed_domain_gap_threshold": selective.get("mixed_domain_gap_threshold", 0.18),
        "hard_mode_min_route_uncertainty": selective.get("hard_mode_min_route_uncertainty", 0.68),
        "hard_mode_min_difficulty_score": selective.get("hard_mode_min_difficulty_score", 0.80),
        "hard_mode_min_proof_burden": selective.get("hard_mode_min_proof_burden", 0.58),
        "hard_mode_min_retrieval_need": selective.get("hard_mode_min_retrieval_need", 0.45),
        "hard_mode_min_repair_need": selective.get("hard_mode_min_repair_need", 0.45),
        "escalated_mode_min_retrieval_need": selective.get("escalated_mode_min_retrieval_need", 0.34),
        "escalated_mode_min_verifier_uncertainty": selective.get("escalated_mode_min_verifier_uncertainty", 0.28),
        "escalated_mode_min_symbolic_burden": selective.get("escalated_mode_min_symbolic_burden", 0.26),
        "escalated_mode_min_critique_uncertainty": selective.get("escalated_mode_min_critique_uncertainty", 0.40),
        "escalated_mode_min_repair_need": selective.get("escalated_mode_min_repair_need", 0.38),
        "escalated_mode_min_search_uncertainty": selective.get("escalated_mode_min_search_uncertainty", 0.30),
        "escalated_mode_min_search_proof_burden": selective.get("escalated_mode_min_search_proof_burden", 0.26),
    }
    return KaggleRunnerConfig.model_validate(config_payload)


def build_inference_engine_config(config: KaggleRunnerConfig) -> InferenceEngineConfig:
    return InferenceEngineConfig(
        deterministic_seed=config.deterministic_seed,
        minimal_mode_self_consistency_samples=config.attempts_per_problem,
        minimal_mode_consensus_stop_count=config.early_stop_threshold,
        minimal_mode_allow_proxy_entropy=config.allow_proxy_entropy,
        prompt_config_path=config.prompt_config_path,
        minimal_prompt_profile=config.minimal_prompt_profile,
        minimal_mode_max_route_uncertainty=config.minimal_mode_max_route_uncertainty,
        minimal_mode_max_difficulty_score=config.minimal_mode_max_difficulty_score,
        minimal_mode_max_proof_burden=config.minimal_mode_max_proof_burden,
        minimal_mode_max_retrieval_need=config.minimal_mode_max_retrieval_need,
        minimal_mode_max_repair_need=config.minimal_mode_max_repair_need,
        minimal_mode_high_entropy_threshold=config.minimal_mode_high_entropy_threshold,
        minimal_mode_high_proxy_uncertainty_threshold=config.minimal_mode_high_proxy_uncertainty_threshold,
        minimal_mode_fragmentation_threshold=config.minimal_mode_fragmentation_threshold,
        minimal_mode_allow_post_attempt_escalation=config.minimal_mode_allow_post_attempt_escalation,
        escalation_min_confidence=config.escalation_min_confidence,
        escalation_min_answer_agreement=config.escalation_min_answer_agreement,
        escalation_low_margin_threshold=config.escalation_low_margin_threshold,
        mixed_domain_gap_threshold=config.mixed_domain_gap_threshold,
        hard_mode_min_route_uncertainty=config.hard_mode_min_route_uncertainty,
        hard_mode_min_difficulty_score=config.hard_mode_min_difficulty_score,
        hard_mode_min_proof_burden=config.hard_mode_min_proof_burden,
        hard_mode_min_retrieval_need=config.hard_mode_min_retrieval_need,
        hard_mode_min_repair_need=config.hard_mode_min_repair_need,
        escalated_mode_min_retrieval_need=config.escalated_mode_min_retrieval_need,
        escalated_mode_min_verifier_uncertainty=config.escalated_mode_min_verifier_uncertainty,
        escalated_mode_min_symbolic_burden=config.escalated_mode_min_symbolic_burden,
        escalated_mode_min_critique_uncertainty=config.escalated_mode_min_critique_uncertainty,
        escalated_mode_min_repair_need=config.escalated_mode_min_repair_need,
        escalated_mode_min_search_uncertainty=config.escalated_mode_min_search_uncertainty,
        escalated_mode_min_search_proof_burden=config.escalated_mode_min_search_proof_burden,
    )


def load_minimal_prompt_profile(
    path: str | Path,
    *,
    profile_name: str = "minimal_default",
) -> MinimalPromptProfile:
    target = Path(path)
    if yaml is None:
        if profile_name != _DEFAULT_MINIMAL_PROMPT_PROFILE.name:
            raise RuntimeError("PyYAML is required to load non-default prompt profiles from configs/prompts.yaml")
        return _DEFAULT_MINIMAL_PROMPT_PROFILE
    payload = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    prompts = payload.get("prompts", {}) if isinstance(payload, Mapping) else {}
    runtime_profiles = prompts.get("runtime_profiles", {}) if isinstance(prompts, Mapping) else {}
    profile = runtime_profiles.get(profile_name, {}) if isinstance(runtime_profiles, Mapping) else {}
    if not isinstance(profile, Mapping):
        raise ValueError(f"Prompt profile '{profile_name}' is malformed in {target}")
    system_prompt = str(profile.get("system_prompt", "")).strip()
    tool_prompt = str(profile.get("tool_prompt", "")).strip()
    preference_prompt = str(profile.get("preference_prompt", "")).strip()
    if not system_prompt:
        raise ValueError(f"Prompt profile '{profile_name}' is missing system_prompt")
    return MinimalPromptProfile(
        name=profile_name,
        system_prompt=system_prompt,
        tool_prompt=tool_prompt,
        preference_prompt=preference_prompt,
    )


class CompetitionAttemptGenerator:
    def __init__(
        self,
        *,
        client: VLLMClient,
        prompt_profile: MinimalPromptProfile,
        deterministic_seed: int = 0,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        top_p: float = 0.95,
        min_p: float = 0.02,
        request_logprobs: int = 5,
        tool_config: PythonToolConfig | None = None,
    ) -> None:
        self.client = client
        self.prompt_profile = prompt_profile
        self.deterministic_seed = int(deterministic_seed)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.min_p = float(min_p)
        self.request_logprobs = max(0, int(request_logprobs))
        self.tool_config = tool_config or PythonToolConfig()

    def run_minimal_attempts(
        self,
        *,
        problem: ParsedProblem,
        route: RouteDecision,
        attempt_count: int,
        early_stop_threshold: int,
    ) -> AttemptBatchResult:
        attempts: list[AttemptRecord] = []
        answer_counts: dict[str, int] = {}
        stopped_reason = "attempt_budget_exhausted"
        for sample_index in range(max(1, int(attempt_count))):
            attempt = self._run_attempt(
                problem=problem,
                route=route,
                sample_index=sample_index,
                prompt=self._compose_minimal_prompt(problem.raw_text),
            )
            attempts.append(attempt)
            if attempt.valid_answer and attempt.extracted_answer is not None:
                answer = str(attempt.extracted_answer)
                answer_counts[answer] = answer_counts.get(answer, 0) + 1
                if answer_counts[answer] >= max(2, int(early_stop_threshold)):
                    attempts[-1] = attempt.model_copy(update={"stopped_reason": "early_stop_consensus"})
                    stopped_reason = "early_stop_consensus"
                    break
        return AttemptBatchResult(
            attempts=attempts,
            stopped_reason=stopped_reason,
            metadata={
                "prompt_profile": self.prompt_profile.name,
                "python_tool_mode": "bounded_stateful",
                "tool_protocol_adapter": "text_protocol_adapter",
                "logprobs_requested": self.request_logprobs,
            },
        )

    def __call__(
        self,
        *,
        problem: ParsedProblem,
        route: RouteDecision,
        branch: Any,
        node: Any,
        operator_name: str,
        retrieval_trace: RetrievedTrace | None,
        depth: int,
        sample_index: int,
    ) -> Any:
        del branch, node, operator_name, depth
        prompt = self._compose_branch_prompt(problem.raw_text, retrieval_trace=retrieval_trace)
        attempt = self._run_attempt(
            problem=problem,
            route=route,
            sample_index=sample_index,
            prompt=prompt,
        )
        answer = attempt.extracted_answer if attempt.valid_answer else None
        confidence = 0.5
        if attempt.mean_token_entropy is not None:
            confidence = max(0.0, min(1.0, 1.0 / (1.0 + float(attempt.mean_token_entropy))))
        return _build_generation_result_from_attempt(attempt, answer=answer, confidence=confidence)

    def _compose_minimal_prompt(self, problem_text: str) -> str:
        pieces = [self.prompt_profile.system_prompt]
        if self.prompt_profile.tool_prompt:
            pieces.append(f"Python Tool:\n{self.prompt_profile.tool_prompt}")
        if self.prompt_profile.preference_prompt:
            pieces.append(f"Preferences:\n{self.prompt_profile.preference_prompt}")
        pieces.append(f"Problem:\n{problem_text.strip()}")
        return "\n\n".join(piece.strip() for piece in pieces if piece and piece.strip())

    def _compose_branch_prompt(self, problem_text: str, *, retrieval_trace: RetrievedTrace | None) -> str:
        prompt = self._compose_minimal_prompt(problem_text)
        if retrieval_trace is None:
            return prompt
        hint = str(getattr(retrieval_trace, "strategy_summary", "") or getattr(retrieval_trace, "solution", "")).strip()
        if not hint:
            return prompt
        return f"{prompt}\n\nStrategy Hint:\n{hint[:400]}"

    def _run_attempt(
        self,
        *,
        problem: ParsedProblem,
        route: RouteDecision,
        sample_index: int,
        prompt: str,
    ) -> AttemptRecord:
        del problem, route
        seed = (self.deterministic_seed + int(sample_index)) ** 2
        attempt_id = f"attempt::{sample_index:03d}"
        session = PythonToolSession(attempt_id=attempt_id, config=self.tool_config)
        assistant_turns: list[str] = []
        tool_feedback: list[str] = []
        turn_entropies: list[float] = []
        per_turn_metadata: list[dict[str, Any]] = []
        tool_protocol = "not_used"
        stopped_reason = "max_turns_reached"
        started = time.perf_counter()

        for turn_index in range(max(1, int(self.tool_config.max_turns_per_attempt))):
            request = GenerationRequest(
                prompt=self._compose_turn_prompt(
                    base_prompt=prompt,
                    assistant_turns=assistant_turns,
                    tool_feedback=tool_feedback,
                ),
                parameters=GenerationParameters(
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    min_p=self.min_p,
                    logprobs=self.request_logprobs if self.request_logprobs > 0 else None,
                    seed=seed,
                ),
                metadata={
                    "prompt_profile": self.prompt_profile.name,
                    "sample_index": int(sample_index),
                    "turn_index": turn_index,
                },
            )
            response = self.client.generate(request)
            if response.status is not RequestStatus.OK or not response.outputs:
                return AttemptRecord(
                    attempt_id=attempt_id,
                    seed=seed,
                    raw_text="\n\n".join(assistant_turns),
                    extracted_answer=None,
                    valid_answer=False,
                    mean_token_entropy=_mean(turn_entropies) if turn_entropies else None,
                    entropy_source=AttemptEntropySource.TRUE_LOGPROBS if turn_entropies else AttemptEntropySource.UNAVAILABLE,
                    runtime_sec=_elapsed(started),
                    stopped_reason=str(response.status.value if hasattr(response.status, "value") else response.status),
                    tool_call_count=session.tool_call_count,
                    tool_error_count=session.tool_error_count,
                    metadata={
                        "error_code": response.error_code,
                        "error_message": response.error_message,
                        "prompt_profile": self.prompt_profile.name,
                        "python_tool_mode": "bounded_stateful",
                        "python_tool_protocol": tool_protocol,
                        "turn_count": len(assistant_turns),
                        "tool_history_cell_count": len(session.history),
                        "per_turn_metadata": per_turn_metadata,
                    },
                )

            output = response.outputs[0]
            raw_text = str(output.text or "")
            assistant_turns.append(raw_text)
            if output.mean_token_entropy is not None:
                turn_entropies.append(float(output.mean_token_entropy))
            per_turn_metadata.append(
                {
                    "turn_index": turn_index,
                    "finish_reason": output.finish_reason,
                    "token_count": output.token_count,
                    "mean_token_entropy": output.mean_token_entropy,
                }
            )

            extracted_answer = _extract_competition_answer(raw_text)
            if extracted_answer is not None:
                stopped_reason = "final_answer"
                mean_entropy = _mean(turn_entropies) if turn_entropies else None
                entropy_source = (
                    AttemptEntropySource.TRUE_LOGPROBS
                    if mean_entropy is not None
                    else AttemptEntropySource.UNAVAILABLE
                )
                return AttemptRecord(
                    attempt_id=attempt_id,
                    seed=seed,
                    raw_text="\n\n".join(assistant_turns),
                    extracted_answer=extracted_answer,
                    valid_answer=True,
                    mean_token_entropy=mean_entropy,
                    entropy_source=entropy_source,
                    runtime_sec=_elapsed(started),
                    stopped_reason=stopped_reason,
                    tool_call_count=session.tool_call_count,
                    tool_error_count=session.tool_error_count,
                    metadata={
                        "finish_reason": output.finish_reason,
                        "token_count": output.token_count,
                        "prompt_profile": self.prompt_profile.name,
                        "python_tool_mode": "bounded_stateful",
                        "python_tool_protocol": tool_protocol,
                        "confidence": 0.0 if mean_entropy is None else max(0.0, min(1.0, 1.0 / (1.0 + float(mean_entropy)))),
                        "turn_count": len(assistant_turns),
                        "tool_history_cell_count": len(session.history),
                        "per_turn_metadata": per_turn_metadata,
                    },
                )

            tool_request = detect_python_tool_request(raw_text, raw_output=output.raw)
            if tool_request is None:
                stopped_reason = "no_answer_or_tool"
                tool_feedback.append(
                    "No final boxed answer was detected. Continue from the current state. "
                    "Either emit <python_tool>...</python_tool> or give the final answer in \\boxed{}."
                )
                continue

            tool_protocol = tool_request.source
            tool_result = session.execute(tool_request)
            tool_feedback.append(render_python_tool_feedback(tool_result))
            per_turn_metadata[-1]["tool_request_source"] = tool_request.source
            per_turn_metadata[-1]["tool_success"] = tool_result.success
            per_turn_metadata[-1]["tool_timeout"] = tool_result.timeout
            if not tool_result.success and tool_result.error_type == "ToolCallLimitReached":
                stopped_reason = "tool_call_limit_reached"
                break
            if not tool_result.success and tool_result.error_type in {"ToolWorkerCrashed", "ToolNoResult"}:
                stopped_reason = "fatal_tool_runtime_failure"
                break
            stopped_reason = "tool_call_executed"

        mean_entropy = _mean(turn_entropies) if turn_entropies else None
        entropy_source = (
            AttemptEntropySource.TRUE_LOGPROBS
            if mean_entropy is not None
            else AttemptEntropySource.UNAVAILABLE
        )
        extracted_answer = _extract_competition_answer("\n\n".join(assistant_turns))
        return AttemptRecord(
            attempt_id=attempt_id,
            seed=seed,
            raw_text="\n\n".join(assistant_turns),
            extracted_answer=extracted_answer,
            valid_answer=extracted_answer is not None,
            mean_token_entropy=mean_entropy,
            entropy_source=entropy_source,
            runtime_sec=_elapsed(started),
            stopped_reason=stopped_reason,
            tool_call_count=session.tool_call_count,
            tool_error_count=session.tool_error_count,
            metadata={
                "prompt_profile": self.prompt_profile.name,
                "python_tool_mode": "bounded_stateful",
                "python_tool_protocol": tool_protocol,
                "turn_count": len(assistant_turns),
                "per_turn_metadata": per_turn_metadata,
                "tool_history_cell_count": len(session.history),
            },
        )

    def _compose_turn_prompt(
        self,
        *,
        base_prompt: str,
        assistant_turns: Sequence[str],
        tool_feedback: Sequence[str],
    ) -> str:
        if not assistant_turns and not tool_feedback:
            return base_prompt
        transcript: list[str] = [base_prompt, "", "Conversation so far:"]
        for turn_index, text in enumerate(assistant_turns):
            transcript.append(f"Assistant turn {turn_index + 1}:\n{text}")
            if turn_index < len(tool_feedback):
                transcript.append(f"Tool feedback {turn_index + 1}:\n{tool_feedback[turn_index]}")
        transcript.append(
            "Continue solving. If you need Python, emit exactly <python_tool>...</python_tool> with only code inside. "
            "Otherwise end with the final answer in \\boxed{}."
        )
        return "\n\n".join(item for item in transcript if item.strip())


class KaggleRunner:
    """
    Competition-safe runner around the canonical inference engine.
    """

    def __init__(
        self,
        *,
        engine: InferenceEngine | None = None,
        engine_config: InferenceEngineConfig | None = None,
        config: KaggleRunnerConfig | None = None,
        generator: Any | None = None,
        generation_client: VLLMClient | None = None,
    ) -> None:
        self.config = config or load_kaggle_runner_config()
        resolved_engine_config = engine_config or build_inference_engine_config(self.config)
        self.engine = engine or InferenceEngine(config=resolved_engine_config)
        self.generator = generator
        if self.generator is None and generation_client is not None:
            profile = load_minimal_prompt_profile(
                self.config.prompt_config_path,
                profile_name=self.config.minimal_prompt_profile,
            )
            self.generator = CompetitionAttemptGenerator(
                client=generation_client,
                prompt_profile=profile,
                deterministic_seed=self.config.deterministic_seed,
                max_tokens=self.config.max_attempt_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                min_p=self.config.min_p,
                request_logprobs=self.config.request_logprobs,
                tool_config=PythonToolConfig(
                    max_turns_per_attempt=self.config.max_turns_per_attempt,
                    max_tool_calls_per_attempt=self.config.max_tool_calls_per_attempt,
                    max_execution_time_sec=self.config.max_tool_execution_time_sec,
                    max_stdout_chars=self.config.max_tool_stdout_chars,
                    max_stderr_chars=self.config.max_tool_stderr_chars,
                ),
            )

    def run_records(
        self,
        records: Sequence[KaggleProblemRecord],
    ) -> KaggleSubmissionBundle:
        self._set_deterministic_seed(self.config.deterministic_seed)

        ordered_records = (
            sort_problem_records(records) if self.config.sort_input_records else list(records)
        )

        tracker = self._build_budget_tracker()
        started = time.perf_counter()

        submission_rows: list[Any] = []
        debug_rows: list[dict[str, Any]] = []
        run_records: list[KaggleRunRecord] = []
        failures: list[KaggleRunFailure] = []

        for idx, record in enumerate(ordered_records):
            remaining = len(ordered_records) - idx
            if self._should_stop_for_budget(tracker):
                failure = KaggleRunFailure(
                    problem_id=record.id,
                    code="global_time_budget_exhausted",
                    message="Stopped before solving because the remaining global time budget was exhausted.",
                    recoverable=True,
                    details={"remaining_problems": remaining},
                )
                failures.append(failure)
                if self.config.emit_fallback_rows_on_failure:
                    row = self._fallback_submission_row(record.id)
                    submission_rows.append(row)
                    run_records.append(
                        KaggleRunRecord(
                            problem_id=record.id,
                            ordinal=idx,
                            elapsed_sec=0.0,
                            status="budget_exhausted_fallback",
                            answer=_extract_answer_value(row),
                            used_fallback_submission=True,
                            failure=failure,
                            metadata={"remaining_problems": remaining},
                        )
                    )
                if not self.config.continue_on_problem_failure:
                    break
                continue

            problem_started = time.perf_counter()
            try:
                result = self.engine.solve_problem(
                    record.problem,
                    problem_id=record.id,
                    remaining_problems=remaining,
                    budget_tracker=tracker,
                    generator=self.generator,
                )
                elapsed = _elapsed(problem_started)
                tracker = self._update_tracker_after_problem(tracker, elapsed)

                row, used_fallback, maybe_failure = self._submission_row_from_result(
                    record=record,
                    result=result,
                )
                if maybe_failure is not None:
                    failures.append(maybe_failure)

                submission_rows.append(row)
                run_records.append(
                    KaggleRunRecord(
                        problem_id=record.id,
                        ordinal=idx,
                        elapsed_sec=elapsed,
                        status=result.status.value if hasattr(result.status, "value") else str(result.status),
                        answer=_extract_answer_value(row),
                        used_fallback_submission=used_fallback,
                        solve_result=result,
                        failure=maybe_failure,
                        metadata={
                            "engine_ok": bool(result.ok),
                            "final_confidence": None
                            if result.final_prediction is None
                            else float(result.final_prediction.confidence),
                            "abstain_recommended": bool(result.metadata.get("abstain_recommended", False)),
                        },
                    )
                )

                if self.config.include_debug_sidecar:
                    debug_rows.append(
                        self._make_debug_row(record, result, row, used_fallback, maybe_failure)
                    )

            except Exception as exc:
                elapsed = _elapsed(problem_started)
                tracker = self._update_tracker_after_problem(tracker, elapsed)

                failure = KaggleRunFailure(
                    problem_id=record.id,
                    code="runner_exception",
                    message=str(exc),
                    recoverable=True,
                    details={
                        "exception_type": type(exc).__name__,
                        "traceback": traceback.format_exc(limit=6),
                    },
                )
                failures.append(failure)

                if self.config.emit_fallback_rows_on_failure:
                    row = self._fallback_submission_row(record.id)
                    submission_rows.append(row)
                    run_records.append(
                        KaggleRunRecord(
                            problem_id=record.id,
                            ordinal=idx,
                            elapsed_sec=elapsed,
                            status="runner_exception_fallback",
                            answer=_extract_answer_value(row),
                            used_fallback_submission=True,
                            failure=failure,
                            metadata={},
                        )
                    )
                    if self.config.include_debug_sidecar:
                        debug_rows.append(
                            {
                                "problem_id": record.id,
                                "status": "runner_exception_fallback",
                                "answer": _extract_answer_value(row),
                                "used_fallback_submission": True,
                                "error_code": failure.code,
                                "error_message": failure.message,
                            }
                        )
                elif not self.config.continue_on_problem_failure:
                    raise

                if not self.config.continue_on_problem_failure:
                    break

        submission_rows = self._sort_submission_rows(submission_rows)

        return KaggleSubmissionBundle(
            submission_rows=submission_rows,
            debug_rows=debug_rows,
            run_records=run_records,
            failures=failures,
            metadata={
                "num_records": len(ordered_records),
                "num_submission_rows": len(submission_rows),
                "num_failures": len(failures),
                "total_elapsed_sec": _elapsed(started),
                "deterministic_seed": self.config.deterministic_seed,
            },
        )

    def run_csv(
        self,
        input_csv_path: str | Path,
    ) -> KaggleSubmissionBundle:
        records = read_kaggle_input_csv(
            input_csv_path,
            validate_columns=self.config.validate_input_columns,
        )
        return self.run_records(records)

    def _submission_row_from_result(
        self,
        *,
        record: KaggleProblemRecord,
        result: SolveResultBundle,
    ) -> tuple[Any, bool, KaggleRunFailure | None]:
        if result.submission_row is not None:
            failure = _kaggle_failure_from_result(record.id, result)
            return result.submission_row, False, failure

        if result.final_selection is not None and format_submission_row is not None:
            try:
                row = format_submission_row(result.final_selection)
                return row, False, _kaggle_failure_from_result(record.id, result)
            except Exception as exc:
                failure = KaggleRunFailure(
                    problem_id=record.id,
                    code="submission_reformat_failed",
                    message=str(exc),
                    recoverable=True,
                    details={"exception_type": type(exc).__name__},
                )
                if self.config.emit_fallback_rows_on_failure:
                    return self._fallback_submission_row(record.id), True, failure
                raise

        failure = KaggleRunFailure(
            problem_id=record.id,
            code="missing_submission_output",
            message="No submission-safe output was available from the inference result.",
            recoverable=True,
            details={"engine_status": str(result.status)},
        )
        if self.config.emit_fallback_rows_on_failure:
            return self._fallback_submission_row(record.id), True, failure
        raise RuntimeError(failure.message)

    def _fallback_submission_row(self, problem_id: str) -> Any:
        if SubmissionRow is not None:
            return SubmissionRow(id=str(problem_id), answer=int(self.config.fallback_answer))
        return {"id": str(problem_id), "answer": int(self.config.fallback_answer)}

    def _make_debug_row(
        self,
        record: KaggleProblemRecord,
        result: SolveResultBundle,
        row: Any,
        used_fallback: bool,
        failure: KaggleRunFailure | None,
    ) -> dict[str, Any]:
        debug = {
            "problem_id": record.id,
            "status": result.status.value if hasattr(result.status, "value") else str(result.status),
            "answer": _extract_answer_value(row),
            "used_fallback_submission": used_fallback,
            "error_code": None if failure is None else failure.code,
            "error_message": None if failure is None else failure.message,
        }
        if result.debug_artifact is not None:
            debug.update(
                {
                    "route_rationale": " | ".join(result.debug_artifact.route_rationale),
                    "budget_summary": _stable_repr(result.debug_artifact.budget_summary),
                    "branch_summary": _stable_repr(result.debug_artifact.branch_summary),
                    "aggregation_summary": _stable_repr(result.debug_artifact.aggregation_summary),
                }
            )
        return debug

    def _sort_submission_rows(self, rows: Sequence[Any]) -> list[Any]:
        if sort_submission_rows is not None:
            try:
                return list(sort_submission_rows(rows))  # type: ignore[arg-type]
            except Exception:
                pass
        return sorted(rows, key=lambda row: _problem_id_sort_key(_extract_row_id(row)))

    def _build_budget_tracker(self) -> Any | None:
        if self.config.total_time_budget_s is None or GlobalBudgetTracker is None:
            return None
        return GlobalBudgetTracker(
            total_budget_s=float(self.config.total_time_budget_s),
            reserve_s=float(self.config.reserve_time_s),
            spent_s=0.0,
        )

    def _update_tracker_after_problem(self, tracker: Any | None, elapsed_sec: float) -> Any | None:
        if tracker is None:
            return None
        if hasattr(tracker, "with_spent"):
            try:
                return tracker.with_spent(float(elapsed_sec))
            except Exception:
                return tracker
        return tracker

    def _should_stop_for_budget(self, tracker: Any | None) -> bool:
        if not self.config.stop_when_out_of_budget:
            return False
        if tracker is None:
            return False
        if hasattr(tracker, "remaining_s"):
            try:
                return float(tracker.remaining_s()) <= 0.0
            except Exception:
                return False
        return False

    def _set_deterministic_seed(self, seed: int) -> None:
        random.seed(int(seed))
        try:  # pragma: no cover
            import numpy as np

            np.random.seed(int(seed))
        except Exception:
            pass


def _extract_competition_answer(text: str) -> str | None:
    candidate_text = str(text or "")
    boxed_matches = _BOXED_ANSWER_RE.findall(candidate_text)
    if boxed_matches:
        candidate = boxed_matches[-1]
        normalized = candidate.replace(",", "").strip()
        if normalized.isdigit():
            value = int(normalized)
            if ANSWER_MIN <= value <= ANSWER_MAX:
                return str(value)
    fallback = _FALLBACK_ANSWER_RE.search(candidate_text)
    if fallback is None:
        return None
    normalized = str(fallback.group(1)).replace(",", "").strip()
    if not normalized.isdigit():
        return None
    value = int(normalized)
    if ANSWER_MIN <= value <= ANSWER_MAX:
        return str(value)
    return None


def _build_generation_result_from_attempt(
    attempt: AttemptRecord,
    *,
    answer: str | None,
    confidence: float,
) -> GenerationResult:
    canonical_answer = canonicalize_competition_answer(answer)
    return GenerationResult(
        reasoning=attempt.raw_text,
        answer=answer,
        answer_canonical=canonical_answer,
        confidence=max(0.0, min(1.0, float(confidence))),
        summary=f"runtime_generation::{attempt.entropy_source.value}",
        partial_solution=answer or "",
        metadata={
            "attempt_record": attempt.model_dump(mode="json"),
            "entropy_source": attempt.entropy_source.value,
        },
    )


def read_kaggle_input_csv(
    input_csv_path: str | Path,
    *,
    validate_columns: bool = True,
) -> list[KaggleProblemRecord]:
    path = Path(input_csv_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if validate_columns:
            required = {"id", "problem"}
            if not required.issubset(set(fieldnames)):
                raise ValueError(
                    f"Kaggle input CSV must contain columns {sorted(required)}. "
                    f"Got columns={fieldnames}"
                )

        records: list[KaggleProblemRecord] = []
        for row in reader:
            problem_id = str(row.get("id", "")).strip()
            problem = str(row.get("problem", "")).strip()
            if not problem_id:
                raise ValueError("Encountered a row with an empty 'id' field.")
            if not problem:
                raise ValueError(
                    f"Encountered a row with an empty 'problem' field for id={problem_id!r}."
                )
            records.append(KaggleProblemRecord(id=problem_id, problem=problem))
    return sort_problem_records(records)


def write_submission_csv(
    rows: Sequence[Any],
    output_csv_path: str | Path,
) -> Path:
    path = Path(output_csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ordered_rows = sorted(rows, key=lambda row: _problem_id_sort_key(_extract_row_id(row)))

    if render_submission_csv is not None:
        try:
            csv_text = render_submission_csv(ordered_rows)  # type: ignore[arg-type]
            path.write_text(csv_text, encoding="utf-8")
            return path
        except Exception:
            pass

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "answer"])
        writer.writeheader()
        for row in ordered_rows:
            writer.writerow(
                {
                    "id": _extract_row_id(row),
                    "answer": _extract_answer_value(row),
                }
            )
    return path


def write_debug_sidecar_csv(
    debug_rows: Sequence[Mapping[str, Any]],
    output_csv_path: str | Path,
) -> Path:
    path = Path(output_csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ordered_rows = sorted(
        debug_rows,
        key=lambda row: _problem_id_sort_key(str(row.get("problem_id", ""))),
    )

    if render_debug_sidecar_csv is not None:
        try:
            csv_text = render_debug_sidecar_csv(ordered_rows)  # type: ignore[arg-type]
            path.write_text(csv_text, encoding="utf-8")
            return path
        except Exception:
            pass

    fieldnames = sorted({key for row in ordered_rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in ordered_rows:
            writer.writerow({k: _stable_repr(v) for k, v in row.items()})
    return path


def run_submission(
    input_csv_path: str | Path,
    output_csv_path: str | Path,
    *,
    debug_output_csv_path: str | Path | None = None,
    runner: KaggleRunner | None = None,
) -> KaggleSubmissionBundle:
    active_runner = runner or KaggleRunner()
    bundle = active_runner.run_csv(input_csv_path)
    write_submission_csv(bundle.submission_rows, output_csv_path)
    if debug_output_csv_path is not None and bundle.debug_rows:
        write_debug_sidecar_csv(bundle.debug_rows, debug_output_csv_path)
    return bundle


def sort_problem_records(records: Sequence[KaggleProblemRecord]) -> list[KaggleProblemRecord]:
    return sorted(records, key=lambda row: _problem_id_sort_key(row.id))


def _kaggle_failure_from_result(problem_id: str, result: SolveResultBundle) -> KaggleRunFailure | None:
    payload = result.to_kaggle_failure()
    if payload is None:
        return None
    return KaggleRunFailure(
        problem_id=problem_id,
        code=str(payload["code"]),
        message=str(payload["message"]),
        recoverable=bool(payload["recoverable"]),
        details=dict(payload["details"] or {}),
    )


def _problem_id_sort_key(problem_id: str) -> tuple[int, int | str, str]:
    value = str(problem_id).strip()
    if value.isdigit():
        return (0, int(value), value)
    return (1, value, value)


def _extract_row_id(row: Any) -> str:
    if hasattr(row, "id"):
        return str(getattr(row, "id"))
    if isinstance(row, Mapping):
        return str(row["id"])
    raise TypeError(f"Unsupported submission row type: {type(row)!r}")


def _extract_answer_value(row: Any) -> int:
    if hasattr(row, "answer"):
        return int(getattr(row, "answer"))
    if isinstance(row, Mapping):
        return int(row["answer"])
    raise TypeError(f"Unsupported submission row type: {type(row)!r}")


def _safe_serialize(value: Any, *, mode: str = "python") -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode=mode)  # type: ignore[misc]
        except TypeError:
            try:
                return value.model_dump()  # type: ignore[misc]
            except Exception:
                pass
        except Exception:
            pass
    if isinstance(value, BaseModel):
        return value.model_dump(mode=mode)
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(k): _safe_serialize(v, mode=mode) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_serialize(v, mode=mode) for v in value]
    return value


def _stable_repr(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, Mapping):
        items = ", ".join(f"{k}={_stable_repr(v)}" for k, v in sorted(value.items()))
        return "{" + items + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_stable_repr(v) for v in value) + "]"
    return str(value)


def _elapsed(started: float) -> float:
    return max(0.0, round(time.perf_counter() - started, 6))


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


__all__ = [
    "CompetitionAttemptGenerator",
    "KaggleProblemRecord",
    "KaggleRunFailure",
    "KaggleRunRecord",
    "KaggleRunner",
    "KaggleRunnerConfig",
    "KaggleSubmissionBundle",
    "MinimalPromptProfile",
    "build_inference_engine_config",
    "load_kaggle_runner_config",
    "load_minimal_prompt_profile",
    "read_kaggle_input_csv",
    "run_submission",
    "sort_problem_records",
    "write_debug_sidecar_csv",
    "write_submission_csv",
]
