from __future__ import annotations

"""
Canonical online solve-loop orchestrator.

This file owns the true runtime control loop:
parser -> router -> state init -> retrieval/operator shaping -> branch controller
-> aggregation/final selection -> submission-safe handoff.

It is intentionally explicit about stage boundaries, deterministic controls,
and failure mapping.
"""

import random
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from src.aggregation.entropy_weighting import EntropyWeightingBundle, WeightedClusterScore
from src.aggregation.final_selector import FinalSelectionResult, SelectionWarning, select_final_answer
from src.branches.branch_controller import (
    AggregationResult,
    BranchController,
    BranchControllerResult,
    ControllerConfig,
    FailureDecision,
)
from src.common.constants import ANSWER_MAX, ANSWER_MIN
from src.common.schemas import (
    AttemptBatchResult,
    AttemptEntropySource,
    AttemptRecord,
    BudgetPlan,
    FinalPrediction,
    ParsedProblem,
    RetrievedTrace,
    RouteDecision,
)
from src.operators.operator_library import OperatorLibrary
from src.operators.priors import OperatorPriorShaper
from src.parsing.parser import ParseFailure, ProblemParser, parse_sync as parse_problem_sync
from src.retrieval.embedder import MathEmbedder
from src.retrieval.index_builder import RetrievalIndexBuilder
from src.retrieval.query import TraceRetriever, build_hint_from_traces
from src.retrieval.retrieval_policy import RetrievalPolicy
from src.routing.dual_router import DualRouter, dual_router
from src.state_graph.state_init import (
    StateGraphInitResult,
    StateGraphInitializer,
    initialize_state_graph,
)

try:
    from src.online.adaptive_budget import (
        RuntimeObservation,
        adapt_budget as runtime_adapt_budget,
        build_initial_budget as runtime_build_initial_budget,
    )
except Exception:  # pragma: no cover
    RuntimeObservation = None
    runtime_adapt_budget = None
    runtime_build_initial_budget = None

try:
    from src.online.submission_formatter import format_submission_row
except Exception:  # pragma: no cover
    format_submission_row = None


class SolveStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED_PARSE = "failed_parse"
    FAILED_ROUTE = "failed_route"
    FAILED_BUDGET = "failed_budget"
    FAILED_STATE_INIT = "failed_state_init"
    FAILED_RETRIEVAL = "failed_retrieval"
    FAILED_BRANCH = "failed_branch"
    FAILED_AGGREGATION = "failed_aggregation"
    FAILED_SUBMISSION_FORMAT = "failed_submission_format"


class StageName(str, Enum):
    PARSE = "parse"
    ROUTE = "route"
    BUDGET = "budget"
    STATE_INIT = "state_init"
    RETRIEVAL = "retrieval"
    BRANCH_CONTROLLER = "branch_controller"
    AGGREGATION = "aggregation"
    SUBMISSION = "submission"


class RuntimePath(str, Enum):
    MINIMAL = "minimal"
    ESCALATED = "escalated"
    HARD = "hard"


@dataclass(slots=True)
class StageFailure:
    stage: StageName
    code: str
    message: str
    recoverable: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StageRecord:
    stage: StageName
    ok: bool
    recoverable: bool = False
    elapsed_sec: float = 0.0
    summary: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.elapsed_sec = max(0.0, float(self.elapsed_sec))

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class InferenceEngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deterministic_seed: int = 0
    enable_debug_artifacts: bool = True
    include_submission_row: bool = True
    strict_stage_failures: bool = True
    retrieval_index_path: str = "data/interim/retrieval_index"
    retrieval_top_k_override: int | None = None
    controller_retain_top_k: int = 8
    default_method_name: str = "online_inference_engine"
    include_trace_hint_text: bool = True
    post_branch_budget_adaptation: bool = True
    selective_runtime_enabled: bool = True
    minimal_mode_max_route_uncertainty: float = 0.34
    minimal_mode_max_difficulty_score: float = 0.48
    minimal_mode_max_proof_burden: float = 0.30
    minimal_mode_max_retrieval_need: float = 0.28
    minimal_mode_max_repair_need: float = 0.28
    minimal_mode_self_consistency_samples: int = 8
    minimal_mode_frontier_width: int = 8
    minimal_mode_max_search_depth: int = 1
    minimal_mode_max_search_nodes: int = 1
    minimal_mode_consensus_stop_count: int = 4
    minimal_mode_entropy_epsilon: float = 0.05
    minimal_mode_high_entropy_threshold: float = 1.10
    minimal_mode_high_proxy_uncertainty_threshold: float = 0.42
    minimal_mode_fragmentation_threshold: float = 0.34
    minimal_mode_allow_proxy_entropy: bool = True
    minimal_mode_allow_post_attempt_escalation: bool = True
    escalation_min_confidence: float = 0.62
    escalation_min_answer_agreement: float = 0.50
    escalation_low_margin_threshold: float = 0.08
    mixed_domain_gap_threshold: float = 0.18
    hard_mode_min_route_uncertainty: float = 0.68
    hard_mode_min_difficulty_score: float = 0.80
    hard_mode_min_proof_burden: float = 0.58
    hard_mode_min_retrieval_need: float = 0.45
    hard_mode_min_repair_need: float = 0.45
    escalated_mode_min_retrieval_need: float = 0.34
    escalated_mode_min_verifier_uncertainty: float = 0.28
    escalated_mode_min_symbolic_burden: float = 0.26
    escalated_mode_min_critique_uncertainty: float = 0.40
    escalated_mode_min_repair_need: float = 0.38
    escalated_mode_min_search_uncertainty: float = 0.30
    escalated_mode_min_search_proof_burden: float = 0.26
    prompt_config_path: str = "configs/prompts.yaml"
    minimal_prompt_profile: str = "minimal_default"
    force_runtime_path: str | None = None


@dataclass(slots=True)
class SolveDebugArtifact:
    problem_text: str
    parse_metadata: dict[str, Any] = field(default_factory=dict)
    route_rationale: list[str] = field(default_factory=list)
    budget_summary: dict[str, Any] = field(default_factory=dict)
    state_summary: dict[str, Any] = field(default_factory=dict)
    retrieval_summary: dict[str, Any] = field(default_factory=dict)
    branch_summary: dict[str, Any] = field(default_factory=dict)
    aggregation_summary: dict[str, Any] = field(default_factory=dict)
    submission_summary: dict[str, Any] = field(default_factory=dict)
    stage_records: list[StageRecord] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SolveResultBundle:
    status: SolveStatus
    problem_id: str
    parsed_problem: ParsedProblem | None = None
    route: RouteDecision | None = None
    budget: Any | None = None
    state_init: StateGraphInitResult | None = None
    retrieved_traces: list[RetrievedTrace] = field(default_factory=list)
    branch_result: BranchControllerResult | None = None
    final_selection: FinalSelectionResult | None = None
    final_prediction: FinalPrediction | None = None
    submission_row: Any | None = None
    stage_records: list[StageRecord] = field(default_factory=list)
    failure: StageFailure | None = None
    debug_artifact: SolveDebugArtifact | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in {SolveStatus.SUCCESS, SolveStatus.PARTIAL_SUCCESS}

    def to_kaggle_failure(self) -> dict[str, Any] | None:
        if self.failure is None:
            return None
        return {
            "problem_id": self.problem_id,
            "code": str(self.failure.code),
            "message": str(self.failure.message),
            "recoverable": bool(self.failure.recoverable),
            "details": dict(self.failure.details or {}),
        }

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MinimalRuntimeResult:
    attempts: tuple[AttemptRecord, ...]
    final_selection: FinalSelectionResult
    early_stop_triggered: bool
    should_escalate: bool
    stopped_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


class InferenceEngine:
    """
    Canonical online solve-loop orchestrator.

    The engine is intentionally conservative and typed:
    - all major stages are explicit
    - failures are mapped to stage-specific statuses
    - route-aware budgeting materially shapes retrieval and branch-controller config
    - final output remains separate from CSV-writing concerns
    """

    def __init__(
        self,
        *,
        parser: ProblemParser | None = None,
        router: DualRouter | None = None,
        state_initializer: StateGraphInitializer | None = None,
        retrieval_embedder: MathEmbedder | None = None,
        retrieval_index: RetrievalIndexBuilder | None = None,
        retrieval_policy: RetrievalPolicy | None = None,
        retriever: TraceRetriever | None = None,
        operator_library: OperatorLibrary | None = None,
        prior_shaper: OperatorPriorShaper | None = None,
        branch_controller: BranchController | None = None,
        config: InferenceEngineConfig | None = None,
    ) -> None:
        self.config = config or InferenceEngineConfig()

        self.parser = parser or ProblemParser()
        self.router = router or DualRouter()
        self.state_initializer = state_initializer or StateGraphInitializer()

        self.operator_library = operator_library or OperatorLibrary()
        self.prior_shaper = prior_shaper or OperatorPriorShaper(self.operator_library)

        self.retrieval_embedder = retrieval_embedder or MathEmbedder()
        self.retrieval_index = retrieval_index or RetrievalIndexBuilder(
            embedder=self.retrieval_embedder,
            index_path=self.config.retrieval_index_path,
        )
        self.retrieval_policy = retrieval_policy or RetrievalPolicy()
        self.retriever = retriever or TraceRetriever(
            embedder=self.retrieval_embedder,
            index=self.retrieval_index,
            policy=self.retrieval_policy,
        )

        # Keep a prototype controller for dependency injection compatibility,
        # but instantiate runtime-configured controllers per solve.
        self.branch_controller = branch_controller or BranchController(
            operator_library=self.operator_library,
            prior_shaper=self.prior_shaper,
        )

    def solve_problem(
        self,
        problem_text: str,
        *,
        problem_id: str,
        remaining_problems: int | None = None,
        budget_tracker: Any | None = None,
        parser_model: Callable[[str], Mapping[str, Any]] | None = None,
        generator: Any | None = None,
        symbolic_hook: Any | None = None,
        verifier_hook: Any | None = None,
        failure_classifier: Any | None = None,
        aggregation_hook: Any | None = None,
        critique_model: object | None = None,
    ) -> SolveResultBundle:
        self._set_deterministic_seed(self.config.deterministic_seed)

        started = time.perf_counter()
        stage_records: list[StageRecord] = []

        parsed_problem: ParsedProblem | None = None
        route: RouteDecision | None = None
        budget: Any | None = None
        state_init: StateGraphInitResult | None = None
        retrieved_traces: list[RetrievedTrace] = []
        branch_result: BranchControllerResult | None = None
        final_selection: FinalSelectionResult | None = None
        submission_row: Any | None = None
        runtime_path = RuntimePath.HARD

        parsed_problem, parse_failure = self._run_parse_stage(
            problem_text=problem_text,
            problem_id=problem_id,
            parser_model=parser_model,
            stage_records=stage_records,
        )
        if parse_failure is not None:
            return self._finalize_failure(
                status=SolveStatus.FAILED_PARSE,
                problem_id=problem_id,
                parsed_problem=parsed_problem,
                route=route,
                budget=budget,
                state_init=state_init,
                retrieved_traces=retrieved_traces,
                branch_result=branch_result,
                final_selection=final_selection,
                submission_row=submission_row,
                stage_records=stage_records,
                failure=parse_failure,
                problem_text=problem_text,
                started=started,
            )

        route, route_failure = self._run_route_stage(
            parsed_problem=parsed_problem,
            stage_records=stage_records,
        )
        if route_failure is not None:
            return self._finalize_failure(
                status=SolveStatus.FAILED_ROUTE,
                problem_id=problem_id,
                parsed_problem=parsed_problem,
                route=route,
                budget=budget,
                state_init=state_init,
                retrieved_traces=retrieved_traces,
                branch_result=branch_result,
                final_selection=final_selection,
                submission_row=submission_row,
                stage_records=stage_records,
                failure=route_failure,
                problem_text=problem_text,
                started=started,
            )

        budget, budget_failure = self._run_budget_stage(
            route=route,
            remaining_problems=remaining_problems,
            budget_tracker=budget_tracker,
            stage_records=stage_records,
        )
        if budget_failure is not None and self.config.strict_stage_failures:
            return self._finalize_failure(
                status=SolveStatus.FAILED_BUDGET,
                problem_id=problem_id,
                parsed_problem=parsed_problem,
                route=route,
                budget=budget,
                state_init=state_init,
                retrieved_traces=retrieved_traces,
                branch_result=branch_result,
                final_selection=final_selection,
                submission_row=submission_row,
                stage_records=stage_records,
                failure=budget_failure,
                problem_text=problem_text,
                started=started,
            )

        runtime_path = self._select_runtime_path(route=route, budget=budget)
        if runtime_path is RuntimePath.MINIMAL and callable(getattr(generator, "run_minimal_attempts", None)):
            minimal_result, minimal_failure = self._run_minimal_stage(
                problem_id=problem_id,
                problem_text=problem_text,
                parsed_problem=parsed_problem,
                route=route,
                budget=budget,
                stage_records=stage_records,
                generator=generator,
            )
            if minimal_failure is not None:
                return self._finalize_failure(
                    status=SolveStatus.FAILED_BRANCH,
                    problem_id=problem_id,
                    parsed_problem=parsed_problem,
                    route=route,
                    budget=budget,
                    state_init=state_init,
                    retrieved_traces=retrieved_traces,
                    branch_result=branch_result,
                    final_selection=final_selection,
                    submission_row=submission_row,
                    stage_records=stage_records,
                    failure=minimal_failure,
                    problem_text=problem_text,
                    started=started,
                )
            if minimal_result is not None and not minimal_result.should_escalate:
                final_selection = minimal_result.final_selection
                submission_row, submission_failure = self._run_submission_stage(
                    final_selection=final_selection,
                    stage_records=stage_records,
                )
                if submission_failure is not None:
                    return self._finalize_failure(
                        status=SolveStatus.FAILED_SUBMISSION_FORMAT,
                        problem_id=problem_id,
                        parsed_problem=parsed_problem,
                        route=route,
                        budget=budget,
                        state_init=state_init,
                        retrieved_traces=retrieved_traces,
                        branch_result=branch_result,
                        final_selection=final_selection,
                        submission_row=submission_row,
                        stage_records=stage_records,
                        failure=submission_failure,
                        problem_text=problem_text,
                        started=started,
                    )
                return self._finalize_success(
                    status=SolveStatus.SUCCESS,
                    problem_id=problem_id,
                    parsed_problem=parsed_problem,
                    route=route,
                    budget=budget,
                    state_init=state_init,
                    retrieved_traces=retrieved_traces,
                    branch_result=branch_result,
                    final_selection=final_selection,
                    submission_row=submission_row,
                    stage_records=stage_records,
                    failure=None,
                    problem_text=problem_text,
                    started=started,
                )
            if minimal_result is not None:
                stage_records.append(
                    StageRecord(
                        stage=StageName.BRANCH_CONTROLLER,
                        ok=True,
                        recoverable=True,
                        elapsed_sec=0.0,
                        summary="minimal_runtime_escalated",
                        details={
                            "runtime_path": runtime_path.value,
                            "stopped_reason": minimal_result.stopped_reason,
                            "attempt_count": len(minimal_result.attempts),
                            "valid_attempt_count": sum(1 for item in minimal_result.attempts if item.valid_answer),
                        },
                    )
                )
                runtime_path = RuntimePath.ESCALATED
        elif runtime_path is RuntimePath.MINIMAL:
            stage_records.append(
                StageRecord(
                    stage=StageName.BRANCH_CONTROLLER,
                    ok=True,
                    recoverable=True,
                    elapsed_sec=0.0,
                    summary="minimal_runtime_unavailable_escalated",
                    details={
                        "runtime_path": RuntimePath.MINIMAL.value,
                        "escalated_to": RuntimePath.ESCALATED.value,
                        "reason": "generator_missing_run_minimal_attempts",
                    },
                )
            )
            runtime_path = RuntimePath.ESCALATED

        state_init, state_failure = self._run_state_init_stage(
            parsed_problem=parsed_problem,
            route=route,
            stage_records=stage_records,
        )
        if state_failure is not None:
            return self._finalize_failure(
                status=SolveStatus.FAILED_STATE_INIT,
                problem_id=problem_id,
                parsed_problem=parsed_problem,
                route=route,
                budget=budget,
                state_init=state_init,
                retrieved_traces=retrieved_traces,
                branch_result=branch_result,
                final_selection=final_selection,
                submission_row=submission_row,
                stage_records=stage_records,
                failure=state_failure,
                problem_text=problem_text,
                started=started,
            )

        route_for_branch, retrieved_traces, retrieval_failure = self._run_retrieval_stage(
            parsed_problem=parsed_problem,
            route=route,
            state_init=state_init,
            budget=budget,
            runtime_path=runtime_path,
            stage_records=stage_records,
        )
        if retrieval_failure is not None and self.config.strict_stage_failures:
            return self._finalize_failure(
                status=SolveStatus.FAILED_RETRIEVAL,
                problem_id=problem_id,
                parsed_problem=parsed_problem,
                route=route_for_branch,
                budget=budget,
                state_init=state_init,
                retrieved_traces=retrieved_traces,
                branch_result=branch_result,
                final_selection=final_selection,
                submission_row=submission_row,
                stage_records=stage_records,
                failure=retrieval_failure,
                problem_text=problem_text,
                started=started,
            )

        branch_result, branch_failure = self._run_branch_stage(
            parsed_problem=parsed_problem,
            route=route_for_branch,
            state_init=state_init,
            retrieved_traces=retrieved_traces,
            budget=budget,
            runtime_path=runtime_path,
            stage_records=stage_records,
            generator=generator,
            symbolic_hook=symbolic_hook,
            verifier_hook=verifier_hook,
            failure_classifier=failure_classifier,
            aggregation_hook=aggregation_hook,
            critique_model=critique_model,
        )
        if branch_failure is not None:
            return self._finalize_failure(
                status=SolveStatus.FAILED_BRANCH,
                problem_id=problem_id,
                parsed_problem=parsed_problem,
                route=route_for_branch,
                budget=budget,
                state_init=state_init,
                retrieved_traces=retrieved_traces,
                branch_result=branch_result,
                final_selection=final_selection,
                submission_row=submission_row,
                stage_records=stage_records,
                failure=branch_failure,
                problem_text=problem_text,
                started=started,
            )

        if runtime_path is RuntimePath.MINIMAL:
            preview_selection = self._preview_branch_selection(
                problem_id=problem_id,
                branch_result=branch_result,
            )
            branch_escalation_reasons = self._branch_escalation_reasons(
                route=route_for_branch,
                branch_result=branch_result,
                preview_selection=preview_selection,
            )
            if self._should_escalate_after_branch(
                route=route_for_branch,
                branch_result=branch_result,
                preview_selection=preview_selection,
            ):
                stage_records.append(
                    StageRecord(
                        stage=StageName.BRANCH_CONTROLLER,
                        ok=True,
                        recoverable=True,
                        elapsed_sec=0.0,
                        summary="selective_escalation_triggered",
                        details={
                            "from_runtime_path": runtime_path.value,
                            "to_runtime_path": RuntimePath.ESCALATED.value,
                            "preview_confidence": preview_selection.prediction.confidence,
                            "preview_warnings": [
                                str(warning.value if hasattr(warning, "value") else warning)
                                for warning in preview_selection.warnings
                            ],
                            "preview_abstain_recommended": preview_selection.abstain_recommended,
                            "escalation_reasons": list(branch_escalation_reasons),
                            "gate_thresholds": self._selective_gate_thresholds(),
                        },
                    )
                )
                runtime_path = RuntimePath.ESCALATED
                route_for_branch, retrieved_traces, retrieval_failure = self._run_retrieval_stage(
                    parsed_problem=parsed_problem,
                    route=route,
                    state_init=state_init,
                    budget=budget,
                    runtime_path=runtime_path,
                    stage_records=stage_records,
                )
                if retrieval_failure is not None and self.config.strict_stage_failures:
                    return self._finalize_failure(
                        status=SolveStatus.FAILED_RETRIEVAL,
                        problem_id=problem_id,
                        parsed_problem=parsed_problem,
                        route=route_for_branch,
                        budget=budget,
                        state_init=state_init,
                        retrieved_traces=retrieved_traces,
                        branch_result=branch_result,
                        final_selection=final_selection,
                        submission_row=submission_row,
                        stage_records=stage_records,
                        failure=retrieval_failure,
                        problem_text=problem_text,
                        started=started,
                    )

                branch_result, branch_failure = self._run_branch_stage(
                    parsed_problem=parsed_problem,
                    route=route_for_branch,
                    state_init=state_init,
                    retrieved_traces=retrieved_traces,
                    budget=budget,
                    runtime_path=runtime_path,
                    stage_records=stage_records,
                    generator=generator,
                    symbolic_hook=symbolic_hook,
                    verifier_hook=verifier_hook,
                    failure_classifier=failure_classifier,
                    aggregation_hook=aggregation_hook,
                    critique_model=critique_model,
                )
                if branch_failure is not None:
                    return self._finalize_failure(
                        status=SolveStatus.FAILED_BRANCH,
                        problem_id=problem_id,
                        parsed_problem=parsed_problem,
                        route=route_for_branch,
                        budget=budget,
                        state_init=state_init,
                        retrieved_traces=retrieved_traces,
                        branch_result=branch_result,
                        final_selection=final_selection,
                        submission_row=submission_row,
                        stage_records=stage_records,
                        failure=branch_failure,
                        problem_text=problem_text,
                        started=started,
                    )

        budget = self._post_branch_budget_update(
            budget=budget,
            branch_result=branch_result,
            stage_records=stage_records,
        )

        final_selection, aggregation_failure = self._run_aggregation_stage(
            problem_id=problem_id,
            branch_result=branch_result,
            stage_records=stage_records,
        )
        if aggregation_failure is not None:
            return self._finalize_failure(
                status=SolveStatus.FAILED_AGGREGATION,
                problem_id=problem_id,
                parsed_problem=parsed_problem,
                route=route_for_branch,
                budget=budget,
                state_init=state_init,
                retrieved_traces=retrieved_traces,
                branch_result=branch_result,
                final_selection=final_selection,
                submission_row=submission_row,
                stage_records=stage_records,
                failure=aggregation_failure,
                problem_text=problem_text,
                started=started,
            )

        submission_row, submission_failure = self._run_submission_stage(
            final_selection=final_selection,
            stage_records=stage_records,
        )

        status = SolveStatus.SUCCESS if submission_failure is None else SolveStatus.PARTIAL_SUCCESS

        return self._finalize_success(
            status=status,
            problem_id=problem_id,
            parsed_problem=parsed_problem,
            route=route_for_branch,
            budget=budget,
            state_init=state_init,
            retrieved_traces=retrieved_traces,
            branch_result=branch_result,
            final_selection=final_selection,
            submission_row=submission_row,
            stage_records=stage_records,
            failure=submission_failure,
            problem_text=problem_text,
            started=started,
        )

    def solve_many(
        self,
        problems: Iterable[tuple[str, str]],
        *,
        budget_tracker: Any | None = None,
    ) -> list[SolveResultBundle]:
        pairs = list(problems)
        results: list[SolveResultBundle] = []
        remaining = len(pairs)
        tracker = budget_tracker
        for problem_id, problem_text in pairs:
            result = self.solve_problem(
                problem_text,
                problem_id=problem_id,
                remaining_problems=remaining,
                budget_tracker=tracker,
            )
            results.append(result)
            remaining = max(0, remaining - 1)
        return results

    def _run_parse_stage(
        self,
        *,
        problem_text: str,
        problem_id: str,
        parser_model: Callable[[str], Mapping[str, Any]] | None,
        stage_records: list[StageRecord],
    ) -> tuple[ParsedProblem | None, StageFailure | None]:
        started = time.perf_counter()
        try:
            parsed = self._parse_problem(
                problem_text,
                problem_id=problem_id,
                parser_model=parser_model,
            )
            stage_records.append(
                StageRecord(
                    stage=StageName.PARSE,
                    ok=True,
                    elapsed_sec=_elapsed(started),
                    summary="parsed_problem",
                    details={
                        "answer_type": getattr(parsed, "answer_type", None),
                        "domain": getattr(getattr(parsed, "domain", None), "value", str(getattr(parsed, "domain", ""))),
                        "constraint_count": len(getattr(parsed, "constraints", []) or []),
                        "unknown_count": len(getattr(parsed, "unknowns", []) or []),
                        "parse_quality_keys": sorted((getattr(parsed, "parse_quality", {}) or {}).keys()),
                    },
                )
            )
            return parsed, None
        except Exception as exc:
            failure = self._stage_failure(
                stage=StageName.PARSE,
                code="parse_exception" if not isinstance(exc, ParseFailure) else "parse_failure",
                message=str(exc),
                recoverable=False,
                exc=exc,
            )
            stage_records.append(
                StageRecord(
                    stage=StageName.PARSE,
                    ok=False,
                    recoverable=False,
                    elapsed_sec=_elapsed(started),
                    summary=failure.code,
                    details=failure.details,
                )
            )
            return None, failure

    def _run_route_stage(
        self,
        *,
        parsed_problem: ParsedProblem,
        stage_records: list[StageRecord],
    ) -> tuple[RouteDecision | None, StageFailure | None]:
        started = time.perf_counter()
        try:
            route = self._route_problem(parsed_problem)
            stage_records.append(
                StageRecord(
                    stage=StageName.ROUTE,
                    ok=True,
                    elapsed_sec=_elapsed(started),
                    summary="route_decision",
                    details={
                        "difficulty": route.difficulty.value,
                        "difficulty_score": route.difficulty_score,
                        "route_uncertainty": route.route_uncertainty,
                        "branch_budget": route.branch_budget,
                        "retrieval_depth": route.retrieval_depth,
                    },
                )
            )
            return route, None
        except Exception as exc:
            failure = self._stage_failure(
                stage=StageName.ROUTE,
                code="route_exception",
                message=str(exc),
                recoverable=False,
                exc=exc,
            )
            stage_records.append(
                StageRecord(
                    stage=StageName.ROUTE,
                    ok=False,
                    elapsed_sec=_elapsed(started),
                    summary=failure.code,
                    details=failure.details,
                )
            )
            return None, failure

    def _run_budget_stage(
        self,
        *,
        route: RouteDecision,
        remaining_problems: int | None,
        budget_tracker: Any | None,
        stage_records: list[StageRecord],
    ) -> tuple[Any | None, StageFailure | None]:
        started = time.perf_counter()
        try:
            if runtime_build_initial_budget is not None:
                budget = runtime_build_initial_budget(
                    route,
                    remaining_problems=remaining_problems,
                    tracker=budget_tracker,
                )
                details = _safe_budget_summary(budget)
            else:
                budget = route.budget_plan
                details = {
                    "mode": "route_budget_plan_fallback",
                    "branch_budget": route.budget_plan.branch_budget,
                    "retrieval_depth": route.budget_plan.retrieval_depth,
                    "max_search_depth": route.budget_plan.max_search_depth,
                    "max_search_nodes": route.budget_plan.max_search_nodes,
                    "repair_budget": route.budget_plan.repair_budget,
                    "self_consistency_samples": route.budget_plan.self_consistency_samples,
                    "critique_top_k": route.budget_plan.critique_top_k,
                }
            stage_records.append(
                StageRecord(
                    stage=StageName.BUDGET,
                    ok=True,
                    elapsed_sec=_elapsed(started),
                    summary="budget_acquired",
                    details=details,
                )
            )
            return budget, None
        except Exception as exc:
            failure = self._stage_failure(
                stage=StageName.BUDGET,
                code="budget_exception",
                message=str(exc),
                recoverable=False,
                exc=exc,
            )
            stage_records.append(
                StageRecord(
                    stage=StageName.BUDGET,
                    ok=False,
                    elapsed_sec=_elapsed(started),
                    summary=failure.code,
                    details=failure.details,
                )
            )
            return None, failure

    def _run_state_init_stage(
        self,
        *,
        parsed_problem: ParsedProblem,
        route: RouteDecision,
        stage_records: list[StageRecord],
    ) -> tuple[StateGraphInitResult | None, StageFailure | None]:
        started = time.perf_counter()
        try:
            state_init = self._initialize_state(parsed_problem, route)
            stage_records.append(
                StageRecord(
                    stage=StageName.STATE_INIT,
                    ok=True,
                    elapsed_sec=_elapsed(started),
                    summary="initial_state_built",
                    details={
                        "root_node_id": state_init.metadata.root_node_id,
                        "canonical_state_hash": state_init.metadata.canonical_state_hash,
                        "top_problem_type": state_init.metadata.top_problem_type,
                        "top_archetypes": list(state_init.metadata.top_archetypes),
                    },
                )
            )
            return state_init, None
        except Exception as exc:
            failure = self._stage_failure(
                stage=StageName.STATE_INIT,
                code="state_init_exception",
                message=str(exc),
                recoverable=False,
                exc=exc,
            )
            stage_records.append(
                StageRecord(
                    stage=StageName.STATE_INIT,
                    ok=False,
                    elapsed_sec=_elapsed(started),
                    summary=failure.code,
                    details=failure.details,
                )
            )
            return None, failure

    def _run_retrieval_stage(
        self,
        *,
        parsed_problem: ParsedProblem,
        route: RouteDecision,
        state_init: StateGraphInitResult,
        budget: Any | None,
        runtime_path: RuntimePath,
        stage_records: list[StageRecord],
    ) -> tuple[RouteDecision, list[RetrievedTrace], StageFailure | None]:
        started = time.perf_counter()
        shaped_route = route
        traces: list[RetrievedTrace] = []

        try:
            retrieval_depth = _budget_retrieval_depth(budget, route.budget_plan)
            should_retrieve = bool(getattr(route, "use_retrieval", True) and retrieval_depth > 0)
            if runtime_path is RuntimePath.MINIMAL:
                should_retrieve = False
            selective_policy = None
            if runtime_path is RuntimePath.ESCALATED:
                selective_policy = self._resolve_escalated_policy(route=route)
                should_retrieve = should_retrieve and bool(selective_policy["retrieval_active"])

            if should_retrieve and self._retrieval_index_loaded():
                query = self.retriever.build_query(
                    parsed_problem,
                    route,
                    reasoning_state=state_init.root_node,
                )
                top_k = self.config.retrieval_top_k_override or retrieval_depth
                traces = self.retriever.retrieve_query(query, route, top_k=top_k)
                shaped_route = self._shape_route_from_retrieval(route, traces)
                trace_hint = build_hint_from_traces(traces) if self.config.include_trace_hint_text else None
                stage_records.append(
                    StageRecord(
                        stage=StageName.RETRIEVAL,
                        ok=True,
                        elapsed_sec=_elapsed(started),
                        summary="retrieval_completed",
                        details={
                            "retrieval_enabled": True,
                            "runtime_path": runtime_path.value,
                            "retrieved_trace_count": len(traces),
                            "top_k": top_k,
                            "trace_ids": [t.trace_id for t in traces[:5]],
                            "trace_hint_available": bool(trace_hint),
                            "operator_prior_shaped": shaped_route.operator_prior != route.operator_prior,
                            "selective_policy": dict(selective_policy or {}),
                            "gate_thresholds": self._selective_gate_thresholds(),
                        },
                    )
                )
                return shaped_route, traces, None

            stage_records.append(
                StageRecord(
                    stage=StageName.RETRIEVAL,
                    ok=True,
                    recoverable=True,
                    elapsed_sec=_elapsed(started),
                    summary=(
                        "retrieval_bypassed_selective_minimal_mode"
                        if runtime_path is RuntimePath.MINIMAL
                        else "retrieval_skipped"
                    ),
                    details={
                        "retrieval_enabled": should_retrieve,
                        "runtime_path": runtime_path.value,
                        "index_loaded": self._retrieval_index_loaded(),
                        "retrieval_depth": retrieval_depth,
                        "selective_policy": dict(selective_policy or {}),
                        "gate_thresholds": self._selective_gate_thresholds(),
                    },
                )
            )
            return shaped_route, traces, None

        except Exception as exc:
            failure = self._stage_failure(
                stage=StageName.RETRIEVAL,
                code="retrieval_exception",
                message=str(exc),
                recoverable=True,
                exc=exc,
            )
            stage_records.append(
                StageRecord(
                    stage=StageName.RETRIEVAL,
                    ok=False,
                    recoverable=True,
                    elapsed_sec=_elapsed(started),
                    summary=failure.code,
                    details=failure.details,
                )
            )
            return route, [], failure

    def _run_minimal_stage(
        self,
        *,
        problem_id: str,
        problem_text: str,
        parsed_problem: ParsedProblem,
        route: RouteDecision,
        budget: Any | None,
        stage_records: list[StageRecord],
        generator: Any,
    ) -> tuple[MinimalRuntimeResult | None, StageFailure | None]:
        started = time.perf_counter()
        try:
            batch = generator.run_minimal_attempts(
                problem=parsed_problem,
                route=route,
                attempt_count=int(self.config.minimal_mode_self_consistency_samples),
                early_stop_threshold=int(self.config.minimal_mode_consensus_stop_count),
            )
            if isinstance(batch, AttemptBatchResult):
                attempts = tuple(batch.attempts)
                batch_metadata = dict(batch.metadata or {})
                stopped_reason = batch.stopped_reason
            else:
                attempts = tuple(getattr(batch, "attempts", ()) or ())
                batch_metadata = dict(getattr(batch, "metadata", {}) or {})
                stopped_reason = str(getattr(batch, "stopped_reason", "completed"))

            selection = self._select_from_attempt_records(
                problem_id=problem_id,
                attempts=attempts,
                stopped_reason=stopped_reason,
            )
            early_stop_triggered = stopped_reason == "early_stop_consensus"
            escalation_reasons = self._attempt_escalation_reasons(
                route=route,
                selection=selection,
                attempts=attempts,
                early_stop_triggered=early_stop_triggered,
            )
            should_escalate = self._should_escalate_after_attempts(
                route=route,
                selection=selection,
                attempts=attempts,
                early_stop_triggered=early_stop_triggered,
            )
            valid_attempts = [item for item in attempts if item.valid_answer]
            stage_records.append(
                StageRecord(
                    stage=StageName.BRANCH_CONTROLLER,
                    ok=True,
                    elapsed_sec=_elapsed(started),
                    summary="minimal_attempt_loop_completed",
                    details={
                        "runtime_path": RuntimePath.MINIMAL.value,
                        "attempt_count": len(attempts),
                        "valid_attempt_count": len(valid_attempts),
                        "stopped_reason": stopped_reason,
                        "early_stop_triggered": early_stop_triggered,
                        "tool_call_count": sum(int(getattr(item, "tool_call_count", 0)) for item in attempts),
                        "tool_error_count": sum(int(getattr(item, "tool_error_count", 0)) for item in attempts),
                        "entropy_sources": {
                            source.value: sum(1 for item in attempts if item.entropy_source is source)
                            for source in AttemptEntropySource
                        },
                        "prompt_profile": batch_metadata.get("prompt_profile", self.config.minimal_prompt_profile),
                        "escalation_reasons": list(escalation_reasons),
                        "gate_thresholds": self._selective_gate_thresholds(),
                    },
                )
            )
            stage_records.append(
                StageRecord(
                    stage=StageName.AGGREGATION,
                    ok=True,
                    elapsed_sec=0.0,
                    summary="minimal_attempt_selection_completed",
                    details={
                        "submission_answer": selection.submission_answer,
                        "confidence": selection.prediction.confidence,
                        "warnings": [str(item.value if hasattr(item, "value") else item) for item in selection.warnings],
                        "abstain_recommended": selection.abstain_recommended,
                        "attempt_weighting_mode": selection.metadata.get("attempt_weighting_mode"),
                        "entropy_source_counts": dict(selection.metadata.get("entropy_source_counts", {}) or {}),
                        "should_escalate": should_escalate,
                        "escalation_reasons": list(escalation_reasons),
                        "gate_thresholds": self._selective_gate_thresholds(),
                    },
                )
            )
            return (
                MinimalRuntimeResult(
                    attempts=attempts,
                    final_selection=selection,
                    early_stop_triggered=early_stop_triggered,
                    should_escalate=should_escalate,
                    stopped_reason=stopped_reason,
                    metadata={
                        **batch_metadata,
                        "problem_text_preview": problem_text[:120],
                        "escalation_reasons": list(escalation_reasons),
                        "gate_thresholds": self._selective_gate_thresholds(),
                    },
                ),
                None,
            )
        except Exception as exc:
            failure = self._stage_failure(
                stage=StageName.BRANCH_CONTROLLER,
                code="minimal_runtime_exception",
                message=str(exc),
                recoverable=False,
                exc=exc,
            )
            stage_records.append(
                StageRecord(
                    stage=StageName.BRANCH_CONTROLLER,
                    ok=False,
                    elapsed_sec=_elapsed(started),
                    summary=failure.code,
                    details=failure.details,
                )
            )
            return None, failure

    def _run_branch_stage(
        self,
        *,
        parsed_problem: ParsedProblem,
        route: RouteDecision,
        state_init: StateGraphInitResult,
        retrieved_traces: Sequence[RetrievedTrace],
        budget: Any | None,
        runtime_path: RuntimePath,
        stage_records: list[StageRecord],
        generator: Any | None,
        symbolic_hook: Any | None,
        verifier_hook: Any | None,
        failure_classifier: Any | None,
        aggregation_hook: Any | None,
        critique_model: object | None,
    ) -> tuple[BranchControllerResult | None, StageFailure | None]:
        started = time.perf_counter()
        try:
            controller_config = self._controller_config_from_budget(
                budget,
                route.budget_plan,
                route=route,
                runtime_path=runtime_path,
            )
            controller = self._make_runtime_controller(controller_config)
            resolved_hooks = self._resolve_branch_runtime_policy(
                route=route,
                runtime_path=runtime_path,
                controller_config=controller_config,
                symbolic_hook=symbolic_hook,
                verifier_hook=verifier_hook,
                failure_classifier=failure_classifier,
                aggregation_hook=aggregation_hook,
                critique_model=critique_model,
            )

            result = controller.solve(
                problem=parsed_problem,
                route=route,
                root_node=state_init.root_node,
                retrieved_traces=retrieved_traces,
                generator=generator,
                symbolic_hook=resolved_hooks["symbolic_hook"],
                verifier_hook=resolved_hooks["verifier_hook"],
                failure_classifier=resolved_hooks["failure_classifier"],
                aggregation_hook=resolved_hooks["aggregation_hook"],
                critique_model=resolved_hooks["critique_model"],
            )
            stage_records.append(
                StageRecord(
                    stage=StageName.BRANCH_CONTROLLER,
                    ok=True,
                    elapsed_sec=_elapsed(started),
                    summary=result.stopped_reason,
                    details={
                        "num_all_branches": len(result.all_branches),
                        "num_surviving_branches": len(result.surviving_branches),
                        "critiqued_branch_ids": list(result.critiqued_branch_ids),
                        "candidate_cluster_count": len(result.candidate_clusters),
                        "selected_for_critique": list(result.selected_for_critique),
                        "final_confidence": result.final_prediction.confidence,
                        "runtime_path": runtime_path.value,
                        "symbolic_hook_active": resolved_hooks["symbolic_hook"] is not None,
                        "verifier_hook_active": resolved_hooks["verifier_hook"] is not None,
                        "critique_active": resolved_hooks["critique_model"] is not None,
                        "selective_policy": dict(resolved_hooks.get("selective_policy") or {}),
                    },
                )
            )
            return result, None
        except Exception as exc:
            failure = self._stage_failure(
                stage=StageName.BRANCH_CONTROLLER,
                code="branch_controller_exception",
                message=str(exc),
                recoverable=False,
                exc=exc,
            )
            stage_records.append(
                StageRecord(
                    stage=StageName.BRANCH_CONTROLLER,
                    ok=False,
                    elapsed_sec=_elapsed(started),
                    summary=failure.code,
                    details=failure.details,
                )
            )
            return None, failure

    def _run_aggregation_stage(
        self,
        *,
        problem_id: str,
        branch_result: BranchControllerResult,
        stage_records: list[StageRecord],
    ) -> tuple[FinalSelectionResult | None, StageFailure | None]:
        started = time.perf_counter()
        try:
            candidate_clusters = tuple(branch_result.candidate_clusters or ())
            if not candidate_clusters:
                fp = branch_result.final_prediction
                candidate_clusters = (
                    fp.winning_cluster.model_copy(
                        update={
                            "cluster_size": max(1, fp.winning_cluster.cluster_size),
                            "answer_agreement": max(fp.winning_cluster.answer_agreement, 1.0),
                            "composite_score": max(fp.winning_cluster.composite_score, fp.confidence),
                        }
                    ),
                )

            selection = select_final_answer(
                problem_id=problem_id,
                weighted_clusters=candidate_clusters,
                num_branches_generated=branch_result.final_prediction.num_branches_generated,
                num_branches_survived=branch_result.final_prediction.num_branches_survived,
                solve_time_sec=branch_result.final_prediction.solve_time_sec,
                method_used=branch_result.final_prediction.method_used,
            )
            stage_records.append(
                StageRecord(
                    stage=StageName.AGGREGATION,
                    ok=True,
                    elapsed_sec=_elapsed(started),
                    summary="final_selection_completed",
                    details={
                        "candidate_cluster_count": len(candidate_clusters),
                        "submission_answer": selection.submission_answer,
                        "confidence": selection.prediction.confidence,
                        "warnings": [str(w.value if hasattr(w, "value") else w) for w in selection.warnings],
                        "abstain_recommended": selection.abstain_recommended,
                    },
                )
            )
            return selection, None
        except Exception as exc:
            failure = self._stage_failure(
                stage=StageName.AGGREGATION,
                code="aggregation_exception",
                message=str(exc),
                recoverable=False,
                exc=exc,
            )
            stage_records.append(
                StageRecord(
                    stage=StageName.AGGREGATION,
                    ok=False,
                    elapsed_sec=_elapsed(started),
                    summary=failure.code,
                    details=failure.details,
                )
            )
            return None, failure

    def _run_submission_stage(
        self,
        *,
        final_selection: FinalSelectionResult,
        stage_records: list[StageRecord],
    ) -> tuple[Any | None, StageFailure | None]:
        started = time.perf_counter()
        try:
            if not self.config.include_submission_row:
                stage_records.append(
                    StageRecord(
                        stage=StageName.SUBMISSION,
                        ok=True,
                        recoverable=True,
                        elapsed_sec=_elapsed(started),
                        summary="submission_skipped_by_config",
                        details={},
                    )
                )
                return None, None

            if format_submission_row is not None:
                row = format_submission_row(final_selection)
            else:
                row = {
                    "id": final_selection.prediction.problem_id,
                    "answer": final_selection.submission_answer,
                }

            stage_records.append(
                StageRecord(
                    stage=StageName.SUBMISSION,
                    ok=True,
                    elapsed_sec=_elapsed(started),
                    summary="submission_row_emitted",
                    details={"row": _stable_row_preview(row)},
                )
            )
            return row, None
        except Exception as exc:
            failure = self._stage_failure(
                stage=StageName.SUBMISSION,
                code="submission_format_exception",
                message=str(exc),
                recoverable=True,
                exc=exc,
            )
            stage_records.append(
                StageRecord(
                    stage=StageName.SUBMISSION,
                    ok=False,
                    recoverable=True,
                    elapsed_sec=_elapsed(started),
                    summary=failure.code,
                    details=failure.details,
                )
            )
            return None, failure

    def _parse_problem(
        self,
        problem_text: str,
        *,
        problem_id: str,
        parser_model: Callable[[str], Mapping[str, Any]] | None,
    ) -> ParsedProblem:
        if hasattr(self.parser, "parse_sync"):
            return self.parser.parse_sync(
                problem_text,
                problem_id=problem_id,
                model_parser=parser_model,
            )
        return parse_problem_sync(
            problem_text,
            problem_id=problem_id,
            model_parser=parser_model,
        )

    def _route_problem(self, parsed_problem: ParsedProblem) -> RouteDecision:
        if hasattr(self.router, "route"):
            return self.router.route(parsed_problem)
        return dual_router(parsed_problem)

    def _initialize_state(self, parsed_problem: ParsedProblem, route: RouteDecision) -> StateGraphInitResult:
        if hasattr(self.state_initializer, "initialize"):
            return self.state_initializer.initialize(parsed_problem, route)
        return initialize_state_graph(parsed_problem, route)

    def _retrieval_index_loaded(self) -> bool:
        value = getattr(self.retrieval_index, "is_loaded", False)
        if callable(value):
            try:
                return bool(value())
            except Exception:
                return False
        return bool(value)

    def _shape_route_from_retrieval(
        self,
        route: RouteDecision,
        traces: Sequence[RetrievedTrace],
    ) -> RouteDecision:
        if not traces:
            return route

        try:
            adjusted_prior = self.retrieval_policy.compute_prior_adjustment(
                list(traces),
                dict(route.operator_prior),
                blend_weight=0.30,
            )
        except Exception:
            adjusted_prior = dict(route.operator_prior)

        diagnostics = dict(route.diagnostics or {})
        diagnostics["retrieval_trace_count"] = len(traces)
        diagnostics["retrieval_trace_ids"] = [t.trace_id for t in traces[:8]]
        diagnostics["operator_prior_shaped_by_retrieval"] = adjusted_prior != route.operator_prior

        return route.model_copy(
            update={
                "operator_prior": adjusted_prior,
                "diagnostics": diagnostics,
            }
        )

    def _route_runtime_metrics(self, *, route: RouteDecision, budget: Any | None = None) -> dict[str, float | bool | str]:
        difficulty = getattr(getattr(route, "difficulty", None), "value", str(getattr(route, "difficulty", ""))).lower()
        difficulty_score = _clamp01(getattr(route, "difficulty_score", 0.0))
        route_uncertainty = float(
            getattr(budget, "route_uncertainty", getattr(route, "route_uncertainty", 0.0))
        )
        signals = dict(getattr(route, "compute_signals", {}) or {})
        proof_burden = _clamp01(signals.get("proof_burden", 0.0))
        retrieval_need = _clamp01(signals.get("retrieval_need", 0.0))
        repair_need = _clamp01(signals.get("repair_need", 0.0))
        top_two_gap = _top_two_gap(getattr(route, "problem_type_probs", {}) or getattr(route, "problem_type", {}))
        mixed_domain = top_two_gap < self.config.mixed_domain_gap_threshold
        return {
            "difficulty": difficulty,
            "difficulty_score": difficulty_score,
            "route_uncertainty": route_uncertainty,
            "proof_burden": proof_burden,
            "retrieval_need": retrieval_need,
            "repair_need": repair_need,
            "top_two_gap": top_two_gap,
            "mixed_domain": mixed_domain,
        }

    def _selective_gate_thresholds(self) -> dict[str, float | bool]:
        return {
            "minimal_mode_max_route_uncertainty": float(self.config.minimal_mode_max_route_uncertainty),
            "minimal_mode_max_difficulty_score": float(self.config.minimal_mode_max_difficulty_score),
            "minimal_mode_max_proof_burden": float(self.config.minimal_mode_max_proof_burden),
            "minimal_mode_max_retrieval_need": float(self.config.minimal_mode_max_retrieval_need),
            "minimal_mode_max_repair_need": float(self.config.minimal_mode_max_repair_need),
            "minimal_mode_high_entropy_threshold": float(self.config.minimal_mode_high_entropy_threshold),
            "minimal_mode_high_proxy_uncertainty_threshold": float(self.config.minimal_mode_high_proxy_uncertainty_threshold),
            "minimal_mode_fragmentation_threshold": float(self.config.minimal_mode_fragmentation_threshold),
            "minimal_mode_allow_post_attempt_escalation": bool(self.config.minimal_mode_allow_post_attempt_escalation),
            "escalation_min_confidence": float(self.config.escalation_min_confidence),
            "escalation_min_answer_agreement": float(self.config.escalation_min_answer_agreement),
            "escalation_low_margin_threshold": float(self.config.escalation_low_margin_threshold),
            "mixed_domain_gap_threshold": float(self.config.mixed_domain_gap_threshold),
            "hard_mode_min_route_uncertainty": float(self.config.hard_mode_min_route_uncertainty),
            "hard_mode_min_difficulty_score": float(self.config.hard_mode_min_difficulty_score),
            "hard_mode_min_proof_burden": float(self.config.hard_mode_min_proof_burden),
            "hard_mode_min_retrieval_need": float(self.config.hard_mode_min_retrieval_need),
            "hard_mode_min_repair_need": float(self.config.hard_mode_min_repair_need),
            "escalated_mode_min_retrieval_need": float(self.config.escalated_mode_min_retrieval_need),
            "escalated_mode_min_verifier_uncertainty": float(self.config.escalated_mode_min_verifier_uncertainty),
            "escalated_mode_min_symbolic_burden": float(self.config.escalated_mode_min_symbolic_burden),
            "escalated_mode_min_critique_uncertainty": float(self.config.escalated_mode_min_critique_uncertainty),
            "escalated_mode_min_repair_need": float(self.config.escalated_mode_min_repair_need),
            "escalated_mode_min_search_uncertainty": float(self.config.escalated_mode_min_search_uncertainty),
            "escalated_mode_min_search_proof_burden": float(self.config.escalated_mode_min_search_proof_burden),
        }

    def _attempt_escalation_reasons(
        self,
        *,
        route: RouteDecision,
        selection: FinalSelectionResult,
        attempts: Sequence[AttemptRecord],
        early_stop_triggered: bool,
    ) -> list[str]:
        if not self.config.minimal_mode_allow_post_attempt_escalation:
            return []
        if early_stop_triggered:
            return []
        valid_attempts = [item for item in attempts if item.valid_answer]
        if not valid_attempts:
            return ["no_valid_attempts"]

        true_entropy = [
            float(item.mean_token_entropy)
            for item in valid_attempts
            if item.entropy_source is AttemptEntropySource.TRUE_LOGPROBS and item.mean_token_entropy is not None
        ]
        proxy_uncertainty = [
            float(item.mean_token_entropy)
            for item in valid_attempts
            if item.entropy_source is AttemptEntropySource.PROXY_CONFIDENCE and item.mean_token_entropy is not None
        ]
        warnings = set(selection.warnings)
        reasons: list[str] = []
        if selection.prediction.confidence < self.config.escalation_min_confidence:
            reasons.append("low_confidence")
        if float(selection.prediction.winning_cluster.answer_agreement) < self.config.escalation_min_answer_agreement:
            reasons.append("low_agreement")
        if selection.metadata.get("margin_to_runner_up", 1.0) < self.config.escalation_low_margin_threshold:
            reasons.append("low_margin")
        if true_entropy and _mean(true_entropy) >= self.config.minimal_mode_high_entropy_threshold:
            reasons.append("high_true_entropy")
        if proxy_uncertainty and _mean(proxy_uncertainty) >= self.config.minimal_mode_high_proxy_uncertainty_threshold:
            reasons.append("high_proxy_uncertainty")
        if selection.prediction.signal_decomposition.get("disagreement_level", 0.0) >= self.config.minimal_mode_fragmentation_threshold:
            reasons.append("fragmented_answer_families")
        if SelectionWarning.HIGH_DISAGREEMENT in warnings:
            reasons.append("warning_high_disagreement")
        if SelectionWarning.FRAGMENTED_SUPPORT in warnings:
            reasons.append("warning_fragmented_support")
        if SelectionWarning.STRONG_EVIDENCE_FRAGMENTED_SUPPORT in warnings:
            reasons.append("warning_strong_evidence_fragmentation")
        if SelectionWarning.HIGH_OBLIGATION_BURDEN in warnings:
            reasons.append("warning_high_obligation_burden")
        if float(getattr(route, "route_uncertainty", 0.0)) >= self.config.minimal_mode_max_route_uncertainty:
            reasons.append("high_route_uncertainty")
        return reasons

    def _branch_escalation_reasons(
        self,
        *,
        route: RouteDecision,
        branch_result: BranchControllerResult,
        preview_selection: FinalSelectionResult,
    ) -> list[str]:
        warnings = set(preview_selection.warnings)
        reasons: list[str] = []
        if preview_selection.abstain_recommended:
            reasons.append("abstain_recommended")
        if preview_selection.prediction.confidence < self.config.escalation_min_confidence:
            reasons.append("low_confidence")
        if float(branch_result.final_prediction.winning_cluster.answer_agreement) < self.config.escalation_min_answer_agreement:
            reasons.append("low_agreement")
        if preview_selection.metadata.get("margin_to_runner_up", 1.0) < self.config.escalation_low_margin_threshold:
            reasons.append("low_margin")
        if SelectionWarning.HIGH_DISAGREEMENT in warnings:
            reasons.append("warning_high_disagreement")
        if SelectionWarning.FRAGMENTED_SUPPORT in warnings:
            reasons.append("warning_fragmented_support")
        if SelectionWarning.STRONG_EVIDENCE_FRAGMENTED_SUPPORT in warnings:
            reasons.append("warning_strong_evidence_fragmentation")
        if SelectionWarning.HIGH_OBLIGATION_BURDEN in warnings:
            reasons.append("warning_high_obligation_burden")
        if len(preview_selection.ranked_clusters) > 2:
            reasons.append("many_answer_families")
        if float(getattr(route, "route_uncertainty", 0.0)) >= self.config.minimal_mode_max_route_uncertainty:
            reasons.append("high_route_uncertainty")
        return reasons

    def _resolve_escalated_policy(self, *, route: RouteDecision) -> dict[str, Any]:
        metrics = self._route_runtime_metrics(route=route)
        route_uncertainty = float(metrics["route_uncertainty"])
        proof_burden = float(metrics["proof_burden"])
        retrieval_need = float(metrics["retrieval_need"])
        repair_need = float(metrics["repair_need"])
        mixed_domain = bool(metrics["mixed_domain"])
        retrieval_active = (
            retrieval_need >= self.config.escalated_mode_min_retrieval_need
            or route_uncertainty >= self.config.escalated_mode_min_verifier_uncertainty
            or mixed_domain
        )
        verifier_active = (
            route_uncertainty >= self.config.escalated_mode_min_verifier_uncertainty
            or proof_burden >= self.config.escalated_mode_min_symbolic_burden
            or repair_need >= self.config.escalated_mode_min_repair_need
        )
        symbolic_active = (
            proof_burden >= self.config.escalated_mode_min_symbolic_burden
            or route_uncertainty >= self.config.escalated_mode_min_verifier_uncertainty
        )
        critique_active = (
            route_uncertainty >= self.config.escalated_mode_min_critique_uncertainty
            or repair_need >= self.config.escalated_mode_min_repair_need
        )
        repair_active = repair_need >= self.config.escalated_mode_min_repair_need
        search_active = (
            route_uncertainty >= self.config.escalated_mode_min_search_uncertainty
            or proof_burden >= self.config.escalated_mode_min_search_proof_burden
            or repair_active
            or mixed_domain
        )
        return {
            "metrics": metrics,
            "retrieval_active": retrieval_active,
            "verifier_active": verifier_active,
            "symbolic_active": symbolic_active,
            "critique_active": critique_active,
            "repair_active": repair_active,
            "search_active": search_active,
        }

    def _select_runtime_path(self, *, route: RouteDecision, budget: Any | None) -> RuntimePath:
        if self.config.force_runtime_path:
            try:
                return RuntimePath(str(self.config.force_runtime_path))
            except Exception:
                pass
        if not self.config.selective_runtime_enabled:
            return RuntimePath.HARD
        metrics = self._route_runtime_metrics(route=route, budget=budget)
        difficulty = str(metrics["difficulty"])
        difficulty_score = float(metrics["difficulty_score"])
        route_uncertainty = float(metrics["route_uncertainty"])
        proof_burden = float(metrics["proof_burden"])
        retrieval_need = float(metrics["retrieval_need"])
        repair_need = float(metrics["repair_need"])
        mixed_domain = bool(metrics["mixed_domain"])

        if (
            difficulty == "very_hard"
            or route_uncertainty >= self.config.hard_mode_min_route_uncertainty
            or difficulty_score >= self.config.hard_mode_min_difficulty_score
            or (
                difficulty == "hard"
                and proof_burden >= self.config.hard_mode_min_proof_burden
                and (
                    retrieval_need >= self.config.hard_mode_min_retrieval_need
                    or repair_need >= self.config.hard_mode_min_repair_need
                    or mixed_domain
                )
            )
        ):
            return RuntimePath.HARD

        if (
            difficulty in {"easy", "medium"}
            and difficulty_score <= self.config.minimal_mode_max_difficulty_score
            and route_uncertainty <= self.config.minimal_mode_max_route_uncertainty
            and proof_burden <= self.config.minimal_mode_max_proof_burden
            and retrieval_need <= self.config.minimal_mode_max_retrieval_need
            and repair_need <= self.config.minimal_mode_max_repair_need
            and not mixed_domain
        ):
            return RuntimePath.MINIMAL

        return RuntimePath.ESCALATED

    def _select_from_attempt_records(
        self,
        *,
        problem_id: str,
        attempts: Sequence[AttemptRecord],
        stopped_reason: str,
    ) -> FinalSelectionResult:
        weighting_mode = self._attempt_weighting_mode(attempts)
        weighting_bundle = self._build_attempt_weighting_bundle(attempts, weighting_mode=weighting_mode)
        method_used = "minimal_early_stop_consensus"
        if stopped_reason != "early_stop_consensus":
            if weighting_mode == "true_entropy":
                method_used = "minimal_entropy_weighted_voting"
            elif weighting_mode == "proxy_uncertainty":
                method_used = "minimal_proxy_weighted_voting"
            else:
                method_used = "minimal_equal_weight_voting"
        selection = select_final_answer(
            problem_id=problem_id,
            weighted_clusters=weighting_bundle,
            num_branches_generated=len(attempts),
            num_branches_survived=sum(1 for item in attempts if item.valid_answer),
            solve_time_sec=sum(float(item.runtime_sec) for item in attempts),
            method_used=method_used,
        )
        return replace(
            selection,
            metadata={
                **dict(selection.metadata or {}),
                "attempt_weighting_mode": weighting_mode,
                "entropy_source_counts": {
                    source.value: sum(1 for item in attempts if item.entropy_source is source)
                    for source in AttemptEntropySource
                },
                "attempt_weighting_fallback": self._attempt_weighting_fallback(attempts, weighting_mode),
                "stopped_reason": stopped_reason,
                "valid_attempt_count": sum(1 for item in attempts if item.valid_answer),
                "attempt_count": len(attempts),
            },
        )

    def _build_attempt_weighting_bundle(
        self,
        attempts: Sequence[AttemptRecord],
        *,
        weighting_mode: str | None = None,
    ) -> EntropyWeightingBundle:
        valid_attempts = [item for item in attempts if item.valid_answer and item.extracted_answer is not None]
        if not valid_attempts:
            return EntropyWeightingBundle(
                winner=None,
                ranked_clusters=(),
                ranked_candidates=(),
                global_entropy=1.0,
                total_branches=len(attempts),
            )

        resolved_mode = weighting_mode or self._attempt_weighting_mode(attempts)
        grouped: dict[str, list[AttemptRecord]] = defaultdict(list)
        total_weight = 0.0
        for attempt in valid_attempts:
            grouped[str(attempt.extracted_answer)].append(attempt)
            total_weight += self._attempt_vote_weight(attempt, weighting_mode=resolved_mode)

        weighted_clusters: list[WeightedClusterScore] = []
        for cluster_index, answer in enumerate(sorted(grouped)):
            members = grouped[answer]
            member_weights = [self._attempt_vote_weight(item, weighting_mode=resolved_mode) for item in members]
            cluster_weight = sum(member_weights)
            support_size = len(members)
            support_share = support_size / max(1, len(valid_attempts))
            diversity_weighted_support = cluster_weight / max(self.config.minimal_mode_entropy_epsilon, total_weight)
            mean_entropy = _mean(
                [
                    _attempt_uncertainty(item)
                    for item in members
                ]
            )
            best_source = _dominant_entropy_source(members)
            mean_confidence = _clamp01(
                _mean(
                    [
                        _attempt_confidence(item)
                        for item in members
                    ]
                )
            )
            caution_flags: list[str] = []
            if support_size == 1:
                caution_flags.append("singleton")
            if best_source is AttemptEntropySource.PROXY_CONFIDENCE:
                caution_flags.append("proxy_entropy_only")
            if best_source is AttemptEntropySource.UNAVAILABLE:
                caution_flags.append("no_entropy")

            weighted_clusters.append(
                WeightedClusterScore(
                    cluster_id=f"attempt::{cluster_index:03d}::{answer}",
                    answer=answer,
                    answer_canonical=answer,
                    branch_ids=tuple(item.attempt_id for item in members),
                    cluster_size=support_size,
                    support_share=_clamp01(support_share),
                    diversity_weighted_support=_clamp01(diversity_weighted_support),
                    unique_branch_families=support_size,
                    family_diversity=1.0,
                    verifier_score=mean_confidence,
                    weak_symbolic_support=0.0,
                    exact_symbolic_support=0.0,
                    retrieval_support=0.0,
                    branch_novelty=0.0,
                    weak_symbolic_only_support=0.0,
                    symbolic_dominance_bonus=0.0,
                    uncertainty=_clamp01(mean_entropy),
                    entropy_penalty=_clamp01(mean_entropy),
                    outlier_penalty=0.0 if support_size > 1 else 0.08,
                    composite_score=_clamp01(
                        0.52 * support_share + 0.33 * diversity_weighted_support + 0.15 * mean_confidence
                    ),
                    mean_provenance_strength=mean_confidence,
                    mean_retrieval_relevance=0.0,
                    mean_evidence_quality=mean_confidence,
                    best_branch_id=members[0].attempt_id,
                    best_member_score=mean_confidence,
                    answer_variants=(answer,),
                    caution_flags=tuple(caution_flags),
                    decomposed_signals={
                        "logical_consistency": mean_confidence,
                        "symbolic_agreement": 0.0,
                        "completeness": _clamp01(0.55 * mean_confidence + 0.45 * support_share),
                        "repairability": _clamp01(1.0 - mean_confidence),
                        "answer_correctness_likelihood": _clamp01(
                            0.50 * diversity_weighted_support + 0.50 * mean_confidence
                        ),
                        "step_quality": mean_confidence,
                        "prefix_quality": mean_confidence,
                        "prm_prefix_quality": mean_confidence,
                        "retrieval_compatibility": 0.0,
                        "retrieval_support": 0.0,
                        "operator_reliability": mean_confidence,
                        "open_obligation_burden": 0.0,
                        "discharge_fraction": 0.0,
                        "confidence_dispersion": _clamp01(
                            max(_attempt_uncertainty(item) for item in members)
                            - min(_attempt_uncertainty(item) for item in members)
                        ),
                        "family_diversity": 1.0,
                    },
                )
            )

        weighted_clusters.sort(
            key=lambda item: (
                float(item.composite_score),
                float(item.diversity_weighted_support),
                float(item.support_share),
                float(item.verifier_score),
                item.answer_canonical,
            ),
            reverse=True,
        )
        return EntropyWeightingBundle(
            winner=weighted_clusters[0] if weighted_clusters else None,
            ranked_clusters=tuple(weighted_clusters),
            ranked_candidates=tuple(item.as_candidate_answer() for item in weighted_clusters),
            global_entropy=_clamp01(_mean([_attempt_uncertainty(item) for item in valid_attempts])),
            total_branches=len(attempts),
        )

    def _attempt_vote_weight(self, attempt: AttemptRecord, *, weighting_mode: str | None = None) -> float:
        resolved_mode = weighting_mode or self._attempt_weighting_mode((attempt,))
        if (
            resolved_mode == "true_entropy"
            and attempt.entropy_source is AttemptEntropySource.TRUE_LOGPROBS
            and attempt.mean_token_entropy is not None
        ):
            return 1.0 / max(float(attempt.mean_token_entropy), self.config.minimal_mode_entropy_epsilon)
        if (
            resolved_mode == "proxy_uncertainty"
            and attempt.entropy_source is AttemptEntropySource.PROXY_CONFIDENCE
            and attempt.mean_token_entropy is not None
            and self.config.minimal_mode_allow_proxy_entropy
        ):
            return 1.0 / max(float(attempt.mean_token_entropy), self.config.minimal_mode_entropy_epsilon)
        return 1.0

    def _attempt_weighting_mode(self, attempts: Sequence[AttemptRecord]) -> str:
        valid_attempts = [item for item in attempts if item.valid_answer]
        if any(
            item.entropy_source is AttemptEntropySource.TRUE_LOGPROBS and item.mean_token_entropy is not None
            for item in valid_attempts
        ):
            return "true_entropy"
        if any(
            item.entropy_source is AttemptEntropySource.PROXY_CONFIDENCE and item.mean_token_entropy is not None
            for item in valid_attempts
        ) and self.config.minimal_mode_allow_proxy_entropy:
            return "proxy_uncertainty"
        return "equal_weight"

    def _attempt_weighting_fallback(self, attempts: Sequence[AttemptRecord], weighting_mode: str) -> str:
        valid_attempts = [item for item in attempts if item.valid_answer]
        if weighting_mode == "true_entropy":
            non_true = [
                item for item in valid_attempts
                if item.entropy_source is not AttemptEntropySource.TRUE_LOGPROBS
                or item.mean_token_entropy is None
            ]
            return "equal_for_non_true_entropy_attempts" if non_true else "none"
        if weighting_mode == "proxy_uncertainty":
            missing_proxy = [
                item for item in valid_attempts
                if item.entropy_source is not AttemptEntropySource.PROXY_CONFIDENCE
                or item.mean_token_entropy is None
            ]
            return "equal_for_non_proxy_attempts" if missing_proxy else "none"
        return "equal_weight_only"

    def _should_escalate_after_attempts(
        self,
        *,
        route: RouteDecision,
        selection: FinalSelectionResult,
        attempts: Sequence[AttemptRecord],
        early_stop_triggered: bool,
    ) -> bool:
        return bool(
            self._attempt_escalation_reasons(
                route=route,
                selection=selection,
                attempts=attempts,
                early_stop_triggered=early_stop_triggered,
            )
        )

    def _resolve_branch_runtime_policy(
        self,
        *,
        route: RouteDecision,
        runtime_path: RuntimePath,
        controller_config: ControllerConfig,
        symbolic_hook: Any | None,
        verifier_hook: Any | None,
        failure_classifier: Any | None,
        aggregation_hook: Any | None,
        critique_model: object | None,
    ) -> dict[str, Any]:
        if runtime_path is not RuntimePath.MINIMAL:
            if runtime_path is RuntimePath.ESCALATED:
                policy = self._resolve_escalated_policy(route=route)
                return {
                    "controller_config": controller_config,
                    "symbolic_hook": symbolic_hook if policy["symbolic_active"] else None,
                    "verifier_hook": verifier_hook if policy["verifier_active"] else None,
                    "failure_classifier": failure_classifier,
                    "aggregation_hook": aggregation_hook,
                    "critique_model": critique_model if policy["critique_active"] else None,
                    "selective_policy": policy,
                }
            return {
                "controller_config": controller_config,
                "symbolic_hook": symbolic_hook,
                "verifier_hook": verifier_hook,
                "failure_classifier": failure_classifier,
                "aggregation_hook": aggregation_hook,
                "critique_model": critique_model,
                "selective_policy": None,
            }

        return {
            "controller_config": controller_config,
            "symbolic_hook": None,
            "verifier_hook": None,
            "failure_classifier": _minimal_failure_classifier,
            "aggregation_hook": aggregation_hook or self._minimal_aggregation_hook,
            "critique_model": None,
            "selective_policy": None,
        }

    def _preview_branch_selection(
        self,
        *,
        problem_id: str,
        branch_result: BranchControllerResult,
    ) -> FinalSelectionResult:
        candidate_clusters = tuple(branch_result.candidate_clusters or ())
        if not candidate_clusters:
            candidate_clusters = (branch_result.final_prediction.winning_cluster,)
        return select_final_answer(
            problem_id=problem_id,
            weighted_clusters=candidate_clusters,
            num_branches_generated=branch_result.final_prediction.num_branches_generated,
            num_branches_survived=branch_result.final_prediction.num_branches_survived,
            solve_time_sec=branch_result.final_prediction.solve_time_sec,
            method_used=branch_result.final_prediction.method_used,
        )

    def _should_escalate_after_branch(
        self,
        *,
        route: RouteDecision,
        branch_result: BranchControllerResult,
        preview_selection: FinalSelectionResult,
    ) -> bool:
        return bool(
            self._branch_escalation_reasons(
                route=route,
                branch_result=branch_result,
                preview_selection=preview_selection,
            )
        )

    def _make_runtime_controller(self, config: ControllerConfig) -> BranchController:
        return BranchController(
            operator_library=self.operator_library,
            prior_shaper=self.prior_shaper,
            config=config,
        )

    def _minimal_aggregation_hook(
        self,
        *,
        problem: ParsedProblem,
        route: RouteDecision,
        branches: Sequence[Any],
    ) -> AggregationResult:
        valid_records: list[dict[str, Any]] = []
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for branch in branches:
            candidate = branch.current_candidate()
            if candidate is None:
                continue

            canonical_answer = str(candidate.canonical_answer or candidate.raw_answer or "").strip()
            if not canonical_answer.isdigit():
                continue

            answer_value = int(canonical_answer)
            if not (ANSWER_MIN <= answer_value <= ANSWER_MAX):
                continue

            breakdown = getattr(branch, "score_breakdown", None)
            confidence = _clamp01(getattr(candidate, "confidence", 0.0))
            quality = _clamp01(
                max(
                    getattr(breakdown, "prefix_quality", 0.0),
                    getattr(breakdown, "step_quality", 0.0),
                    confidence,
                )
            )
            open_obligation_burden = _clamp01(getattr(breakdown, "open_obligation_burden", 0.0))
            penalty = _clamp01(getattr(breakdown, "penalty", 0.0))
            weight = max(
                0.05,
                0.70 * confidence + 0.20 * quality - 0.15 * penalty - 0.10 * open_obligation_burden,
            )
            record = {
                "branch_id": str(getattr(branch, "branch_id", "")),
                "answer": canonical_answer,
                "confidence": confidence,
                "quality": quality,
                "weight": weight,
                "open_obligation_burden": open_obligation_burden,
            }
            valid_records.append(record)
            grouped[canonical_answer].append(record)

        if not valid_records:
            selection = select_final_answer(
                problem.problem_id,
                (),
                num_branches_generated=len(branches),
                num_branches_survived=len(branches),
                solve_time_sec=0.0,
                method_used="minimal_weighted_self_consistency",
            )
            return AggregationResult(
                final_prediction=selection.prediction,
                winning_branch_ids=(),
                candidate_clusters=(),
                metadata={
                    "minimal_mode": True,
                    "selector_status": selection.metadata.get("selector_status", "fallback"),
                    "global_entropy": 1.0,
                },
            )

        total_weight = sum(record["weight"] for record in valid_records)
        weighted_clusters: list[WeightedClusterScore] = []
        sorted_answers = sorted(grouped)
        for cluster_index, answer in enumerate(sorted_answers):
            members = grouped[answer]
            support_size = len(members)
            cluster_weight = sum(member["weight"] for member in members)
            mean_confidence = _clamp01(
                sum(member["confidence"] for member in members) / max(1, support_size)
            )
            mean_quality = _clamp01(
                sum(member["quality"] for member in members) / max(1, support_size)
            )
            mean_open_obligation_burden = _clamp01(
                sum(member["open_obligation_burden"] for member in members) / max(1, support_size)
            )
            dispersion = _clamp01(
                max(member["confidence"] for member in members) - min(member["confidence"] for member in members)
            )
            support_share = support_size / max(1, len(valid_records))
            diversity_weighted_support = cluster_weight / max(1e-6, total_weight)
            uncertainty = _clamp01(
                0.55 * (1.0 - mean_confidence)
                + 0.25 * dispersion
                + 0.20 * mean_open_obligation_burden
            )
            caution_flags = []
            if support_size == 1:
                caution_flags.append("singleton")
            if diversity_weighted_support < 0.20:
                caution_flags.append("low_support_outlier")

            weighted_clusters.append(
                WeightedClusterScore(
                    cluster_id=f"minimal::{cluster_index:03d}::{answer}",
                    answer=answer,
                    answer_canonical=answer,
                    branch_ids=tuple(member["branch_id"] for member in members),
                    cluster_size=support_size,
                    support_share=_clamp01(support_share),
                    diversity_weighted_support=_clamp01(diversity_weighted_support),
                    unique_branch_families=support_size,
                    family_diversity=1.0,
                    verifier_score=mean_confidence,
                    weak_symbolic_support=0.0,
                    exact_symbolic_support=0.0,
                    retrieval_support=0.0,
                    branch_novelty=0.0,
                    weak_symbolic_only_support=0.0,
                    symbolic_dominance_bonus=0.0,
                    uncertainty=uncertainty,
                    entropy_penalty=dispersion,
                    outlier_penalty=0.0 if support_size > 1 else 0.08,
                    composite_score=_clamp01(
                        0.52 * support_share + 0.33 * diversity_weighted_support + 0.15 * mean_confidence
                    ),
                    mean_provenance_strength=mean_quality,
                    mean_retrieval_relevance=0.0,
                    mean_evidence_quality=mean_quality,
                    best_branch_id=members[0]["branch_id"],
                    best_member_score=mean_confidence,
                    answer_variants=(answer,),
                    caution_flags=tuple(caution_flags),
                    decomposed_signals={
                        "logical_consistency": mean_confidence,
                        "symbolic_agreement": 0.0,
                        "completeness": _clamp01(0.60 * mean_confidence + 0.40 * support_share),
                        "repairability": _clamp01(1.0 - mean_confidence),
                        "answer_correctness_likelihood": _clamp01(
                            0.50 * diversity_weighted_support + 0.50 * mean_confidence
                        ),
                        "step_quality": mean_quality,
                        "prefix_quality": mean_quality,
                        "prm_prefix_quality": mean_quality,
                        "retrieval_compatibility": 0.0,
                        "retrieval_support": 0.0,
                        "operator_reliability": mean_quality,
                        "open_obligation_burden": mean_open_obligation_burden,
                        "discharge_fraction": 0.0,
                        "confidence_dispersion": dispersion,
                        "family_diversity": 1.0,
                    },
                )
            )

        weighted_clusters.sort(
            key=lambda item: (
                float(item.composite_score),
                float(item.diversity_weighted_support),
                float(item.support_share),
                float(item.verifier_score),
                item.answer_canonical,
            ),
            reverse=True,
        )
        weighting_bundle = EntropyWeightingBundle(
            winner=weighted_clusters[0] if weighted_clusters else None,
            ranked_clusters=tuple(weighted_clusters),
            ranked_candidates=tuple(item.as_candidate_answer() for item in weighted_clusters),
            global_entropy=_clamp01(
                sum(item.uncertainty for item in weighted_clusters) / max(1, len(weighted_clusters))
            ),
            total_branches=len(valid_records),
        )
        selection = select_final_answer(
            problem.problem_id,
            weighting_bundle,
            num_branches_generated=len(branches),
            num_branches_survived=len(branches),
            solve_time_sec=0.0,
            method_used="minimal_weighted_self_consistency",
        )
        return AggregationResult(
            final_prediction=selection.prediction,
            winning_branch_ids=tuple(selection.selected_cluster.candidate.branch_ids),
            candidate_clusters=tuple(weighting_bundle.ranked_clusters),
            metadata={
                "minimal_mode": True,
                "cluster_count": len(weighted_clusters),
                "global_entropy": weighting_bundle.global_entropy,
                "selection_warnings": [warning.value for warning in selection.warnings],
                "abstain_recommended": selection.abstain_recommended,
            },
        )

    def _controller_config_from_budget(
        self,
        budget: Any | None,
        fallback_plan: BudgetPlan,
        *,
        route: RouteDecision,
        runtime_path: RuntimePath,
    ) -> ControllerConfig:
        overrides = _budget_branch_overrides(budget)
        config = ControllerConfig(
            self_consistency_samples=int(
                overrides.get("self_consistency_samples", fallback_plan.self_consistency_samples)
            ),
            frontier_width=int(overrides.get("frontier_width", fallback_plan.branch_budget)),
            max_search_depth=int(overrides.get("max_search_depth", fallback_plan.max_search_depth)),
            max_search_nodes=int(overrides.get("max_search_nodes", fallback_plan.max_search_nodes)),
            resample_budget=int(
                overrides.get("resample_budget", max(1, fallback_plan.branch_budget // 16 or 1))
            ),
            repair_budget=int(overrides.get("repair_budget", fallback_plan.repair_budget)),
            critique_top_k=int(overrides.get("critique_top_k", fallback_plan.critique_top_k)),
            retain_top_k=self.config.controller_retain_top_k,
            deterministic=True,
        )
        if runtime_path is RuntimePath.MINIMAL:
            config = replace(
                config,
                self_consistency_samples=int(self.config.minimal_mode_self_consistency_samples),
                frontier_width=int(self.config.minimal_mode_frontier_width),
                max_search_depth=int(self.config.minimal_mode_max_search_depth),
                max_search_nodes=int(self.config.minimal_mode_max_search_nodes),
                resample_budget=0,
                repair_budget=0,
                critique_top_k=0,
                consensus_stop_count=int(self.config.minimal_mode_consensus_stop_count),
                retain_top_k=min(self.config.controller_retain_top_k, int(self.config.minimal_mode_frontier_width)),
                enable_mid_search_critique=False,
            )
        elif runtime_path is RuntimePath.ESCALATED:
            policy = self._resolve_escalated_policy(route=route)
            if not policy["search_active"]:
                config = replace(
                    config,
                    max_search_depth=min(int(config.max_search_depth), 1),
                    max_search_nodes=min(int(config.max_search_nodes), max(1, int(config.frontier_width))),
                    resample_budget=0,
                )
            if not policy["repair_active"]:
                config = replace(config, repair_budget=0)
            if not policy["critique_active"]:
                config = replace(config, critique_top_k=0, enable_mid_search_critique=False)
        return config

    def _post_branch_budget_update(
        self,
        *,
        budget: Any | None,
        branch_result: BranchControllerResult,
        stage_records: list[StageRecord],
    ) -> Any | None:
        if (
            budget is None
            or runtime_adapt_budget is None
            or RuntimeObservation is None
            or not self.config.post_branch_budget_adaptation
        ):
            return budget

        started = time.perf_counter()
        try:
            decomposition = dict(getattr(branch_result.final_prediction, "signal_decomposition", {}) or {})
            observation = RuntimeObservation(
                elapsed_s=float(branch_result.final_prediction.solve_time_sec),
                estimated_tokens_used=0,
                branches_generated=int(branch_result.final_prediction.num_branches_generated),
                solved_branches=len(branch_result.surviving_branches),
                active_frontier=0,
                top_answer_confidence=float(branch_result.final_prediction.confidence),
                verifier_agreement=float(decomposition.get("logical_consistency", branch_result.final_prediction.confidence)),
                symbolic_pass_rate=float(decomposition.get("symbolic_agreement", 1.0 if branch_result.final_prediction.winning_cluster.symbolic_check > 0 else 0.0)),
                answer_entropy=max(0.0, float(branch_result.final_prediction.winning_cluster.entropy_penalty)),
                best_cluster_ratio=max(0.0, min(1.0, float(branch_result.final_prediction.winning_cluster.answer_agreement))),
                search_nodes_expanded=int(branch_result.metadata.get("search_nodes_expanded", 0)),
                max_depth_reached=int(branch_result.metadata.get("max_depth_reached", 0)),
                final_candidate_available=True,
                metadata={
                    "open_obligation_burden": float(decomposition.get("open_obligation_burden", 0.0)),
                    "prm_prefix_quality": float(decomposition.get("prefix_quality", 0.0)),
                    "verifier_repairability": float(decomposition.get("repairability", 0.0)),
                    "logical_consistency": float(decomposition.get("logical_consistency", 0.0)),
                    "completeness": float(decomposition.get("completeness", 0.0)),
                    "retrieval_compatibility": float(decomposition.get("retrieval_compatibility", 0.0)),
                    "retrieval_support": float(decomposition.get("retrieval_support", 0.0)),
                    "operator_reliability": float(decomposition.get("operator_reliability", 0.0)),
                    "branch_disagreement": max(0.0, 1.0 - float(branch_result.final_prediction.winning_cluster.answer_agreement)),
                },
            )
            decision = runtime_adapt_budget(budget, observation)
            stage_records.append(
                StageRecord(
                    stage=StageName.BUDGET,
                    ok=True,
                    recoverable=True,
                    elapsed_sec=_elapsed(started),
                    summary="post_branch_budget_update",
                    details={
                        "stop": bool(getattr(decision, "stop", False)),
                        "widen": bool(getattr(decision, "widen", False)),
                        "degrade": bool(getattr(decision, "degrade", False)),
                    },
                )
            )
            return getattr(decision, "updated_bundle", budget)
        except Exception as exc:
            stage_records.append(
                StageRecord(
                    stage=StageName.BUDGET,
                    ok=False,
                    recoverable=True,
                    elapsed_sec=_elapsed(started),
                    summary="post_branch_budget_update_failed",
                    details={"message": str(exc)},
                )
            )
            return budget

    def _finalize_failure(
        self,
        *,
        status: SolveStatus,
        problem_id: str,
        parsed_problem: ParsedProblem | None,
        route: RouteDecision | None,
        budget: Any | None,
        state_init: StateGraphInitResult | None,
        retrieved_traces: Sequence[RetrievedTrace],
        branch_result: BranchControllerResult | None,
        final_selection: FinalSelectionResult | None,
        submission_row: Any | None,
        stage_records: Sequence[StageRecord],
        failure: StageFailure,
        problem_text: str,
        started: float,
    ) -> SolveResultBundle:
        debug_artifact = self._build_debug_artifact(
            problem_text=problem_text,
            parsed_problem=parsed_problem,
            route=route,
            budget=budget,
            state_init=state_init,
            retrieved_traces=retrieved_traces,
            branch_result=branch_result,
            final_selection=final_selection,
            submission_row=submission_row,
            stage_records=stage_records,
        )
        return SolveResultBundle(
            status=status,
            problem_id=problem_id,
            parsed_problem=parsed_problem,
            route=route,
            budget=budget,
            state_init=state_init,
            retrieved_traces=list(retrieved_traces),
            branch_result=branch_result,
            final_selection=final_selection,
            final_prediction=(final_selection.prediction if final_selection is not None else None),
            submission_row=submission_row,
            stage_records=list(stage_records),
            failure=failure,
            debug_artifact=debug_artifact,
            metadata={
                "total_elapsed_sec": _elapsed(started),
                "stage_count": len(stage_records),
            },
        )

    def _finalize_success(
        self,
        *,
        status: SolveStatus,
        problem_id: str,
        parsed_problem: ParsedProblem,
        route: RouteDecision,
        budget: Any | None,
        state_init: StateGraphInitResult,
        retrieved_traces: Sequence[RetrievedTrace],
        branch_result: BranchControllerResult,
        final_selection: FinalSelectionResult,
        submission_row: Any | None,
        stage_records: Sequence[StageRecord],
        failure: StageFailure | None,
        problem_text: str,
        started: float,
    ) -> SolveResultBundle:
        debug_artifact = self._build_debug_artifact(
            problem_text=problem_text,
            parsed_problem=parsed_problem,
            route=route,
            budget=budget,
            state_init=state_init,
            retrieved_traces=retrieved_traces,
            branch_result=branch_result,
            final_selection=final_selection,
            submission_row=submission_row,
            stage_records=stage_records,
        )
        return SolveResultBundle(
            status=status,
            problem_id=problem_id,
            parsed_problem=parsed_problem,
            route=route,
            budget=budget,
            state_init=state_init,
            retrieved_traces=list(retrieved_traces),
            branch_result=branch_result,
            final_selection=final_selection,
            final_prediction=final_selection.prediction,
            submission_row=submission_row,
            stage_records=list(stage_records),
            failure=failure,
            debug_artifact=debug_artifact,
            metadata={
                "total_elapsed_sec": _elapsed(started),
                "stage_count": len(stage_records),
                "abstain_recommended": final_selection.abstain_recommended,
            },
        )

    def _build_debug_artifact(
        self,
        *,
        problem_text: str,
        parsed_problem: ParsedProblem | None,
        route: RouteDecision | None,
        budget: Any | None,
        state_init: StateGraphInitResult | None,
        retrieved_traces: Sequence[RetrievedTrace],
        branch_result: BranchControllerResult | None,
        final_selection: FinalSelectionResult | None,
        submission_row: Any | None,
        stage_records: Sequence[StageRecord],
    ) -> SolveDebugArtifact | None:
        if not self.config.enable_debug_artifacts:
            return None

        parse_metadata: dict[str, Any] = {}
        if parsed_problem is not None:
            parse_metadata = {
                "problem_id": parsed_problem.problem_id,
                "domain": getattr(parsed_problem.domain, "value", str(parsed_problem.domain)),
                "answer_type": parsed_problem.answer_type,
                "parse_quality": dict(parsed_problem.parse_quality or {}),
                "constraint_count": len(parsed_problem.constraints),
            }

        route_rationale = list(route.route_rationale) if route is not None else []
        budget_summary = _safe_budget_summary(budget)

        state_summary: dict[str, Any] = {}
        if state_init is not None:
            state_summary = {
                "root_node_id": state_init.metadata.root_node_id,
                "canonical_state_hash": state_init.metadata.canonical_state_hash,
                "top_problem_type": state_init.metadata.top_problem_type,
                "top_archetypes": list(state_init.metadata.top_archetypes),
            }

        retrieval_summary = {
            "trace_count": len(retrieved_traces),
            "trace_ids": [t.trace_id for t in retrieved_traces[:8]],
            "trace_domains": [t.domain for t in retrieved_traces[:8]],
            "trace_operators": [list(t.operators_used[:5]) for t in retrieved_traces[:5]],
        }

        branch_summary: dict[str, Any] = {}
        if branch_result is not None:
            branch_summary = {
                "stopped_reason": branch_result.stopped_reason,
                "num_all_branches": len(branch_result.all_branches),
                "num_surviving_branches": len(branch_result.surviving_branches),
                "critiqued_branch_ids": list(branch_result.critiqued_branch_ids),
                "selected_for_critique": list(branch_result.selected_for_critique),
                "metadata": dict(branch_result.metadata or {}),
            }

        aggregation_summary: dict[str, Any] = {}
        if final_selection is not None:
            aggregation_summary = {
                "submission_answer": final_selection.submission_answer,
                "confidence": final_selection.prediction.confidence,
                "warnings": [str(w.value if hasattr(w, "value") else w) for w in final_selection.warnings],
                "abstain_recommended": final_selection.abstain_recommended,
                "ranked_cluster_count": len(final_selection.ranked_clusters),
            }

        submission_summary = {
            "available": submission_row is not None,
            "row_preview": _stable_row_preview(submission_row) if submission_row is not None else None,
        }

        return SolveDebugArtifact(
            problem_text=problem_text,
            parse_metadata=parse_metadata,
            route_rationale=route_rationale,
            budget_summary=budget_summary,
            state_summary=state_summary,
            retrieval_summary=retrieval_summary,
            branch_summary=branch_summary,
            aggregation_summary=aggregation_summary,
            submission_summary=submission_summary,
            stage_records=list(stage_records),
        )

    def _stage_failure(
        self,
        *,
        stage: StageName,
        code: str,
        message: str,
        recoverable: bool,
        exc: Exception | None = None,
    ) -> StageFailure:
        details: dict[str, Any] = {}
        if exc is not None:
            details["exception_type"] = type(exc).__name__
            details["traceback"] = traceback.format_exc(limit=6)
        return StageFailure(
            stage=stage,
            code=code,
            message=message,
            recoverable=recoverable,
            details=details,
        )

    def _set_deterministic_seed(self, seed: int) -> None:
        random.seed(int(seed))
        try:  # pragma: no cover
            import numpy as np

            np.random.seed(int(seed))
        except Exception:
            pass


def solve_problem(
    problem_text: str,
    *,
    problem_id: str,
    remaining_problems: int | None = None,
    budget_tracker: Any | None = None,
    config: InferenceEngineConfig | None = None,
) -> SolveResultBundle:
    return InferenceEngine(config=config).solve_problem(
        problem_text,
        problem_id=problem_id,
        remaining_problems=remaining_problems,
        budget_tracker=budget_tracker,
    )


def _elapsed(started: float) -> float:
    return max(0.0, round(time.perf_counter() - started, 6))


def _safe_budget_summary(budget: Any | None) -> dict[str, Any]:
    if budget is None:
        return {}
    summary: dict[str, Any] = {}
    for key in ("mode", "route_uncertainty", "diagnostics"):
        if hasattr(budget, key):
            value = getattr(budget, key)
            summary[key] = value.value if hasattr(value, "value") else value
    if hasattr(budget, "effective_plan"):
        plan = getattr(budget, "effective_plan")
        summary["effective_plan"] = _budget_plan_summary(plan)
    elif isinstance(budget, BudgetPlan):
        summary["effective_plan"] = _budget_plan_summary(budget)
    if hasattr(budget, "ceilings"):
        summary["ceilings"] = {
            k: getattr(budget.ceilings, k)
            for k in (
                "wall_clock_s",
                "token_budget",
                "branch_ceiling",
                "search_node_ceiling",
                "search_depth_ceiling",
                "retrieval_depth_ceiling",
                "critique_ceiling",
                "repair_ceiling",
                "resample_ceiling",
            )
            if hasattr(budget.ceilings, k)
        }
    return summary


def _budget_plan_summary(plan: Any) -> dict[str, Any]:
    return {
        "branch_budget": int(getattr(plan, "branch_budget", 0)),
        "retrieval_depth": int(getattr(plan, "retrieval_depth", 0)),
        "max_search_depth": int(getattr(plan, "max_search_depth", 0)),
        "max_search_nodes": int(getattr(plan, "max_search_nodes", 0)),
        "repair_budget": int(getattr(plan, "repair_budget", 0)),
        "self_consistency_samples": int(getattr(plan, "self_consistency_samples", 0)),
        "critique_top_k": int(getattr(plan, "critique_top_k", 0)),
        "use_retrieval": bool(getattr(plan, "use_retrieval", False)),
        "use_symbolic": bool(getattr(plan, "use_symbolic", False)),
        "use_brute_force": bool(getattr(plan, "use_brute_force", False)),
    }


def _budget_retrieval_depth(budget: Any | None, fallback_plan: BudgetPlan) -> int:
    if budget is None:
        return int(fallback_plan.retrieval_depth)
    if hasattr(budget, "effective_plan"):
        return int(getattr(budget.effective_plan, "retrieval_depth", fallback_plan.retrieval_depth))
    return int(getattr(budget, "retrieval_depth", fallback_plan.retrieval_depth))


def _budget_branch_overrides(budget: Any | None) -> dict[str, Any]:
    if budget is None:
        return {}
    if hasattr(budget, "branch_controller_overrides"):
        try:
            return dict(budget.branch_controller_overrides())
        except Exception:
            return {}
    if hasattr(budget, "effective_plan"):
        return {
            "self_consistency_samples": int(getattr(budget.effective_plan, "self_consistency_samples", 0)),
            "frontier_width": int(getattr(budget.effective_plan, "branch_budget", 0)),
            "max_search_depth": int(getattr(budget.effective_plan, "max_search_depth", 0)),
            "max_search_nodes": int(getattr(budget.effective_plan, "max_search_nodes", 0)),
            "repair_budget": int(getattr(budget.effective_plan, "repair_budget", 0)),
            "critique_top_k": int(getattr(budget.effective_plan, "critique_top_k", 0)),
        }
    return {}


def _stable_row_preview(row: Any) -> Any:
    if row is None:
        return None
    if hasattr(row, "model_dump"):
        return row.model_dump()
    if isinstance(row, Mapping):
        return dict(row)
    return str(row)


def _minimal_failure_classifier(**_: Any) -> FailureDecision:
    return FailureDecision(
        failure_type=None,
        route_to_repair=False,
        route_to_resample=False,
        summary="selective_minimal_mode_no_retry",
    )


def _top_two_gap(distribution: Mapping[str, Any]) -> float:
    values = sorted((float(value) for value in distribution.values()), reverse=True)
    if len(values) < 2:
        return 1.0
    return max(0.0, values[0] - values[1])


def _attempt_confidence(attempt: AttemptRecord) -> float:
    metadata = dict(attempt.metadata or {})
    if "confidence" in metadata:
        return _clamp01(metadata.get("confidence"))
    if attempt.mean_token_entropy is not None:
        return _clamp01(1.0 / (1.0 + float(attempt.mean_token_entropy)))
    return 0.5 if attempt.valid_answer else 0.0


def _attempt_uncertainty(attempt: AttemptRecord) -> float:
    if attempt.mean_token_entropy is not None:
        return _clamp01(float(attempt.mean_token_entropy))
    return _clamp01(1.0 - _attempt_confidence(attempt))


def _dominant_entropy_source(attempts: Sequence[AttemptRecord]) -> AttemptEntropySource:
    counts: dict[AttemptEntropySource, int] = {
        AttemptEntropySource.TRUE_LOGPROBS: 0,
        AttemptEntropySource.PROXY_CONFIDENCE: 0,
        AttemptEntropySource.UNAVAILABLE: 0,
    }
    for attempt in attempts:
        counts[attempt.entropy_source] = counts.get(attempt.entropy_source, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0].value))[0][0]


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


__all__ = [
    "InferenceEngine",
    "InferenceEngineConfig",
    "RuntimePath",
    "SolveDebugArtifact",
    "SolveResultBundle",
    "SolveStatus",
    "StageFailure",
    "StageName",
    "StageRecord",
    "solve_problem",
]
