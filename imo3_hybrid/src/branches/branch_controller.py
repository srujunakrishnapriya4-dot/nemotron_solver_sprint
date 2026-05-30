from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
from typing import Mapping, Protocol, Sequence
import inspect

from src.aggregation.canonicalize import canonicalize_competition_answer
from src.aggregation.clustering import cluster_candidates
from src.aggregation.entropy_weighting import score_answer_clusters
from src.aggregation.final_selector import select_final_answer
from src.common.constants import ANSWER_MAX, ANSWER_MIN, MAX_REPAIR_ATTEMPTS
from src.common.schemas import CandidateAnswer, FailureType, FinalPrediction, ParsedProblem, RetrievedTrace, RouteDecision
from src.operators.operator_library import OperatorLibrary
from src.operators.operator_types import OperatorPolicyScore
from src.operators.priors import OperatorPriorShaper
from src.prm import ProcessObligationInput, ProcessScoreInput, ProcessStepInput, score_process
from src.state_graph.node import (
    ConstraintRecord,
    EvidenceRecord,
    GoalRecord,
    InvariantRecord,
    OperatorApplicationRecord,
    ReasoningStateNode,
)

from .branch_state import BranchPhase, BranchState, BranchStepKind
from .failure_classifier import BranchFailureDiagnosis, classify_branch_failure
from .repair import RepairBudget, repair_branch
from .resampler import ResampleBudget, resample_branches
from . import self_critique as _self_critique_module
from .self_critique import SelfCritiqueBatchResult, SelfCritiqueConfig, critique_top_k_branches, select_top_k_for_critique


class BranchGeneratorProtocol(Protocol):
    def __call__(
        self,
        *,
        problem: ParsedProblem,
        route: RouteDecision,
        branch: BranchState,
        node: ReasoningStateNode,
        operator_name: str,
        retrieval_trace: RetrievedTrace | None,
        depth: int,
        sample_index: int,
    ) -> "GenerationResult": ...


class SymbolicHookProtocol(Protocol):
    def __call__(
        self,
        *,
        problem: ParsedProblem,
        route: RouteDecision,
        branch: BranchState,
        node: ReasoningStateNode,
        operator_name: str,
        generation: "GenerationResult",
    ) -> "SymbolicHookResult": ...


class VerifierHookProtocol(Protocol):
    def __call__(
        self,
        *,
        problem: ParsedProblem,
        route: RouteDecision,
        branch: BranchState,
        node: ReasoningStateNode,
        symbolic: "SymbolicHookResult",
        generation: "GenerationResult",
    ) -> "VerifierHookResult": ...


class FailureClassifierProtocol(Protocol):
    def __call__(
        self,
        *,
        problem: ParsedProblem,
        route: RouteDecision,
        branch: BranchState,
        symbolic: "SymbolicHookResult",
        verifier: "VerifierHookResult",
        generation: "GenerationResult",
    ) -> "FailureDecision": ...


class AggregationHookProtocol(Protocol):
    def __call__(
        self,
        *,
        problem: ParsedProblem,
        route: RouteDecision,
        branches: Sequence[BranchState],
    ) -> "AggregationResult": ...


@dataclass(frozen=True)
class ControllerConfig:
    self_consistency_samples: int | None = None
    frontier_width: int | None = None
    max_search_depth: int | None = None
    max_search_nodes: int | None = None
    resample_budget: int | None = None
    repair_budget: int | None = None
    critique_top_k: int | None = None
    consensus_stop_count: int | None = None
    retain_top_k: int = 8
    solved_cluster_stop_ratio: float = 0.68
    min_branch_score_for_expansion: float = 0.18
    novelty_bonus_weight: float = 0.18
    exploration_bonus_weight: float = 0.14
    depth_penalty_weight: float = 0.08
    repair_penalty_weight: float = 0.05
    disagreement_bonus_weight: float = 0.05
    retrieval_blend_weight: float = 0.30
    enable_mid_search_critique: bool = True
    mid_search_critique_top_k: int = 2
    mid_search_critique_every_n_expansions: int = 6
    mid_search_critique_min_expansions: int = 4
    mid_search_critique_min_frontier: int = 3
    deterministic: bool = True


@dataclass(frozen=True)
class GenerationResult:
    reasoning: str
    answer: str | None = None
    answer_canonical: str | None = None
    confidence: float = 0.0
    summary: str = ""
    partial_solution: str = ""
    added_constraints: tuple[str, ...] = ()
    added_invariants: tuple[str, ...] = ()
    added_goals: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SymbolicHookResult:
    passed: bool
    score: float
    summary: str
    exact_match: bool = False
    added_constraints: tuple[str, ...] = ()
    added_invariants: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class VerifierHookResult:
    probability: float
    logical_consistency: float
    completeness: float
    repairability: float
    summary: str
    symbolic_agreement: float = 0.0
    answer_correctness_likelihood: float | None = None
    branch_score: float | None = None
    step_quality: float | None = None
    prefix_quality: float | None = None
    open_obligation_burden: float | None = None
    failure_step_index: int | None = None
    failure_step_id: str | None = None
    failure_node_id: str | None = None
    failure_obligation_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class FailureDecision:
    failure_type: FailureType | None
    route_to_repair: bool
    route_to_resample: bool
    summary: str


@dataclass(frozen=True)
class AggregationResult:
    final_prediction: FinalPrediction
    winning_branch_ids: tuple[str, ...]
    candidate_clusters: tuple[CandidateAnswer, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BranchSelectionRecord:
    branch_id: str
    depth: int
    score: float
    priority: float
    operator_name: str | None
    reason: str


@dataclass(frozen=True)
class BranchControllerResult:
    problem_id: str
    route: RouteDecision
    final_prediction: FinalPrediction
    all_branches: tuple[BranchState, ...]
    surviving_branches: tuple[BranchState, ...]
    critiqued_branch_ids: tuple[str, ...]
    candidate_clusters: tuple[CandidateAnswer, ...]
    selected_for_critique: tuple[str, ...]
    search_history: tuple[BranchSelectionRecord, ...]
    stopped_reason: str
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(order=True)
class _FrontierEntry:
    sort_key: tuple[float, int, str]
    branch: BranchState = field(compare=False)
    node: ReasoningStateNode = field(compare=False)
    depth: int = field(compare=False)
    operator_name: str | None = field(compare=False, default=None)
    novelty: float = field(compare=False, default=0.0)
    retrieval_support: float = field(compare=False, default=0.0)


@dataclass(frozen=True)
class _ProcessedBranch:
    branch: BranchState
    node: ReasoningStateNode
    novelty: float
    retrieval_support: float


class BranchController:
    """
    Concrete branch/search controller.

    Owns:
    - self-consistency-first branch generation
    - bounded priority-frontier search
    - route/retrieval-conditioned operator proposal handoff
    - failure classification into repair vs resample
    - critique-before-aggregation handoff
    """

    def __init__(
        self,
        *,
        operator_library: OperatorLibrary | None = None,
        prior_shaper: OperatorPriorShaper | None = None,
        config: ControllerConfig | None = None,
    ) -> None:
        self.operator_library = operator_library or OperatorLibrary()
        self.prior_shaper = prior_shaper or OperatorPriorShaper(self.operator_library)
        self.config = config or ControllerConfig()

    def solve(
        self,
        *,
        problem: ParsedProblem,
        route: RouteDecision,
        root_node: ReasoningStateNode | None = None,
        retrieved_traces: Sequence[RetrievedTrace] | None = None,
        generator: BranchGeneratorProtocol | None = None,
        symbolic_hook: SymbolicHookProtocol | None = None,
        verifier_hook: VerifierHookProtocol | None = None,
        failure_classifier: FailureClassifierProtocol | None = None,
        aggregation_hook: AggregationHookProtocol | None = None,
        critique_model: object | None = None,
    ) -> BranchControllerResult:
        root = root_node or _build_root_node(problem=problem, route=route)
        traces = tuple(retrieved_traces or ())
        search_history: list[BranchSelectionRecord] = []
        all_branches: dict[str, BranchState] = {}
        active_branches: dict[str, BranchState] = {}
        frontier: list[_FrontierEntry] = []
        seen_branch_ids: set[str] = set()
        solved_branch_ids: list[str] = []
        expanded_nodes = 0
        stopped_reason = "budget_exhausted"
        mid_search_critiqued_branch_ids: list[str] = []
        mid_search_selected_for_critique: list[str] = []

        initial_entries = self._generate_initial_frontier(
            problem=problem,
            route=route,
            root=root,
            retrieved_traces=traces,
            generator=generator,
            symbolic_hook=symbolic_hook,
            verifier_hook=verifier_hook,
            failure_classifier=failure_classifier,
            all_branches=all_branches,
            active_branches=active_branches,
            seen_branch_ids=seen_branch_ids,
            solved_branch_ids=solved_branch_ids,
        )
        for entry in initial_entries:
            heapq.heappush(frontier, entry)

        max_nodes = self._max_search_nodes(route)
        while frontier and expanded_nodes < max_nodes:
            current = heapq.heappop(frontier)
            branch = current.branch
            if branch.phase in {BranchPhase.FAILED, BranchPhase.PRUNED}:
                continue

            expanded_nodes += 1
            search_history.append(
                BranchSelectionRecord(
                    branch_id=branch.branch_id,
                    depth=current.depth,
                    score=branch.composite_score(),
                    priority=-current.sort_key[0],
                    operator_name=current.operator_name,
                    reason="priority_frontier_selection",
                )
            )

            if self._should_stop_early(route=route, all_branches=all_branches, solved_branch_ids=solved_branch_ids):
                stopped_reason = "consensus_converged"
                break

            if current.depth >= self._max_depth(route):
                branch.set_phase(BranchPhase.PRUNED, reason="max_search_depth_reached")
                continue

            children = self._expand_branch(
                problem=problem,
                route=route,
                branch=branch,
                node=current.node,
                retrieved_traces=traces,
                depth=current.depth + 1,
                generator=generator,
                symbolic_hook=symbolic_hook,
                verifier_hook=verifier_hook,
                failure_classifier=failure_classifier,
                all_branches=all_branches,
                active_branches=active_branches,
                seen_branch_ids=seen_branch_ids,
                solved_branch_ids=solved_branch_ids,
            )
            for child in children:
                heapq.heappush(frontier, child)

            frontier = self._prune_frontier(frontier=frontier, route=route)
            mid_search_batch, frontier = self._maybe_mid_search_critique(
                problem=problem,
                route=route,
                current=current,
                frontier=frontier,
                active_branches=active_branches,
                all_branches=all_branches,
                critique_model=critique_model,
                expanded_nodes=expanded_nodes,
            )
            if mid_search_batch.critiques:
                mid_search_critiqued_branch_ids.extend(list(mid_search_batch.critiqued_branch_ids))
                mid_search_selected_for_critique.extend(
                    [result.original_branch.branch_id for result in mid_search_batch.critiques]
                )

        surviving = self._select_survivors(route=route, branches=tuple(active_branches.values()))
        critique_batch = self._critique_survivors(problem=problem, surviving=surviving, critique_model=critique_model, route=route)
        critiqued_by_id = {
            result.critiqued_branch.branch_id: result.critiqued_branch
            for result in critique_batch.critiques
        }
        selected_for_critique = tuple(result.original_branch.branch_id for result in critique_batch.critiques)

        for branch_id, branch in list(active_branches.items()):
            critiqued = critiqued_by_id.get(branch_id)
            if critiqued is not None:
                active_branches[branch_id] = self._merge_critiqued_branch(branch, critiqued)
                all_branches[branch_id] = active_branches[branch_id]

        final_survivors = tuple(active_branches[branch.branch_id] for branch in surviving if branch.branch_id in active_branches)
        aggregation = self._aggregate(
            problem=problem,
            route=route,
            branches=final_survivors,
            aggregation_hook=aggregation_hook,
        )

        if stopped_reason == "budget_exhausted" and frontier:
            stopped_reason = "search_budget_hit"
        elif stopped_reason == "budget_exhausted":
            stopped_reason = "frontier_drained"

        return BranchControllerResult(
            problem_id=problem.problem_id,
            route=route,
            final_prediction=aggregation.final_prediction,
            all_branches=tuple(all_branches.values()),
            surviving_branches=final_survivors,
            critiqued_branch_ids=tuple(dict.fromkeys(mid_search_critiqued_branch_ids + list(critique_batch.critiqued_branch_ids))),
            candidate_clusters=aggregation.candidate_clusters,
            selected_for_critique=tuple(dict.fromkeys(mid_search_selected_for_critique + list(selected_for_critique))),
            search_history=tuple(search_history),
            stopped_reason=stopped_reason,
            metadata={
                "expanded_nodes": expanded_nodes,
                "initial_branches": len(initial_entries),
                "retrieval_count": len(traces),
                "winning_branch_ids": aggregation.winning_branch_ids,
                "mid_search_critiqued_count": len(mid_search_critiqued_branch_ids),
                "mid_search_selected_for_critique": tuple(dict.fromkeys(mid_search_selected_for_critique)),
            },
        )

    def _generate_initial_frontier(
        self,
        *,
        problem: ParsedProblem,
        route: RouteDecision,
        root: ReasoningStateNode,
        retrieved_traces: Sequence[RetrievedTrace],
        generator: BranchGeneratorProtocol | None,
        symbolic_hook: SymbolicHookProtocol | None,
        verifier_hook: VerifierHookProtocol | None,
        failure_classifier: FailureClassifierProtocol | None,
        all_branches: dict[str, BranchState],
        active_branches: dict[str, BranchState],
        seen_branch_ids: set[str],
        solved_branch_ids: list[str],
    ) -> list[_FrontierEntry]:
        operator_rows, retrieval_signal = self._shape_operator_rows(
            node=root,
            route=route,
            retrieved_traces=retrieved_traces,
            branch=None,
        )
        sample_count = self._self_consistency_samples(route)
        entries: list[_FrontierEntry] = []
        for sample_index in range(sample_count):
            operator_row = operator_rows[sample_index % max(1, len(operator_rows))]
            retrieval_trace = retrieved_traces[sample_index % len(retrieved_traces)] if retrieved_traces else None
            branch = BranchState.create(
                route=route,
                parsed_problem=problem,
                branch_id=f"{problem.problem_id}::seed::{sample_index:03d}",
                parent_branch_id=None,
                root_node=root,
                metadata={"sample_index": sample_index, "origin": "self_consistency"},
            )
            if retrieval_trace is not None:
                self._attach_retrieval(branch=branch, trace=retrieval_trace, retrieval_signal=retrieval_signal)
            processed = self._process_branch_attempt(
                problem=problem,
                route=route,
                branch=branch,
                node=root,
                operator_name=operator_row.operator_name,
                retrieval_trace=retrieval_trace,
                depth=0,
                sample_index=sample_index,
                generator=generator,
                symbolic_hook=symbolic_hook,
                verifier_hook=verifier_hook,
                failure_classifier=failure_classifier,
            )
            all_branches[processed.branch.branch_id] = processed.branch
            active_branches[processed.branch.branch_id] = processed.branch
            seen_branch_ids.add(processed.branch.branch_id)
            if processed.branch.phase is BranchPhase.SOLVED:
                solved_branch_ids.append(processed.branch.branch_id)
            entries.append(
                self._frontier_entry(
                    branch=processed.branch,
                    node=processed.node,
                    depth=0,
                    operator_name=operator_row.operator_name,
                    novelty=processed.novelty,
                    retrieval_support=processed.retrieval_support,
                )
            )
        return self._dedupe_frontier(entries)

    def _expand_branch(
        self,
        *,
        problem: ParsedProblem,
        route: RouteDecision,
        branch: BranchState,
        node: ReasoningStateNode,
        retrieved_traces: Sequence[RetrievedTrace],
        depth: int,
        generator: BranchGeneratorProtocol | None,
        symbolic_hook: SymbolicHookProtocol | None,
        verifier_hook: VerifierHookProtocol | None,
        failure_classifier: FailureClassifierProtocol | None,
        all_branches: dict[str, BranchState],
        active_branches: dict[str, BranchState],
        seen_branch_ids: set[str],
        solved_branch_ids: list[str],
    ) -> list[_FrontierEntry]:
        operator_rows, retrieval_signal = self._shape_operator_rows(
            node=node,
            route=route,
            retrieved_traces=retrieved_traces,
            branch=branch,
        )
        expansion_width = min(3, len(operator_rows))
        children: list[_FrontierEntry] = []
        for offset in range(expansion_width):
            operator_row = operator_rows[offset]
            retrieval_trace = self._pick_retrieval_for_operator(
                operator_name=operator_row.operator_name,
                retrieved_traces=retrieved_traces,
                offset=offset,
            )
            child = branch.fork(branch_label=f"{operator_row.operator_name}@{depth}", root_node=node)
            child.metadata["expansion_depth"] = depth
            if retrieval_trace is not None and child.active_cursor.retrieval_count == 0:
                self._attach_retrieval(branch=child, trace=retrieval_trace, retrieval_signal=retrieval_signal)
            processed = self._process_branch_attempt(
                problem=problem,
                route=route,
                branch=child,
                node=node,
                operator_name=operator_row.operator_name,
                retrieval_trace=retrieval_trace,
                depth=depth,
                sample_index=offset,
                generator=generator,
                symbolic_hook=symbolic_hook,
                verifier_hook=verifier_hook,
                failure_classifier=failure_classifier,
            )
            if processed.branch.branch_id in seen_branch_ids:
                continue
            all_branches[processed.branch.branch_id] = processed.branch
            active_branches[processed.branch.branch_id] = processed.branch
            seen_branch_ids.add(processed.branch.branch_id)
            if processed.branch.phase is BranchPhase.SOLVED:
                solved_branch_ids.append(processed.branch.branch_id)
            children.append(
                self._frontier_entry(
                    branch=processed.branch,
                    node=processed.node,
                    depth=depth,
                    operator_name=operator_row.operator_name,
                    novelty=processed.novelty,
                    retrieval_support=processed.retrieval_support,
                )
            )
        return self._dedupe_frontier(children)

    def _process_branch_attempt(
        self,
        *,
        problem: ParsedProblem,
        route: RouteDecision,
        branch: BranchState,
        node: ReasoningStateNode,
        operator_name: str,
        retrieval_trace: RetrievedTrace | None,
        depth: int,
        sample_index: int,
        generator: BranchGeneratorProtocol | None,
        symbolic_hook: SymbolicHookProtocol | None,
        verifier_hook: VerifierHookProtocol | None,
        failure_classifier: FailureClassifierProtocol | None,
    ) -> _ProcessedBranch:
        branch.add_step(
            kind=BranchStepKind.OPERATOR,
            phase=BranchPhase.REASONING,
            description=f"apply_operator::{operator_name}",
            operator_name=operator_name,
            node=node,
            summary_text=node.summary_text,
        )

        generation = self._run_generation(
            problem=problem,
            route=route,
            branch=branch,
            node=node,
            operator_name=operator_name,
            retrieval_trace=retrieval_trace,
            depth=depth,
            sample_index=sample_index,
            generator=generator,
        )
        supporting_steps = tuple(step.step_id for step in branch.active_steps()[-2:])
        branch.set_candidate_answer(
            raw_answer=generation.answer or "",
            canonical_answer=generation.answer_canonical or canonicalize_competition_answer(generation.answer),
            answer_source=f"generated::{operator_name}",
            confidence=generation.confidence,
            supporting_step_ids=supporting_steps,
        )

        updated_node = self._apply_generation_to_node(
            problem=problem,
            branch=branch,
            node=node,
            operator_name=operator_name,
            generation=generation,
            depth=depth,
        )
        branch.add_step(
            kind=BranchStepKind.OPERATOR,
            phase=BranchPhase.REASONING,
            description=generation.summary or generation.reasoning or f"generated reasoning via {operator_name}",
            operator_name=operator_name,
            node=updated_node,
            summary_text=updated_node.summary_text,
        )

        symbolic = self._run_symbolic(
            problem=problem,
            route=route,
            branch=branch,
            node=updated_node,
            operator_name=operator_name,
            generation=generation,
            symbolic_hook=symbolic_hook,
        )
        verifier = self._run_verifier(
            problem=problem,
            route=route,
            branch=branch,
            node=updated_node,
            symbolic=symbolic,
            generation=generation,
            verifier_hook=verifier_hook,
        )

        failure, diagnosis = self._classify_failure(
            problem=problem,
            route=route,
            branch=branch,
            symbolic=symbolic,
            verifier=verifier,
            generation=generation,
            failure_classifier=failure_classifier,
        )
        retrieval_support = _retrieval_support_score(branch)
        novelty = self._compute_branch_novelty(branch)
        answer_agreement = self._estimate_answer_agreement(branch)
        exact_symbolic = 1.0 if symbolic.exact_match else symbolic.score
        penalty = self._failure_penalty(branch=branch, failure=failure, depth=depth)

        branch.update_score_breakdown(
            verifier_probability=verifier.probability,
            tool_consistency=_clamp01((symbolic.score + max(symbolic.score, verifier.symbolic_agreement)) / 2.0),
            answer_agreement=answer_agreement,
            branch_novelty=novelty,
            exact_symbolic_check=exact_symbolic,
            retrieval_support=retrieval_support,
            logical_consistency=verifier.logical_consistency,
            completeness=verifier.completeness,
            repairability=verifier.repairability,
            step_quality=verifier.step_quality,
            prefix_quality=verifier.prefix_quality,
            open_obligation_burden=verifier.open_obligation_burden,
            penalty=penalty,
        )

        if self._is_solved(branch=branch, symbolic=symbolic, verifier=verifier):
            branch.set_phase(BranchPhase.SOLVED, reason="solver_acceptance")
        elif failure.route_to_repair:
            branch.metadata["failure_reason"] = failure.summary
            branch, updated_node = self._apply_local_repair(
                branch=branch,
                node=updated_node,
                failure=failure,
                diagnosis=diagnosis,
                route=route,
                retrieval_trace=retrieval_trace,
            )
        elif failure.route_to_resample:
            branch.metadata["failure_reason"] = failure.summary
            branch = self._apply_resample_guidance(
                branch=branch,
                failure=failure,
                diagnosis=diagnosis,
                route=route,
            )
        else:
            branch.set_phase(BranchPhase.REASONING, reason="branch_remains_expandable")

        retrieval_support = _retrieval_support_score(branch)
        novelty = self._compute_branch_novelty(branch)

        return _ProcessedBranch(
            branch=branch,
            node=updated_node,
            novelty=novelty,
            retrieval_support=retrieval_support,
        )

    def _run_generation(
        self,
        *,
        problem: ParsedProblem,
        route: RouteDecision,
        branch: BranchState,
        node: ReasoningStateNode,
        operator_name: str,
        retrieval_trace: RetrievedTrace | None,
        depth: int,
        sample_index: int,
        generator: BranchGeneratorProtocol | None,
    ) -> GenerationResult:
        if generator is not None:
            result = generator(
                problem=problem,
                route=route,
                branch=branch,
                node=node,
                operator_name=operator_name,
                retrieval_trace=retrieval_trace,
                depth=depth,
                sample_index=sample_index,
            )
            return GenerationResult(
                reasoning=result.reasoning,
                answer=result.answer,
                answer_canonical=result.answer_canonical or canonicalize_competition_answer(result.answer),
                confidence=_clamp01(result.confidence),
                summary=result.summary,
                partial_solution=result.partial_solution,
                added_constraints=tuple(result.added_constraints),
                added_invariants=tuple(result.added_invariants),
                added_goals=tuple(result.added_goals),
                metadata=dict(result.metadata),
            )

        retrieval_hint = retrieval_trace.solution[:200] if retrieval_trace else ""
        fallback_reasoning = (
            f"operator={operator_name}; depth={depth}; "
            f"route_top={_top_distribution_key(route.problem_type)}; "
            f"archetype_top={_top_distribution_key(route.archetypes)}"
        )
        if retrieval_hint:
            fallback_reasoning += f"; retrieval_hint={retrieval_hint}"
        answer = retrieval_trace.answer if retrieval_trace and retrieval_trace.answer else None
        return GenerationResult(
            reasoning=fallback_reasoning,
            answer=answer,
            answer_canonical=canonicalize_competition_answer(answer),
            confidence=_clamp01(0.40 + 0.20 * _prior_confidence(route, operator_name)),
            summary=f"fallback_generation::{operator_name}",
            partial_solution=answer or "",
            metadata={"mode": "deterministic_fallback"},
        )

    def _run_symbolic(
        self,
        *,
        problem: ParsedProblem,
        route: RouteDecision,
        branch: BranchState,
        node: ReasoningStateNode,
        operator_name: str,
        generation: GenerationResult,
        symbolic_hook: SymbolicHookProtocol | None,
    ) -> SymbolicHookResult:
        if symbolic_hook is not None:
            result = symbolic_hook(
                problem=problem,
                route=route,
                branch=branch,
                node=node,
                operator_name=operator_name,
                generation=generation,
            )
        else:
            answer = canonicalize_competition_answer(generation.answer)
            valid_answer = answer is not None and answer.isdigit() and ANSWER_MIN <= int(answer) <= ANSWER_MAX
            if valid_answer:
                result = SymbolicHookResult(
                    passed=True,
                    score=0.80,
                    summary="deterministic_symbolic_fallback::range_checked_answer",
                    exact_match=True,
                    metadata={"symbolic_status": "success"},
                )
            elif generation.answer:
                result = SymbolicHookResult(
                    passed=False,
                    score=0.05,
                    summary="deterministic_symbolic_fallback::invalid_answer",
                    exact_match=False,
                    metadata={"symbolic_status": "malformed_input"},
                )
            elif generation.reasoning:
                result = SymbolicHookResult(
                    passed=False,
                    score=0.18,
                    summary="deterministic_symbolic_fallback::unsupported",
                    exact_match=False,
                    metadata={"symbolic_status": "unsupported", "partial_support": False},
                )
            else:
                result = SymbolicHookResult(
                    passed=False,
                    score=0.0,
                    summary="deterministic_symbolic_fallback::no_signal",
                    exact_match=False,
                    metadata={"symbolic_status": "unsupported", "partial_support": False},
                )
        branch.attach_symbolic_evidence(
            passed=result.passed,
            score=result.score,
            check_name="symbolic_hook",
            summary=result.summary,
            supporting_step_ids=tuple(step.step_id for step in branch.active_steps()[-2:]),
            discharged_obligation_ids=tuple(result.metadata.get("discharged_obligation_ids", []) or ()),
            contradicted_obligation_ids=tuple(result.metadata.get("contradicted_obligation_ids", []) or ()),
            proof_obligations=tuple(result.metadata.get("proof_obligations", []) or ()) or None,
        )
        branch.add_step(
            kind=BranchStepKind.SYMBOLIC,
            phase=BranchPhase.VERIFIED,
            description=result.summary,
            operator_name=operator_name,
            node=node,
            summary_text=node.summary_text,
        )
        return result

    def _run_verifier(
        self,
        *,
        problem: ParsedProblem,
        route: RouteDecision,
        branch: BranchState,
        node: ReasoningStateNode,
        symbolic: SymbolicHookResult,
        generation: GenerationResult,
        verifier_hook: VerifierHookProtocol | None,
    ) -> VerifierHookResult:
        prm_summary = self._score_process_state(
            branch=branch,
            route=route,
            symbolic=symbolic,
            generation=generation,
        )
        if verifier_hook is not None:
            result = self._coerce_verifier_hook_result(
                verifier_hook(
                    problem=problem,
                    route=route,
                    branch=branch,
                    node=node,
                    symbolic=symbolic,
                    generation=generation,
                ),
                symbolic=symbolic,
                prm_summary=prm_summary,
            )
        else:
            retrieval_support = _retrieval_support_score(branch)
            symbolic_status = str((symbolic.metadata or {}).get("symbolic_status", "success" if symbolic.passed else "unsupported"))
            logical_consistency = _clamp01(
                0.55 * symbolic.score
                + 0.25 * generation.confidence
                + 0.20 * (1.0 - route.route_uncertainty)
            )
            completeness = _clamp01(0.40 + 0.30 * bool(generation.reasoning) + 0.30 * bool(generation.answer))
            probability = _clamp01(
                0.35 * generation.confidence
                + 0.35 * symbolic.score
                + 0.20 * retrieval_support
                + 0.10 * (1.0 - route.route_uncertainty)
            )
            if symbolic_status == "unsupported":
                logical_consistency = min(logical_consistency, 0.46)
                completeness = min(completeness, 0.58)
                probability = min(probability, 0.38)
            elif symbolic_status in {"malformed_input", "execution_failure", "contradiction"}:
                logical_consistency = min(logical_consistency, 0.28)
                completeness = min(completeness, 0.42)
                probability = min(probability, 0.18)
            result = VerifierHookResult(
                probability=probability,
                logical_consistency=logical_consistency,
                completeness=completeness,
                repairability=_clamp01(0.75 - 0.40 * symbolic.score if not symbolic.passed else 0.25),
                summary=f"deterministic_verifier_fallback::{symbolic_status}",
                symbolic_agreement=symbolic.score,
                answer_correctness_likelihood=probability,
                branch_score=_clamp01(0.65 * probability + 0.20 * prm_summary.prefix_quality + 0.15 * prm_summary.mean_step_quality),
                step_quality=prm_summary.mean_step_quality,
                prefix_quality=prm_summary.prefix_quality,
                open_obligation_burden=prm_summary.open_obligation_burden,
                metadata={
                    "runtime_mode": "controller_fallback",
                    "symbolic_status": symbolic_status,
                    "step_quality": prm_summary.mean_step_quality,
                    "prefix_quality": prm_summary.prefix_quality,
                    "open_obligation_burden": prm_summary.open_obligation_burden,
                    "verifier_decomposition": {
                        "logical_consistency": logical_consistency,
                        "symbolic_agreement": symbolic.score,
                        "completeness": completeness,
                        "answer_correctness_likelihood": probability,
                        "repairability": _clamp01(0.75 - 0.40 * symbolic.score if not symbolic.passed else 0.25),
                        "step_quality": prm_summary.mean_step_quality,
                        "prefix_quality": prm_summary.prefix_quality,
                        "open_obligation_burden": prm_summary.open_obligation_burden,
                    },
                },
            )
        branch.attach_verifier_evidence(
            probability=result.probability,
            logical_consistency=result.logical_consistency,
            symbolic_agreement=result.symbolic_agreement,
            completeness=result.completeness,
            repairability=result.repairability,
            summary=result.summary,
            answer_correctness_likelihood=result.answer_correctness_likelihood,
            step_quality=result.step_quality if result.step_quality is not None else prm_summary.mean_step_quality,
            prefix_quality=result.prefix_quality if result.prefix_quality is not None else prm_summary.prefix_quality,
            open_obligation_burden=(
                result.open_obligation_burden
                if result.open_obligation_burden is not None
                else prm_summary.open_obligation_burden
            ),
            supporting_obligation_ids=tuple(
                result.metadata.get("supporting_obligation_ids", [])
                or result.metadata.get("discharged_obligation_ids", [])
                or ()
            ),
            failure_step_index=result.failure_step_index,
            failure_step_id=result.failure_step_id,
            failure_node_id=result.failure_node_id,
            failure_obligation_id=result.failure_obligation_id,
            decomposition=(
                dict(result.metadata.get("verifier_decomposition", {}) or {})
                if isinstance(result.metadata.get("verifier_decomposition"), Mapping)
                else {}
            ),
        )
        branch.add_step(
            kind=BranchStepKind.VERIFIER,
            phase=BranchPhase.VERIFIED,
            description=result.summary,
            operator_name=None,
            node=node,
            summary_text=node.summary_text,
        )
        return result

    def _coerce_verifier_hook_result(
        self,
        raw_result: object,
        *,
        symbolic: SymbolicHookResult,
        prm_summary,
    ) -> VerifierHookResult:
        if isinstance(raw_result, VerifierHookResult):
            answer_likelihood = (
                raw_result.answer_correctness_likelihood
                if raw_result.answer_correctness_likelihood is not None
                else raw_result.probability
            )
            branch_score = (
                raw_result.branch_score
                if raw_result.branch_score is not None
                else raw_result.probability
            )
            return VerifierHookResult(
                probability=_clamp01(raw_result.probability),
                logical_consistency=_clamp01(raw_result.logical_consistency),
                completeness=_clamp01(raw_result.completeness),
                repairability=_clamp01(raw_result.repairability),
                summary=raw_result.summary,
                symbolic_agreement=_clamp01(raw_result.symbolic_agreement or symbolic.score),
                answer_correctness_likelihood=_clamp01(answer_likelihood),
                branch_score=_clamp01(branch_score),
                step_quality=_clamp01(raw_result.step_quality if raw_result.step_quality is not None else prm_summary.mean_step_quality),
                prefix_quality=_clamp01(raw_result.prefix_quality if raw_result.prefix_quality is not None else prm_summary.prefix_quality),
                open_obligation_burden=_clamp01(
                    raw_result.open_obligation_burden
                    if raw_result.open_obligation_burden is not None
                    else prm_summary.open_obligation_burden
                ),
                failure_step_index=raw_result.failure_step_index,
                failure_step_id=raw_result.failure_step_id,
                failure_node_id=raw_result.failure_node_id,
                failure_obligation_id=raw_result.failure_obligation_id,
                metadata=dict(raw_result.metadata),
            )

        if isinstance(raw_result, Mapping):
            metadata = dict(raw_result.get("metadata", {}) or {})
            symbolic_agreement = raw_result.get(
                "symbolic_agreement",
                raw_result.get("symbolic_consistency", metadata.get("symbolic_agreement", symbolic.score)),
            )
            answer_likelihood = raw_result.get(
                "answer_correctness_likelihood",
                raw_result.get("probability", metadata.get("answer_correctness_likelihood", 0.0)),
            )
            branch_score = raw_result.get(
                "branch_score",
                raw_result.get("overall_score", metadata.get("branch_score", answer_likelihood)),
            )
            return VerifierHookResult(
                probability=_clamp01(raw_result.get("probability", answer_likelihood)),
                logical_consistency=_clamp01(raw_result.get("logical_consistency", 0.0)),
                completeness=_clamp01(raw_result.get("completeness", 0.0)),
                repairability=_clamp01(raw_result.get("repairability", 0.0)),
                summary=str(raw_result.get("summary", "") or ""),
                symbolic_agreement=_clamp01(symbolic_agreement),
                answer_correctness_likelihood=_clamp01(answer_likelihood),
                branch_score=_clamp01(branch_score),
                step_quality=_clamp01(raw_result.get("step_quality", metadata.get("step_quality", prm_summary.mean_step_quality))),
                prefix_quality=_clamp01(raw_result.get("prefix_quality", metadata.get("prefix_quality", prm_summary.prefix_quality))),
                open_obligation_burden=_clamp01(
                    raw_result.get("open_obligation_burden", metadata.get("open_obligation_burden", prm_summary.open_obligation_burden))
                ),
                failure_step_index=int(raw_result["failure_step_index"]) if raw_result.get("failure_step_index") is not None else None,
                failure_step_id=str(raw_result.get("failure_step_id")) if raw_result.get("failure_step_id") is not None else None,
                failure_node_id=str(raw_result.get("failure_node_id")) if raw_result.get("failure_node_id") is not None else None,
                failure_obligation_id=str(raw_result.get("failure_obligation_id")) if raw_result.get("failure_obligation_id") is not None else None,
                metadata=metadata,
            )

        metadata = dict(getattr(raw_result, "metadata", {}) or {})
        symbolic_agreement = metadata.get("symbolic_agreement", metadata.get("symbolic_consistency", symbolic.score))
        answer_likelihood = metadata.get("answer_correctness_likelihood", getattr(raw_result, "probability", 0.0))
        branch_score = metadata.get("branch_score", metadata.get("overall_score", answer_likelihood))
        return VerifierHookResult(
            probability=_clamp01(getattr(raw_result, "probability", answer_likelihood)),
            logical_consistency=_clamp01(getattr(raw_result, "logical_consistency", 0.0)),
            completeness=_clamp01(getattr(raw_result, "completeness", 0.0)),
            repairability=_clamp01(getattr(raw_result, "repairability", 0.0)),
            summary=str(getattr(raw_result, "summary", "") or ""),
            symbolic_agreement=_clamp01(symbolic_agreement),
            answer_correctness_likelihood=_clamp01(answer_likelihood),
            branch_score=_clamp01(branch_score),
            step_quality=_clamp01(metadata.get("step_quality", prm_summary.mean_step_quality)),
            prefix_quality=_clamp01(metadata.get("prefix_quality", prm_summary.prefix_quality)),
            open_obligation_burden=_clamp01(metadata.get("open_obligation_burden", prm_summary.open_obligation_burden)),
            failure_step_index=metadata.get("failure_step_index") if isinstance(metadata.get("failure_step_index"), int) else None,
            failure_step_id=metadata.get("failure_step_id") if isinstance(metadata.get("failure_step_id"), str) else None,
            failure_node_id=metadata.get("failure_node_id") if isinstance(metadata.get("failure_node_id"), str) else None,
            failure_obligation_id=metadata.get("failure_obligation_id") if isinstance(metadata.get("failure_obligation_id"), str) else None,
            metadata=metadata,
        )

    def _score_process_state(
        self,
        *,
        branch: BranchState,
        route: RouteDecision,
        symbolic: SymbolicHookResult,
        generation: GenerationResult,
    ):
        return score_process(
            ProcessScoreInput(
                problem_id=branch.problem_id,
                branch_id=branch.branch_id,
                steps=tuple(
                    ProcessStepInput(
                        step_index=step.index + 1,
                        description=step.description,
                        operator_name=step.operator_name,
                        symbolic_valid=bool(symbolic.passed) if step.kind == BranchStepKind.SYMBOLIC else True,
                        proof_obligation_ids=tuple(step.proof_obligation_ids),
                        state_fingerprint=step.state_fingerprint,
                    )
                    for step in branch.active_steps()
                ),
                proof_obligations=tuple(
                    ProcessObligationInput(
                        obligation_id=item.obligation_id,
                        status=item.status.value,
                        claim=item.claim,
                        evidence_kind_required=item.evidence_kind_required.value,
                        originating_node_id=item.originating_node_id,
                        target_goal_id=item.target_goal_id,
                    )
                    for item in branch.active_proof_obligations()
                ),
                route_uncertainty=route.route_uncertainty,
                retrieval_support=_retrieval_support_score(branch),
                symbolic_score=symbolic.score,
                symbolic_passed=symbolic.passed,
                exact_symbolic_match=symbolic.exact_match,
                answer_present=bool(generation.answer or (branch.current_candidate() and branch.current_candidate().canonical_answer)),
                contradiction_count=sum(
                    1 for item in branch.active_proof_obligations() if item.status.value == "contradicted"
                ),
            )
        )

    def _classify_failure(
        self,
        *,
        problem: ParsedProblem,
        route: RouteDecision,
        branch: BranchState,
        symbolic: SymbolicHookResult,
        verifier: VerifierHookResult,
        generation: GenerationResult,
        failure_classifier: FailureClassifierProtocol | None,
    ) -> tuple[FailureDecision, BranchFailureDiagnosis | None]:
        if failure_classifier is not None:
            return (
                failure_classifier(
                    problem=problem,
                    route=route,
                    branch=branch,
                    symbolic=symbolic,
                    verifier=verifier,
                    generation=generation,
                ),
                None,
            )

        diagnosis = classify_branch_failure(branch)
        return self._decision_from_diagnosis(diagnosis), diagnosis

    def _decision_from_diagnosis(self, diagnosis: BranchFailureDiagnosis) -> FailureDecision:
        route_to_repair = diagnosis.can_repair_in_place and diagnosis.should_retry_locally
        route_to_resample = not route_to_repair and diagnosis.recoverability > 0.0
        return FailureDecision(
            failure_type=diagnosis.schema_failure_type,
            route_to_repair=route_to_repair,
            route_to_resample=route_to_resample,
            summary=diagnosis.summary,
        )

    def _legacy_failure_decision(
        self,
        *,
        route: RouteDecision,
        symbolic: SymbolicHookResult,
        verifier: VerifierHookResult,
        generation: GenerationResult,
    ) -> FailureDecision:
        if not symbolic.passed:
            return FailureDecision(
                failure_type=FailureType.SYMBOLIC_MISMATCH,
                route_to_repair=True,
                route_to_resample=False,
                summary="symbolic_failure_prefers_local_repair",
            )
        if not generation.answer:
            return FailureDecision(
                failure_type=FailureType.COVERAGE_GAP,
                route_to_repair=False,
                route_to_resample=True,
                summary="missing_answer_prefers_resample",
            )
        answer = canonicalize_competition_answer(generation.answer)
        if answer is None:
            return FailureDecision(
                failure_type=FailureType.ARITHMETIC_ERROR,
                route_to_repair=True,
                route_to_resample=False,
                summary="answer_not_canonical_prefers_repair",
            )
        if verifier.probability < max(0.30, route.repair_threshold - 0.10):
            if verifier.repairability >= 0.40:
                return FailureDecision(
                    failure_type=FailureType.LOGIC_ERROR,
                    route_to_repair=True,
                    route_to_resample=False,
                    summary="low_verifier_score_prefers_repair",
                )
            return FailureDecision(
                failure_type=FailureType.MISSING_CASE,
                route_to_repair=False,
                route_to_resample=True,
                summary="low_verifier_score_prefers_resample",
            )
        return FailureDecision(
            failure_type=None,
            route_to_repair=False,
            route_to_resample=False,
            summary="no_failure",
        )

    def _apply_local_repair(
        self,
        *,
        branch: BranchState,
        node: ReasoningStateNode,
        failure: FailureDecision,
        diagnosis: BranchFailureDiagnosis | None,
        route: RouteDecision,
        retrieval_trace: RetrievedTrace | None,
    ) -> tuple[BranchState, ReasoningStateNode]:
        repair_budget = min(self._repair_budget(route), MAX_REPAIR_ATTEMPTS)
        if branch.active_cursor.repair_count >= repair_budget:
            branch.set_phase(BranchPhase.FAILED, reason="repair_budget_exhausted")
            return branch, node

        trace = self._to_branch_trace(branch)
        trace = trace.model_copy(
            update={
                "failure_type": diagnosis.schema_failure_type if diagnosis is not None else failure.failure_type,
                "failure_location": (
                    diagnosis.location.summary
                    if diagnosis is not None and diagnosis.location.summary
                    else trace.failure_location
                ),
                "failure_step_index": (
                    int(getattr(diagnosis.location, "step_index", 0))
                    if diagnosis is not None and getattr(diagnosis, "location", None) is not None
                    else trace.failure_step_index
                ),
                "failure_step_id": (
                    branch.active_steps()[int(getattr(diagnosis.location, "step_index", 0))].step_id
                    if diagnosis is not None
                    and getattr(diagnosis, "location", None) is not None
                    and isinstance(getattr(diagnosis.location, "step_index", None), int)
                    and 0 <= int(diagnosis.location.step_index) < len(branch.active_steps())
                    else trace.failure_step_id
                ),
                "failure_node_id": branch.latest_node_id or trace.failure_node_id,
                "failure_obligation_id": trace.failure_obligation_id
                or next(
                    (
                        item.obligation_id
                        for item in branch.active_proof_obligations()
                        if item.status.value == "contradicted"
                    ),
                    next(
                        (
                            item.obligation_id
                            for item in branch.active_proof_obligations()
                            if item.is_open()
                        ),
                        None,
                    ),
                ),
            }
        )

        retrieval_hints = list(route.repair_neighbors)
        if retrieval_trace is not None:
            retrieval_hints.extend(list(retrieval_trace.operators_used[:4]))
            retrieval_hints.extend(list(getattr(retrieval_trace, "repair_operator_hints", [])[:3]))
            retrieval_hints.extend(list(getattr(retrieval_trace, "failure_mode_support", [])[:2]))
            retrieval_hints.extend(list(getattr(retrieval_trace, "relevant_evidence_kinds", [])[:2]))
        if diagnosis is not None:
            retrieval_hints.extend(list(diagnosis.suggested_operator_bias))

        result = repair_branch(
            trace,
            node=node,
            route=route,
            failure_type=trace.failure_type,
            budget=RepairBudget(
                max_attempts=repair_budget,
                max_candidates=6,
                max_mutation_window=2,
                max_rollback_steps=3,
                local_window_radius=1,
                allow_operator_substitution=True,
                allow_mutation=True,
                allow_rollback=True,
                allow_full_restart=False,
                deterministic=True,
                retrieval_hints=tuple(dict.fromkeys(hint for hint in retrieval_hints if hint)),
            ),
            retrieval_hints=retrieval_hints,
            operator_library=self.operator_library,
        )
        if result.repaired_branch is None or result.selected_candidate is None:
            branch.set_phase(BranchPhase.FAILED, reason=f"repair_unavailable::{failure.summary}")
            return branch, node

        selected = result.selected_candidate
        preserve_prefix = max(0, int(selected.action.preserve_prefix_until))
        target_step_index = selected.action.target_step_index
        branch.record_repair(
            reason=failure.summary,
            from_step_index=max(0, target_step_index if target_step_index is not None else max(0, branch.active_cursor.step_count - 2)),
            to_step_index=preserve_prefix,
            success=result.outcome.value == "applied",
            failure_type=trace.failure_type.value if trace.failure_type else None,
            notes=selected.action.description,
        )
        branch.add_step(
            kind=BranchStepKind.REPAIR,
            phase=BranchPhase.REPAIRED,
            description=selected.action.description or f"repair::{failure.summary}",
            operator_name=selected.action.replacement_operator,
            node=result.repaired_node or node,
            summary_text=(result.repaired_node.summary_text if result.repaired_node is not None else node.summary_text),
        )
        if result.repaired_branch.answer:
            branch.set_candidate_answer(
                raw_answer=result.repaired_branch.answer,
                canonical_answer=result.repaired_branch.answer_canonical or canonicalize_competition_answer(result.repaired_branch.answer),
                answer_source="repair",
                confidence=max(branch.composite_score(), selected.score),
                supporting_step_ids=tuple(step.step_id for step in branch.active_steps()[-2:]),
            )
        branch.metadata.update(dict(result.history_update))
        branch.update_score_breakdown(
            penalty=_clamp01(branch.score_breakdown.penalty + self.config.repair_penalty_weight),
            branch_novelty=_clamp01(max(branch.score_breakdown.branch_novelty, selected.prefix_preservation_ratio)),
            exact_symbolic_check=max(
                branch.score_breakdown.exact_symbolic_check,
                1.0 if result.repaired_branch.symbolic_valid else branch.score_breakdown.exact_symbolic_check,
            ),
        )
        branch.set_phase(BranchPhase.REPAIRED, reason="module_repair_applied")
        return branch, result.repaired_node or node

    def _apply_resample_guidance(
        self,
        *,
        branch: BranchState,
        failure: FailureDecision,
        diagnosis: BranchFailureDiagnosis | None,
        route: RouteDecision,
    ) -> BranchState:
        resample_count = int(branch.metadata.get("resample_count", 0))
        if resample_count >= self._resample_budget(route):
            branch.set_phase(BranchPhase.FAILED, reason="resample_budget_exhausted")
            return branch

        result = resample_branches(
            [branch],
            route=route,
            failure_diagnoses=[diagnosis if diagnosis is not None else failure.failure_type],
            budget=ResampleBudget(
                max_new_branches=1,
                max_total_branches=max(route.branch_budget, 1),
                max_attempts_per_focus=1,
                max_duplicates_per_signature=0,
                novelty_floor=0.10,
                deterministic_seed=resample_count,
                prefer_self_consistency_fill=False,
                suppress_answer_duplicates=False,
            ),
        )
        if not result.created_branches:
            branch.set_phase(BranchPhase.FAILED, reason=f"resample_unavailable::{failure.summary}")
            return branch

        resampled = result.created_branches[0]
        resampled.metadata["resample_count"] = resample_count + 1
        resampled.update_score_breakdown(
            penalty=_clamp01(resampled.score_breakdown.penalty + 0.06),
            branch_novelty=_clamp01(max(resampled.score_breakdown.branch_novelty, 0.05)),
        )
        return resampled

    def _shape_operator_rows(
        self,
        *,
        node: ReasoningStateNode,
        route: RouteDecision,
        retrieved_traces: Sequence[RetrievedTrace],
        branch: BranchState | None,
    ) -> tuple[list[OperatorPolicyScore], dict[str, float]]:
        retrieval_signal = _retrieval_operator_support(retrieved_traces)
        retrieval_context = _retrieval_context(retrieved_traces, branch=branch)
        for focus_operator in _branch_focus_operators(branch):
            retrieval_signal[focus_operator] = retrieval_signal.get(focus_operator, 0.0) + 0.20
        total = sum(retrieval_signal.values())
        if total > 0.0:
            retrieval_signal = {name: score / total for name, score in sorted(retrieval_signal.items())}
        _, rows = self.prior_shaper.shape_priors(
            node=node,
            route=route,
            retrieved_operator_hints=retrieval_signal,
            retrieved_traces=retrieved_traces,
            retrieval_context=retrieval_context,
        )
        ordered = [row for row in rows if row.normalized_score > 0][: max(4, min(len(rows), self._frontier_width(route)))]
        if not ordered:
            ordered = rows[:4]
        return ordered, retrieval_signal

    def _attach_retrieval(
        self,
        *,
        branch: BranchState,
        trace: RetrievedTrace,
        retrieval_signal: Mapping[str, float],
    ) -> None:
        merged_operator_support = dict(retrieval_signal)
        for name, score in dict(getattr(trace, "operator_support", {}) or {}).items():
            merged_operator_support[name] = max(float(score), merged_operator_support.get(name, 0.0))
        branch.attach_retrieval_evidence(
            trace_id=trace.trace_id,
            score=trace.similarity_score,
            compatibility_score=float(getattr(trace, "compatibility_score", 0.0) or 0.0),
            operator_support=merged_operator_support,
            problem=trace.problem,
            solution=trace.solution,
            answer=trace.answer,
            domain=trace.domain,
            archetypes=tuple(trace.archetypes),
            operators_used=tuple(trace.operators_used),
            repair_operator_hints=tuple(getattr(trace, "repair_operator_hints", []) or ()),
            branch_continuation_bias=dict(getattr(trace, "branch_continuation_bias", {}) or {}),
            relevant_obligation_claims=tuple(getattr(trace, "relevant_obligation_claims", []) or ()),
            relevant_evidence_kinds=tuple(getattr(trace, "relevant_evidence_kinds", []) or ()),
            failure_mode_support=tuple(getattr(trace, "failure_mode_support", []) or ()),
            compatibility_reasons=tuple(getattr(trace, "compatibility_reasons", []) or ()),
            strategy_summary=str(getattr(trace, "strategy_summary", "") or ""),
        )
        branch.add_step(
            kind=BranchStepKind.RETRIEVAL,
            phase=BranchPhase.RETRIEVED,
            description=f"retrieved_trace::{trace.trace_id}",
            summary_text=(str(getattr(trace, "strategy_summary", "") or "")[:160] or trace.problem[:160]),
        )

    def _apply_generation_to_node(
        self,
        *,
        problem: ParsedProblem,
        branch: BranchState,
        node: ReasoningStateNode,
        operator_name: str,
        generation: GenerationResult,
        depth: int,
    ) -> ReasoningStateNode:
        added_constraints = tuple(
            ConstraintRecord.create(kind="other", lhs=text, relation=None, rhs=None, origin="generated", strength="soft")
            for text in generation.added_constraints
        )
        added_invariants = tuple(
            InvariantRecord.create(invariant_type="generated", expression=text, proof_source="branch_generator", confidence=generation.confidence)
            for text in generation.added_invariants
        )
        added_goals = tuple(
            GoalRecord.create(kind="subgoal", text=text, priority=50)
            for text in generation.added_goals
        )
        op_record = OperatorApplicationRecord.create(
            operator_name=operator_name,
            rationale=generation.summary or generation.reasoning[:180],
            pre_state_fingerprint=node.state_fingerprint,
            success=bool(generation.reasoning),
        )
        evidence = EvidenceRecord.create(
            kind="other",
            source="branch_controller",
            summary=generation.summary or f"generated via {operator_name}",
            score=generation.confidence,
            payload={"reasoning": generation.reasoning[:400]},
        )
        summary = generation.summary or generation.reasoning[:200] or node.summary_text
        return node.with_updates(
            constraints=node.constraints + added_constraints,
            invariants=node.invariants + added_invariants,
            goals=node.goals + added_goals,
            partial_solution=generation.partial_solution or generation.answer or node.partial_solution,
            summary_text=summary,
            operator_history=node.operator_history + (op_record,),
            evidence=node.evidence + (evidence,),
            confidence=_clamp01(max(node.confidence, generation.confidence)),
            verifier_score=node.verifier_score,
            symbolic_valid=node.symbolic_valid,
            merge_ready=True,
            is_terminal=bool(generation.answer),
            parent_node_ids=(node.node_id,),
            source_tags=node.provenance.source_tags + ("controller",),
            depth=max(node.provenance.depth, depth),
            lineage=node.provenance.lineage + (node.node_id,),
            metadata={**dict(node.metadata), "problem_id": problem.problem_id, "last_operator": operator_name, "branch_id": branch.branch_id},
        )

    def _frontier_entry(
        self,
        *,
        branch: BranchState,
        node: ReasoningStateNode,
        depth: int,
        operator_name: str | None,
        novelty: float,
        retrieval_support: float,
    ) -> _FrontierEntry:
        priority = self._priority_score(
            branch=branch,
            depth=depth,
            novelty=novelty,
            retrieval_support=retrieval_support,
        )
        return _FrontierEntry(
            sort_key=(-priority, depth, branch.branch_id),
            branch=branch,
            node=node,
            depth=depth,
            operator_name=operator_name,
            novelty=novelty,
            retrieval_support=retrieval_support,
        )

    def _priority_score(
        self,
        *,
        branch: BranchState,
        depth: int,
        novelty: float,
        retrieval_support: float,
    ) -> float:
        base = branch.composite_score()
        search_value = self._branch_search_value(branch=branch, retrieval_support=retrieval_support)
        exploration_bonus = self.config.exploration_bonus_weight / max(1.0, 1.0 + depth)
        disagreement_bonus = self.config.disagreement_bonus_weight * (1.0 - branch.score_breakdown.answer_agreement)
        depth_penalty = self.config.depth_penalty_weight * (depth / max(1, self._max_depth(branch.route)))
        novelty_bonus = self.config.novelty_bonus_weight * novelty
        retrieval_bonus = self.config.retrieval_blend_weight * retrieval_support
        obligation_penalty = 0.14 * branch.score_breakdown.open_obligation_burden
        return (
            0.52 * base
            + 0.30 * search_value
            + exploration_bonus
            + disagreement_bonus
            + novelty_bonus
            + retrieval_bonus
            - depth_penalty
            - obligation_penalty
        )

    def _branch_search_value(self, *, branch: BranchState, retrieval_support: float) -> float:
        breakdown = branch.score_breakdown
        return _clamp01(
            0.24 * breakdown.verifier_probability
            + 0.18 * breakdown.logical_consistency
            + 0.10 * breakdown.completeness
            + 0.10 * breakdown.exact_symbolic_check
            + 0.12 * breakdown.step_quality
            + 0.18 * breakdown.prefix_quality
            + 0.08 * retrieval_support
            + 0.04 * breakdown.repairability
            - 0.18 * breakdown.open_obligation_burden
        )

    def _prune_frontier(self, *, frontier: list[_FrontierEntry], route: RouteDecision) -> list[_FrontierEntry]:
        width = self._frontier_width(route)
        if len(frontier) <= width:
            return frontier
        kept = heapq.nsmallest(width, frontier)
        pruned = [entry for entry in frontier if entry not in kept]
        for entry in pruned:
            entry.branch.set_phase(BranchPhase.PRUNED, reason="frontier_width_pruning")
        heapq.heapify(kept)
        return kept

    def _dedupe_frontier(self, entries: Sequence[_FrontierEntry]) -> list[_FrontierEntry]:
        deduped: dict[tuple[str, str | None], _FrontierEntry] = {}
        for entry in entries:
            candidate = entry.branch.current_candidate()
            canonical = (candidate.canonical_answer if candidate else "") or entry.branch.branch_id
            key = (canonical, entry.operator_name)
            prior = deduped.get(key)
            if prior is None or entry.sort_key < prior.sort_key:
                deduped[key] = entry
        return list(deduped.values())

    def _select_survivors(self, *, route: RouteDecision, branches: Sequence[BranchState]) -> tuple[BranchState, ...]:
        eligible = [
            branch for branch in branches
            if branch.phase not in {BranchPhase.FAILED, BranchPhase.PRUNED}
        ]
        ranked = sorted(
            eligible,
            key=lambda branch: (
                -self._branch_search_value(branch=branch, retrieval_support=_retrieval_support_score(branch)),
                -branch.composite_score(),
                branch.active_cursor.repair_count,
                branch.branch_id,
            ),
        )
        return tuple(ranked[: max(1, min(self.config.retain_top_k, route.branch_budget))])

    def _maybe_mid_search_critique(
        self,
        *,
        problem: ParsedProblem,
        route: RouteDecision,
        current: _FrontierEntry,
        frontier: list[_FrontierEntry],
        active_branches: dict[str, BranchState],
        all_branches: dict[str, BranchState],
        critique_model: object | None,
        expanded_nodes: int,
    ) -> tuple[SelfCritiqueBatchResult, list[_FrontierEntry]]:
        if not self.config.enable_mid_search_critique:
            return SelfCritiqueBatchResult(branches=[], critiques=[], critiqued_branch_ids=[], top_k=0), frontier
        if expanded_nodes < self.config.mid_search_critique_min_expansions:
            return SelfCritiqueBatchResult(branches=[], critiques=[], critiqued_branch_ids=[], top_k=0), frontier
        if self.config.mid_search_critique_every_n_expansions <= 0:
            return SelfCritiqueBatchResult(branches=[], critiques=[], critiqued_branch_ids=[], top_k=0), frontier
        if expanded_nodes % self.config.mid_search_critique_every_n_expansions != 0:
            return SelfCritiqueBatchResult(branches=[], critiques=[], critiqued_branch_ids=[], top_k=0), frontier
        if len(frontier) < self.config.mid_search_critique_min_frontier:
            return SelfCritiqueBatchResult(branches=[], critiques=[], critiqued_branch_ids=[], top_k=0), frontier

        candidate_entries = self._mid_search_critique_entries(current=current, frontier=frontier)
        traces = [self._to_branch_trace(entry.branch) for entry in candidate_entries]
        if not traces:
            return SelfCritiqueBatchResult(branches=[], critiques=[], critiqued_branch_ids=[], top_k=0), frontier

        top_k = max(0, min(self.config.mid_search_critique_top_k, len(traces)))
        if top_k <= 0:
            return SelfCritiqueBatchResult(branches=[], critiques=[], critiqued_branch_ids=[], top_k=0), frontier

        ordered = self._select_for_critique(traces, k=top_k, stage="mid_search")
        if not ordered:
            return SelfCritiqueBatchResult(branches=traces, critiques=[], critiqued_branch_ids=[], top_k=0), frontier

        critique_batch = self._run_critique_batch(
            traces,
            top_k=top_k,
            critique_model=critique_model,
            stage="mid_search",
        )
        if not critique_batch.critiques:
            return critique_batch, frontier

        node_map = {entry.branch.branch_id: entry.node for entry in candidate_entries}
        updated_frontier: list[_FrontierEntry] = []
        critiqued_by_id = {result.critiqued_branch.branch_id: result.critiqued_branch for result in critique_batch.critiques}
        for branch_id, critiqued_trace in critiqued_by_id.items():
            original_branch = active_branches.get(branch_id)
            if original_branch is None:
                continue
            merged = self._merge_critiqued_branch(original_branch, critiqued_trace)
            merged.metadata["mid_search_critique_applied"] = True
            active_branches[branch_id] = merged
            all_branches[branch_id] = merged

        for entry in frontier:
            replacement = active_branches.get(entry.branch.branch_id)
            if replacement is not None and replacement is not entry.branch:
                updated_frontier.append(
                    self._frontier_entry(
                        branch=replacement,
                        node=node_map.get(entry.branch.branch_id, entry.node),
                        depth=entry.depth,
                        operator_name=entry.operator_name,
                        novelty=self._compute_branch_novelty(replacement),
                        retrieval_support=_retrieval_support_score(replacement),
                    )
                )
            else:
                updated_frontier.append(entry)
        updated_frontier = self._dedupe_frontier(updated_frontier)
        updated_frontier = self._prune_frontier(frontier=updated_frontier, route=route)
        return critique_batch, updated_frontier

    def _mid_search_critique_entries(
        self,
        *,
        current: _FrontierEntry,
        frontier: Sequence[_FrontierEntry],
    ) -> tuple[_FrontierEntry, ...]:
        ordered = [current] + list(heapq.nsmallest(max(0, self.config.mid_search_critique_top_k * 3), list(frontier)))
        deduped: dict[str, _FrontierEntry] = {}
        for entry in ordered:
            branch = entry.branch
            if branch.phase in {BranchPhase.FAILED, BranchPhase.PRUNED, BranchPhase.SOLVED}:
                continue
            if branch.metadata.get("mid_search_critique_applied"):
                continue
            prior = deduped.get(branch.branch_id)
            if prior is None or entry.sort_key < prior.sort_key:
                deduped[branch.branch_id] = entry
        return tuple(deduped.values())

    def _select_for_critique(self, traces, *, k: int, stage: str):
        try:
            signature = inspect.signature(select_top_k_for_critique)
            if "stage" in signature.parameters:
                return select_top_k_for_critique(
                    traces,
                    k=k,
                    stage=stage,
                    config=self._critique_config(top_k=k),
                )
        except Exception:
            pass
        return select_top_k_for_critique(traces, k=k)

    def _run_critique_batch(self, traces, *, top_k: int, critique_model: object | None, stage: str) -> SelfCritiqueBatchResult:
        config = self._critique_config(top_k=top_k)
        try:
            signature = inspect.signature(critique_top_k_branches)
            if "stage" in signature.parameters:
                return critique_top_k_branches(
                    traces,
                    model=critique_model,
                    config=config,
                    stage=stage,
                )
        except Exception:
            pass
        return critique_top_k_branches(
            traces,
            model=critique_model,
            config=config,
        )

    def _critique_config(self, *, top_k: int) -> SelfCritiqueConfig:
        kwargs = {"top_k": top_k}
        try:
            fields = getattr(SelfCritiqueConfig, "model_fields", None) or getattr(SelfCritiqueConfig, "__fields__", {})
            if "max_mid_search_branches" in fields:
                kwargs["max_mid_search_branches"] = max(0, min(top_k, self.config.mid_search_critique_top_k))
            if "max_pre_aggregation_branches" in fields:
                kwargs["max_pre_aggregation_branches"] = max(0, top_k)
        except Exception:
            pass
        return SelfCritiqueConfig(**kwargs)

    def _critique_survivors(
        self,
        *,
        problem: ParsedProblem,
        surviving: Sequence[BranchState],
        critique_model: object | None,
        route: RouteDecision,
    ) -> SelfCritiqueBatchResult:
        traces = [self._to_branch_trace(branch) for branch in surviving]
        if not traces:
            return SelfCritiqueBatchResult(branches=[], critiques=[], critiqued_branch_ids=[], top_k=0)
        top_k = self._critique_top_k(route)
        if top_k <= 0:
            return SelfCritiqueBatchResult(branches=traces, critiques=[], critiqued_branch_ids=[], top_k=0)
        ordered = self._select_for_critique(traces, k=top_k, stage="pre_aggregation")
        if not ordered:
            return SelfCritiqueBatchResult(branches=traces, critiques=[], critiqued_branch_ids=[], top_k=0)
        return self._run_critique_batch(
            traces,
            top_k=top_k,
            critique_model=critique_model,
            stage="pre_aggregation",
        )

    def _merge_critiqued_branch(self, branch: BranchState, critiqued_trace) -> BranchState:
        summary = critiqued_trace.full_reasoning[:220] if critiqued_trace.full_reasoning else "self_critique"
        branch.attach_critique_evidence(
            changed_answer=critiqued_trace.answer != (branch.current_candidate().raw_answer if branch.current_candidate() else None),
            summary=summary,
            critique_score=_clamp01(critiqued_trace.branch_score),
            recommended_answer=critiqued_trace.answer,
        )
        if critiqued_trace.answer:
            branch.set_candidate_answer(
                raw_answer=critiqued_trace.answer,
                canonical_answer=critiqued_trace.answer_canonical or canonicalize_competition_answer(critiqued_trace.answer),
                answer_source="self_critique",
                confidence=_clamp01(critiqued_trace.branch_score),
                supporting_step_ids=tuple(step.step_id for step in branch.active_steps()[-2:]),
            )
        branch.add_step(
            kind=BranchStepKind.CRITIQUE,
            phase=BranchPhase.CRITIQUED,
            description="self_critique_applied",
            summary_text=summary,
        )
        branch.set_phase(BranchPhase.CRITIQUED, reason="top_k_self_critique")
        branch.update_score_breakdown(
            verifier_probability=max(branch.score_breakdown.verifier_probability, _clamp01(critiqued_trace.verifier_score)),
            answer_agreement=max(branch.score_breakdown.answer_agreement, 0.5),
        )
        return branch

    def _aggregate(
        self,
        *,
        problem: ParsedProblem,
        route: RouteDecision,
        branches: Sequence[BranchState],
        aggregation_hook: AggregationHookProtocol | None,
    ) -> AggregationResult:
        if aggregation_hook is not None:
            return aggregation_hook(problem=problem, route=route, branches=branches)

        traces = [self._to_branch_trace(branch) for branch in branches]
        clustering_result = cluster_candidates(traces)
        weighting_bundle = score_answer_clusters(traces)
        selection = select_final_answer(
            problem.problem_id,
            weighting_bundle,
            num_branches_generated=len(traces),
            num_branches_survived=len(branches),
            solve_time_sec=0.0,
            method_used="self_consistency_priority_frontier",
        )
        return AggregationResult(
            final_prediction=selection.prediction,
            winning_branch_ids=tuple(selection.selected_cluster.candidate.branch_ids),
            candidate_clusters=tuple(cluster.candidate for cluster in selection.ranked_clusters),
            metadata={
                "cluster_count": len(clustering_result.clusters),
                "excluded_candidates": len(clustering_result.excluded),
                "excluded_reasons": {
                    reason: sum(1 for item in clustering_result.excluded if item.reason == reason)
                    for reason in sorted({item.reason for item in clustering_result.excluded})
                },
                "global_entropy": weighting_bundle.global_entropy,
                "selection_warnings": [warning.value for warning in selection.warnings],
                "abstain_recommended": selection.abstain_recommended,
            },
        )


    def _to_branch_trace(self, branch: BranchState):
        from src.common.schemas import BranchTrace, ReasoningStep

        candidate = branch.current_candidate()
        latest_verifier = branch.active_verifier_evidence()[-1] if branch.active_verifier_evidence() else None

        retrieval_metadata = {}
        if isinstance(getattr(branch, "metadata", None), dict):
            retrieval_metadata = dict(branch.metadata)

        active_retrieval = branch.active_retrieval_evidence()

        retrieval_compatibility = getattr(branch, "retrieval_compatibility", None)
        if retrieval_compatibility is None and getattr(branch, "score_breakdown", None) is not None:
            retrieval_compatibility = getattr(branch.score_breakdown, "retrieval_support", None)
        if retrieval_compatibility is None and isinstance(getattr(branch, "metadata", None), dict):
            retrieval_compatibility = (
                branch.metadata.get("retrieval_compatibility")
                or branch.metadata.get("retrieval_support")
                or (
                    branch.metadata.get("retrieval", {}).get("retrieval_compatibility")
                    if isinstance(branch.metadata.get("retrieval"), dict)
                    else None
                )
                or (
                    branch.metadata.get("retrieval", {}).get("compatibility_score")
                    if isinstance(branch.metadata.get("retrieval"), dict)
                    else None
                )
            )
        if retrieval_compatibility is None:
            if active_retrieval:
                retrieval_compatibility = max(
                    float(getattr(item, "compatibility_score", 0.0) or 0.0)
                    for item in active_retrieval
                )
            else:
                retrieval_compatibility = 0.0
        retrieval_compatibility = _clamp01(retrieval_compatibility)

        answer_correctness_likelihood = None
        if latest_verifier is not None:
            answer_correctness_likelihood = getattr(
                latest_verifier,
                "answer_correctness_likelihood",
                None,
            )
        if answer_correctness_likelihood is None and getattr(branch, "score_breakdown", None) is not None:
            answer_correctness_likelihood = getattr(
                branch.score_breakdown,
                "verifier_probability",
                None,
            )
        if answer_correctness_likelihood is None and isinstance(getattr(branch, "metadata", None), dict):
            answer_correctness_likelihood = (
                branch.metadata.get("answer_correctness_likelihood")
                or (
                    branch.metadata.get("verifier_decomposition", {}).get("answer_correctness_likelihood")
                    if isinstance(branch.metadata.get("verifier_decomposition"), dict)
                    else None
                )
                or (
                    branch.metadata.get("signal_decomposition", {}).get("answer_correctness_likelihood")
                    if isinstance(branch.metadata.get("signal_decomposition"), dict)
                    else None
                )
            )
        if answer_correctness_likelihood is None:
            answer_correctness_likelihood = 0.0
        answer_correctness_likelihood = _clamp01(answer_correctness_likelihood)

        source_split = None
        if isinstance(getattr(branch, "metadata", None), dict):
            source_split = (
                branch.metadata.get("source_split")
                or branch.metadata.get("calibration_safe_split")
                or (
                    branch.metadata.get("retrieval", {}).get("source_split")
                    if isinstance(branch.metadata.get("retrieval"), dict)
                    else None
                )
            )
        if source_split is None:
            source_split = getattr(branch.route, "calibration_safe_split", None)

        retrieval_compatibility_reasons = []
        if active_retrieval:
            for item in active_retrieval:
                reasons = getattr(item, "compatibility_reasons", None)
                if reasons:
                    retrieval_compatibility_reasons.extend([str(reason) for reason in reasons if str(reason).strip()])

        if not retrieval_compatibility_reasons and isinstance(getattr(branch, "metadata", None), dict):
            retrieval_compatibility_reasons = list(
                branch.metadata.get("retrieval_compatibility_reasons", [])
                or (
                    branch.metadata.get("retrieval", {}).get("compatibility_reasons", [])
                    if isinstance(branch.metadata.get("retrieval"), dict)
                    else []
                )
            )

        trace_metadata = dict(retrieval_metadata)
        trace_metadata["retrieval_compatibility_reasons"] = list(dict.fromkeys(retrieval_compatibility_reasons))

        failure_type = None
        if branch.phase is BranchPhase.FAILED:
            raw_failure_type = branch.metadata.get("failure_type")
            if isinstance(raw_failure_type, FailureType):
                failure_type = raw_failure_type
            else:
                try:
                    failure_type = FailureType(str(raw_failure_type))
                except Exception:
                    failure_type = FailureType.LOGIC_ERROR

        verifier_decomposition = (
            {
                "logical_consistency": branch.score_breakdown.logical_consistency,
                "symbolic_agreement": branch.score_breakdown.tool_consistency,
                "completeness": branch.score_breakdown.completeness,
                "answer_correctness_likelihood": answer_correctness_likelihood,
                "repairability": branch.score_breakdown.repairability,
                "step_quality": branch.score_breakdown.step_quality,
                "prefix_quality": branch.score_breakdown.prefix_quality,
                "open_obligation_burden": branch.score_breakdown.open_obligation_burden,
            }
            if latest_verifier is not None
            else {}
        )

        return BranchTrace(
            branch_id=branch.branch_id,
            problem_id=branch.problem_id,
            steps=[
                ReasoningStep(
                    step_num=index + 1,
                    description=step.description,
                    operator_used=step.operator_name,
                    symbolic_valid=branch.score_breakdown.exact_symbolic_check >= 0.5,
                )
                for index, step in enumerate(branch.active_steps())
            ],
            full_reasoning=" ".join(step.description for step in branch.active_steps()),
            answer=candidate.raw_answer if candidate else None,
            answer_canonical=candidate.canonical_answer if candidate else None,
            failure_type=failure_type,
            symbolic_valid=branch.score_breakdown.exact_symbolic_check >= 0.5,
            branch_score=branch.composite_score(),
            verifier_score=branch.score_breakdown.verifier_probability,
            answer_correctness_likelihood=answer_correctness_likelihood,
            tool_consistency=branch.score_breakdown.tool_consistency,
            logical_consistency=branch.score_breakdown.logical_consistency,
            completeness=branch.score_breakdown.completeness,
            repairability=branch.score_breakdown.repairability,
            step_quality=branch.score_breakdown.step_quality,
            prefix_quality=branch.score_breakdown.prefix_quality,
            prm_prefix_quality=branch.score_breakdown.prefix_quality,
            open_obligation_burden=branch.score_breakdown.open_obligation_burden,
            verifier_decomposition=verifier_decomposition,
            self_critiqued=branch.phase is BranchPhase.CRITIQUED,
            repaired=branch.active_cursor.repair_count > 0,
            repair_count=branch.active_cursor.repair_count,
            operator_sequence=[step.operator_name for step in branch.active_steps() if step.operator_name],
            archetype_used=_top_distribution_key(branch.route.archetypes),
            retrieval_used=branch.active_cursor.retrieval_count > 0,
            retrieval_compatibility=retrieval_compatibility,
            generation_time_sec=0.0,
            failure_step_index=(
                latest_verifier.failure_step_index
                if latest_verifier is not None and latest_verifier.failure_step_index is not None
                else branch.metadata.get("failure_step_index")
                if isinstance(branch.metadata.get("failure_step_index"), int)
                else None
            ),
            failure_step_id=(
                latest_verifier.failure_step_id
                if latest_verifier is not None and latest_verifier.failure_step_id
                else branch.metadata.get("failure_step_id")
                if isinstance(branch.metadata.get("failure_step_id"), str)
                else (branch.active_steps()[-1].step_id if branch.active_steps() else None)
            ),
            failure_node_id=(
                latest_verifier.failure_node_id
                if latest_verifier is not None and latest_verifier.failure_node_id
                else branch.latest_node_id
            ),
            failure_obligation_id=(
                latest_verifier.failure_obligation_id
                if latest_verifier is not None and latest_verifier.failure_obligation_id
                else branch.metadata.get("failure_obligation_id")
                if isinstance(branch.metadata.get("failure_obligation_id"), str)
                else next(
                    (
                        item.obligation_id
                        for item in branch.active_proof_obligations()
                        if item.status.value == "contradicted"
                    ),
                    None,
                )
            ),
            proof_state_fingerprint=branch.proof_state_fingerprint,
            proof_obligations=[
                {
                    "obligation_id": item.obligation_id,
                    "originating_node_id": item.originating_node_id,
                    "claim": item.claim,
                    "evidence_kind_required": item.evidence_kind_required.value,
                    "status": item.status.value,
                    "discharged_by": list(item.discharged_by),
                    "target_goal_id": item.target_goal_id,
                    "source_constraint_ids": list(item.source_constraint_ids),
                    "notes": list(item.notes),
                    "metadata": dict(item.metadata),
                }
                for item in branch.active_proof_obligations()
            ],
            source_split=source_split,
            provenance={},
            metadata=trace_metadata,
        )   


    def _pick_retrieval_for_operator(
        self,
        *,
        operator_name: str,
        retrieved_traces: Sequence[RetrievedTrace],
        offset: int,
    ) -> RetrievedTrace | None:
        if not retrieved_traces:
            return None
        ordered = sorted(
            retrieved_traces,
            key=lambda trace: (
                -(
                    0.52 * float(trace.similarity_score)
                    + 0.24 * float(getattr(trace, "compatibility_score", 0.0) or 0.0)
                    + 0.16 * float((getattr(trace, "operator_support", {}) or {}).get(operator_name, 0.0))
                    + 0.08 * (1.0 if operator_name in getattr(trace, "repair_operator_hints", []) else 0.0)
                ),
                trace.trace_id,
            ),
        )
        return ordered[offset % len(ordered)]

    def _compute_branch_novelty(self, branch: BranchState) -> float:
        candidate = branch.current_candidate()
        op_names = tuple(step.operator_name for step in branch.active_steps() if step.operator_name)
        unique_ops = len(set(op_names))
        answer_bonus = 0.15 if candidate and candidate.answer_source == "self_critique" else 0.0
        diversity = unique_ops / max(1, len(op_names))
        repair_bonus = 0.08 if branch.active_cursor.repair_count > 0 else 0.0
        return _clamp01(0.55 * diversity + answer_bonus + repair_bonus)

    def _estimate_answer_agreement(self, branch: BranchState) -> float:
        candidate = branch.current_candidate()
        if candidate is None:
            return 0.0
        supporting = max(1, len(candidate.supporting_step_ids))
        retrieval_bonus = 0.10 if branch.active_cursor.retrieval_count > 0 else 0.0
        return _clamp01(0.25 + 0.15 * min(supporting, 4) + retrieval_bonus)

    def _failure_penalty(self, *, branch: BranchState, failure: FailureDecision, depth: int) -> float:
        penalty = 0.02 * depth
        if branch.active_cursor.repair_count > 0:
            penalty += 0.03 * branch.active_cursor.repair_count
        if failure.failure_type is not None:
            penalty += 0.08
        return _clamp01(penalty)

    def _is_solved(self, *, branch: BranchState, symbolic: SymbolicHookResult, verifier: VerifierHookResult) -> bool:
        candidate = branch.current_candidate()
        return (
            candidate is not None
            and bool(candidate.canonical_answer)
            and symbolic.passed
            and verifier.probability >= max(0.45, branch.route.repair_threshold)
        )

    def _should_stop_early(
        self,
        *,
        route: RouteDecision,
        all_branches: Mapping[str, BranchState],
        solved_branch_ids: Sequence[str],
    ) -> bool:
        if len(solved_branch_ids) < 2:
            return False
        grouped: dict[str, int] = {}
        for branch_id in solved_branch_ids:
            branch = all_branches.get(branch_id)
            if branch is None or branch.current_candidate() is None:
                continue
            canonical = branch.current_candidate().canonical_answer
            grouped[canonical] = grouped.get(canonical, 0) + 1
        if not grouped:
            return False
        best = max(grouped.values())
        if self.config.consensus_stop_count is not None:
            target = max(2, int(self.config.consensus_stop_count))
        else:
            target = max(
                2,
                math.ceil(self._self_consistency_samples(route) * self.config.solved_cluster_stop_ratio),
            )
        return best >= target

    def _frontier_width(self, route: RouteDecision) -> int:
        if self.config.frontier_width is not None:
            return int(self.config.frontier_width)
        return max(4, min(route.branch_budget, route.budget_plan.max_search_nodes))

    def _max_depth(self, route: RouteDecision) -> int:
        if self.config.max_search_depth is not None:
            return int(self.config.max_search_depth)
        return route.budget_plan.max_search_depth

    def _max_search_nodes(self, route: RouteDecision) -> int:
        if self.config.max_search_nodes is not None:
            return int(self.config.max_search_nodes)
        return route.budget_plan.max_search_nodes

    def _repair_budget(self, route: RouteDecision) -> int:
        if self.config.repair_budget is not None:
            return int(self.config.repair_budget)
        return route.budget_plan.repair_budget

    def _resample_budget(self, route: RouteDecision) -> int:
        if self.config.resample_budget is not None:
            return int(self.config.resample_budget)
        return max(1, min(3, route.branch_budget // 16))

    def _self_consistency_samples(self, route: RouteDecision) -> int:
        if self.config.self_consistency_samples is not None:
            return int(self.config.self_consistency_samples)
        return route.budget_plan.self_consistency_samples

    def _critique_top_k(self, route: RouteDecision) -> int:
        if self.config.critique_top_k is not None:
            return int(self.config.critique_top_k)
        return route.budget_plan.critique_top_k


def _build_root_node(*, problem: ParsedProblem, route: RouteDecision) -> ReasoningStateNode:
    constraints = tuple(
        ConstraintRecord.create(kind="other", lhs=text, relation=None, rhs=None, origin="parser", strength="hard")
        for text in problem.constraint_texts()
    )
    invariants = tuple(
        InvariantRecord.create(invariant_type="seed", expression=text, proof_source="parser", confidence=0.7)
        for text in ((problem.parity_cues or []) + (problem.symmetries or []))
    )
    goals = [GoalRecord.create(kind="final_answer", text=problem.target or "determine_final_answer", priority=10)]
    if problem.answer_type:
        goals.append(GoalRecord.create(kind="verification", text=f"answer_type::{problem.answer_type}", priority=20))
    summary = f"problem={problem.problem_id} | domain={getattr(problem.domain, 'value', problem.domain)} | target={problem.target or 'final_answer'}"
    return ReasoningStateNode.create(
        problem_id=problem.problem_id,
        branch_id="root",
        constraints=constraints,
        invariants=invariants,
        goals=tuple(goals),
        extracted_objects={
            "knowns": list(problem.knowns),
            "unknowns": list(problem.unknowns),
            "domain": getattr(problem.domain, "value", str(problem.domain)),
            "route_problem_type": dict(route.problem_type),
            "route_archetypes": dict(route.archetypes),
        },
        partial_solution="",
        summary_text=summary,
        confidence=_clamp01(0.60 - 0.20 * route.difficulty_score),
        verifier_score=0.0,
        symbolic_valid=True,
        merge_ready=True,
        is_terminal=False,
        source_tags=("root", "branch_controller"),
        depth=0,
        lineage=(),
        metadata={"route_rationale": list(route.route_rationale)},
        node_id=f"root::{problem.problem_id}",
    )


def _top_distribution_key(distribution: Mapping[str, float]) -> str:
    if not distribution:
        return "unknown"
    return sorted(distribution.items(), key=lambda item: (-float(item[1]), item[0]))[0][0]


def _prior_confidence(route: RouteDecision, operator_name: str) -> float:
    return _clamp01(float((route.operator_prior or {}).get(operator_name, 0.0)) * 3.0)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _retrieval_operator_support(traces: Sequence[RetrievedTrace]) -> dict[str, float]:
    if not traces:
        return {}
    weights: dict[str, float] = {}
    total = 0.0
    for trace in traces:
        base = max(
            0.0,
            0.65 * float(trace.similarity_score) + 0.35 * float(getattr(trace, "compatibility_score", 0.0) or 0.0),
        )
        direct_support = dict(getattr(trace, "operator_support", {}) or {})
        if direct_support:
            for operator_name, score in direct_support.items():
                value = base * max(0.0, float(score))
                weights[operator_name] = weights.get(operator_name, 0.0) + value
                total += value
            for operator_name, score in dict(getattr(trace, "branch_continuation_bias", {}) or {}).items():
                value = 0.45 * base * max(0.0, float(score))
                weights[operator_name] = weights.get(operator_name, 0.0) + value
                total += value
            continue
        for operator_name in trace.operators_used:
            weights[operator_name] = weights.get(operator_name, 0.0) + base
            total += base
    if total <= 0.0:
        return {}
    return {name: score / total for name, score in sorted(weights.items())}


def _retrieval_support_score(branch: BranchState) -> float:
    active = branch.active_retrieval_evidence()
    if not active:
        return 0.0
    score = sum(0.55 * item.score + 0.45 * item.compatibility_score for item in active) / len(active)
    operator_bonus = 0.0
    obligation_bonus = 0.0
    current = branch.current_candidate()
    if current is not None:
        operator_bonus = 0.08 if any(current.canonical_answer == canonicalize_competition_answer(item.answer) for item in active) else 0.0
    if any(item.relevant_evidence_kinds for item in active):
        obligation_bonus = 0.06
    return _clamp01(score + operator_bonus + obligation_bonus)


def _retrieval_context(traces: Sequence[RetrievedTrace], *, branch: BranchState | None) -> dict[str, object]:
    context: dict[str, object] = {
        "archetypes": [],
        "evidence_kinds": [],
        "failure_modes": [],
        "repair_operators": [],
        "operator_prefix": [],
        "tags": [],
    }

    def _extend_unique(key: str, values: Sequence[object], limit: int) -> None:
        existing = [str(item) for item in list(context.get(key, []) or []) if str(item).strip()]
        for item in values:
            token = str(item).strip()
            if token and token not in existing:
                existing.append(token)
            if len(existing) >= limit:
                break
        context[key] = existing[:limit]

    for trace in traces:
        _extend_unique("archetypes", list(getattr(trace, "archetypes", []) or ()), 6)
        _extend_unique("evidence_kinds", list(getattr(trace, "relevant_evidence_kinds", []) or ()), 4)
        _extend_unique("failure_modes", list(getattr(trace, "failure_mode_support", []) or ()), 4)
        _extend_unique("repair_operators", list(getattr(trace, "repair_operator_hints", []) or ()), 4)
        _extend_unique("operator_prefix", list(getattr(trace, "operators_used", [])[:2] or ()), 4)
        _extend_unique("tags", list(getattr(trace, "compatibility_reasons", []) or ()), 6)

    if branch is not None:
        _extend_unique(
            "operator_prefix",
            [step.operator_name for step in branch.active_steps() if step.operator_name][-3:],
            4,
        )
    return context


def _branch_focus_operators(branch: BranchState | None) -> tuple[str, ...]:
    if branch is None:
        return ()
    profile = branch.metadata.get("resample_profile")
    if isinstance(profile, Mapping):
        focus = profile.get("focus_operators", ())
        if isinstance(focus, Sequence) and not isinstance(focus, (str, bytes)):
            return tuple(str(item).strip() for item in focus if str(item).strip())
    return ()


def build_controller(
    *,
    operator_library: OperatorLibrary | None = None,
    prior_shaper: OperatorPriorShaper | None = None,
    config: ControllerConfig | None = None,
) -> BranchController:
    return BranchController(
        operator_library=operator_library,
        prior_shaper=prior_shaper,
        config=config,
    )


__all__ = [
    "AggregationHookProtocol",
    "AggregationResult",
    "BranchController",
    "BranchControllerResult",
    "BranchGeneratorProtocol",
    "BranchSelectionRecord",
    "ControllerConfig",
    "FailureClassifierProtocol",
    "FailureDecision",
    "GenerationResult",
    "SymbolicHookProtocol",
    "SymbolicHookResult",
    "VerifierHookProtocol",
    "VerifierHookResult",
    "build_controller",
]
