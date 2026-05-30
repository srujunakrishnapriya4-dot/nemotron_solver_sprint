from __future__ import annotations

from src.branches.branch_controller import (
    BranchController,
    BranchControllerResult,
    ControllerConfig,
    GenerationResult,
    SymbolicHookResult,
    VerifierHookResult,
)
from src.branches.branch_state import BranchPhase, BranchState
from src.common.schemas import BudgetPlan, Difficulty, ParsedProblem, ProblemDomain, RetrievedTrace, RouteDecision
from src.parsing.parser import ProblemParser
from src.state_graph.proof_obligations import ProofEvidenceKind, ProofObligation, ProofObligationStatus
from src.state_graph.state_init import StateGraphInitializer
from src.symbolic import validators
from src.symbolic.sympy_runner import SymbolicRunRequest, SymbolicRunner, SymbolicTarget


def _problem() -> ParsedProblem:
    return ParsedProblem(
        problem_id="imo3-branch-problem",
        raw_text="Find the non-negative integer answer using modular reasoning and explicit verification.",
        knowns=["n is a non-negative integer"],
        unknowns=["n"],
        constraints=["n is non-negative", "n satisfies a modular condition"],
        domain=ProblemDomain.NUMBER_THEORY,
        target="final_answer",
        answer_type="non_negative_integer",
        difficulty_seed=0.62,
        likely_archetypes=["modular", "case_work"],
        parity_cues=["parity"],
        parse_quality={"confidence": 0.88},
    )


def _route(
    *,
    self_consistency_samples: int = 3,
    branch_budget: int = 3,
    repair_threshold: float = 0.55,
) -> RouteDecision:
    budget = BudgetPlan(
        branch_budget=branch_budget,
        retrieval_depth=2,
        max_search_depth=1,
        max_search_nodes=3,
        repair_budget=1,
        self_consistency_samples=self_consistency_samples,
        critique_top_k=1,
        widen_on_uncertainty=False,
        use_retrieval=True,
        use_symbolic=True,
        use_brute_force=False,
    )
    return RouteDecision(
        problem_id="imo3-branch-problem",
        problem_type_probs={"number_theory": 0.74, "algebra": 0.26},
        archetype_probs={"modular": 0.58, "case_work": 0.24, "invariant": 0.18},
        difficulty=Difficulty.HARD,
        difficulty_score=0.68,
        operator_prior={
            "modular_arithmetic": 0.34,
            "case_work": 0.24,
            "substitution": 0.22,
            "contradiction": 0.20,
        },
        budget_plan=budget,
        retrieval_depth=budget.retrieval_depth,
        branch_budget=budget.branch_budget,
        repair_threshold=repair_threshold,
        route_uncertainty=0.27,
        verifier_mode="strict",
        problem_type={"number_theory": 0.74, "algebra": 0.26},
        archetypes={"modular": 0.58, "case_work": 0.24, "invariant": 0.18},
        use_retrieval=True,
        use_symbolic=True,
        use_brute_force=False,
        retrieval_tags=["modular_patterns", "case_split_examples"],
        repair_neighbors=["contradiction", "invariant"],
        route_rationale=["top_domain=number_theory", "top_archetype=modular"],
    )


def _retrieved_trace() -> RetrievedTrace:
    return RetrievedTrace(
        trace_id="trace-1",
        problem="Example modular arithmetic problem",
        solution="Reduce modulo 3, prune cases, then verify the remaining candidate.",
        answer="42",
        domain="number_theory",
        archetypes=["modular", "case_work"],
        operators_used=["modular_arithmetic", "case_work", "substitution"],
        similarity_score=0.83,
        operator_support={"modular_arithmetic": 0.72, "case_work": 0.28},
        repair_operator_hints=["contradiction", "case_work"],
        branch_continuation_bias={"case_work": 0.61, "substitution": 0.39},
        compatibility_score=0.79,
        compatibility_reasons=["problem_structure_match", "proof_obligation_support"],
        relevant_obligation_claims=["justify the modular pruning step"],
        relevant_evidence_kinds=["symbolic_check"],
        failure_mode_support=["symbolic_mismatch"],
        strategy_summary="why=problem_structure_match,proof_obligation_support | ops=modular_arithmetic->case_work",
    )


def _controller(**overrides: object) -> BranchController:
    config = ControllerConfig(
        self_consistency_samples=int(overrides.pop("self_consistency_samples", 3)),
        frontier_width=int(overrides.pop("frontier_width", 2)),
        max_search_depth=int(overrides.pop("max_search_depth", 1)),
        max_search_nodes=int(overrides.pop("max_search_nodes", 1)),
        resample_budget=int(overrides.pop("resample_budget", 1)),
        repair_budget=int(overrides.pop("repair_budget", 1)),
        critique_top_k=int(overrides.pop("critique_top_k", 1)),
        retain_top_k=int(overrides.pop("retain_top_k", 2)),
        deterministic=True,
    )
    return BranchController(config=config)


def _success_generator(**kwargs: object) -> GenerationResult:
    operator_name = str(kwargs["operator_name"])
    sample_index = int(kwargs["sample_index"])
    return GenerationResult(
        reasoning=f"reasoning::{operator_name}::sample={sample_index}",
        answer="42",
        answer_canonical="42",
        confidence=0.72 + 0.03 * sample_index,
        summary=f"gen::{operator_name}",
        partial_solution="42",
        metadata={"sample_index": sample_index},
    )


def _symbolic_success(**kwargs: object) -> SymbolicHookResult:
    operator_name = str(kwargs["operator_name"])
    return SymbolicHookResult(
        passed=True,
        score=0.87,
        summary=f"symbolic_ok::{operator_name}",
        exact_match=True,
    )


def _verifier_success(**kwargs: object) -> VerifierHookResult:
    return VerifierHookResult(
        probability=0.91,
        logical_consistency=0.84,
        completeness=0.82,
        repairability=0.18,
        summary="verifier_accept",
        symbolic_agreement=0.86,
        answer_correctness_likelihood=0.91,
        branch_score=0.89,
        step_quality=0.79,
        prefix_quality=0.81,
        open_obligation_burden=0.08,
        metadata={"source": "unit_test"},
    )


def _repair_generator(**kwargs: object) -> GenerationResult:
    operator_name = str(kwargs["operator_name"])
    return GenerationResult(
        reasoning=f"repairable::{operator_name}",
        answer="17",
        answer_canonical="17",
        confidence=0.54,
        summary="repairable_generation",
        partial_solution="17",
    )


def _symbolic_failure(**kwargs: object) -> SymbolicHookResult:
    return SymbolicHookResult(
        passed=False,
        score=0.14,
        summary="symbolic contradiction on latest step",
        exact_match=False,
    )


def _verifier_repairable(**kwargs: object) -> VerifierHookResult:
    return VerifierHookResult(
        probability=0.19,
        logical_consistency=0.28,
        completeness=0.51,
        repairability=0.81,
        summary="local symbolic mismatch",
        symbolic_agreement=0.12,
        answer_correctness_likelihood=0.19,
        branch_score=0.18,
    )


def _resample_generator(**kwargs: object) -> GenerationResult:
    operator_name = str(kwargs["operator_name"])
    return GenerationResult(
        reasoning=f"resampleable::{operator_name}",
        answer="12",
        answer_canonical="12",
        confidence=0.61,
        summary="resample_generation",
        partial_solution="12",
    )


def _symbolic_neutral(**kwargs: object) -> SymbolicHookResult:
    return SymbolicHookResult(
        passed=True,
        score=0.58,
        summary="symbolic_partial_support",
        exact_match=False,
    )


def _verifier_rejects_for_resample(**kwargs: object) -> VerifierHookResult:
    return VerifierHookResult(
        probability=0.11,
        logical_consistency=0.33,
        completeness=0.69,
        repairability=0.42,
        summary="model rejected branch",
        symbolic_agreement=0.56,
        answer_correctness_likelihood=0.11,
        branch_score=0.14,
    )


def _value_guided_verifier(**kwargs: object) -> VerifierHookResult:
    generation = kwargs["generation"]
    sample_index = int(getattr(generation, "metadata", {}).get("sample_index", 0))
    if sample_index == 1:
        return VerifierHookResult(
            probability=0.55,
            logical_consistency=0.76,
            completeness=0.73,
            repairability=0.22,
            summary="value_guided_high_prefix",
            symbolic_agreement=0.78,
            answer_correctness_likelihood=0.55,
            branch_score=0.57,
            step_quality=0.84,
            prefix_quality=0.88,
            open_obligation_burden=0.05,
            metadata={"source": "unit_test"},
        )
    return VerifierHookResult(
        probability=0.43,
        logical_consistency=0.52,
        completeness=0.64,
        repairability=0.36,
        summary="value_guided_low_prefix",
        symbolic_agreement=0.54,
        answer_correctness_likelihood=0.43,
        branch_score=0.41,
        step_quality=0.34,
        prefix_quality=0.29,
        open_obligation_burden=0.58,
        metadata={"source": "unit_test"},
    )


def _result_signature(result: BranchControllerResult) -> dict[str, object]:
    return {
        "final": result.final_prediction.model_dump(mode="json"),
        "stopped_reason": result.stopped_reason,
        "metadata": dict(result.metadata),
        "search_history": [
            {
                "branch_id": item.branch_id,
                "depth": item.depth,
                "operator_name": item.operator_name,
                "score": round(item.score, 6),
            }
            for item in result.search_history
        ],
        "survivors": [
            {
                "branch_id": branch.branch_id,
                "phase": branch.phase.value,
                "candidate": branch.current_candidate().canonical_answer if branch.current_candidate() else None,
                "repair_count": branch.active_cursor.repair_count,
                "critique_count": branch.active_cursor.critique_count,
                "score": round(branch.composite_score(), 6),
            }
            for branch in result.surviving_branches
        ],
    }


def test_branch_controller_self_consistency_first_is_bounded_and_deterministic() -> None:
    controller = _controller(self_consistency_samples=3, max_search_nodes=1, frontier_width=2, retain_top_k=2)
    problem = _problem()
    route = _route(self_consistency_samples=3, branch_budget=3, repair_threshold=0.55)

    first = controller.solve(
        problem=problem,
        route=route,
        retrieved_traces=[_retrieved_trace()],
        generator=_success_generator,
        symbolic_hook=_symbolic_success,
        verifier_hook=_verifier_success,
    )
    second = controller.solve(
        problem=problem,
        route=route,
        retrieved_traces=[_retrieved_trace()],
        generator=_success_generator,
        symbolic_hook=_symbolic_success,
        verifier_hook=_verifier_success,
    )

    assert first.metadata["initial_branches"] == 3
    assert first.metadata["expanded_nodes"] <= 1
    assert len(first.search_history) <= 1
    assert len(first.surviving_branches) <= 2
    assert _result_signature(first) == _result_signature(second)


def test_branch_controller_honors_explicit_consensus_stop_count() -> None:
    controller = _controller(
        self_consistency_samples=8,
        max_search_nodes=4,
        frontier_width=4,
        retain_top_k=4,
    )
    controller.config = ControllerConfig(
        self_consistency_samples=8,
        frontier_width=4,
        max_search_depth=1,
        max_search_nodes=4,
        resample_budget=1,
        repair_budget=1,
        critique_top_k=0,
        retain_top_k=4,
        consensus_stop_count=4,
        enable_mid_search_critique=False,
        deterministic=True,
    )

    result = controller.solve(
        problem=_problem(),
        route=_route(self_consistency_samples=8, branch_budget=8, repair_threshold=0.55),
        retrieved_traces=[_retrieved_trace()],
        generator=_success_generator,
        symbolic_hook=_symbolic_success,
        verifier_hook=_verifier_success,
    )

    assert result.stopped_reason == "consensus_converged"
    assert result.metadata["initial_branches"] == 4
    assert result.metadata["expanded_nodes"] <= 1
    assert not result.selected_for_critique


def test_branch_controller_routes_symbolic_failure_into_local_repair() -> None:
    controller = _controller(self_consistency_samples=1, max_search_nodes=1, repair_budget=1, resample_budget=1)
    result = controller.solve(
        problem=_problem(),
        route=_route(self_consistency_samples=1, branch_budget=2, repair_threshold=0.55),
        retrieved_traces=[_retrieved_trace()],
        generator=_repair_generator,
        symbolic_hook=_symbolic_failure,
        verifier_hook=_verifier_repairable,
    )

    repaired = [branch for branch in result.all_branches if branch.active_cursor.repair_count > 0]

    assert repaired
    assert all(branch.active_cursor.repair_count == 1 for branch in repaired)
    assert any("selected_candidate_id" in branch.metadata for branch in repaired)
    assert not any(branch.metadata.get("resample_count") for branch in repaired)


def test_branch_controller_routes_low_verifier_rejection_into_resample() -> None:
    controller = _controller(self_consistency_samples=1, max_search_nodes=1, repair_budget=1, resample_budget=1)
    result = controller.solve(
        problem=_problem(),
        route=_route(self_consistency_samples=1, branch_budget=2, repair_threshold=0.8),
        generator=_resample_generator,
        symbolic_hook=_symbolic_neutral,
        verifier_hook=_verifier_rejects_for_resample,
    )

    resampled = [branch for branch in result.all_branches if int(branch.metadata.get("resample_count", 0)) > 0]

    assert resampled
    assert all(branch.active_cursor.repair_count == 0 for branch in resampled)
    assert all(branch.metadata["resample_count"] == 1 for branch in resampled)
    assert all("resample_profile" in branch.metadata for branch in resampled)


def test_branch_controller_fallback_symbolic_does_not_treat_reasoning_only_as_pass() -> None:
    controller = _controller(self_consistency_samples=1, max_search_nodes=1, repair_budget=0, resample_budget=0)

    def _reasoning_only_generator(**kwargs: object) -> GenerationResult:
        operator_name = str(kwargs["operator_name"])
        return GenerationResult(
            reasoning=f"plausible::{operator_name}",
            answer=None,
            answer_canonical=None,
            confidence=0.72,
            summary="reasoning_only_generation",
            partial_solution="candidate family sketched",
        )

    result = controller.solve(
        problem=_problem(),
        route=_route(self_consistency_samples=1, branch_budget=1, repair_threshold=0.7),
        generator=_reasoning_only_generator,
        symbolic_hook=None,
        verifier_hook=None,
    )

    branch = result.all_branches[0]
    symbolic = branch.active_symbolic_evidence()[-1]
    verifier = branch.active_verifier_evidence()[-1]

    assert symbolic.passed is False
    assert "unsupported" in symbolic.summary
    assert branch.score_breakdown.exact_symbolic_check <= 0.18
    assert verifier.probability <= 0.38
    assert verifier.summary.endswith("unsupported")


def test_branch_controller_uses_rich_score_breakdown_and_critiques_top_survivors() -> None:
    controller = _controller(self_consistency_samples=3, max_search_nodes=1, critique_top_k=1, retain_top_k=2)
    result = controller.solve(
        problem=_problem(),
        route=_route(self_consistency_samples=3, branch_budget=3, repair_threshold=0.55),
        retrieved_traces=[_retrieved_trace()],
        generator=_success_generator,
        symbolic_hook=_symbolic_success,
        verifier_hook=_verifier_success,
    )

    assert result.selected_for_critique
    assert tuple(result.critiqued_branch_ids) == result.selected_for_critique

    critiqued = [branch for branch in result.surviving_branches if branch.branch_id in result.selected_for_critique]
    assert critiqued
    assert all(branch.active_cursor.critique_count == 1 for branch in critiqued)
    assert all(branch.phase is BranchPhase.CRITIQUED for branch in critiqued)

    assert any(branch.score_breakdown.retrieval_support > 0.0 for branch in result.surviving_branches)
    assert any(branch.score_breakdown.tool_consistency > 0.0 for branch in result.surviving_branches)
    assert any(branch.score_breakdown.answer_agreement > 0.0 for branch in result.surviving_branches)
    assert any(branch.score_breakdown.branch_novelty > 0.0 for branch in result.surviving_branches)
    assert any(branch.score_breakdown.exact_symbolic_check > 0.0 for branch in result.surviving_branches)
    assert any(branch.score_breakdown.prefix_quality > 0.0 for branch in result.surviving_branches)
    assert any(branch.score_breakdown.step_quality > 0.0 for branch in result.surviving_branches)
    assert any(branch.composite_score() != branch.score_breakdown.verifier_probability for branch in result.surviving_branches)
    assert any(branch.active_retrieval_evidence()[0].compatibility_score > 0.0 for branch in result.surviving_branches if branch.active_retrieval_evidence())
    assert any(branch.active_retrieval_evidence()[0].relevant_evidence_kinds for branch in result.surviving_branches if branch.active_retrieval_evidence())


def test_state_init_preserves_typed_constraints_and_proof_obligations_from_parser() -> None:
    parser = ProblemParser()
    problem = parser.parse_sync(
        "Let x be a positive integer such that x+1=4. Find x."
    )
    route = _route().model_copy(update={"problem_id": problem.problem_id})

    init = StateGraphInitializer().initialize(problem, route)
    branch = BranchState.create(route=route, parsed_problem=problem, root_node=init.root_node)

    assert init.root_node.constraints
    assert any(constraint.metadata.get("typed_constraint") for constraint in init.root_node.constraints)
    assert init.root_node.proof_obligations
    assert any(item.source_constraint_ids for item in init.root_node.proof_obligations)
    assert init.metadata.proof_obligation_count == len(init.root_node.proof_obligations)
    assert branch.proof_obligations
    assert branch.proof_state_fingerprint


def test_symbolic_runner_propagates_proof_obligation_discharge_and_contradiction() -> None:
    runner = SymbolicRunner()
    obligation = ProofObligation.create(
        originating_node_id="root::imo3-proof",
        claim="x + 1 = 4",
        evidence_kind_required=ProofEvidenceKind.SYMBOLIC_CHECK,
    )

    success_result = validators.CompositeValidationResult(
        status=validators.CompositeValidationStatus.SUCCESS,
        target=validators.ValidationTarget.BRANCH_STEP,
        summary="symbolic success",
        score=0.92,
        exact=True,
        applied_domains=(validators.ValidatorDomain.ALGEBRA,),
        proof_obligations=(obligation.discharge(discharged_by="symbolic_validation"),),
        discharged_obligation_ids=(obligation.obligation_id,),
    )
    contradiction_result = validators.CompositeValidationResult(
        status=validators.CompositeValidationStatus.CONTRADICTION,
        target=validators.ValidationTarget.BRANCH_STEP,
        summary="symbolic contradiction",
        score=0.0,
        exact=False,
        applied_domains=(validators.ValidatorDomain.ALGEBRA,),
        contradiction_found=True,
        proof_obligations=(
            obligation.contradict(
                source="symbolic_validation",
                summary="symbolic contradiction",
                failure_type="symbolic_mismatch",
            ),
        ),
        contradicted_obligation_ids=(obligation.obligation_id,),
    )

    runner._run_with_timeout = lambda fn, timeout_ms: success_result  # type: ignore[method-assign]
    discharged = runner.validate(
        SymbolicRunRequest(
            target=SymbolicTarget.BRANCH_STEP,
            statement="x + 1 = 4",
            constraints=("x + 1 = 4",),
            proof_obligations=(obligation,),
            problem_id="imo3-proof",
            branch_id="branch-proof",
            node_id="root::imo3-proof",
        )
    )

    runner._run_with_timeout = lambda fn, timeout_ms: contradiction_result  # type: ignore[method-assign]
    contradicted = runner.validate(
        SymbolicRunRequest(
            target=SymbolicTarget.BRANCH_STEP,
            statement="x = x + 1",
            constraints=("x = x + 1",),
            proof_obligations=(obligation,),
            problem_id="imo3-proof",
            branch_id="branch-proof",
            node_id="root::imo3-proof",
        )
    )

    assert discharged.discharged_obligation_ids == (obligation.obligation_id,)
    assert discharged.proof_obligations[0].status is ProofObligationStatus.DISCHARGED
    assert contradicted.contradicted_obligation_ids == (obligation.obligation_id,)
    assert contradicted.proof_obligations[0].status is ProofObligationStatus.CONTRADICTED


def test_branch_controller_ranks_frontier_with_prefix_quality_and_obligation_burden() -> None:
    controller = _controller(self_consistency_samples=2, max_search_nodes=1, frontier_width=2, retain_top_k=2)
    result = controller.solve(
        problem=_problem(),
        route=_route(self_consistency_samples=2, branch_budget=2, repair_threshold=0.55),
        retrieved_traces=[_retrieved_trace()],
        generator=_success_generator,
        symbolic_hook=_symbolic_success,
        verifier_hook=_value_guided_verifier,
    )

    assert result.search_history
    assert result.search_history[0].branch_id.endswith("001")
    ranked = {branch.branch_id: branch for branch in result.all_branches}
    assert ranked["imo3-branch-problem::seed::001"].score_breakdown.prefix_quality > ranked["imo3-branch-problem::seed::000"].score_breakdown.prefix_quality
    assert ranked["imo3-branch-problem::seed::001"].score_breakdown.open_obligation_burden < ranked["imo3-branch-problem::seed::000"].score_breakdown.open_obligation_burden


def test_bounded_search_preserves_typed_proof_state_during_value_guided_ranking() -> None:
    parser = ProblemParser()
    problem = parser.parse_sync("Let x be a positive integer such that x+1=4. Find x.")
    route = _route(self_consistency_samples=2, branch_budget=2, repair_threshold=0.55).model_copy(update={"problem_id": problem.problem_id})
    root = StateGraphInitializer().initialize(problem, route).root_node
    controller = _controller(self_consistency_samples=2, max_search_nodes=1, frontier_width=2, retain_top_k=2)

    result = controller.solve(
        problem=problem,
        route=route,
        root_node=root,
        generator=_success_generator,
        symbolic_hook=_symbolic_success,
        verifier_hook=_value_guided_verifier,
    )

    assert result.metadata["expanded_nodes"] <= 1
    assert all(branch.proof_state_fingerprint for branch in result.all_branches)
    assert any(branch.score_breakdown.open_obligation_burden >= 0.0 for branch in result.all_branches)



def test_branch_trace_exports_richer_calibration_signals() -> None:
    controller = _controller(self_consistency_samples=2, max_search_nodes=1, frontier_width=2, retain_top_k=2)
    result = controller.solve(
        problem=_problem(),
        route=_route(self_consistency_samples=2, branch_budget=2, repair_threshold=0.55).model_copy(update={"calibration_safe_split": "valid"}),
        retrieved_traces=[_retrieved_trace()],
        generator=_success_generator,
        symbolic_hook=_symbolic_success,
        verifier_hook=_verifier_success,
    )

    trace = controller._to_branch_trace(result.surviving_branches[0])
    assert trace.prm_prefix_quality >= 0.0
    assert trace.retrieval_compatibility > 0.0
    assert trace.answer_correctness_likelihood > 0.0
    assert trace.source_split == "valid"
    assert "retrieval_compatibility_reasons" in trace.metadata
    assert result.final_prediction.signal_decomposition["retrieval_compatibility"] >= 0.0
