from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.branches.branch_controller import BranchController, ControllerConfig, GenerationResult
from src.common.schemas import AttemptBatchResult, AttemptEntropySource, AttemptRecord, BudgetPlan, Difficulty, ParsedProblem, ProblemDomain, RouteDecision
from src.online.inference_engine import InferenceEngine, InferenceEngineConfig, RuntimePath
from src.online.kaggle_runner import KaggleRunner, load_kaggle_runner_config, load_minimal_prompt_profile


def _problem() -> ParsedProblem:
    return ParsedProblem(
        problem_id="selective-runtime-problem",
        raw_text="Find the non-negative integer answer.",
        knowns=["n is a non-negative integer"],
        unknowns=["n"],
        constraints=["n is a non-negative integer"],
        domain=ProblemDomain.NUMBER_THEORY,
        target="final_answer",
        answer_type="non_negative_integer",
        difficulty_seed=0.35,
        likely_archetypes=["modular"],
        parse_quality={"confidence": 0.88},
    )


def _route(
    *,
    difficulty: Difficulty,
    difficulty_score: float,
    route_uncertainty: float,
    proof_burden: float,
    retrieval_need: float,
    repair_need: float,
    self_consistency_samples: int = 4,
    branch_budget: int = 4,
    problem_type_probs: dict[str, float] | None = None,
) -> RouteDecision:
    budget = BudgetPlan(
        branch_budget=branch_budget,
        retrieval_depth=2,
        max_search_depth=1,
        max_search_nodes=4,
        repair_budget=1,
        self_consistency_samples=self_consistency_samples,
        critique_top_k=1,
        widen_on_uncertainty=False,
        use_retrieval=True,
        use_symbolic=True,
        use_brute_force=False,
    )
    probs = problem_type_probs or {"number_theory": 0.78, "algebra": 0.22}
    return RouteDecision(
        problem_id="selective-runtime-problem",
        problem_type_probs=probs,
        archetype_probs={"modular": 0.66, "case_work": 0.20, "invariant": 0.14},
        difficulty=difficulty,
        difficulty_score=difficulty_score,
        operator_prior={"modular_arithmetic": 0.58, "case_work": 0.42},
        budget_plan=budget,
        retrieval_depth=budget.retrieval_depth,
        branch_budget=budget.branch_budget,
        repair_threshold=0.55,
        route_uncertainty=route_uncertainty,
        verifier_mode="default",
        problem_type=probs,
        archetypes={"modular": 0.66, "case_work": 0.20, "invariant": 0.14},
        use_retrieval=True,
        use_symbolic=True,
        use_brute_force=False,
        retrieval_tags=["modular_patterns"],
        repair_neighbors=["case_work"],
        route_rationale=["top_domain=number_theory", "top_archetype=modular"],
        compute_signals={
            "proof_burden": proof_burden,
            "retrieval_need": retrieval_need,
            "repair_need": repair_need,
        },
    )


def _sampled_generator(*, sample_index: int, **_: object) -> GenerationResult:
    answer_map = {0: "42", 1: "42", 2: "42", 3: "17", 4: "19"}
    answer = answer_map.get(sample_index, "42")
    confidence = 0.84 if answer == "42" else 0.46
    return GenerationResult(
        reasoning=f"reasoning::{sample_index}",
        answer=answer,
        answer_canonical=answer,
        confidence=confidence,
        summary=f"sample::{sample_index}",
        partial_solution=answer,
        metadata={"sample_index": sample_index},
    )


class _FakeMinimalGenerator:
    def __init__(self, attempts: list[AttemptRecord]) -> None:
        self.attempts = attempts
        self.minimal_calls = 0

    def run_minimal_attempts(self, **_: object) -> AttemptBatchResult:
        self.minimal_calls += 1
        stopped_reason = "early_stop_consensus" if sum(1 for item in self.attempts if item.valid_answer and item.extracted_answer == "42") >= 4 else "attempt_budget_exhausted"
        return AttemptBatchResult(
            attempts=self.attempts,
            stopped_reason=stopped_reason,
            metadata={"prompt_profile": "minimal_default"},
        )


def test_easy_low_risk_route_uses_minimal_runtime_path() -> None:
    engine = InferenceEngine()
    route = _route(
        difficulty=Difficulty.EASY,
        difficulty_score=0.34,
        route_uncertainty=0.18,
        proof_burden=0.12,
        retrieval_need=0.10,
        repair_need=0.08,
    )

    runtime_path = engine._select_runtime_path(route=route, budget=None)
    controller_config = engine._controller_config_from_budget(
        None,
        route.budget_plan,
        route=route,
        runtime_path=runtime_path,
    )

    assert runtime_path is RuntimePath.MINIMAL
    assert controller_config.self_consistency_samples == engine.config.minimal_mode_self_consistency_samples
    assert controller_config.max_search_depth == engine.config.minimal_mode_max_search_depth
    assert controller_config.max_search_nodes == engine.config.minimal_mode_max_search_nodes
    assert controller_config.critique_top_k == 0
    assert controller_config.repair_budget == 0


def test_minimal_prompt_profile_loads_from_config() -> None:
    profile = load_minimal_prompt_profile("configs/prompts.yaml", profile_name="minimal_default")

    assert "Solve the following math problem step by step." in profile.system_prompt
    assert "Execute Python code in a stateful Jupyter notebook." in profile.tool_prompt
    assert "<python_tool>...</python_tool>" in profile.tool_prompt
    assert "sympy" in profile.preference_prompt


def test_kaggle_runner_loads_minimal_runtime_defaults_from_online_config() -> None:
    config = load_kaggle_runner_config("configs/online.yaml")
    runner = KaggleRunner(config=config)

    assert runner.config.attempts_per_problem == 8
    assert runner.config.early_stop_threshold == 4
    assert runner.config.prompt_config_path == "configs/prompts.yaml"
    assert runner.config.minimal_prompt_profile == "minimal_default"
    assert runner.config.escalation_min_confidence == 0.66
    assert runner.config.minimal_mode_fragmentation_threshold == 0.28


def test_non_severe_hard_route_uses_escalated_not_hard() -> None:
    engine = InferenceEngine()
    route = _route(
        difficulty=Difficulty.HARD,
        difficulty_score=0.69,
        route_uncertainty=0.36,
        proof_burden=0.34,
        retrieval_need=0.28,
        repair_need=0.24,
    )

    assert engine._select_runtime_path(route=route, budget=None) is RuntimePath.ESCALATED


def test_severe_route_uses_hard_runtime_path() -> None:
    engine = InferenceEngine()
    route = _route(
        difficulty=Difficulty.HARD,
        difficulty_score=0.83,
        route_uncertainty=0.74,
        proof_burden=0.66,
        retrieval_need=0.54,
        repair_need=0.52,
    )

    assert engine._select_runtime_path(route=route, budget=None) is RuntimePath.HARD


def test_minimal_policy_disables_heavy_hooks_and_installs_lightweight_aggregation() -> None:
    engine = InferenceEngine()
    route = _route(
        difficulty=Difficulty.EASY,
        difficulty_score=0.34,
        route_uncertainty=0.18,
        proof_burden=0.12,
        retrieval_need=0.10,
        repair_need=0.08,
    )
    policy = engine._resolve_branch_runtime_policy(
        route=route,
        runtime_path=RuntimePath.MINIMAL,
        controller_config=ControllerConfig(),
        symbolic_hook=object(),
        verifier_hook=object(),
        failure_classifier=None,
        aggregation_hook=None,
        critique_model=object(),
    )

    failure_decision = policy["failure_classifier"]()

    assert policy["symbolic_hook"] is None
    assert policy["verifier_hook"] is None
    assert policy["critique_model"] is None
    assert policy["aggregation_hook"] is not None
    assert failure_decision.route_to_repair is False
    assert failure_decision.route_to_resample is False


def test_escalated_policy_keeps_some_heavy_components_off_when_route_signals_are_weak() -> None:
    engine = InferenceEngine()
    route = _route(
        difficulty=Difficulty.HARD,
        difficulty_score=0.62,
        route_uncertainty=0.22,
        proof_burden=0.18,
        retrieval_need=0.12,
        repair_need=0.16,
    )

    policy = engine._resolve_branch_runtime_policy(
        route=route,
        runtime_path=RuntimePath.ESCALATED,
        controller_config=ControllerConfig(repair_budget=2, critique_top_k=2, max_search_depth=4, max_search_nodes=32),
        symbolic_hook=object(),
        verifier_hook=object(),
        failure_classifier=None,
        aggregation_hook=None,
        critique_model=object(),
    )

    assert policy["symbolic_hook"] is None
    assert policy["verifier_hook"] is None
    assert policy["critique_model"] is None
    assert policy["selective_policy"]["retrieval_active"] is False


def test_minimal_aggregation_prefers_consensus_answer_and_flags_fragmentation() -> None:
    engine = InferenceEngine()
    controller = BranchController(
        config=ControllerConfig(
            self_consistency_samples=5,
            frontier_width=5,
            max_search_depth=1,
            max_search_nodes=0,
            resample_budget=0,
            repair_budget=0,
            critique_top_k=0,
            consensus_stop_count=6,
            retain_top_k=5,
            enable_mid_search_critique=False,
            deterministic=True,
        )
    )
    problem = _problem()
    route = _route(
        difficulty=Difficulty.EASY,
        difficulty_score=0.34,
        route_uncertainty=0.18,
        proof_burden=0.12,
        retrieval_need=0.10,
        repair_need=0.08,
    ).model_copy(
        update={
            "budget_plan": BudgetPlan(
                branch_budget=5,
                retrieval_depth=2,
                max_search_depth=1,
                max_search_nodes=0,
                repair_budget=0,
                self_consistency_samples=5,
                critique_top_k=0,
                use_retrieval=True,
                use_symbolic=True,
                use_brute_force=False,
            ),
            "branch_budget": 5,
        }
    )

    result = controller.solve(
        problem=problem,
        route=route,
        generator=_sampled_generator,
        symbolic_hook=None,
        verifier_hook=None,
        aggregation_hook=engine._minimal_aggregation_hook,
        critique_model=None,
    )

    preview = engine._preview_branch_selection(
        problem_id=problem.problem_id,
        branch_result=SimpleNamespace(
            candidate_clusters=result.candidate_clusters,
            final_prediction=result.final_prediction,
        ),
    )

    assert result.final_prediction.final_answer == 42
    assert result.final_prediction.method_used == "minimal_weighted_self_consistency"
    assert preview.submission_answer == 42
    assert preview.prediction.method_used == "minimal_weighted_self_consistency"


def test_low_risk_problem_skips_heavy_architecture_in_solve_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = InferenceEngine()
    generator = _FakeMinimalGenerator(
        [
            AttemptRecord(attempt_id="a0", seed=0, raw_text="\\boxed{42}", extracted_answer="42", valid_answer=True, mean_token_entropy=0.30, entropy_source=AttemptEntropySource.TRUE_LOGPROBS, runtime_sec=0.1),
            AttemptRecord(attempt_id="a1", seed=1, raw_text="\\boxed{42}", extracted_answer="42", valid_answer=True, mean_token_entropy=0.28, entropy_source=AttemptEntropySource.TRUE_LOGPROBS, runtime_sec=0.1),
            AttemptRecord(attempt_id="a2", seed=2, raw_text="\\boxed{42}", extracted_answer="42", valid_answer=True, mean_token_entropy=0.32, entropy_source=AttemptEntropySource.TRUE_LOGPROBS, runtime_sec=0.1),
            AttemptRecord(attempt_id="a3", seed=3, raw_text="\\boxed{42}", extracted_answer="42", valid_answer=True, mean_token_entropy=0.29, entropy_source=AttemptEntropySource.TRUE_LOGPROBS, runtime_sec=0.1),
        ]
    )
    problem = _problem()
    route = _route(
        difficulty=Difficulty.EASY,
        difficulty_score=0.34,
        route_uncertainty=0.18,
        proof_burden=0.12,
        retrieval_need=0.10,
        repair_need=0.08,
    )

    monkeypatch.setattr(engine, "_run_parse_stage", lambda **kwargs: (problem, None))
    monkeypatch.setattr(engine, "_run_route_stage", lambda **kwargs: (route, None))
    monkeypatch.setattr(engine, "_run_budget_stage", lambda **kwargs: (None, None))
    monkeypatch.setattr(engine, "_run_state_init_stage", lambda **kwargs: (_ for _ in ()).throw(AssertionError("state init should not run")))
    monkeypatch.setattr(engine, "_run_retrieval_stage", lambda **kwargs: (_ for _ in ()).throw(AssertionError("retrieval should not run")))
    monkeypatch.setattr(engine, "_run_branch_stage", lambda **kwargs: (_ for _ in ()).throw(AssertionError("branch stage should not run")))
    monkeypatch.setattr(engine, "_run_submission_stage", lambda **kwargs: ({"id": problem.problem_id, "answer": 42}, None))

    result = engine.solve_problem("Find the answer.", problem_id=problem.problem_id, generator=generator)

    assert result.ok
    assert result.final_prediction is not None
    assert result.final_prediction.final_answer == 42
    assert generator.minimal_calls == 1


def test_high_disagreement_minimal_attempts_trigger_escalation() -> None:
    engine = InferenceEngine()
    attempts = (
        AttemptRecord(attempt_id="a0", seed=0, raw_text="\\boxed{42}", extracted_answer="42", valid_answer=True, mean_token_entropy=1.25, entropy_source=AttemptEntropySource.TRUE_LOGPROBS, runtime_sec=0.1),
        AttemptRecord(attempt_id="a1", seed=1, raw_text="\\boxed{17}", extracted_answer="17", valid_answer=True, mean_token_entropy=1.18, entropy_source=AttemptEntropySource.TRUE_LOGPROBS, runtime_sec=0.1),
        AttemptRecord(attempt_id="a2", seed=2, raw_text="\\boxed{19}", extracted_answer="19", valid_answer=True, mean_token_entropy=1.22, entropy_source=AttemptEntropySource.TRUE_LOGPROBS, runtime_sec=0.1),
    )
    selection = engine._select_from_attempt_records(
        problem_id="selective-runtime-problem",
        attempts=attempts,
        stopped_reason="attempt_budget_exhausted",
    )
    route = _route(
        difficulty=Difficulty.EASY,
        difficulty_score=0.34,
        route_uncertainty=0.18,
        proof_burden=0.12,
        retrieval_need=0.10,
        repair_need=0.08,
    )

    assert engine._should_escalate_after_attempts(
        route=route,
        selection=selection,
        attempts=attempts,
        early_stop_triggered=False,
    )
    reasons = engine._attempt_escalation_reasons(
        route=route,
        selection=selection,
        attempts=attempts,
        early_stop_triggered=False,
    )
    assert "high_true_entropy" in reasons


def test_low_risk_problem_without_minimal_generator_escalates_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = InferenceEngine(config=InferenceEngineConfig(enable_debug_artifacts=False))
    problem = _problem()
    route = _route(
        difficulty=Difficulty.EASY,
        difficulty_score=0.34,
        route_uncertainty=0.18,
        proof_burden=0.12,
        retrieval_need=0.10,
        repair_need=0.08,
    )

    branch_result = SimpleNamespace(
        final_prediction=SimpleNamespace(
            final_answer=42,
            confidence=0.81,
            winning_cluster=SimpleNamespace(answer_agreement=0.81),
            num_branches_generated=4,
            num_branches_survived=4,
            solve_time_sec=0.1,
            method_used="weighted_cluster_final_selector",
            signal_decomposition={"disagreement_level": 0.10},
        ),
        candidate_clusters=(),
    )

    monkeypatch.setattr(engine, "_run_parse_stage", lambda **kwargs: (problem, None))
    monkeypatch.setattr(engine, "_run_route_stage", lambda **kwargs: (route, None))
    monkeypatch.setattr(engine, "_run_budget_stage", lambda **kwargs: (None, None))
    monkeypatch.setattr(engine, "_run_state_init_stage", lambda **kwargs: (SimpleNamespace(root_node=object()), None))
    monkeypatch.setattr(engine, "_run_retrieval_stage", lambda **kwargs: (route, [], None))
    monkeypatch.setattr(engine, "_run_branch_stage", lambda **kwargs: (branch_result, None))
    monkeypatch.setattr(
        engine,
        "_run_aggregation_stage",
        lambda **kwargs: (
            SimpleNamespace(
                submission_answer=42,
                prediction=branch_result.final_prediction,
                abstain_recommended=False,
                warnings=(),
                metadata={},
            ),
            None,
        ),
    )
    monkeypatch.setattr(engine, "_run_submission_stage", lambda **kwargs: ({"id": problem.problem_id, "answer": 42}, None))

    result = engine.solve_problem("Find the answer.", problem_id=problem.problem_id, generator=None)

    assert result.ok
    summaries = [record.summary for record in result.stage_records]
    assert "minimal_runtime_unavailable_escalated" in summaries
