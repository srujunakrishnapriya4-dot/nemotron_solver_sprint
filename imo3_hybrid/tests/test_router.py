from __future__ import annotations

from math import isclose

import pytest

from src.common.schemas import (
    ArchetypePrediction,
    BudgetPlan,
    Difficulty,
    DifficultyEstimate,
    ParsedProblem,
    ProblemDomain,
    ProblemTypePrediction,
    RouteDecision,
)
from src.routing.archetype_predictor import (
    ArchetypePredictor,
    get_archetype_registry,
    get_supported_archetypes,
    predict_archetypes,
)
from src.routing.difficulty_estimator import DifficultyEstimator, estimate_difficulty
from src.routing.dual_router import DualRouter, dual_router
from src.routing.problem_type import ProblemTypeClassifier, predict_problem_type


def _assert_probability_distribution(values: dict[str, float], *, tolerance: float = 1e-5) -> None:
    assert values, "expected a non-empty probability distribution"
    assert all(0.0 <= value <= 1.0 for value in values.values())
    assert isclose(sum(values.values()), 1.0, rel_tol=tolerance, abs_tol=tolerance)


def _problem(
    *,
    problem_id: str,
    raw_text: str,
    domain: ProblemDomain = ProblemDomain.UNKNOWN,
    constraints: list[str],
    likely_archetypes: list[str] | None = None,
    symmetries: list[str] | None = None,
    parity_cues: list[str] | None = None,
    integrality_constraints: list[str] | None = None,
    unknowns: list[str] | None = None,
    parse_confidence: float = 0.92,
    difficulty_seed: float = 0.5,
    metadata: dict[str, object] | None = None,
) -> ParsedProblem:
    return ParsedProblem(
        problem_id=problem_id,
        raw_text=raw_text,
        constraints=constraints,
        domain=domain,
        likely_archetypes=likely_archetypes or [],
        symmetries=symmetries or [],
        parity_cues=parity_cues or [],
        integrality_constraints=integrality_constraints or [],
        unknowns=unknowns or ["x"],
        parse_quality={"overall_confidence": parse_confidence},
        difficulty_seed=difficulty_seed,
        metadata=metadata or {},
    )


@pytest.fixture
def number_theory_problem() -> ParsedProblem:
    return _problem(
        problem_id="nt_modular",
        raw_text=(
            "Find all integers n such that n is prime or divisible by 3, "
            "and determine the remainder of n^2 + 1 modulo 5."
        ),
        domain=ProblemDomain.NUMBER_THEORY,
        constraints=[
            "n is integer",
            "n is prime",
            "n ≡ 1 (mod 5)",
            "3 divides n^2 + 1",
        ],
        likely_archetypes=["modular", "parity", "contradiction"],
        parity_cues=["parity"],
        integrality_constraints=["n is integer"],
        unknowns=["n"],
        difficulty_seed=0.56,
        metadata={"seed_family": "olympiad_nt"},
    )


@pytest.fixture
def geometry_construction_problem() -> ParsedProblem:
    return _problem(
        problem_id="geo_construct",
        raw_text=(
            "In triangle ABC with AB = AC, construct a point P on the circumcircle such that "
            "angle BPC is maximal. Prove the resulting configuration is symmetric."
        ),
        domain=ProblemDomain.GEOMETRY,
        constraints=[
            "triangle ABC",
            "AB = AC",
            "P lies on circle(ABC)",
            "angle BPC is maximal",
            "AB is congruent to AC",
        ],
        likely_archetypes=["construction", "symmetry", "extremal"],
        symmetries=["AB and AC are exchangeable"],
        unknowns=["P"],
        difficulty_seed=0.74,
        metadata={"diagram": "isosceles"},
    )


@pytest.fixture
def easy_algebra_problem() -> ParsedProblem:
    return _problem(
        problem_id="easy_algebra",
        raw_text="Compute the value of x if x + 2 = 5.",
        domain=ProblemDomain.ALGEBRA,
        constraints=["x + 2 = 5"],
        likely_archetypes=["symbolic_manipulation"],
        unknowns=["x"],
        difficulty_seed=0.05,
        metadata={"shape": "single_equation"},
    )


@pytest.fixture
def hard_mixed_problem() -> ParsedProblem:
    return _problem(
        problem_id="hard_mixed",
        raw_text=(
            "For all positive integers n, determine all functions f such that there exists a partition of the "
            "integers with a symmetric extremal property, and prove the maximum possible value is attained."
        ),
        domain=ProblemDomain.MIXED,
        constraints=[
            "for all positive integers n",
            "there exists a function f",
            "f(n + 1) >= f(n)",
            "n is integer",
            "maximum possible value exists",
            "partition of integers",
        ],
        likely_archetypes=["induction", "symmetry", "bounding", "case_work"],
        symmetries=["exchange on partition blocks"],
        integrality_constraints=["n is integer"],
        unknowns=["f", "n"],
        parse_confidence=0.88,
        difficulty_seed=0.84,
        metadata={"quantified": True, "family": "mixed_quantified"},
    )


def test_problem_type_prediction_is_probabilistic_and_schema_rich(number_theory_problem: ParsedProblem) -> None:
    classifier = ProblemTypeClassifier()

    prediction = classifier.predict(number_theory_problem)

    assert isinstance(prediction, ProblemTypePrediction)
    _assert_probability_distribution(prediction.problem_type_probs)
    assert prediction.top_domain == ProblemDomain.NUMBER_THEORY
    assert prediction.problem_type_probs["number_theory"] == max(prediction.problem_type_probs.values())
    assert prediction.problem_type_probs["number_theory"] > prediction.problem_type_probs["algebra"]
    assert 0.0 <= prediction.confidence <= 1.0
    assert prediction.entropy >= 0.0
    assert prediction.is_mixed is False
    assert "features" in prediction.diagnostics
    assert "raw_scores" in prediction.diagnostics
    assert "reasons" in prediction.diagnostics
    assert prediction.diagnostics["features"]["has_parity"] is True


def test_archetype_predictor_supports_multi_label_and_registry_backed_outputs(
    geometry_construction_problem: ParsedProblem,
) -> None:
    predictor = ArchetypePredictor()
    problem_type_prediction = predict_problem_type(geometry_construction_problem)

    prediction = predictor.predict(
        geometry_construction_problem,
        problem_type_prediction=problem_type_prediction,
    )

    supported = set(get_supported_archetypes())
    registry = get_archetype_registry()

    assert isinstance(prediction, ArchetypePrediction)
    _assert_probability_distribution(prediction.archetype_probs)
    assert len(prediction.top_archetypes) >= 2
    assert {"construction", "symmetry"} <= set(prediction.top_archetypes)
    assert set(prediction.top_archetypes).issubset(supported)
    assert prediction.archetype_probs["construction"] > 0.0
    assert prediction.archetype_probs["symmetry"] > 0.0
    assert prediction.confidence >= 0.0
    assert prediction.entropy >= 0.0
    assert "registry" in prediction.diagnostics
    for name in prediction.top_archetypes[:2]:
        assert name in registry
        assert prediction.diagnostics["registry"][name]["operator_tags"]
        assert prediction.diagnostics["registry"][name]["retrieval_tags"]
        assert prediction.diagnostics["registry"][name]["neighbors"]
        assert prediction.diagnostics["registry"][name]["compatible_domains"]


def test_difficulty_estimator_changes_budget_behavior_with_problem_hardness(
    easy_algebra_problem: ParsedProblem,
    hard_mixed_problem: ParsedProblem,
) -> None:
    estimator = DifficultyEstimator()

    easy_type = predict_problem_type(easy_algebra_problem)
    easy_arch = predict_archetypes(easy_algebra_problem, problem_type_prediction=easy_type)
    easy_estimate = estimator.estimate(
        easy_algebra_problem,
        problem_type_prediction=easy_type,
        archetype_prediction=easy_arch,
    )

    hard_type = predict_problem_type(hard_mixed_problem)
    hard_arch = predict_archetypes(hard_mixed_problem, problem_type_prediction=hard_type)
    hard_estimate = estimator.estimate(
        hard_mixed_problem,
        problem_type_prediction=hard_type,
        archetype_prediction=hard_arch,
    )

    assert isinstance(easy_estimate, DifficultyEstimate)
    assert isinstance(hard_estimate, DifficultyEstimate)
    assert easy_estimate.difficulty == Difficulty.EASY
    assert hard_estimate.difficulty in {Difficulty.HARD, Difficulty.VERY_HARD}
    assert isinstance(easy_estimate.budget_plan, BudgetPlan)
    assert isinstance(hard_estimate.budget_plan, BudgetPlan)
    assert hard_estimate.difficulty_score > easy_estimate.difficulty_score
    assert hard_estimate.budget_plan.branch_budget > easy_estimate.budget_plan.branch_budget
    assert hard_estimate.budget_plan.max_search_depth >= easy_estimate.budget_plan.max_search_depth
    assert hard_estimate.budget_plan.max_search_nodes > easy_estimate.budget_plan.max_search_nodes
    assert hard_estimate.budget_plan.self_consistency_samples >= hard_estimate.budget_plan.branch_budget
    assert hard_estimate.budget_plan.retrieval_depth >= easy_estimate.budget_plan.retrieval_depth


def test_dual_router_combines_signals_into_control_plane_rich_route_decision(
    geometry_construction_problem: ParsedProblem,
) -> None:
    router = DualRouter()
    problem_type_prediction = ProblemTypePrediction(
        problem_id=geometry_construction_problem.problem_id,
        problem_type_probs={
            "algebra": 0.10,
            "number_theory": 0.03,
            "combinatorics": 0.04,
            "geometry": 0.52,
            "functional_equation": 0.01,
            "mixed": 0.20,
            "unknown": 0.10,
        },
        top_domain=ProblemDomain.GEOMETRY,
        confidence=0.21,
        entropy=1.82,
        is_mixed=True,
        diagnostics={"source": "test"},
    )
    archetype_prediction = ArchetypePrediction(
        problem_id=geometry_construction_problem.problem_id,
        archetype_probs={
            "construction": 0.18,
            "symmetry": 0.17,
            "extremal": 0.15,
            "invariant": 0.14,
            "case_work": 0.12,
            "bounding": 0.10,
            "contradiction": 0.08,
            "parity": 0.01,
            "modular": 0.01,
            "bijection": 0.01,
            "induction": 0.01,
            "pigeonhole": 0.01,
            "symbolic_manipulation": 0.0,
            "brute_force_smallspace": 0.0,
            "generating_function": 0.0,
        },
        top_archetypes=["construction", "symmetry", "extremal", "invariant"],
        confidence=0.18,
        entropy=2.55,
        diagnostics={"source": "test"},
    )
    difficulty_estimate = DifficultyEstimate(
        problem_id=geometry_construction_problem.problem_id,
        difficulty=Difficulty.HARD,
        difficulty_score=0.72,
        uncertainty=0.82,
        budget_plan=BudgetPlan(
            branch_budget=64,
            retrieval_depth=4,
            max_search_depth=7,
            max_search_nodes=128,
            repair_budget=2,
            self_consistency_samples=64,
            critique_top_k=5,
            widen_on_uncertainty=False,
            use_retrieval=True,
            use_symbolic=True,
            use_brute_force=False,
        ),
        diagnostics={"source": "test"},
    )

    decision = router.route(
        geometry_construction_problem,
        problem_type_prediction=problem_type_prediction,
        archetype_prediction=archetype_prediction,
        difficulty_estimate=difficulty_estimate,
    )

    assert isinstance(decision, RouteDecision)
    _assert_probability_distribution(decision.operator_prior, tolerance=1e-4)
    assert decision.problem_type_probs == problem_type_prediction.problem_type_probs
    assert decision.problem_type == problem_type_prediction.problem_type_probs
    assert decision.archetype_probs == archetype_prediction.archetype_probs
    assert decision.archetypes == archetype_prediction.archetype_probs
    assert decision.difficulty == Difficulty.HARD
    assert decision.difficulty_score == difficulty_estimate.difficulty_score
    assert decision.budget_plan.branch_budget == decision.branch_budget
    assert decision.budget_plan.retrieval_depth == decision.retrieval_depth
    assert decision.use_retrieval == decision.budget_plan.use_retrieval
    assert decision.use_symbolic == decision.budget_plan.use_symbolic
    assert decision.use_brute_force == decision.budget_plan.use_brute_force
    assert decision.branch_budget > difficulty_estimate.budget_plan.branch_budget
    assert decision.retrieval_depth > difficulty_estimate.budget_plan.retrieval_depth
    assert decision.budget_plan.widen_on_uncertainty is True
    assert decision.route_uncertainty >= 0.72
    assert decision.repair_threshold > 0.45
    assert decision.retrieval_tags
    assert "construction_traces" in decision.retrieval_tags
    assert "diagram_structure" in decision.retrieval_tags
    assert decision.repair_neighbors
    assert {"case_work", "extremal", "invariant"} & set(decision.repair_neighbors)
    assert decision.route_rationale
    assert any(item.startswith("top_domain=") for item in decision.route_rationale)
    assert any(item.startswith("top_archetype=") for item in decision.route_rationale)
    assert any(item.startswith("route_uncertainty=") for item in decision.route_rationale)
    assert "mixed_domain_handling_enabled" in decision.route_rationale
    assert decision.operator_prior["construction_reasoning"] > 0.0
    assert decision.operator_prior["geometry_transform"] > 0.0
    assert decision.diagnostics["problem_type_prediction"]["problem_id"] == geometry_construction_problem.problem_id


def test_routing_uncertainty_widens_downstream_policy(number_theory_problem: ParsedProblem) -> None:
    difficulty_estimate = DifficultyEstimate(
        problem_id=number_theory_problem.problem_id,
        difficulty=Difficulty.MEDIUM,
        difficulty_score=0.44,
        uncertainty=0.91,
        budget_plan=BudgetPlan(
            branch_budget=32,
            retrieval_depth=3,
            max_search_depth=6,
            max_search_nodes=64,
            repair_budget=1,
            self_consistency_samples=32,
            critique_top_k=5,
            widen_on_uncertainty=False,
            use_retrieval=True,
            use_symbolic=True,
            use_brute_force=False,
        ),
        diagnostics={"source": "test"},
    )
    problem_type_prediction = ProblemTypePrediction(
        problem_id=number_theory_problem.problem_id,
        problem_type_probs={
            "algebra": 0.20,
            "number_theory": 0.22,
            "combinatorics": 0.18,
            "geometry": 0.10,
            "functional_equation": 0.05,
            "mixed": 0.15,
            "unknown": 0.10,
        },
        top_domain=ProblemDomain.MIXED,
        confidence=0.17,
        entropy=1.90,
        is_mixed=True,
        diagnostics={},
    )
    archetype_prediction = ArchetypePrediction(
        problem_id=number_theory_problem.problem_id,
        archetype_probs={
            "modular": 0.14,
            "parity": 0.14,
            "invariant": 0.13,
            "contradiction": 0.11,
            "case_work": 0.10,
            "bounding": 0.09,
            "construction": 0.07,
            "induction": 0.06,
            "symbolic_manipulation": 0.06,
            "extremal": 0.05,
            "bijection": 0.02,
            "pigeonhole": 0.01,
            "brute_force_smallspace": 0.01,
            "generating_function": 0.005,
            "symmetry": 0.005,
        },
        top_archetypes=["modular", "parity", "invariant"],
        confidence=0.12,
        entropy=2.60,
        diagnostics={},
    )

    decision = dual_router(
        number_theory_problem,
        problem_type_prediction=problem_type_prediction,
        archetype_prediction=archetype_prediction,
        difficulty_estimate=difficulty_estimate,
    )

    assert decision.route_uncertainty >= 0.72
    assert decision.budget_plan.widen_on_uncertainty is True
    assert decision.branch_budget > 32
    assert decision.retrieval_depth > 3
    assert decision.budget_plan.max_search_nodes > 64
    assert decision.budget_plan.repair_budget > 1


def test_routing_is_deterministic_for_fixed_inputs(
    number_theory_problem: ParsedProblem,
) -> None:
    first_problem_type = predict_problem_type(number_theory_problem)
    second_problem_type = predict_problem_type(number_theory_problem)
    assert first_problem_type.model_dump() == second_problem_type.model_dump()

    first_archetypes = predict_archetypes(
        number_theory_problem,
        problem_type_prediction=first_problem_type,
    )
    second_archetypes = predict_archetypes(
        number_theory_problem,
        problem_type_prediction=second_problem_type,
    )
    assert first_archetypes.model_dump() == second_archetypes.model_dump()

    first_difficulty = estimate_difficulty(
        number_theory_problem,
        problem_type_prediction=first_problem_type,
        archetype_prediction=first_archetypes,
    )
    second_difficulty = estimate_difficulty(
        number_theory_problem,
        problem_type_prediction=second_problem_type,
        archetype_prediction=second_archetypes,
    )
    assert first_difficulty.model_dump() == second_difficulty.model_dump()

    first_route = dual_router(
        number_theory_problem,
        problem_type_prediction=first_problem_type,
        archetype_prediction=first_archetypes,
        difficulty_estimate=first_difficulty,
    )
    second_route = dual_router(
        number_theory_problem,
        problem_type_prediction=second_problem_type,
        archetype_prediction=second_archetypes,
        difficulty_estimate=second_difficulty,
    )
    assert first_route.model_dump() == second_route.model_dump()
