from __future__ import annotations

import pytest

from src.common import schemas as S
from src.parsing import parser as parser_module


@pytest.fixture
def parser_runtime():
    return parser_module, S


@pytest.fixture
def algebraic_problem_text():
    return (
        "Find all integers x such that x^2 + x = 6 and "
        "x \\equiv 0 (mod 2)."
    )


def _constraint_signature(constraint):
    quantifiers = tuple(
        (
            q.quantifier_type.value,
            tuple(q.variables),
            q.raw_text,
        )
        for q in constraint.quantified_variables
    )
    return (
        constraint.relation_type.value,
        constraint.operator,
        constraint.lhs.normalized_text if constraint.lhs else None,
        constraint.rhs.normalized_text if constraint.rhs else None,
        tuple(constraint.domain_assumptions),
        quantifiers,
    )


def _parsed_problem_signature(problem):
    return {
        "problem_id": problem.problem_id,
        "normalized_text": problem.normalized_text,
        "variables": [
            (
                item.symbol.symbol_id,
                item.symbol.canonical_name,
                item.declared_domain.domain if item.declared_domain else None,
            )
            for item in problem.variables
        ],
        "constraints": [_constraint_signature(c) for c in problem.constraints],
        "objective_mode": problem.objective.objective_mode.value if problem.objective else None,
        "objective_target": problem.objective.normalized_target_text if problem.objective else None,
        "answer_type": problem.answer_type.value,
        "problem_domain_seed": problem.problem_domain_seed.value,
        "archetype_cues": [item.value for item in problem.archetype_cues],
        "graph_nodes": tuple(problem.canonical_problem_graph.node_ids),
        "graph_edges": [
            (edge.source_node_id, edge.target_node_id, edge.relation_label)
            for edge in problem.canonical_problem_graph.edges
        ],
    }


def _assert_typed_problem(problem, S):
    assert isinstance(problem, S.ParsedProblem)
    assert all(isinstance(span, S.LatexSpan) for span in problem.latex_spans)
    assert problem.variables
    assert all(isinstance(item, S.VariableDecl) for item in problem.variables)
    assert problem.constraints
    assert all(isinstance(item, S.ConstraintNode) for item in problem.constraints)
    assert all(
        item.lhs is None or isinstance(item.lhs, S.MathOperand)
        for item in problem.constraints
    )
    assert all(
        item.rhs is None or isinstance(item.rhs, S.MathOperand)
        for item in problem.constraints
    )
    assert problem.objective is None or isinstance(problem.objective, S.ObjectiveSpec)
    assert isinstance(problem.parse_quality, S.ParseQualityReport)
    assert isinstance(problem.canonical_problem_graph, S.CanonicalProblemGraph)
    assert not all(isinstance(item, str) for item in problem.constraints)


def test_parse_entrypoint_returns_typed_parsed_problem(parser_runtime, algebraic_problem_text):
    parser_module, S = parser_runtime

    problem = parser_module.parse_sync(algebraic_problem_text, problem_id="p_main")

    _assert_typed_problem(problem, S)
    assert problem.problem_id == "p_main"
    assert problem.answer_type == S.AnswerType.SET
    assert problem.objective is not None
    assert problem.objective.objective_mode == S.QuantifierType.FIND_ALL
    assert "x" in problem.objective.normalized_target_text
    assert any(c.relation_type == S.RelationType.EQUALITY for c in problem.constraints)
    assert any(c.relation_type == S.RelationType.CONGRUENCE for c in problem.constraints)
    assert problem.problem_domain_seed in {
        S.ProblemDomain.ALGEBRA,
        S.ProblemDomain.NUMBER_THEORY,
    }
    assert problem.parse_quality.confidence_by_field["constraints"] > 0.0
    assert problem.parse_quality.confidence_by_field["answer_type"] > 0.0


def test_parser_output_is_downstream_ready_without_reparsing(parser_runtime, algebraic_problem_text):
    parser_module, S = parser_runtime

    problem = parser_module.parse_sync(algebraic_problem_text, problem_id="p_seed")

    _assert_typed_problem(problem, S)
    graph_node_ids = set(problem.canonical_problem_graph.node_ids)
    assert graph_node_ids
    assert any(node_id.startswith("sym_") for node_id in graph_node_ids)
    assert any(node_id.startswith("cst_") for node_id in graph_node_ids)
    assert all(edge.source_node_id in graph_node_ids for edge in problem.canonical_problem_graph.edges)
    assert all(edge.target_node_id in graph_node_ids for edge in problem.canonical_problem_graph.edges)
    assert any(item.symbol.canonical_name == "x" for item in problem.variables)
    assert all(
        item.lhs is None or item.lhs.normalized_text
        for item in problem.constraints
    )
    assert all(
        item.rhs is None or item.rhs.normalized_text
        for item in problem.constraints
    )
    assert not isinstance(problem.parse_quality, dict)
    assert not isinstance(problem.canonical_problem_graph, dict)


def test_fallback_parser_keeps_same_schema_family_as_model_assisted_path(
    parser_runtime,
    algebraic_problem_text,
):
    parser_module, S = parser_runtime

    def model_parser(text):
        return {
            "normalized_text": text,
            "variable_candidates": ("x",),
            "answer_type": "set",
            "domain_seed": "number_theory",
            "archetype_cues": ("modular",),
        }

    main_path = parser_module.parse_with_fallback(
        algebraic_problem_text,
        problem_id="p_compare",
        model_parser=model_parser,
    )
    fallback_path = parser_module.parse_with_fallback(
        algebraic_problem_text,
        problem_id="p_compare",
    )

    _assert_typed_problem(main_path, S)
    _assert_typed_problem(fallback_path, S)
    assert type(main_path) is type(fallback_path)
    assert type(main_path.parse_quality) is type(fallback_path.parse_quality)
    assert type(main_path.canonical_problem_graph) is type(fallback_path.canonical_problem_graph)
    assert [_constraint_signature(c) for c in main_path.constraints] == [
        _constraint_signature(c) for c in fallback_path.constraints
    ]
    assert main_path.answer_type == fallback_path.answer_type
    assert main_path.objective == fallback_path.objective
    assert main_path.parse_quality.fallback_used is False
    assert fallback_path.parse_quality.fallback_used is True
    assert "canonical_problem_graph_builder_absent" in fallback_path.parse_quality.semantic_gaps


def test_parser_reports_explicit_fallback_reason_on_model_exception(
    parser_runtime,
    algebraic_problem_text,
):
    parser_module, S = parser_runtime

    def broken_model_parser(_text):
        raise RuntimeError("boom")

    problem = parser_module.parse_with_fallback(
        algebraic_problem_text,
        problem_id="p_exception",
        model_parser=broken_model_parser,
    )

    _assert_typed_problem(problem, S)
    assert problem.parse_quality.fallback_used is True
    assert "fallback_reason:model_exception" in problem.parse_quality.semantic_gaps
    assert problem.parse_quality.suggested_downstream_penalties["routing_confidence_penalty"] > 0.0


def test_parse_quality_exposes_confidence_and_missing_field_metadata(parser_runtime):
    parser_module, S = parser_runtime

    problem = parser_module.parse_sync(
        "Let x be an integer. x = 4.",
        problem_id="p_quality",
    )

    _assert_typed_problem(problem, S)
    assert isinstance(problem.parse_quality.confidence_by_field, dict)
    assert {"normalized_text", "variables", "constraints", "answer_type"} <= set(
        problem.parse_quality.confidence_by_field
    )
    assert isinstance(problem.parse_quality.missing_fields, list)
    assert "objective" in problem.parse_quality.missing_fields


def test_parser_is_deterministic_for_fixed_input(parser_runtime, algebraic_problem_text):
    parser_module, _ = parser_runtime

    first = parser_module.parse_sync(algebraic_problem_text, problem_id="p_repeat")
    second = parser_module.parse_sync(algebraic_problem_text, problem_id="p_repeat")

    assert _parsed_problem_signature(first) == _parsed_problem_signature(second)


def test_parser_failure_modes_are_explicit(parser_runtime):
    parser_module, _ = parser_runtime

    with pytest.raises(ValueError, match="must not be empty"):
        parser_module.parse_sync("   ")

    with pytest.raises(parser_module.ParseFailure, match="zero constraints"):
        parser_module.parse_sync(
            "Let x be an integer and describe the variable.",
            problem_id="p_failure",
        )
