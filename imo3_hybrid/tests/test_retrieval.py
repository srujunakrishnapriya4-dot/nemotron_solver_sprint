from __future__ import annotations

from dataclasses import asdict
from math import isclose

import numpy as np
import pytest

from src.common.schemas import BudgetPlan, Difficulty, ParsedProblem, ProblemDomain, RetrievedTrace, RouteDecision
from src.operators.operator_library import OperatorLibrary
from src.operators.operator_miner import OperatorMiner
from src.retrieval.index_builder import IndexedTraceRecord, RetrievalIndexBuilder
from src.retrieval.query import (
    RetrievalQuery,
    build_repair_query,
    build_retrieval_query,
    build_subproblem_query,
)
from src.retrieval.retrieval_policy import RetrievalHit, RetrievalPolicy
from src.retrieval.trace_prompting import (
    build_hint_from_traces,
    build_hint_only_prompt,
    build_repair_prompt,
    build_trace_conditioned_prompt,
    select_conditioning_strategy,
)
from src.state_graph.node import (
    ConstraintRecord,
    GoalRecord,
    InvariantRecord,
    OperatorApplicationRecord,
    ReasoningStateNode,
)
from src.state_graph.proof_obligations import ProofEvidenceKind, ProofObligation


class StubEmbedder:
    def __init__(self, dimension: int = 12) -> None:
        self._dimension = dimension
        self.backend = "deterministic_test"

    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        rows = [self.embed_single(text, normalize=normalize) for text in texts]
        if not rows:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return np.stack(rows, axis=0).astype(np.float32)

    def embed_single(self, text: str, normalize: bool = True) -> np.ndarray:
        vector = np.zeros(self._dimension, dtype=np.float32)
        clean = (text or "").lower()
        features = {
            0: ["number_theory", "integer", "prime", "divides", "mod"],
            1: ["modular", "congruence", "residue", "modular_patterns"],
            2: ["parity", "even", "odd"],
            3: ["invariant", "preserved", "unchanged", "invariant_traces"],
            4: ["divisibility", "divides", "prime"],
            5: ["construction", "geometry", "triangle", "circle"],
            6: ["repair", "failure", "candidate_answer"],
            7: ["operator_sequence", "modular_reasoning", "divisibility_reasoning"],
            8: ["contradiction", "impossible"],
            9: ["subproblem", "goal"],
            10: ["symbolic", "equation"],
            11: ["count", "bijection"],
        }
        for idx, keywords in features.items():
            vector[idx] = float(sum(clean.count(keyword) for keyword in keywords))
        if not normalize:
            return vector
        norm = float(np.linalg.norm(vector))
        return vector if norm == 0.0 else (vector / norm).astype(np.float32)


def _problem() -> ParsedProblem:
    return ParsedProblem(
        problem_id="p_nt_01",
        raw_text=(
            "Find all positive integers n such that n is prime or divisible by 3, "
            "and determine whether n^2 + 1 is congruent to 0 modulo 5."
        ),
        constraints=[
            "n is integer",
            "n is prime",
            "3 divides n^2 + 1",
            "n ≡ 1 (mod 5)",
        ],
        domain=ProblemDomain.NUMBER_THEORY,
        target="find all values of n",
        likely_archetypes=["modular", "parity", "invariant"],
        parity_cues=["parity"],
        integrality_constraints=["n is integer"],
        unknowns=["n"],
        parse_quality={"overall_confidence": 0.91},
        metadata={"family": "number_theory"},
    )


def _route() -> RouteDecision:
    budget = BudgetPlan(
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
        use_brute_force=True,
    )
    return RouteDecision(
        problem_id="p_nt_01",
        problem_type_probs={
            "algebra": 0.06,
            "number_theory": 0.63,
            "combinatorics": 0.05,
            "geometry": 0.03,
            "functional_equation": 0.01,
            "mixed": 0.18,
            "unknown": 0.04,
        },
        archetype_probs={
            "modular": 0.31,
            "parity": 0.21,
            "invariant": 0.16,
            "contradiction": 0.12,
            "case_work": 0.08,
            "bounding": 0.07,
            "construction": 0.02,
            "induction": 0.01,
            "bijection": 0.01,
            "symbolic_manipulation": 0.01,
        },
        difficulty=Difficulty.HARD,
        difficulty_score=0.72,
        operator_prior={
            "modular_reasoning": 0.28,
            "divisibility_reasoning": 0.24,
            "invariant_reasoning": 0.16,
            "contradiction_reasoning": 0.12,
            "symbolic_execution": 0.10,
            "case_split": 0.10,
        },
        budget_plan=budget,
        retrieval_depth=budget.retrieval_depth,
        branch_budget=budget.branch_budget,
        repair_threshold=0.61,
        route_uncertainty=0.34,
        verifier_mode="strict",
        problem_type={
            "algebra": 0.06,
            "number_theory": 0.63,
            "combinatorics": 0.05,
            "geometry": 0.03,
            "functional_equation": 0.01,
            "mixed": 0.18,
            "unknown": 0.04,
        },
        archetypes={
            "modular": 0.31,
            "parity": 0.21,
            "invariant": 0.16,
            "contradiction": 0.12,
            "case_work": 0.08,
            "bounding": 0.07,
            "construction": 0.02,
            "induction": 0.01,
            "bijection": 0.01,
            "symbolic_manipulation": 0.01,
        },
        use_retrieval=True,
        use_symbolic=True,
        use_brute_force=True,
        retrieval_tags=["modular_patterns", "divisibility_traces", "invariant_traces"],
        repair_neighbors=["contradiction", "invariant", "parity"],
        route_rationale=["top_domain=number_theory", "top_archetype=modular"],
    )


def _reasoning_state() -> ReasoningStateNode:
    return ReasoningStateNode.create(
        problem_id="p_nt_01",
        branch_id="branch_01",
        constraints=(
            ConstraintRecord.create(
                kind="congruence",
                lhs="n^2 + 1",
                relation="equiv",
                rhs="0 (mod 5)",
            ),
            ConstraintRecord.create(
                kind="divisibility",
                lhs="3",
                relation="divides",
                rhs="n^2 + 1",
            ),
        ),
        invariants=(
            InvariantRecord.create(
                invariant_type="modular",
                expression="n mod 2 is stable",
                confidence=0.82,
            ),
        ),
        goals=(
            GoalRecord.create(
                kind="subgoal",
                text="show the modular residue class is impossible",
                priority=10,
            ),
        ),
        proof_obligations=(
            ProofObligation.create(
                originating_node_id="root::p_nt_01",
                claim="justify the modular pruning step",
                evidence_kind_required=ProofEvidenceKind.SYMBOLIC_CHECK,
            ),
        ),
        operator_history=(
            OperatorApplicationRecord.create(
                operator_name="modular_reasoning",
                rationale="reduce modulo 5",
                success=True,
            ),
            OperatorApplicationRecord.create(
                operator_name="divisibility_reasoning",
                rationale="use divisibility structure",
                success=True,
            ),
        ),
        partial_solution="n cannot satisfy both modular constraints",
        summary_text="number theory branch with modular and divisibility focus",
        symbolic_valid=True,
        metadata={"candidate_answer": "none", "branch_phase": "reasoning", "failure_type": "symbolic_mismatch"},
    )


def _records() -> list[IndexedTraceRecord]:
    long_solution = (
        "Use modular arithmetic first. Reduce the expression modulo 5 and modulo 2, "
        "then compare residue classes with the divisibility condition. "
        "After pruning impossible residues, close with a contradiction."
    )
    return [
        IndexedTraceRecord(
            trace_id="trace_modular_primary",
            problem_id="r1",
            problem_text="Find integers n with n ≡ 1 (mod 5) and 3 divides n^2 + 1.",
            solution_text=long_solution,
            answer="0",
            domain="number_theory",
            difficulty="hard",
            archetypes=["modular", "parity"],
            operators_used=["modular_reasoning", "divisibility_reasoning", "contradiction_reasoning"],
            source="mined",
            tags=["modular_patterns", "divisibility_traces", "parity"],
            subproblem_snippets=["n ≡ 1 (mod 5)", "3 divides n^2 + 1"],
            repair_snippets=["incorrect residue class"],
            failure_modes=["symbolic_mismatch"],
            proof_obligation_hints=["symbolic_check", "justify the modular pruning step"],
            repair_neighbors=["contradiction_reasoning", "divisibility_reasoning"],
            continuation_operators=["case_split", "contradiction_reasoning"],
        ),
        IndexedTraceRecord(
            trace_id="trace_modular_secondary",
            problem_id="r2",
            problem_text="Determine residues of primes under divisibility constraints.",
            solution_text=long_solution + " Recheck residues before concluding.",
            answer="1",
            domain="number_theory",
            difficulty="hard",
            archetypes=["modular", "contradiction"],
            operators_used=["modular_reasoning", "contradiction_reasoning"],
            source="mined",
            tags=["modular_patterns", "divisibility_traces"],
            subproblem_snippets=["prime residue classes", "contradiction from divisibility"],
            repair_snippets=["repair by checking a smaller modulus"],
            failure_modes=["symbolic_mismatch"],
            proof_obligation_hints=["symbolic_check"],
            repair_neighbors=["modular_reasoning", "contradiction_reasoning"],
            continuation_operators=["contradiction_reasoning"],
        ),
        IndexedTraceRecord(
            trace_id="trace_invariant_diverse",
            problem_id="r3",
            problem_text="Track a preserved parity and modular invariant for integer states.",
            solution_text=(
                "Introduce an invariant on parity classes, then use contradiction only after the invariant "
                "forces the remaining cases into a tiny search space."
            ),
            answer="2",
            domain="number_theory",
            difficulty="hard",
            archetypes=["invariant", "parity"],
            operators_used=["invariant_reasoning", "contradiction_reasoning"],
            source="mined",
            tags=["invariant_traces", "parity"],
            subproblem_snippets=["preserved parity", "tiny search space"],
            repair_snippets=["switch from modular to invariant view"],
            failure_modes=["coverage_gap"],
            proof_obligation_hints=["invariant_check"],
            repair_neighbors=["invariant_reasoning"],
            continuation_operators=["contradiction_reasoning"],
        ),
        IndexedTraceRecord(
            trace_id="trace_geometry_irrelevant",
            problem_id="r4",
            problem_text="Construct a point on a circumcircle of a triangle.",
            solution_text="Use symmetry and angle chasing on the circle.",
            answer="3",
            domain="geometry",
            difficulty="hard",
            archetypes=["construction", "symmetry"],
            operators_used=["geometry_transform", "construction_reasoning"],
            source="mined",
            tags=["construction_traces", "diagram_structure"],
        ),
    ]


def _build_index() -> tuple[StubEmbedder, RetrievalIndexBuilder]:
    embedder = StubEmbedder()
    index = RetrievalIndexBuilder(embedder=embedder, index_path="data/interim/test_retrieval_index")
    index.build_from_records(_records(), save=False)
    return embedder, index


def test_retrieval_query_builds_from_problem_route_and_state_purely(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.retrieval.query as query_module

    class _UnexpectedRuntimeInstantiation:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("query construction should not instantiate retrieval runtime systems")

    monkeypatch.setattr(query_module, "MathEmbedder", _UnexpectedRuntimeInstantiation)
    monkeypatch.setattr(query_module, "RetrievalIndexBuilder", _UnexpectedRuntimeInstantiation)

    problem = _problem()
    route = _route()
    state = _reasoning_state()

    query = build_retrieval_query(
        problem,
        route,
        reasoning_state=state,
        subproblem_text="Reduce the contradiction to a residue-class subproblem.",
        operator_sequence_hint=["modular_reasoning", "contradiction_reasoning"],
    )

    assert isinstance(query, RetrievalQuery)
    assert query.problem_id == problem.problem_id
    assert query.domain == "number_theory"
    assert query.difficulty == "hard"
    assert query.repair_mode is False
    assert query.source == "problem_route_state"
    assert query.archetypes[:3] == ("modular", "parity", "invariant")
    assert query.operator_hints[0] == "modular_reasoning"
    assert "divisibility_reasoning" in query.operator_hints
    assert "modular_patterns" in query.tags
    assert "parity" in query.tags
    assert query.open_obligation_claims
    assert "symbolic_check" in query.required_evidence_kinds
    assert "symbolic_mismatch" in query.failure_modes
    assert query.branch_phase == "reasoning"
    assert any("operator_sequence:" in snippet for snippet in query.subproblem_snippets)
    assert "OPERATOR HINTS:" in query.query_text
    assert "OPEN OBLIGATIONS:" in query.query_text
    lexical_text = query.lexical_text()
    assert lexical_text.startswith("domain: number_theory")
    assert "difficulty: hard" in lexical_text
    assert "Reduce the contradiction" in lexical_text


def test_subproblem_and_repair_query_variants_are_typed_and_deterministic() -> None:
    problem = _problem()
    route = _route()
    state = _reasoning_state()

    subproblem_query = build_subproblem_query(
        problem,
        route,
        "Check the residue class modulo 5 separately.",
        reasoning_state=state,
        operator_sequence_hint=["modular_reasoning"],
    )
    repair_query = build_repair_query(
        problem,
        route,
        "The previous branch used the wrong residue class.",
        reasoning_state=state,
        operator_sequence_hint=["contradiction_reasoning"],
    )
    repair_query_repeat = build_repair_query(
        problem,
        route,
        "The previous branch used the wrong residue class.",
        reasoning_state=state,
        operator_sequence_hint=["contradiction_reasoning"],
    )

    assert isinstance(subproblem_query, RetrievalQuery)
    assert isinstance(repair_query, RetrievalQuery)
    assert subproblem_query.repair_mode is False
    assert repair_query.repair_mode is True
    assert repair_query.source == "repair"
    assert "repair" in repair_query.tags
    assert repair_query == repair_query_repeat


def test_retrieval_policy_returns_typed_hits_and_operator_support_signals() -> None:
    problem = _problem()
    route = _route()
    state = _reasoning_state()
    embedder, index = _build_index()
    policy = RetrievalPolicy()
    query = build_retrieval_query(problem, route, reasoning_state=state)

    hits = policy.retrieve(query=query, route=route, index=index, embedder=embedder, top_k=3)

    assert hits
    assert all(isinstance(hit, RetrievalHit) for hit in hits)
    assert len(hits) <= 3
    assert hits[0].retrieval_method == "hybrid"
    assert hits[0].domain == "number_theory"
    assert hits[0].confidence >= 0.10
    assert hits[0].fused_score >= 0.0
    assert hits[0].compatibility_score > 0.0
    assert hits[0].operator_support
    assert hits[0].compatibility_reasons
    assert "symbolic_check" in hits[0].relevant_evidence_kinds
    assert any("modular_reasoning" in hit.operator_support for hit in hits)

    support = policy.operator_support_signal(hits)
    assert support
    assert isclose(sum(support.values()), 1.0, rel_tol=1e-5, abs_tol=1e-5)
    assert max(support, key=support.get) in {"modular_reasoning", "divisibility_reasoning", "contradiction_reasoning"}

    traces = [policy.hit_to_trace(hit) for hit in hits]
    assert all(isinstance(trace, RetrievedTrace) for trace in traces)
    assert traces[0].similarity_score == hits[0].confidence
    assert traces[0].operators_used
    assert traces[0].operator_support
    assert traces[0].compatibility_score > 0.0
    assert traces[0].strategy_summary


def test_retrieval_policy_uses_bounded_diversifying_selection_when_available() -> None:
    policy = RetrievalPolicy()
    hits = [
        RetrievalHit(
            trace_id="a_modular",
            problem_id="r1",
            score_dense=0.70,
            score_lexical=0.40,
            score_structural=0.35,
            fused_score=0.60,
            compatibility_score=0.76,
            confidence=0.80,
            retrieval_method="hybrid",
            domain="number_theory",
            archetypes=("modular", "parity"),
            operators_used=("modular_reasoning", "divisibility_reasoning"),
            problem="A",
            solution="A solution",
            answer="0",
            source="mined",
            tags=("modular_patterns", "parity"),
        ),
        RetrievalHit(
            trace_id="b_near_duplicate",
            problem_id="r2",
            score_dense=0.68,
            score_lexical=0.39,
            score_structural=0.34,
            fused_score=0.58,
            compatibility_score=0.74,
            confidence=0.79,
            retrieval_method="hybrid",
            domain="number_theory",
            archetypes=("modular", "parity"),
            operators_used=("modular_reasoning", "divisibility_reasoning"),
            problem="B",
            solution="B solution",
            answer="1",
            source="mined",
            tags=("modular_patterns", "parity"),
        ),
        RetrievalHit(
            trace_id="c_diverse",
            problem_id="r3",
            score_dense=0.64,
            score_lexical=0.32,
            score_structural=0.31,
            fused_score=0.54,
            compatibility_score=0.62,
            confidence=0.74,
            retrieval_method="hybrid",
            domain="number_theory",
            archetypes=("invariant", "contradiction"),
            operators_used=("invariant_reasoning", "contradiction_reasoning"),
            problem="C",
            solution="C solution",
            answer="2",
            source="mined",
            tags=("invariant_traces", "contradiction"),
        ),
    ]

    diversified = policy._mmr_diversify(hits, top_k=2)

    assert len(diversified) == 2
    assert diversified[0].trace_id == "a_modular"
    assert {hit.trace_id for hit in diversified} == {"a_modular", "c_diverse"}


def test_trace_prompting_stays_compact_and_strategy_oriented() -> None:
    problem = _problem()
    long_solution = "Use modular reasoning. " * 120 + "FORBIDDEN_TAIL_TEXT"
    trace = RetrievedTrace(
        trace_id="trace_long",
        problem="Determine residues for an integer divisibility problem.",
        solution=long_solution,
        answer="7",
        domain="number_theory",
        archetypes=["modular", "parity", "invariant"],
        operators_used=["modular_reasoning", "divisibility_reasoning", "contradiction_reasoning"],
        similarity_score=0.88,
        source="mined",
        compatibility_score=0.81,
        compatibility_reasons=["problem_structure_match", "proof_obligation_support"],
        relevant_obligation_claims=["justify the modular pruning step"],
        relevant_evidence_kinds=["symbolic_check"],
        failure_mode_support=["symbolic_mismatch"],
        repair_operator_hints=["contradiction_reasoning"],
    )

    hint = build_hint_from_traces([trace])
    prompt = build_trace_conditioned_prompt(
        problem,
        trace,
        operator_hint="modular_reasoning",
        max_example_solution_chars=180,
    )
    repair_prompt = build_repair_prompt(
        problem,
        "A residue-class assumption contradicted the divisibility step.",
        [trace],
        operator_hint="contradiction_reasoning",
    )
    hint_only_prompt = build_hint_only_prompt(
        problem,
        [trace],
        failure_text="Need a safer strategy for the modular subproblem.",
    )

    assert hint is not None
    assert "Likely operators:" in hint
    assert "Operator sequence hint:" in hint
    assert "Why retrieved:" in hint
    assert "FORBIDDEN_TAIL_TEXT" not in hint

    assert "Use the retrieved example only as a strategy reference." in prompt
    assert "Do not copy the derivation or the final answer." in prompt
    assert "TRANSFER HINTS:" in prompt
    assert "RETRIEVAL SUPPORT:" in prompt
    assert "Primary operator bias: modular_reasoning" in prompt
    assert "FORBIDDEN_TAIL_TEXT" not in prompt
    assert len(prompt) < len(long_solution) + len(problem.raw_text)
    assert "FINAL_ANSWER: [integer]" in prompt

    assert "Repair the current solution locally." in repair_prompt
    assert "avoid restarting from scratch" in repair_prompt
    assert "FORBIDDEN_TAIL_TEXT" not in repair_prompt
    assert "REPAIR HINTS:" in repair_prompt
    assert "RETRIEVAL SUPPORT:" in repair_prompt

    assert "STRATEGY HINTS:" in hint_only_prompt
    assert "RETRIEVAL SUPPORT:" in hint_only_prompt
    assert "avoid copying any retrieved solution text" in hint_only_prompt
    assert "FORBIDDEN_TAIL_TEXT" not in hint_only_prompt


def test_conditioning_strategy_and_retrieval_results_are_deterministic() -> None:
    problem = _problem()
    route = _route()
    state = _reasoning_state()
    embedder, index = _build_index()
    policy = RetrievalPolicy()
    query = build_retrieval_query(problem, route, reasoning_state=state)

    hits_first = policy.retrieve(query=query, route=route, index=index, embedder=embedder, top_k=3)
    hits_second = policy.retrieve(query=query, route=route, index=index, embedder=embedder, top_k=3)

    assert [asdict(hit) for hit in hits_first] == [asdict(hit) for hit in hits_second]

    traces = [policy.hit_to_trace(hit) for hit in hits_first]
    best_trace = policy.select_best_trace_for_conditioning(traces, min_similarity=0.2)
    assert best_trace is not None
    assert best_trace.trace_id == policy.select_best_trace_for_conditioning(traces, min_similarity=0.2).trace_id

    strategy = select_conditioning_strategy(traces, route, branch_index=0, total_branches=4)
    strategy_repeat = select_conditioning_strategy(traces, route, branch_index=0, total_branches=4)
    assert strategy == strategy_repeat
    assert strategy in {"full_trace", "hint_only"}


def test_operator_applicability_uses_typed_state_and_retrieval_context() -> None:
    library = OperatorLibrary()
    applicability = library.check_applicability(
        "modular_arithmetic",
        _reasoning_state(),
        _route(),
        retrieval_context={
            "archetypes": ["modular"],
            "evidence_kinds": ["symbolic_check"],
            "failure_modes": ["symbolic_mismatch"],
            "repair_operators": ["contradiction_reasoning"],
            "operator_prefix": ["modular_reasoning"],
            "tags": ["problem_structure_match"],
        },
    )

    assert applicability.score > 0.0
    assert applicability.retrieval_compatibility_score > 0.0
    assert applicability.compatible_retrieval_signals
    assert applicability.diagnostics["open_obligation_count"] >= 1


def test_mined_operators_preserve_provenance_and_thresholds() -> None:
    miner = OperatorMiner(min_support=2, min_symbolic_success_rate=0.5, min_mean_verifier_gain=0.0)
    trace_payload = {
        "problem_id": "p_nt_01",
        "branch_id": "branch_mined",
        "operator_sequence": ["modular_arithmetic", "contradiction"],
        "steps": [
            {"step_num": 1, "description": "Reduce modulo 5.", "operator_used": "modular_arithmetic"},
            {"step_num": 2, "description": "Derive contradiction.", "operator_used": "contradiction"},
        ],
        "symbolic_valid": True,
        "verifier_score": 0.82,
        "branch_score": 0.44,
        "archetype_used": "modular",
        "retrieval_used": True,
        "failure_type": "symbolic_mismatch",
    }
    miner.ingest_trace(trace_payload)
    miner.ingest_trace({**trace_payload, "branch_id": "branch_mined_2"})

    candidates = miner.emit_candidates()
    modular = next(candidate for candidate in candidates if candidate.operator_name == "modular_arithmetic")

    assert modular.support_count >= 2
    assert modular.operator_purity > 0.0
    assert modular.reliability > 0.0
    assert modular.promotion_thresholds["min_support"] == 2
    assert modular.provenance_summary["dominant_archetypes"]
