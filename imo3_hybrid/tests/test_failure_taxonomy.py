from __future__ import annotations

from src.common.schemas import BranchTrace, FailureBucket, HistoricalSolveRecord
from src.offline.failure_mining import FailureMiningConfig, mine_failures


def _branch(problem_id: str, branch_id: str, answer: str) -> BranchTrace:
    return BranchTrace(
        branch_id=branch_id,
        problem_id=problem_id,
        steps=[],
        full_reasoning=f"{branch_id} reasoning",
        answer=answer,
        answer_canonical=answer,
        symbolic_valid=False,
        branch_score=0.37,
        verifier_score=0.34,
        logical_consistency=0.40,
        completeness=0.45,
        repairability=0.30,
        answer_correctness_likelihood=0.33,
        symbolic_agreement=0.18,
        step_quality=0.28,
        prefix_quality=0.31,
        operator_sequence=["guess_and_check"],
        metadata={"symbolic_status": "unsupported"},
    )


def test_failure_bucket_values_cover_expected_taxonomy() -> None:
    assert {bucket.value for bucket in FailureBucket} == {
        "search_failure",
        "verifier_failure",
        "symbolic_failure",
        "retrieval_failure",
        "operator_failure",
        "aggregation_failure",
        "routing_budget_failure",
        "unknown_failure",
    }


def test_failure_taxonomy_labels_underbudget_search_slice() -> None:
    record = HistoricalSolveRecord(
        problem_id="p_budget",
        domain="geometry",
        difficulty="hard",
        gold_answer="11",
        predicted_answer="7",
        branch_traces=[_branch("p_budget", "b_wrong", "7")],
        route_snapshot={
            "branch_budget": 12,
            "max_search_depth": 3,
            "max_search_nodes": 24,
            "retrieval_depth": 0,
            "critique_top_k": 0,
            "verifier_mode": "fast",
        },
    )

    result = mine_failures([record], config=FailureMiningConfig(min_hard_problems=1))
    assert len(result.failures) == 1
    failure = result.failures[0]
    assert failure.primary_failure is FailureBucket.ROUTING_BUDGET_FAILURE
    assert failure.metadata["primary_subtype"] == "under_budgeted_hard_problem"
    subtypes = {item.subtype for item in failure.attributions}
    assert "insufficient_depth" in subtypes
    assert "critique_disabled_too_early" in subtypes
