from __future__ import annotations

import json
from pathlib import Path

from src.common.schemas import BranchTrace, FailureBucket, HistoricalSolveRecord
from src.offline.failure_mining import (
    FailureMiningConfig,
    build_failure_dataset,
    load_failure_input_path,
    mine_failures,
    write_failure_artifacts,
)


def _branch(
    *,
    problem_id: str,
    branch_id: str,
    answer: str,
    branch_score: float,
    verifier_score: float,
    logical_consistency: float,
    symbolic_agreement: float,
    step_quality: float,
    prefix_quality: float,
    symbolic_valid: bool = True,
    answer_correctness_likelihood: float | None = None,
    retrieval_support: float = 0.0,
    retrieval_compatibility: float = 0.0,
    operator_reliability: float = 0.5,
    open_obligation_burden: float = 0.0,
    metadata: dict[str, object] | None = None,
    operator_sequence: list[str] | None = None,
) -> BranchTrace:
    return BranchTrace(
        branch_id=branch_id,
        problem_id=problem_id,
        steps=[],
        full_reasoning=f"branch {branch_id} reasoning",
        answer=answer,
        answer_canonical=answer,
        symbolic_valid=symbolic_valid,
        branch_score=branch_score,
        verifier_score=verifier_score,
        logical_consistency=logical_consistency,
        completeness=0.64,
        repairability=0.22,
        answer_correctness_likelihood=verifier_score if answer_correctness_likelihood is None else answer_correctness_likelihood,
        symbolic_agreement=symbolic_agreement,
        step_quality=step_quality,
        prefix_quality=prefix_quality,
        retrieval_support=retrieval_support,
        retrieval_compatibility=retrieval_compatibility,
        operator_reliability=operator_reliability,
        open_obligation_burden=open_obligation_burden,
        operator_sequence=list(operator_sequence or []),
        metadata=dict(metadata or {}),
    )


def _verifier_failure_record() -> HistoricalSolveRecord:
    problem_id = "p_verifier"
    wrong = _branch(
        problem_id=problem_id,
        branch_id="b_wrong",
        answer="17",
        branch_score=0.88,
        verifier_score=0.86,
        logical_consistency=0.81,
        symbolic_agreement=0.74,
        step_quality=0.44,
        prefix_quality=0.78,
        metadata={"symbolic_status": "success"},
        operator_sequence=["smooth_guess", "light_check"],
    )
    correct = _branch(
        problem_id=problem_id,
        branch_id="b_correct",
        answer="42",
        branch_score=0.42,
        verifier_score=0.31,
        logical_consistency=0.53,
        symbolic_agreement=0.79,
        step_quality=0.70,
        prefix_quality=0.56,
        metadata={"symbolic_status": "success", "phase": "active"},
        operator_sequence=["case_split", "exact_check"],
    )
    return HistoricalSolveRecord(
        problem_id=problem_id,
        domain="algebra",
        difficulty="hard",
        gold_answer="42",
        predicted_answer="17",
        branch_traces=[wrong, correct],
        route_snapshot={
            "branch_budget": 32,
            "max_search_depth": 6,
            "max_search_nodes": 128,
            "retrieval_depth": 2,
            "critique_top_k": 2,
            "verifier_mode": "strong",
        },
        verifier_snapshot={"status": "success"},
        symbolic_snapshot={"status": "success"},
        final_selector_diagnostics={"confidence": 0.88},
    )


def _symbolic_failure_record() -> HistoricalSolveRecord:
    problem_id = "p_symbolic"
    wrong = _branch(
        problem_id=problem_id,
        branch_id="b_symbolic_wrong",
        answer="9",
        branch_score=0.62,
        verifier_score=0.66,
        logical_consistency=0.58,
        symbolic_agreement=0.12,
        step_quality=0.52,
        prefix_quality=0.54,
        symbolic_valid=False,
        metadata={"symbolic_status": "malformed_input", "symbolic_error": "parser failure on congruence"},
        operator_sequence=["modular_arithmetic"],
    )
    return HistoricalSolveRecord(
        problem_id=problem_id,
        domain="number_theory",
        difficulty="hard",
        gold_answer="12",
        predicted_answer="9",
        branch_traces=[wrong],
        route_snapshot={
            "branch_budget": 40,
            "max_search_depth": 7,
            "max_search_nodes": 128,
            "retrieval_depth": 1,
            "critique_top_k": 2,
            "verifier_mode": "strong",
        },
        verifier_snapshot={"status": "success"},
        symbolic_snapshot={"status": "malformed_input", "summary": "parse error in modular expression"},
        final_selector_diagnostics={"confidence": 0.63},
    )


def test_failure_mining_is_deterministic_and_emits_artifacts(tmp_path) -> None:
    records = [_verifier_failure_record(), _symbolic_failure_record()]
    config = FailureMiningConfig(min_hard_problems=1)

    result_one = mine_failures(records, config=config)
    result_two = mine_failures(records, config=config)

    assert result_one.model_dump(mode="json") == result_two.model_dump(mode="json")

    out_dir = tmp_path / "failure_mining"
    paths = write_failure_artifacts(result_one, out_dir)
    expected_names = {
        "manifest",
        "failures",
        "failure_summary",
        "slice_metrics",
        "suggested_repairs",
    }
    assert set(paths) == expected_names
    for path in paths.values():
        assert out_dir.joinpath(Path(path).name).exists()

    failures_path = out_dir / "failures.jsonl"
    rows = [json.loads(line) for line in failures_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    assert all("secondary_failures" in row for row in rows)
    assert all("metadata" in row for row in rows)

    dataset_rows = build_failure_dataset(result_one)
    assert tuple(row["problem_id"] for row in dataset_rows) == ("p_symbolic", "p_verifier")


def test_failure_mining_handles_missing_gold_gracefully() -> None:
    record = HistoricalSolveRecord(
        problem_id="p_unlabeled",
        domain="algebra",
        difficulty="hard",
        gold_answer=None,
        predicted_answer="7",
        branch_traces=[
            _branch(
                problem_id="p_unlabeled",
                branch_id="b1",
                answer="7",
                branch_score=0.51,
                verifier_score=0.50,
                logical_consistency=0.55,
                symbolic_agreement=0.41,
                step_quality=0.48,
                prefix_quality=0.51,
            )
        ],
        route_snapshot={"branch_budget": 24},
    )
    result = mine_failures([record], config=FailureMiningConfig(min_hard_problems=1))
    assert result.manifest.failed_records == 0
    assert result.manifest.skipped_missing_gold == 1
    assert result.failures == ()


def test_failure_mining_replays_nested_final_prediction_payload() -> None:
    row = {
        "problem_id": "p_nested",
        "gold_answer": "42",
        "branch_traces": [
            {
                "branch_id": "b_nested_wrong",
                "answer": "17",
                "branch_score": 0.74,
                "verifier_score": 0.79,
                "logical_consistency": 0.77,
                "symbolic_agreement": 0.66,
                "step_quality": 0.41,
                "prefix_quality": 0.73,
                "symbolic_valid": True,
                "metadata": {"symbolic_status": "success"},
            },
            {
                "branch_id": "b_nested_correct",
                "answer": "42",
                "branch_score": 0.46,
                "verifier_score": 0.30,
                "logical_consistency": 0.49,
                "symbolic_agreement": 0.74,
                "step_quality": 0.69,
                "prefix_quality": 0.55,
                "symbolic_valid": True,
                "metadata": {"symbolic_status": "success"},
            },
        ],
        "final_prediction": {
            "final_answer": 17,
            "confidence": 0.83,
            "method_used": "hybrid",
            "signal_decomposition": {"logical_consistency": 0.77},
            "provenance": {
                "difficulty": "hard",
                "domain": "algebra",
                "route_snapshot": {"branch_budget": 28, "max_search_depth": 6, "critique_top_k": 2},
                "verifier_snapshot": {"status": "success"},
                "symbolic_snapshot": {"status": "success"},
            },
        },
    }
    result = mine_failures([row], config=FailureMiningConfig(min_hard_problems=1))
    assert len(result.failures) == 1
    failure = result.failures[0]
    assert failure.route_snapshot["branch_budget"] == 28
    assert failure.verifier_snapshot["status"] == "success"
    assert failure.symbolic_snapshot["selected_symbolic_status"] == "success"


def test_failure_mining_identifies_verifier_dominant_synthetic_failure() -> None:
    result = mine_failures([_verifier_failure_record()], config=FailureMiningConfig(min_hard_problems=1))
    assert len(result.failures) == 1
    failure = result.failures[0]
    assert failure.primary_failure is FailureBucket.VERIFIER_FAILURE
    assert failure.metadata["primary_subtype"] == "wrong_branch_scored_high"
    assert "aggregation_failure:correct_cluster_existed_but_not_selected" in failure.secondary_failures


def test_failure_mining_identifies_symbolic_dominant_synthetic_failure() -> None:
    result = mine_failures([_symbolic_failure_record()], config=FailureMiningConfig(min_hard_problems=1))
    assert len(result.failures) == 1
    failure = result.failures[0]
    assert failure.primary_failure is FailureBucket.SYMBOLIC_FAILURE
    assert failure.metadata["primary_subtype"] == "parser_syntax_issue"
    assert failure.symbolic_snapshot["parser_issue_detected"] is True


def test_failure_mining_accepts_minimal_prediction_records_jsonl(tmp_path) -> None:
    input_path = tmp_path / "records.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "problem_id": "p_min_1",
                        "problem": "Find the non-negative integer answer.",
                        "predicted_answer": "42",
                        "gold_answer": "42",
                        "metadata": {},
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "id": "p_min_2",
                        "question": "Find the non-negative integer answer.",
                        "answer": "17",
                        "expected_answer": "19",
                        "metadata": {},
                    },
                    sort_keys=True,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_failure_input_path(input_path)
    assert len(records) == 2
    assert records[0].raw_problem_text == "Find the non-negative integer answer."
    assert records[1].problem_id == "p_min_2"
    assert records[1].predicted_answer == "17"
    assert records[1].gold_answer == "19"
    assert records[1].verifier_snapshot["status"] == "unavailable"
    assert records[1].symbolic_snapshot["status"] == "unavailable"

    result = mine_failures(records, config=FailureMiningConfig(min_hard_problems=1))
    assert result.manifest.total_records == 2
    assert result.manifest.labeled_records == 2
    assert result.manifest.failed_records == 1
