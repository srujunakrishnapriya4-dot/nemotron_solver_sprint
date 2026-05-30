from __future__ import annotations

import json

from scripts.calibrate_search import CalibrationScriptConfig, run_calibration
from src.common.schemas import BranchTrace, FailureType, ParsedProblem, ProblemDomain, ReasoningStep
from src.offline.cluster_and_mine import ClusterAndMineOrchestrator
from src.offline.training_data_builder import TrainingDataBuilderConfig, build_training_artifact
from src.routing.difficulty_estimator import DifficultyEstimator


def _trace(split: str) -> BranchTrace:
    return BranchTrace(
        branch_id=f"branch::{split}",
        problem_id=f"problem::{split}",
        steps=[ReasoningStep(step_num=1, description="Reduce modulo 3.", operator_used="modular_arithmetic", symbolic_valid=True)],
        full_reasoning="Reduce modulo 3 and verify the candidate.",
        answer="42",
        answer_canonical="42",
        failure_type=FailureType.LOGIC_ERROR if split == "calibration" else None,
        symbolic_valid=True,
        branch_score=0.78,
        verifier_score=0.81,
        tool_consistency=0.77,
        logical_consistency=0.76,
        completeness=0.72,
        repairability=0.31,
        answer_correctness_likelihood=0.80,
        symbolic_agreement=0.79,
        step_quality=0.69,
        prefix_quality=0.74,
        prm_prefix_quality=0.74,
        retrieval_support=0.43,
        retrieval_compatibility=0.61,
        operator_reliability=0.57,
        discharge_fraction=0.5,
        open_obligation_burden=0.18,
        verifier_decomposition={"logical_consistency": 0.76, "prefix_quality": 0.74, "retrieval_compatibility": 0.61},
        operator_sequence=["modular_arithmetic"],
        archetype_used="modular",
        retrieval_used=True,
        source_split=split,
        provenance={"source": "unit_test"},
        metadata={"note": split},
        proof_obligations=[{"obligation_id": f"obl::{split}", "originating_node_id": "root", "claim": "justify modular pruning", "evidence_kind_required": "symbolic_check", "status": "discharged", "discharged_by": ["sympy"], "notes": [], "source_constraint_ids": [], "metadata": {}}],
    )


def test_offline_builders_preserve_richer_fields() -> None:
    traces = [_trace("train"), _trace("valid")]
    artifact, _ = build_training_artifact(traces, config=TrainingDataBuilderConfig())
    calibration_records = [record for record in artifact.records if record.task_family.value == "calibration"]
    assert calibration_records
    features = calibration_records[0].feature_fields
    assert "decomposed_signals" in features
    assert "proof_obligation_summary" in features
    assert features["retrieval_compatibility"] > 0.0

    mining = ClusterAndMineOrchestrator().run(traces)
    assert mining.records[0].retrieval_compatibility > 0.0
    assert mining.records[0].operator_reliability > 0.0
    assert "verifier_decomposition" in mining.records[0].metadata


def test_calibration_emits_split_safe_robust_summary(tmp_path) -> None:
    rows = [
        {
            "problem_id": "p_train",
            "gold_answer": "42",
            "raw_text": "Find the answer.",
            "source_metadata": {"split": "train"},
            "branches": [_trace("train").model_dump(mode="json")],
        },
        {
            "problem_id": "p_valid",
            "gold_answer": "42",
            "raw_text": "Find the answer.",
            "source_metadata": {"split": "valid"},
            "branches": [_trace("valid").model_dump(mode="json")],
        },
    ]
    input_path = tmp_path / "examples.json"
    input_path.write_text(json.dumps(rows), encoding="utf-8")
    status, manifest = run_calibration(
        CalibrationScriptConfig(
            input_paths=[str(input_path)],
            output_dir=str(tmp_path / "out"),
            overwrite=True,
            verifier_thresholds=[0.5],
            branch_budgets=[1],
            entropy_betas=[0.2],
            min_branch_scores=[0.1],
            retrieval_compatibility_floors=[0.0],
            repairability_biases=[0.0],
        )
    )
    assert status.value == "completed"
    assert manifest.best_summary["split_summaries"]["train"]["accuracy"] >= 0.0
    assert manifest.split_summary["valid"]["accuracy"] >= 0.0
    assert "robustness_penalty" in manifest.best_summary


def test_difficulty_estimator_exports_compute_signals_and_bounded_budget() -> None:
    estimate = DifficultyEstimator().estimate(
        ParsedProblem(
            problem_id="p_route_budget",
            raw_text=(
                "Find all positive integers n such that a modular condition holds, "
                "prove the remaining cases, and justify every residue-class elimination."
            ),
            constraints=[
                "n is a positive integer",
                "n satisfies a modular condition",
                "justify every eliminated case",
            ],
            knowns=["n is positive"],
            unknowns=["n"],
            domain=ProblemDomain.NUMBER_THEORY,
            target="find all n",
            answer_type="positive_integer",
            difficulty_seed=0.58,
            likely_archetypes=["modular", "case_work", "contradiction"],
            parity_cues=["parity"],
            symmetries=["residue symmetry"],
            integrality_constraints=["n is integer"],
            proof_targets=["justify modular pruning", "prove completeness"],
            proof_obligation_hints=["symbolic_check", "case_split"],
            parse_quality={"overall_confidence": 0.72},
        )
    )
    compute_signals = estimate.diagnostics.get("compute_signals", {})
    assert compute_signals["difficulty_intensity"] > 0.0
    assert compute_signals["proof_burden"] > 0.0
    assert estimate.budget_plan.branch_budget > 0
    assert estimate.budget_plan.self_consistency_samples >= estimate.budget_plan.branch_budget
    assert 0.0 <= estimate.budget_plan.repair_aggressiveness <= 1.0
    assert 0.0 <= estimate.budget_plan.resample_aggressiveness <= 1.0
    assert 0.0 <= estimate.budget_plan.critique_aggressiveness <= 1.0