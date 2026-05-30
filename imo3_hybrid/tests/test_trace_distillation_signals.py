
from src.common.schemas import BranchTrace, FailureType, ReasoningStep
from src.offline.trace_distillation import distill_traces


def test_trace_distillation_preserves_richer_signals_and_provenance() -> None:
    trace = BranchTrace(
        branch_id="b1",
        problem_id="p1",
        steps=[ReasoningStep(step_num=1, description="Reduce modulo 3", operator_used="modular_arithmetic", symbolic_valid=True)],
        full_reasoning="Reduce modulo 3 and verify the final candidate.",
        answer="42",
        verifier_score=0.81,
        symbolic_valid=True,
        logical_consistency=0.77,
        completeness=0.69,
        repairability=0.28,
        answer_correctness_likelihood=0.80,
        symbolic_agreement=0.84,
        step_quality=0.70,
        prefix_quality=0.72,
        prm_step_quality=0.68,
        prm_prefix_quality=0.71,
        retrieval_support=0.44,
        retrieval_compatibility=0.63,
        operator_reliability=0.59,
        discharge_fraction=0.50,
        open_obligation_burden=0.22,
        verifier_decomposition={"logical_consistency": 0.77, "completeness": 0.69},
        proof_obligations=[{"id": "obl1", "status": "open"}],
        failure_type=FailureType.LOGIC_ERROR,
        source_split="calibration",
        metadata={"route_compute_signals": {"proof_burden": 0.62}},
    )
    artifact, report = distill_traces([trace])
    assert report.kept_count == 1
    rec = artifact.records[0]
    assert rec.outcome.metadata["retrieval_compatibility"] == 0.63
    assert rec.outcome.metadata["prm_prefix_quality"] == 0.71
    assert rec.metadata["classification_basis"]["route_compute_signals"]["proof_burden"] == 0.62
    assert rec.provenance.metadata["normalization_metadata"]["proof_obligation_summary"]["count"] == 1