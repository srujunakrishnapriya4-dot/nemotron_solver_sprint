from __future__ import annotations

import json

from src.branches.branch_controller import VerifierHookResult
from src.branches.branch_state import BranchPhase
from src.common.schemas import VerifierLabel
from src.prm import ProcessObligationInput, ProcessScoreInput, ProcessStepInput, score_process
from src.verifier import scoring
from src.verifier.deepseek_verifier import (
    DeepSeekVerifier,
    PackedReasoningStep,
    PackedProofObligation,
    StepVerificationResult,
    VerifierConfig,
    VerifierHookPayload,
    VerifierOutput,
    VerifierRuntimeStatus,
    VerifierTraceInput,
)
from src.verifier.distilled_verifier import (
    DistilledVerifier,
    DistilledVerifierConfig,
    PackedVerifierInput,
    PackedVerifierStep,
    VerifierArtifactState,
    VerifierExecutionMode,
    VerifierScoreBundle,
)


class _StaticModel:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt: str) -> str:
        return self.response


def _trace_input() -> VerifierTraceInput:
    return VerifierTraceInput(
        problem_id="imo3-verifier-trace",
        branch_id="branch-main",
        raw_problem_text="Find the non-negative integer answer to a divisibility problem.",
        target="final_answer",
        domain="number_theory",
        route_problem_type={"number_theory": 0.78, "algebra": 0.22},
        route_archetypes={"modular": 0.71, "invariant": 0.29},
        route_uncertainty=0.19,
        node_summary="Track modular residues and close the remaining cases.",
        reasoning=(
            "Assume n is minimal. Reduce modulo 3, verify the candidate family, "
            "and then check the final answer against the divisibility constraint."
        ),
        answer="42",
        answer_canonical="42",
        generation_confidence=0.73,
        symbolic_passed=True,
        symbolic_score=0.81,
        symbolic_summary="mod-3 and direct substitution checks passed",
        exact_symbolic_match=True,
        retrieval_support=0.34,
        branch_phase=BranchPhase.REASONING.value,
        steps=[
            PackedReasoningStep(
                step_index=1,
                description="Reduce the condition modulo 3.",
                operator_used="modular_arithmetic",
                symbolic_valid=True,
            ),
            PackedReasoningStep(
                step_index=2,
                description="Eliminate impossible residues and keep the viable branch.",
                operator_used="case_work",
                symbolic_valid=True,
            ),
            PackedReasoningStep(
                step_index=3,
                description="Substitute the candidate answer and verify the original constraint.",
                operator_used="substitution",
                symbolic_valid=True,
            ),
        ],
    )


def _packed_input(
    *,
    symbolic_score: float = 0.82,
    symbolic_passed: bool = True,
    symbolic_exact_match: bool = False,
) -> PackedVerifierInput:
    return PackedVerifierInput(
        problem_id="imo3-packed-problem",
        branch_id="branch-packed",
        branch_phase=BranchPhase.REASONING.value,
        domain="number_theory",
        difficulty="hard",
        route_uncertainty=0.22,
        repair_threshold=0.58,
        answer_raw="42",
        answer_canonical="42",
        answer_in_range=True,
        reasoning_excerpt="Use modular arithmetic to prune cases, then verify the surviving candidate.",
        step_count=3,
        retrieval_count=1,
        symbolic_count=1,
        verifier_count=0,
        critique_count=0,
        repair_count=0,
        candidate_confidence=0.69,
        generation_confidence=0.66,
        latest_operator="modular_arithmetic",
        operator_sequence=("modular_arithmetic", "case_work", "substitution"),
        constraints=("n is a non-negative integer", "n mod 3 determines the feasible residue"),
        invariants=("residue class preserved",),
        goals=("final_answer", "answer_type::non_negative_integer"),
        steps=(
            PackedVerifierStep(
                index=0,
                description="Reduce modulo 3.",
                operator_name="modular_arithmetic",
                phase=BranchPhase.REASONING.value,
                state_fingerprint="state-0",
                summary_text="mod reduction",
            ),
            PackedVerifierStep(
                index=1,
                description="Split the remaining cases.",
                operator_name="case_work",
                phase=BranchPhase.REASONING.value,
                state_fingerprint="state-1",
                summary_text="case split",
            ),
            PackedVerifierStep(
                index=2,
                description="Substitute the candidate answer.",
                operator_name="substitution",
                phase=BranchPhase.REASONING.value,
                state_fingerprint="state-2",
                summary_text="direct check",
            ),
        ),
        symbolic_passed=symbolic_passed,
        symbolic_score=symbolic_score,
        symbolic_exact_match=symbolic_exact_match,
        symbolic_summary="symbolic validation summary",
        retrieval_support=0.27,
        retrieval_answer_support=0.35,
        prior_confidence=0.52,
        existing_verifier_probability=0.0,
        contradiction_flag=False,
        truncation_notes=(),
        metadata={"fixture": "deterministic"},
    )


def test_deepseek_verifier_parses_structured_model_output_into_typed_label_and_hook() -> None:
    trace = _trace_input()
    verifier = DeepSeekVerifier(
        model=_StaticModel(
            json.dumps(
                {
                    "logical_consistency": 0.91,
                    "symbolic_agreement": 0.88,
                    "completeness": 0.79,
                    "answer_correctness_likelihood": 0.84,
                    "repairability": 0.21,
                    "failure_type": "none",
                    "repair_type": "none",
                    "summary_tags": ["consistent", "symbolic support"],
                    "step_correctness": [1, 1, 1],
                    "evidence": [
                        {
                            "kind": "symbolic",
                            "message": "symbolic checks agree",
                            "step_index": 3,
                            "score": 0.91,
                            "source": "unit_test",
                        }
                    ],
                    "overall_score": 0.86,
                }
            )
        ),
        config=VerifierConfig(),
    )

    output = verifier.verify_trace(trace)
    label = output.to_label()
    hook = output.to_hook_payload()

    assert isinstance(output, VerifierOutput)
    assert output.runtime_status is VerifierRuntimeStatus.MODEL
    assert output.degraded_mode is False
    assert all(isinstance(step, StepVerificationResult) for step in output.step_results)
    assert output.step_correctness == [1, 1, 1]

    assert isinstance(label, VerifierLabel)
    assert label.branch_id == trace.branch_id
    assert label.step_correctness == [1, 1, 1]
    assert label.logical_consistency == output.logical_consistency
    assert label.symbolic_agreement == output.symbolic_agreement
    assert label.final_answer_correct_probability == output.answer_correctness_likelihood

    assert isinstance(hook, VerifierHookPayload)
    assert hook.probability == output.answer_correctness_likelihood
    assert hook.branch_score == output.overall_score
    assert hook.metadata["runtime_status"] == VerifierRuntimeStatus.MODEL.value
    assert hook.metadata["branch_score"] == output.overall_score


def test_deepseek_unavailable_model_fallback_is_explicit_and_schema_rich() -> None:
    trace = _trace_input()
    output = DeepSeekVerifier(model=None, config=VerifierConfig(enable_model=True)).verify_trace(trace)

    assert output.runtime_status is VerifierRuntimeStatus.MODEL_UNAVAILABLE
    assert output.degraded_mode is True
    assert output.metadata["fallback_reason"] == "model_not_configured"
    assert output.metadata["step_count"] == len(trace.steps)
    assert output.probability == output.answer_correctness_likelihood
    assert output.symbolic_agreement != output.logical_consistency
    assert output.summary.startswith("verifier::model_unavailable|")
    assert output.evidence_bundle.runtime_status is VerifierRuntimeStatus.MODEL_UNAVAILABLE

    label = output.to_label()
    hook = output.to_hook_payload()

    assert isinstance(label, VerifierLabel)
    assert isinstance(hook, VerifierHookPayload)
    assert hook.metadata["degraded_mode"] is True
    assert hook.metadata["overall_score"] == output.overall_score


def test_deepseek_parse_error_fallback_is_explicit_and_deterministic() -> None:
    trace = _trace_input()
    verifier = DeepSeekVerifier(model=_StaticModel("not valid json"), config=VerifierConfig())

    first = verifier.verify_trace(trace)
    second = verifier.verify_trace(trace)

    assert first.runtime_status is VerifierRuntimeStatus.PARSE_ERROR
    assert first.degraded_mode is True
    assert first.metadata["fallback_reason"].startswith("parse_error::")
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert isinstance(first.to_label(), VerifierLabel)


def test_distilled_verifier_missing_artifact_fallback_is_explicit_and_branch_controller_ready() -> None:
    packed = _packed_input()
    verifier = DistilledVerifier(
        DistilledVerifierConfig(
            checkpoint_path="C:/missing/distilled-verifier-manifest.json",
            search_default_artifacts=False,
            allow_fallback=True,
        )
    )

    first = verifier.verify_packed(packed)
    second = verifier.verify_packed(packed)
    hook = first.to_branch_controller_hook_result()
    scoring_payload = first.to_scoring_payload()
    diagnostic_codes = {item.code for item in first.diagnostics}

    assert isinstance(first, VerifierScoreBundle)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.mode is VerifierExecutionMode.FALLBACK
    assert first.artifact_state is VerifierArtifactState.MISSING
    assert "missing_verifier_artifact" in diagnostic_codes
    assert "verifier_fallback_active" in diagnostic_codes
    assert isinstance(first.label, VerifierLabel)

    assert isinstance(hook, VerifierHookResult)
    assert hook.probability == first.probability
    assert hook.branch_score == first.branch_score
    assert hook.step_quality == first.step_quality
    assert hook.prefix_quality == first.prefix_quality
    assert hook.metadata["artifact_state"] == VerifierArtifactState.MISSING.value
    assert hook.metadata["mode"] == VerifierExecutionMode.FALLBACK.value
    assert hook.metadata["label"]["branch_id"] == packed.branch_id

    assert scoring_payload["branch_id"] == packed.branch_id
    assert scoring_payload["symbolic_consistency"] == first.symbolic_consistency
    assert scoring_payload["logical_consistency"] == first.logical_consistency
    assert scoring_payload["repairability"] == first.repairability


def test_distilled_fallback_keeps_symbolic_signal_separable_in_scoring() -> None:
    verifier = DistilledVerifier(
        DistilledVerifierConfig(
            checkpoint_path="C:/missing/distilled-verifier-manifest.json",
            search_default_artifacts=False,
            allow_fallback=True,
        )
    )
    weak_symbolic = verifier.verify_packed(
        _packed_input(symbolic_score=0.14, symbolic_passed=False, symbolic_exact_match=False)
    )
    strong_symbolic = verifier.verify_packed(
        _packed_input(symbolic_score=0.97, symbolic_passed=True, symbolic_exact_match=True)
    )

    assert weak_symbolic.logical_consistency == strong_symbolic.logical_consistency
    assert strong_symbolic.symbolic_consistency > weak_symbolic.symbolic_consistency
    assert strong_symbolic.probability > weak_symbolic.probability
    assert strong_symbolic.branch_score > weak_symbolic.branch_score
    assert weak_symbolic.label.symbolic_agreement == weak_symbolic.symbolic_consistency
    assert strong_symbolic.label.symbolic_agreement == strong_symbolic.symbolic_consistency


def test_scoring_module_exposes_multi_component_bundle_and_aggregation_features() -> None:
    bundle = scoring.compute_verifier_score(
        {
            "problem_id": "imo3-score-problem",
            "branch_id": "score-branch",
            "model_confidence": 0.61,
            "logical_consistency": 0.74,
            "symbolic_agreement": 0.57,
            "completeness": 0.68,
            "answer_correctness_likelihood": 0.72,
            "repairability": 0.31,
            "step_quality": 0.63,
            "prefix_quality": 0.67,
            "open_obligation_burden": 0.18,
            "route_uncertainty": 0.26,
            "mode": "llm_backed",
            "artifact_state": "ready",
        }
    )

    component_names = {component.name.value for component in bundle.components}
    aggregation_features = bundle.to_aggregation_features()
    branch_breakdown = bundle.to_branch_score_breakdown(answer_agreement=0.41, branch_novelty=0.27)

    assert isinstance(bundle, scoring.VerifierScoreBundle)
    assert len(bundle.components) >= 6
    assert component_names >= {
        "logical_consistency",
        "symbolic_agreement",
        "completeness",
        "answer_correctness_likelihood",
        "repairability",
        "step_quality",
        "prefix_quality",
        "uncertainty",
    }
    assert bundle.summary.startswith("verdict=")
    assert aggregation_features["symbolic_agreement"] == bundle.symbolic_agreement
    assert aggregation_features["logical_consistency"] == bundle.logical_consistency
    assert aggregation_features["step_quality"] == bundle.step_quality
    assert aggregation_features["prefix_quality"] == bundle.prefix_quality
    assert branch_breakdown.verifier_probability == bundle.answer_correctness_likelihood
    assert branch_breakdown.tool_consistency == bundle.symbolic_agreement
    assert branch_breakdown.step_quality == bundle.step_quality
    assert branch_breakdown.prefix_quality == bundle.prefix_quality


def test_scoring_penalizes_smooth_but_unsupported_branch_below_supported_branch() -> None:
    smooth_unsupported = scoring.compute_verifier_score(
        {
            "problem_id": "imo3-score-problem",
            "branch_id": "smooth-unsupported",
            "model_confidence": 0.84,
            "logical_consistency": 0.86,
            "symbolic_agreement": 0.72,
            "completeness": 0.79,
            "answer_correctness_likelihood": 0.88,
            "repairability": 0.22,
            "step_quality": 0.31,
            "prefix_quality": 0.27,
            "open_obligation_burden": 0.62,
            "route_uncertainty": 0.18,
            "symbolic_status": "unsupported",
            "symbolic_partial_support": False,
            "quality_tag": scoring.ReasoningQualityTag.WRONG_ANSWER_PLAUSIBLE_REASONING,
        }
    )
    supported = scoring.compute_verifier_score(
        {
            "problem_id": "imo3-score-problem",
            "branch_id": "supported-branch",
            "model_confidence": 0.76,
            "logical_consistency": 0.72,
            "symbolic_agreement": 0.79,
            "completeness": 0.70,
            "answer_correctness_likelihood": 0.74,
            "repairability": 0.26,
            "step_quality": 0.71,
            "prefix_quality": 0.78,
            "open_obligation_burden": 0.12,
            "route_uncertainty": 0.16,
            "symbolic_status": "success",
            "quality_tag": scoring.ReasoningQualityTag.PARTIAL_CORRECT_PREFIX,
        }
    )

    assert smooth_unsupported.branch_score < supported.branch_score
    assert smooth_unsupported.metadata["smooth_unsupported_penalty"] > 0.0
    assert smooth_unsupported.metadata["shallow_penalty"] > 0.0
    assert smooth_unsupported.flags.strong_pass is False


def test_scoring_penalizes_symbolic_contradiction_and_contradicted_obligations() -> None:
    contradicted = scoring.compute_verifier_score(
        {
            "problem_id": "imo3-score-problem",
            "branch_id": "contradicted-branch",
            "model_confidence": 0.82,
            "logical_consistency": 0.88,
            "symbolic_agreement": 0.91,
            "completeness": 0.84,
            "answer_correctness_likelihood": 0.87,
            "repairability": 0.20,
            "step_quality": 0.79,
            "prefix_quality": 0.83,
            "open_obligation_burden": 0.18,
            "route_uncertainty": 0.10,
            "symbolic_status": "contradiction",
            "contradicted_obligation_count": 1,
            "quality_tag": scoring.ReasoningQualityTag.UNKNOWN,
        }
    )

    assert contradicted.symbolic_agreement <= 0.02
    assert contradicted.answer_correctness_likelihood <= 0.20
    assert contradicted.open_obligation_burden >= 0.55
    assert contradicted.branch_score <= 0.20
    assert contradicted.verdict in {
        scoring.VerifierVerdict.REJECT,
        scoring.VerifierVerdict.ESCALATE,
    }


def test_deepseek_fallback_tracks_typed_proof_obligations() -> None:
    base_trace = _trace_input()
    typed_trace = base_trace.model_copy(
        update={
            "proof_obligations": [
                PackedProofObligation(
                    obligation_id="obl-open",
                    claim="justify the modular pruning step",
                    evidence_kind_required="symbolic_check",
                    status="open",
                    originating_node_id="root::imo3-verifier-trace",
                ),
                PackedProofObligation(
                    obligation_id="obl-contradicted",
                    claim="the current branch remains consistent",
                    evidence_kind_required="symbolic_check",
                    status="contradicted",
                    originating_node_id="root::imo3-verifier-trace",
                ),
            ]
        }
    )

    output = DeepSeekVerifier(model=None, config=VerifierConfig(enable_model=True)).verify_trace(typed_trace)

    assert output.runtime_status is VerifierRuntimeStatus.MODEL_UNAVAILABLE
    assert output.failure_type == "symbolic_mismatch"
    assert output.metadata["proof_obligation_count"] == 2
    assert output.metadata["open_proof_obligation_count"] == 1
    assert output.metadata["contradicted_proof_obligation_count"] == 1
    assert output.step_quality > 0.0
    assert output.prefix_quality > 0.0
    assert output.open_obligation_burden >= 0.5
    assert output.to_hook_payload().metadata["verifier_decomposition"]["prefix_quality"] == output.prefix_quality
    assert any("formal contradiction" in tag for tag in output.evidence_bundle.summary_tags)


def test_prm_scores_step_and_prefix_quality_from_typed_process_state() -> None:
    summary = score_process(
        ProcessScoreInput(
            problem_id="imo3-prm",
            branch_id="branch-prm",
            steps=(
                ProcessStepInput(
                    step_index=1,
                    description="Reduce the expression modulo 3.",
                    operator_name="modular_arithmetic",
                    symbolic_valid=True,
                    proof_obligation_ids=("obl-1",),
                ),
                ProcessStepInput(
                    step_index=2,
                    description="Assume the remaining case is obvious.",
                    operator_name="case_work",
                    symbolic_valid=False,
                    proof_obligation_ids=("obl-2",),
                ),
            ),
            proof_obligations=(
                ProcessObligationInput(obligation_id="obl-1", status="discharged"),
                ProcessObligationInput(obligation_id="obl-2", status="open"),
            ),
            symbolic_score=0.78,
            symbolic_passed=True,
            retrieval_support=0.22,
            answer_present=True,
        )
    )

    assert len(summary.step_scores) == 2
    assert summary.step_scores[0].step_quality > summary.step_scores[1].step_quality
    assert summary.step_scores[0].prefix_quality >= summary.step_scores[0].step_quality
    assert summary.open_obligation_burden > 0.0
    assert 0.0 <= summary.branch_search_value <= 1.0
