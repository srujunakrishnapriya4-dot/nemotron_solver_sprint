from __future__ import annotations

import json
from pathlib import Path

import pytest

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.evaluation import (
    PrivateLikeReport,
    PromotionDecision,
    PromotionGateReport,
    SubmissionExactReport,
    TransferEvaluationReport,
    TransferSliceResult,
)
from nemotron_engine.inference_eval.evaluation_manifest import (
    InferenceEvaluationManifest,
    compute_inference_evaluation_manifest_id,
)
from nemotron_engine.packaging import FileEntry, PackageBuildReport, PreflightGateReport, SubmissionManifest
from nemotron_engine.packaging.submission_manifest import compute_package_hash
from nemotron_engine.release import ReleaseCandidateReport
from nemotron_engine.rehearsal import (
    AdapterEvidenceBundle,
    PackageRehearsalConfig,
    RehearsalConfig,
    RuntimeRehearsalConfig,
    build_adapter_evidence_bundle,
    build_final_rehearsal_manifest,
    build_promotion_rehearsal_report,
    run_package_rehearsal,
    run_runtime_rehearsal,
    run_submission_rehearsal,
)
from nemotron_engine.rehearsal.rehearsal_config import RehearsalConfigError
from nemotron_engine.training_backend import AdapterValidationReport, TrainingRunManifest, validate_adapter_output
from nemotron_engine.training_backend.run_manifest import compute_training_run_manifest_id


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def make_config(**kwargs: object) -> RehearsalConfig:
    payload = {
        "rehearsal_id": "pass14-rehearsal",
        "dry_run": True,
        "allow_zip_build": False,
        "allow_runtime_unloaded": True,
        "allow_warnings": True,
        "require_smoke_evidence": False,
        "require_completion_evaluation": True,
        "require_private_like": True,
        "require_release_candidate": False,
        "output_dir": None,
        "package_name": "dry_run_package.zip",
        "seed": 14,
        "metadata": {},
    }
    payload.update(kwargs)
    return RehearsalConfig(**payload)


def make_adapter_report(tmp_path: Path | None = None) -> AdapterValidationReport:
    if tmp_path is not None:
        adapter_root = tmp_path / "adapter"
        adapter_root.mkdir(parents=True, exist_ok=True)
        (adapter_root / "adapter_config.json").write_text(json.dumps({"r": 8}) + "\n", encoding="utf-8")
        (adapter_root / "adapter_model.safetensors").write_bytes(b"pass14-temp-adapter-artifact")
        return validate_adapter_output(adapter_root)
    adapter_dir = "validated-adapter"
    hashes = {"adapter_config.json": SHA_A, "adapter_model.safetensors": SHA_B}
    adapter_hash = stable_hash({"artifact_hashes": hashes, "rank": 8})
    return AdapterValidationReport(
        adapter_dir=adapter_dir,
        passed=True,
        adapter_config_valid=True,
        model_file_present=True,
        adapter_model_files=("adapter_model.safetensors",),
        artifact_hashes=hashes,
        adapter_hash=adapter_hash,
        rank=8,
    )


def make_training_manifest(adapter_report: AdapterValidationReport | None = None, *, adapter_hash: str | None = None) -> TrainingRunManifest:
    report = adapter_report or make_adapter_report()
    payload = {
        "invocation_hash": SHA_A,
        "backend_result_hash": SHA_B,
        "adapter_validation_hash": report.report_hash,
        "training_log_hash": SHA_C,
        "training_plan_hash": "training-plan-hash",
        "lora_config_hash": "lora-config-hash",
        "dataset_manifest_hash": "dataset-manifest-hash",
        "stage": "sft",
        "trained": True,
        "adapter_hash": adapter_hash or report.adapter_hash,
    }
    payload["manifest_id"] = compute_training_run_manifest_id(payload)
    return TrainingRunManifest(artifact_hashes=report.artifact_hashes, metadata={}, **payload)


def make_adapter_bundle(tmp_path: Path | None = None) -> AdapterEvidenceBundle:
    report = make_adapter_report(tmp_path)
    return build_adapter_evidence_bundle(adapter_validation_report=report, lora_config_hash="lora-config-hash")


def make_submission_exact(passed: bool = True) -> SubmissionExactReport:
    errors = () if passed else ("submission_exact_failed",)
    return SubmissionExactReport(
        config_hash="submission-exact-config",
        serving_config_hash="serving-config-hash",
        passed=passed,
        dry_run=True,
        runtime_success=False,
        required_hashes_present=passed,
        adapter_checked=True,
        adapter_rank=8,
        critical_serving_fields={
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 512,
            "stop": (),
            "num_samples": 1,
            "majority_vote": False,
            "batch_size": 1,
            "prompt_template_hash": "prompt",
            "tokenizer_hash": "tokenizer",
            "model_hash": "model",
            "adapter_hash": "adapter",
        },
        errors=errors,
    )


def make_transfer_report(passed: bool = True) -> TransferEvaluationReport:
    slice_result = TransferSliceResult(
        slice_name="private_like",
        total=2,
        correct=2 if passed else 1,
        accuracy=1.0 if passed else 0.5,
        baseline_accuracy=1.0,
        transfer_gap=0.0 if passed else 0.5,
        passed=passed,
        failures=() if passed else ("private_like:transfer_gap",),
        example_ids=("a", "b"),
    )
    return TransferEvaluationReport(
        config_hash="transfer-config",
        total_examples=2,
        slice_results=(slice_result,),
        overall_accuracy=1.0 if passed else 0.5,
        passed=passed,
        failure_reasons=() if passed else ("slice_failed:private_like",),
        required_slices=("private_like",),
        allow_small_slices=True,
    )


def make_private_like_report(passed: bool = True) -> PrivateLikeReport:
    return PrivateLikeReport(
        total=2,
        correct=2 if passed else 1,
        accuracy=1.0 if passed else 0.5,
        baseline_accuracy=1.0,
        regression_vs_baseline=0.0 if passed else 0.5,
        format_error_rate=0.0,
        extraction_error_rate=0.0,
        contamination_count=0,
        passed=passed,
        failure_reasons=() if passed else ("accuracy_below_threshold",),
        example_ids=("a", "b"),
    )


def make_inference_manifest(passed: bool = True) -> InferenceEvaluationManifest:
    payload = {
        "prompt_batch_hash": "prompt-batch",
        "serving_config_hash": "serving-config-hash",
        "inference_invocation_hash": "inference-invocation",
        "inference_result_hash": "inference-result",
        "completion_capture_hash": "completion-capture",
        "transfer_report_hash": "completion-transfer" if passed else None,
        "private_like_report_hash": "completion-private-like" if passed else None,
        "non_submission_exact": False,
        "backend_kind": "external",
        "dry_run": False,
        "evaluated": passed,
        "passed": passed,
    }
    payload["manifest_id"] = compute_inference_evaluation_manifest_id(payload)
    return InferenceEvaluationManifest(errors=() if passed else ("inference_failed",), warnings=(), metadata={}, **payload)


def make_promotion_report(adapter_hash: str, *, passed: bool = True) -> PromotionGateReport:
    return PromotionGateReport(
        decision=PromotionDecision.ALLOW if passed else PromotionDecision.REJECT,
        passed=passed,
        failure_reasons=() if passed else ("transfer_failed",),
        input_report_hashes={
            "submission_exact_report": make_submission_exact().report_hash,
            "transfer_report": make_transfer_report().report_hash,
            "private_like_report": make_private_like_report().report_hash,
        },
        required_hashes={
            "training_plan_hash": "training-plan-hash",
            "dataset_manifest_hash": "dataset-manifest-hash",
            "trace_manifest_hash": "trace-manifest-hash",
            "lora_config_hash": "lora-config-hash",
            "adapter_hash": adapter_hash,
        },
        adapter_rank=8,
    )


def make_bridge(tmp_path: Path | None = None):
    config = make_config()
    bundle = make_adapter_bundle(tmp_path)
    return build_promotion_rehearsal_report(
        promotion_gate_report=make_promotion_report(bundle.adapter_hash),
        submission_exact_report=make_submission_exact(),
        transfer_report=make_transfer_report(),
        private_like_report=make_private_like_report(),
        inference_evaluation_manifest=make_inference_manifest(),
        adapter_evidence_bundle=bundle,
        rehearsal_config=config,
    )


def make_package_report(config: RehearsalConfig, bundle: AdapterEvidenceBundle, bridge) :
    return run_package_rehearsal(
        rehearsal_config=config,
        adapter_evidence_bundle=bundle,
        promotion_bridge_report=bridge,
        submission_manifest=make_submission_manifest(bundle, bridge),
        config=PackageRehearsalConfig(),
    )


def make_runtime_report(config: RehearsalConfig, bundle: AdapterEvidenceBundle, package_report):
    return run_runtime_rehearsal(
        rehearsal_config=config,
        adapter_evidence_bundle=bundle,
        package_rehearsal_report=package_report,
        runtime_validation_report=None,
        config=RuntimeRehearsalConfig(expected_adapter_hash=bundle.adapter_hash),
    )


def make_submission_manifest(bundle: AdapterEvidenceBundle, bridge) -> SubmissionManifest:
    entry = FileEntry(path="run.py", size_bytes=1, sha256=SHA_D, required=True, role="code")
    files = (entry,)
    package_hash = compute_package_hash(files)
    manifest_id = stable_hash(
        {
            "package_name": "pkg",
            "created_by": "test",
            "package_hash": package_hash,
            "serving_config_hash": "serving-config-hash",
            "promotion_decision_hash": bridge.promotion_decision_hash,
            "training_plan_hash": "training-plan-hash",
            "dataset_manifest_hash": "dataset-manifest-hash",
            "trace_manifest_hash": "trace-manifest-hash",
            "lora_config_hash": "lora-config-hash",
            "adapter_hash": bundle.adapter_hash,
        }
    )
    return SubmissionManifest(
        manifest_id=manifest_id,
        package_name="pkg",
        created_by="test",
        files=files,
        serving_config_hash="serving-config-hash",
        lora_config_hash="lora-config-hash",
        adapter_hash=bundle.adapter_hash,
        promotion_decision_hash=bridge.promotion_decision_hash,
        training_plan_hash="training-plan-hash",
        dataset_manifest_hash="dataset-manifest-hash",
        trace_manifest_hash="trace-manifest-hash",
        package_hash=package_hash,
    )


def make_preflight(passed: bool = True) -> PreflightGateReport:
    return PreflightGateReport(
        passed=passed,
        failure_reasons=() if passed else ("preflight_failed",),
        input_report_hashes={"x": "y"},
        required_hashes={"x": "y"},
        warnings_allowed=True,
    )


def make_release(ready: bool = True) -> ReleaseCandidateReport:
    return ReleaseCandidateReport(
        candidate_id="candidate",
        repository_root=None,
        locked_registry_hash="locked",
        system_audit_hash="audit",
        scoped_suite_result_summary="787 passed, 2 skipped",
        ready=ready,
        release_level="submission_dry_run_candidate",
        blockers=() if ready else ("blocker",),
        system_audit_passed=ready,
    )


def test_end_to_end_dry_run_rehearsal_is_deterministic_and_no_package_written(tmp_path: Path) -> None:
    config = make_config()
    bundle = make_adapter_bundle(tmp_path)
    bridge = build_promotion_rehearsal_report(
        promotion_gate_report=make_promotion_report(bundle.adapter_hash),
        submission_exact_report=make_submission_exact(),
        transfer_report=make_transfer_report(),
        private_like_report=make_private_like_report(),
        inference_evaluation_manifest=make_inference_manifest(),
        adapter_evidence_bundle=bundle,
        rehearsal_config=config,
    )
    submission_manifest = make_submission_manifest(bundle, bridge)
    package = run_package_rehearsal(
        rehearsal_config=config,
        adapter_evidence_bundle=bundle,
        promotion_bridge_report=bridge,
        submission_manifest=submission_manifest,
    )
    runtime = run_runtime_rehearsal(
        rehearsal_config=config,
        adapter_evidence_bundle=bundle,
        package_rehearsal_report=package,
        runtime_validation_report=None,
    )
    submission = run_submission_rehearsal(
        rehearsal_config=config,
        promotion_bridge_report=bridge,
        package_rehearsal_report=package,
        runtime_rehearsal_report=runtime,
    )
    manifest = build_final_rehearsal_manifest(
        rehearsal_config=config,
        adapter_evidence_bundle=bundle,
        promotion_bridge_report=bridge,
        package_rehearsal_report=package,
        runtime_rehearsal_report=runtime,
        submission_rehearsal_report=submission,
    )
    manifest2 = build_final_rehearsal_manifest(
        rehearsal_config=config,
        adapter_evidence_bundle=bundle,
        promotion_bridge_report=bridge,
        package_rehearsal_report=package,
        runtime_rehearsal_report=runtime,
        submission_rehearsal_report=submission,
    )
    assert package.built is False
    assert package.package_path is None
    assert runtime.unloaded_dry_run is True
    assert submission.submitted is False
    assert manifest.ready_for_external_submission is True
    assert manifest.manifest_hash == manifest2.manifest_hash
    assert package.package_path is None


def test_end_to_end_fake_kaggle_metadata_blocks() -> None:
    bundle = make_adapter_bundle()
    with pytest.raises(RehearsalConfigError):
        make_config(metadata={"kaggle_success": True})
    with pytest.raises(RehearsalConfigError):
        make_config(metadata={"leaderboard_ready": True})
    with pytest.raises(RehearsalConfigError):
        make_config(metadata={"score": "95+ guaranteed"})
