from __future__ import annotations


def test_pass14_api_imports() -> None:
    from nemotron_engine.rehearsal import (  # noqa: F401
        AdapterEvidenceBundle,
        FinalRehearsalManifest,
        PackageRehearsalConfig,
        PackageRehearsalReport,
        PromotionBridgeConfig,
        PromotionBridgeReport,
        RehearsalConfig,
        RuntimeRehearsalConfig,
        RuntimeRehearsalReport,
        SubmissionRehearsalConfig,
        SubmissionRehearsalReport,
        build_adapter_evidence_bundle,
        build_final_rehearsal_manifest,
        build_promotion_rehearsal_report,
        compute_adapter_evidence_hash,
        compute_final_rehearsal_manifest_hash,
        compute_rehearsal_config_hash,
        run_package_rehearsal,
        run_runtime_rehearsal,
        run_submission_rehearsal,
        validate_adapter_evidence_bundle,
        validate_final_rehearsal_manifest,
        validate_package_rehearsal_report,
        validate_promotion_rehearsal_report,
        validate_rehearsal_config,
        validate_runtime_rehearsal_report,
        validate_submission_rehearsal_report,
    )


def test_representative_pass1_to_pass13_imports_still_work() -> None:
    from nemotron_engine.core.schemas import CanonicalProblem, stable_hash  # noqa: F401
    from nemotron_engine.data import assign_splits  # noqa: F401
    from nemotron_engine.evaluation import PromotionGateReport, evaluate_promotion  # noqa: F401
    from nemotron_engine.inference_eval import InferenceEvaluationManifest  # noqa: F401
    from nemotron_engine.packaging import PreflightGateReport, RuntimeValidationReport  # noqa: F401
    from nemotron_engine.programs import execute_program  # noqa: F401
    from nemotron_engine.release import ReleaseCandidateReport, SystemAuditReport  # noqa: F401
    from nemotron_engine.runtime import ServingConfig  # noqa: F401
    from nemotron_engine.scoring import score_completion  # noqa: F401
    from nemotron_engine.smoke_training import SmokeTrainingReport  # noqa: F401
    from nemotron_engine.solvers import solve_numeric  # noqa: F401
    from nemotron_engine.training import LoRAConfig, build_sft_dataset  # noqa: F401
    from nemotron_engine.training_backend import AdapterValidationReport, TrainingRunManifest  # noqa: F401
