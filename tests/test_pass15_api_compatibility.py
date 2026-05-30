from __future__ import annotations


def test_pass15_api_imports() -> None:
    from nemotron_engine.runbook import (  # noqa: F401
        BlockerGateConfig,
        BlockerGateReport,
        ChecklistItem,
        CommandManifest,
        FailureRecoveryPlan,
        FinalOperationsManifest,
        HumanApprovalStatement,
        OperatorChecklist,
        RecoveryAction,
        ReproducibilityManifest,
        RunbookCommand,
        RunbookConfig,
        SubmissionRecord,
        build_default_command_manifest,
        build_default_failure_recovery_plan,
        build_final_operations_manifest,
        build_human_approval_statement,
        build_manual_submission_record,
        build_operator_checklist,
        build_reproducibility_manifest,
        evaluate_blocker_gate,
        validate_command_manifest,
        validate_runbook_config,
    )


def test_representative_imports_from_passes_1_to_14_still_work() -> None:
    from nemotron_engine.core.schemas import CanonicalProblem, stable_hash  # noqa: F401
    from nemotron_engine.data import assign_splits  # noqa: F401
    from nemotron_engine.evaluation import PromotionGateReport, evaluate_promotion  # noqa: F401
    from nemotron_engine.inference_eval import InferenceEvaluationManifest  # noqa: F401
    from nemotron_engine.packaging import PreflightGateReport, RuntimeValidationReport  # noqa: F401
    from nemotron_engine.programs import execute_program  # noqa: F401
    from nemotron_engine.release import ReleaseCandidateReport, SystemAuditReport  # noqa: F401
    from nemotron_engine.rehearsal import FinalRehearsalManifest  # noqa: F401
    from nemotron_engine.runtime import ServingConfig  # noqa: F401
    from nemotron_engine.scoring import score_completion  # noqa: F401
    from nemotron_engine.smoke_training import SmokeTrainingReport  # noqa: F401
    from nemotron_engine.solvers import solve_numeric  # noqa: F401
    from nemotron_engine.training import LoRAConfig, build_sft_dataset  # noqa: F401
    from nemotron_engine.training_backend import AdapterValidationReport, TrainingRunManifest  # noqa: F401
