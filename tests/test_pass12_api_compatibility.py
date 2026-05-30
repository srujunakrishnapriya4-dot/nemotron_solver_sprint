from __future__ import annotations


def test_pass12_api_imports() -> None:
    from nemotron_engine.smoke_training import (  # noqa: F401
        SmokeBackendResult,
        SmokeDataset,
        SmokeDatasetExample,
        SmokeHandoffReport,
        SmokeRunConfig,
        SmokeRunReport,
        SmokeTrainingConfig,
        SmokeTrainingReport,
        build_smoke_handoff_to_pass11,
        build_smoke_training_report,
        build_tiny_dpo_smoke_dataset,
        build_tiny_sft_smoke_dataset,
        compute_smoke_config_hash,
        compute_smoke_dataset_hash,
        normalize_smoke_backend_result,
        run_smoke_training,
        validate_smoke_backend_result,
        validate_smoke_dataset,
        validate_smoke_handoff_report,
        validate_smoke_run_report,
        validate_smoke_training_config,
        validate_smoke_training_report,
    )


def test_representative_imports_from_passes_1_to_11_still_work() -> None:
    from nemotron_engine.core.schemas import CanonicalProblem, stable_hash  # noqa: F401
    from nemotron_engine.data.split_builder import SplitManifest  # noqa: F401
    from nemotron_engine.evaluation.promotion_gates import PromotionGateReport  # noqa: F401
    from nemotron_engine.packaging.preflight_gate import PreflightGateReport  # noqa: F401
    from nemotron_engine.programs.executor import execute_program  # noqa: F401
    from nemotron_engine.release.system_audit import SystemAuditReport  # noqa: F401
    from nemotron_engine.runtime.serving_config import ServingConfig  # noqa: F401
    from nemotron_engine.scoring.local_scorer import score_completion  # noqa: F401
    from nemotron_engine.solvers.known_numeric import solve  # noqa: F401
    from nemotron_engine.submission.submission_validator import validate_submission_zip  # noqa: F401
    from nemotron_engine.traces.trace_compiler import TraceRecord  # noqa: F401
    from nemotron_engine.training.lora_config import LoRAConfig  # noqa: F401
    from nemotron_engine.training.sft_builder import build_sft_dataset  # noqa: F401
    from nemotron_engine.training_backend import invoke_training_backend, validate_adapter_output  # noqa: F401
