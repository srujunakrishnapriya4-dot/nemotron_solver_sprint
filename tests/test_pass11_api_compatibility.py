from __future__ import annotations


def test_pass11_api_imports() -> None:
    from nemotron_engine.training_backend import (  # noqa: F401
        AdapterValidationConfig,
        BackendInvocation,
        BackendResult,
        BackendRunnerConfig,
        TrainingRunManifest,
        build_invocation_from_sft_plan,
        invoke_training_backend,
        validate_adapter_output,
        validate_backend_result,
        validate_training_log,
    )


def test_representative_imports_from_passes_1_to_10_still_work() -> None:
    from nemotron_engine.scoring.local_scorer import score_answer  # noqa: F401
    from nemotron_engine.core.schemas import CanonicalProblem, stable_hash  # noqa: F401
    from nemotron_engine.programs.executor import execute_program  # noqa: F401
    from nemotron_engine.solvers.known_numeric import solve  # noqa: F401
    from nemotron_engine.data.split_builder import SplitManifest  # noqa: F401
    from nemotron_engine.traces.trace_compiler import TraceRecord  # noqa: F401
    from nemotron_engine.training.sft_builder import build_sft_dataset  # noqa: F401
    from nemotron_engine.training.lora_config import LoRAConfig  # noqa: F401
    from nemotron_engine.evaluation.promotion_gates import PromotionGateReport  # noqa: F401
    from nemotron_engine.packaging.preflight_gate import PreflightGateReport  # noqa: F401
    from nemotron_engine.release.system_audit import SystemAuditReport  # noqa: F401
