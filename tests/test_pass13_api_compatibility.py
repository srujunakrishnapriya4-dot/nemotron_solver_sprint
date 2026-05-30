from __future__ import annotations


def test_pass13_api_imports() -> None:
    from nemotron_engine.inference_eval import (  # noqa: F401
        CompletionCaptureReport,
        CompletionPrivateLikeReport,
        CompletionTransferReport,
        InferenceBackendInvocation,
        InferenceBackendResult,
        InferenceCompletion,
        InferenceEvaluationManifest,
        InferenceRunReport,
        InferenceRunnerConfig,
        PromptBatch,
        PromptExample,
        build_inference_evaluation_manifest,
        build_prompt_batch_from_transfer_examples,
        build_transfer_examples_from_completions,
        capture_completions,
        evaluate_completions_with_transfer_harness,
        evaluate_private_like_from_completions,
        invoke_inference_backend,
        validate_inference_result,
    )


def test_representative_imports_from_passes_1_to_12_still_work() -> None:
    from nemotron_engine.core.schemas import CanonicalProblem, stable_hash  # noqa: F401
    from nemotron_engine.data.split_builder import SplitManifest  # noqa: F401
    from nemotron_engine.evaluation.private_like import evaluate_private_like_gate  # noqa: F401
    from nemotron_engine.evaluation.transfer_harness import evaluate_transfer_slices  # noqa: F401
    from nemotron_engine.packaging.preflight_gate import PreflightGateReport  # noqa: F401
    from nemotron_engine.release.system_audit import SystemAuditReport  # noqa: F401
    from nemotron_engine.runtime.serving_config import ServingConfig  # noqa: F401
    from nemotron_engine.scoring.local_scorer import score_completion  # noqa: F401
    from nemotron_engine.smoke_training import run_smoke_training  # noqa: F401
    from nemotron_engine.solvers.known_numeric import solve  # noqa: F401
    from nemotron_engine.submission.submission_validator import validate_submission_zip  # noqa: F401
    from nemotron_engine.traces.trace_compiler import TraceRecord  # noqa: F401
    from nemotron_engine.training.lora_config import LoRAConfig  # noqa: F401
    from nemotron_engine.training.sft_builder import build_sft_dataset  # noqa: F401
    from nemotron_engine.training_backend import invoke_training_backend, validate_backend_result  # noqa: F401
