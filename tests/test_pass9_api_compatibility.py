from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_pass9_api_imports() -> None:
    from nemotron_engine.packaging import (  # noqa: F401
        ArtifactAuditConfig,
        FileEntry,
        PackageBuildConfig,
        PreflightGateConfig,
        RuntimeValidationConfig,
        SubmissionManifest,
        audit_artifacts,
        build_submission_manifest,
        build_submission_package,
        evaluate_preflight_gate,
        validate_offline_runtime,
    )


def test_locked_pass_representative_imports_still_work() -> None:
    from nemotron_engine.core.schemas import CanonicalProblem, stable_hash  # noqa: F401
    from nemotron_engine.evaluation.promotion_gates import evaluate_promotion  # noqa: F401
    from nemotron_engine.evaluation.transfer_harness import evaluate_transfer_slices  # noqa: F401
    from nemotron_engine.programs.executor import execute_program  # noqa: F401
    from nemotron_engine.runtime.serving_config import ServingConfig  # noqa: F401
    from nemotron_engine.scoring.local_scorer import score_completion  # noqa: F401
    from nemotron_engine.solvers.known_numeric import solve  # noqa: F401
    from nemotron_engine.training.sft_builder import build_sft_dataset  # noqa: F401
    from nemotron_engine.training.lora_config import LoRAConfig  # noqa: F401
