from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_pass10_api_imports() -> None:
    from nemotron_engine.release import (  # noqa: F401
        ArtifactSafetyConfig,
        ImportSafetyConfig,
        LockedPassRegistry,
        ReleaseCandidateConfig,
        SystemAuditConfig,
        audit_import_safety,
        build_locked_pass_registry,
        build_release_candidate_report,
        run_system_audit,
        scan_release_artifacts,
    )


def test_pass1_to_pass9_representative_imports_still_work() -> None:
    from nemotron_engine.core.schemas import CanonicalProblem, stable_hash  # noqa: F401
    from nemotron_engine.data.split_builder import SplitManifest  # noqa: F401
    from nemotron_engine.evaluation.promotion_gates import PromotionGateReport  # noqa: F401
    from nemotron_engine.packaging.preflight_gate import PreflightGateReport  # noqa: F401
    from nemotron_engine.programs.executor import execute_program  # noqa: F401
    from nemotron_engine.runtime.serving_config import ServingConfig  # noqa: F401
    from nemotron_engine.scoring.local_scorer import score_completion  # noqa: F401
    from nemotron_engine.solvers.known_numeric import solve  # noqa: F401
    from nemotron_engine.submission.submission_validator import validate_submission_zip  # noqa: F401
    from nemotron_engine.traces.trace_compiler import TraceRecord  # noqa: F401
    from nemotron_engine.training.lora_config import LoRAConfig  # noqa: F401
    from nemotron_engine.training.sft_builder import build_sft_dataset  # noqa: F401
