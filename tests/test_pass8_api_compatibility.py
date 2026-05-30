from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_pass8_imports() -> None:
    from nemotron_engine.evaluation import (  # noqa: F401
        PrivateLikeGateConfig,
        PromotionGateConfig,
        PublicFirewallConfig,
        RegressionMetric,
        SubmissionExactConfig,
        TransferExample,
        build_regression_report,
        evaluate_promotion,
        evaluate_transfer_slices,
        validate_submission_exact_config,
    )


def test_representative_locked_pass_imports_still_work() -> None:
    from nemotron_engine.core.schemas import ImmutableProblemRecord  # noqa: F401
    from nemotron_engine.data import assign_splits  # noqa: F401
    from nemotron_engine.scoring.answer_extractor import extract_boxed_answer  # noqa: F401
    from nemotron_engine.solvers import solve_numeric  # noqa: F401
    from nemotron_engine.traces import compile_trace_from_solver_result  # noqa: F401
    from nemotron_engine.training import LoRAConfig, build_sft_row  # noqa: F401
