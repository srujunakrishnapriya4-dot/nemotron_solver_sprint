from __future__ import annotations


def test_pass7_imports() -> None:
    from nemotron_engine.training import (  # noqa: F401
        LoRAConfig,
        RewardAuditReport,
        TrainingPlan,
        grpo_admission_decision,
        plan_dpo_training,
        plan_grpo_training,
        plan_sft_training,
        validate_lora_config,
        validate_training_plan,
    )


def test_representative_locked_pass_imports_still_work() -> None:
    from nemotron_engine.scoring.answer_extractor import extract_boxed_answer  # noqa: F401
    from nemotron_engine.core.schemas import ImmutableProblemRecord  # noqa: F401
    from nemotron_engine.solvers import solve_numeric  # noqa: F401
    from nemotron_engine.data import assign_splits  # noqa: F401
    from nemotron_engine.traces import compile_trace_from_solver_result  # noqa: F401
    from nemotron_engine.training import build_sft_row  # noqa: F401
