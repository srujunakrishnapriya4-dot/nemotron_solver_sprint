from __future__ import annotations

import pytest

from nemotron_engine.parsing import canonicalize_prompt
from nemotron_engine.solvers import solve_numeric
from nemotron_engine.traces import compile_trace_from_solver_result
from nemotron_engine.training import build_sft_row
from nemotron_engine.training.dpo_builder import build_dpo_pair
from nemotron_engine.training.lora_config import LoRAConfig
from nemotron_engine.training.negative_generator import generate_negative_for_trace
from nemotron_engine.training.reward_audit import audit_reward_model_signals
from nemotron_engine.training.train_dpo import DPOTrainingConfig, plan_dpo_training
from nemotron_engine.training.train_grpo import GRPOTrainingConfig, GRPOTrainingError, plan_grpo_training
from nemotron_engine.training.train_sft import SFTTrainingConfig, plan_sft_training


def rows_and_pairs():
    problem = canonicalize_prompt("p", "1 -> 2\n2 -> 3\n4 -> ?")
    trace = compile_trace_from_solver_result(solve_numeric(problem), prompt=problem.raw_prompt)
    row = build_sft_row(trace)
    neg = generate_negative_for_trace(trace, "false_rule").trace
    pair = build_dpo_pair(prompt=problem.raw_prompt, chosen_trace=trace, rejected_trace=neg, negative_type="false_rule")
    return row, pair


def test_end_to_end_training_plans_are_dry_run_and_deterministic() -> None:
    row, pair = rows_and_pairs()
    lora = LoRAConfig("base", ("q_proj",), 32, 64, 0.0)
    sft_out = "tests/pass7_e2e_sft_out_should_not_exist"
    dpo_out = "tests/pass7_e2e_dpo_out_should_not_exist"

    sft_plan, _, sft_report = plan_sft_training([row], SFTTrainingConfig(lora, output_dir=str(sft_out)))
    dpo_plan, _, dpo_report = plan_dpo_training([pair], DPOTrainingConfig(lora, output_dir=str(dpo_out)))
    sft_plan2, _, _ = plan_sft_training([row], SFTTrainingConfig(lora, output_dir=str(sft_out)))

    assert sft_plan.plan_hash == sft_plan2.plan_hash
    assert not sft_report.trained
    assert not dpo_report.trained
    assert sft_report.adapter_path is None
    assert dpo_report.checkpoint_path is None
    from pathlib import Path

    assert not Path(sft_out).exists()
    assert not Path(dpo_out).exists()

    failed_audit = audit_reward_model_signals(
        sample_count=1,
        false_positive_rate=0.0,
        format_failure_rate=0.0,
        leakage_reward_rate=0.0,
        length_gaming_rate=0.0,
        heldout_regression_rate=0.0,
    )
    with pytest.raises(GRPOTrainingError):
        plan_grpo_training(failed_audit, GRPOTrainingConfig(lora, enabled=True))

    accepted_audit = audit_reward_model_signals(
        sample_count=100,
        false_positive_rate=0.0,
        format_failure_rate=0.0,
        leakage_reward_rate=0.0,
        length_gaming_rate=0.0,
        heldout_regression_rate=0.0,
    )
    grpo_plan, grpo_report = plan_grpo_training(accepted_audit, GRPOTrainingConfig(lora, enabled=True))
    assert grpo_plan.dry_run
    assert not grpo_report.trained
