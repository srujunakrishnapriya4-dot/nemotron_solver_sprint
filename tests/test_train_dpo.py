from __future__ import annotations

import pytest

from nemotron_engine.parsing import canonicalize_prompt
from nemotron_engine.solvers import solve_numeric
from nemotron_engine.traces import compile_trace_from_solver_result
from nemotron_engine.training.dpo_builder import build_dpo_pair
from nemotron_engine.training.lora_config import LoRAConfig
from nemotron_engine.training.negative_generator import generate_negative_for_trace
from nemotron_engine.training.train_dpo import DPOTrainingConfig, DPOTrainingError, plan_dpo_training, run_dpo_training
from nemotron_engine.training.training_contracts import TrainingRunReport


def pair():
    problem = canonicalize_prompt("p", "1 -> 2\n2 -> 3\n4 -> ?")
    trace = compile_trace_from_solver_result(solve_numeric(problem), prompt=problem.raw_prompt)
    neg = generate_negative_for_trace(trace, "false_rule").trace
    return build_dpo_pair(prompt=problem.raw_prompt, chosen_trace=trace, rejected_trace=neg, negative_type="false_rule")


def lora() -> LoRAConfig:
    return LoRAConfig("base", ("q_proj",), 8, 16, 0.0)


def test_plan_dpo_training_dry_run_and_no_fake_artifacts() -> None:
    output_dir = "tests/pass7_dpo_out_should_not_exist"
    plan, manifest, report = plan_dpo_training([pair()], DPOTrainingConfig(lora(), output_dir=str(output_dir)))

    assert plan.dry_run
    assert manifest.input_count == 1
    assert not report.trained
    assert report.adapter_path is None
    assert report.checkpoint_path is None
    from pathlib import Path

    assert not Path(output_dir).exists()


def test_run_dpo_training_backend_gates() -> None:
    plan, _, _ = plan_dpo_training([pair()], DPOTrainingConfig(lora(), dry_run=False))
    with pytest.raises(DPOTrainingError):
        run_dpo_training(plan)

    def success_backend(p):
        return TrainingRunReport("run-real", p.plan_hash, p.stage, True, False, checkpoint_path="real-checkpoint")

    assert run_dpo_training(plan, backend=success_backend).trained

    def failed_backend(_):
        raise RuntimeError("boom")

    with pytest.raises(DPOTrainingError):
        run_dpo_training(plan, backend=failed_backend)


def test_dpo_backend_adapter_config_rank_rejected() -> None:
    plan, _, _ = plan_dpo_training([pair()], DPOTrainingConfig(lora(), dry_run=False))

    def bad_rank_backend(p):
        return TrainingRunReport("run-real", p.plan_hash, p.stage, True, False, checkpoint_path="real", metadata={"adapter_config": {"r": 64}})

    with pytest.raises(DPOTrainingError):
        run_dpo_training(plan, backend=bad_rank_backend)

    def string_rank_backend(p):
        return TrainingRunReport("run-real", p.plan_hash, p.stage, True, False, checkpoint_path="real", metadata={"adapter_config": {"rank": "32"}})

    with pytest.raises(DPOTrainingError):
        run_dpo_training(plan, backend=string_rank_backend)


def test_dpo_invalid_pairs_rejected() -> None:
    with pytest.raises(Exception):
        plan_dpo_training([], DPOTrainingConfig(lora()))
