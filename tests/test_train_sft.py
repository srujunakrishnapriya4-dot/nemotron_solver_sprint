from __future__ import annotations

import pytest

from nemotron_engine.parsing import canonicalize_prompt
from nemotron_engine.solvers import solve_numeric
from nemotron_engine.traces import compile_trace_from_solver_result
from nemotron_engine.training import build_sft_row
from nemotron_engine.training.lora_config import LoRAConfig
from nemotron_engine.training.train_sft import SFTTrainingConfig, SFTTrainingError, plan_sft_training, run_sft_training
from nemotron_engine.training.training_contracts import TrainingRunReport


def row():
    problem = canonicalize_prompt("p", "1 -> 2\n2 -> 3\n4 -> ?")
    return build_sft_row(compile_trace_from_solver_result(solve_numeric(problem), prompt=problem.raw_prompt))


def lora(rank: int = 8) -> LoRAConfig:
    return LoRAConfig("base", ("q_proj",), rank, 16, 0.0)


def test_plan_sft_training_dry_run_report_and_no_output_dir() -> None:
    output_dir = "tests/pass7_sft_out_should_not_exist"
    plan, manifest, report = plan_sft_training([row()], SFTTrainingConfig(lora(), output_dir=str(output_dir)))

    assert plan.dry_run
    assert manifest.input_count == 1
    assert not report.trained
    assert report.adapter_path is None
    assert report.checkpoint_path is None
    from pathlib import Path

    assert not Path(output_dir).exists()


def test_run_sft_training_dry_run_does_not_call_backend() -> None:
    plan, _, _ = plan_sft_training([row()], SFTTrainingConfig(lora()))

    def backend(_):
        raise AssertionError("backend should not be called")

    report = run_sft_training(plan, backend=backend)
    assert report.trained is False


def test_sft_non_dry_run_requires_backend_and_validates_result() -> None:
    plan, _, _ = plan_sft_training([row()], SFTTrainingConfig(lora(), dry_run=False))
    with pytest.raises(SFTTrainingError):
        run_sft_training(plan)

    def success_backend(p):
        return TrainingRunReport("run-real", p.plan_hash, p.stage, True, False, adapter_path="real-adapter")

    assert run_sft_training(plan, backend=success_backend).trained

    def malformed_backend(_):
        return {"trained": True}

    with pytest.raises(SFTTrainingError):
        run_sft_training(plan, backend=malformed_backend)


def test_sft_backend_adapter_config_rank_rejected() -> None:
    plan, _, _ = plan_sft_training([row()], SFTTrainingConfig(lora(), dry_run=False))

    def bad_rank_backend(p):
        return TrainingRunReport("run-real", p.plan_hash, p.stage, True, False, adapter_path="real", metadata={"adapter_config": {"r": 64}})

    with pytest.raises(SFTTrainingError):
        run_sft_training(plan, backend=bad_rank_backend)

    def string_rank_backend(p):
        return TrainingRunReport("run-real", p.plan_hash, p.stage, True, False, adapter_path="real", metadata={"adapter_config": {"rank": "32"}})

    with pytest.raises(SFTTrainingError):
        run_sft_training(plan, backend=string_rank_backend)


def test_sft_rejects_rank_over_32_and_invalid_rows() -> None:
    with pytest.raises(Exception):
        lora(33)
    with pytest.raises(Exception):
        plan_sft_training([], SFTTrainingConfig(lora()))
