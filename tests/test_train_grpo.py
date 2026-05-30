from __future__ import annotations

import pytest

from nemotron_engine.training.lora_config import LoRAConfig
from nemotron_engine.training.reward_audit import RewardAuditError, RewardAuditReport, audit_reward_model_signals
from nemotron_engine.training.train_grpo import GRPOTrainingConfig, GRPOTrainingError, plan_grpo_training, run_grpo_training
from nemotron_engine.training.training_contracts import TrainingRunReport


def lora(rank: int = 8) -> LoRAConfig:
    return LoRAConfig("base", ("q_proj",), rank, 16, 0.0)


def accepted_audit():
    return audit_reward_model_signals(
        sample_count=100,
        false_positive_rate=0.0,
        format_failure_rate=0.0,
        leakage_reward_rate=0.0,
        length_gaming_rate=0.0,
        heldout_regression_rate=0.0,
    )


def test_grpo_disabled_by_default_and_failed_audit_blocks() -> None:
    with pytest.raises(GRPOTrainingError, match="disabled"):
        plan_grpo_training(accepted_audit(), GRPOTrainingConfig(lora()))
    failed = audit_reward_model_signals(
        sample_count=1,
        false_positive_rate=0.0,
        format_failure_rate=0.0,
        leakage_reward_rate=0.0,
        length_gaming_rate=0.0,
        heldout_regression_rate=0.0,
    )
    with pytest.raises(GRPOTrainingError, match="accepted reward audit"):
        plan_grpo_training(failed, GRPOTrainingConfig(lora(), enabled=True))


def test_accepted_audit_allows_dry_run_only_and_non_dry_run_needs_backend() -> None:
    audit = accepted_audit()
    plan, report = plan_grpo_training(audit, GRPOTrainingConfig(lora(), enabled=True))
    assert not report.trained
    assert report.adapter_path is None

    plan2, _ = plan_grpo_training(audit, GRPOTrainingConfig(lora(), enabled=True, dry_run=False))
    with pytest.raises(GRPOTrainingError):
        run_grpo_training(plan2, reward_audit=audit)


def test_grpo_rank_over_32_rejected() -> None:
    with pytest.raises(Exception):
        lora(33)


def test_grpo_rejects_forged_accepted_audit_threshold_failures() -> None:
    with pytest.raises(RewardAuditError):
        RewardAuditReport(99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, True)
    with pytest.raises(RewardAuditError):
        RewardAuditReport(100, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, True)


def test_grpo_backend_adapter_config_rank_rejected() -> None:
    audit = accepted_audit()
    plan, _ = plan_grpo_training(audit, GRPOTrainingConfig(lora(), enabled=True, dry_run=False))

    def bad_rank_backend(p):
        return TrainingRunReport("run-real", p.plan_hash, p.stage, True, False, adapter_path="real", metadata={"adapter_config": {"r": 64}})

    with pytest.raises(GRPOTrainingError):
        run_grpo_training(plan, reward_audit=audit, backend=bad_rank_backend)

    def string_rank_backend(p):
        return TrainingRunReport("run-real", p.plan_hash, p.stage, True, False, adapter_path="real", metadata={"adapter_config": {"rank": "32"}})

    with pytest.raises(GRPOTrainingError):
        run_grpo_training(plan, reward_audit=audit, backend=string_rank_backend)
