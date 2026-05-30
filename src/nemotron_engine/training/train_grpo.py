"""Admission-gated GRPO dry-run interface for Pass 7."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.training.lora_config import LoRAConfig, validate_lora_config
from nemotron_engine.training.reward_audit import RewardAuditReport, grpo_admission_decision
from nemotron_engine.training.training_contracts import (
    TrainingContractError,
    TrainingPlan,
    TrainingRunReport,
    build_training_plan,
    make_dry_run_report,
    validate_backend_report,
    TrainingInputManifest,
)


class GRPOTrainingError(TrainingContractError):
    """Raised when GRPO is not admitted or cannot run safely."""


@dataclass(frozen=True)
class GRPOTrainingConfig:
    lora_config: LoRAConfig
    enabled: bool = False
    seed: int = 3
    max_steps: int = 50
    batch_size: int = 1
    learning_rate: float = 5e-5
    gradient_accumulation_steps: int = 1
    precision: str = "bf16"
    dry_run: bool = True
    output_dir: str = "outputs/grpo"
    metadata: Mapping[str, Any] = field(default_factory=dict)


GRPOTrainingReport = TrainingRunReport


def plan_grpo_training(
    reward_audit: RewardAuditReport,
    config: GRPOTrainingConfig,
) -> tuple[TrainingPlan, GRPOTrainingReport]:
    if config.enabled is not True:
        raise GRPOTrainingError("GRPO is disabled by default.")
    if grpo_admission_decision(reward_audit) is not True:
        raise GRPOTrainingError("GRPO requires accepted reward audit.")
    validate_lora_config(config.lora_config)
    item_ids = ("reward-audit-" + reward_audit.report_hash[:16],)
    item_hashes = (reward_audit.report_hash,)
    manifest = TrainingInputManifest(
        dataset_type="dpo",
        item_ids=item_ids,
        item_hashes=item_hashes,
        input_count=1,
        dataset_hash=stable_hash({"dataset_type": "dpo", "item_ids": item_ids, "item_hashes": item_hashes}),
        metadata={"stage": "grpo_admission"},
    )
    plan = build_training_plan(
        stage="grpo",
        manifest=manifest,
        lora_config=config.lora_config,
        seed=config.seed,
        max_steps=config.max_steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        precision=config.precision,
        dry_run=config.dry_run,
        output_dir=config.output_dir,
        metadata={**dict(config.metadata), "reward_audit_hash": reward_audit.report_hash},
    )
    return plan, make_dry_run_report(plan)


def run_grpo_training(
    plan: TrainingPlan,
    *,
    reward_audit: RewardAuditReport,
    backend: Callable[[TrainingPlan], TrainingRunReport] | None = None,
) -> GRPOTrainingReport:
    if grpo_admission_decision(reward_audit) is not True:
        raise GRPOTrainingError("GRPO requires accepted reward audit.")
    if plan.dry_run:
        return make_dry_run_report(plan)
    if backend is None:
        raise GRPOTrainingError("dry_run=False requires an explicit backend.")
    try:
        report = backend(plan)
    except Exception as exc:
        raise GRPOTrainingError("GRPO backend failed.") from exc
    try:
        return validate_backend_report(report, expected_plan=plan)
    except TrainingContractError as exc:
        raise GRPOTrainingError(str(exc)) from exc


__all__ = ["GRPOTrainingConfig", "GRPOTrainingError", "GRPOTrainingReport", "plan_grpo_training", "run_grpo_training"]
