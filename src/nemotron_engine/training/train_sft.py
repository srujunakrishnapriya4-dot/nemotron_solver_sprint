"""Dry-run-first SFT training interfaces for Pass 7."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from nemotron_engine.training.lora_config import LoRAConfig, validate_lora_config
from nemotron_engine.training.sft_builder import SFTDatasetRow
from nemotron_engine.training.training_contracts import (
    TrainingContractError,
    TrainingInputManifest,
    TrainingPlan,
    TrainingRunReport,
    build_training_input_manifest,
    build_training_plan,
    make_dry_run_report,
    validate_backend_report,
)


class SFTTrainingError(TrainingContractError):
    """Raised when SFT training cannot be planned or run safely."""


@dataclass(frozen=True)
class SFTTrainingConfig:
    lora_config: LoRAConfig
    seed: int = 1
    max_steps: int = 100
    batch_size: int = 1
    learning_rate: float = 2e-4
    gradient_accumulation_steps: int = 1
    precision: str = "bf16"
    dry_run: bool = True
    output_dir: str = "outputs/sft"
    allow_empty_dry_run: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


SFTTrainingReport = TrainingRunReport


def plan_sft_training(rows: Sequence[SFTDatasetRow], config: SFTTrainingConfig) -> tuple[TrainingPlan, TrainingInputManifest, SFTTrainingReport]:
    validate_lora_config(config.lora_config)
    manifest = build_training_input_manifest(
        rows,
        dataset_type="sft",
        metadata={"stage": "sft"},
        allow_empty_dry_run=config.allow_empty_dry_run and config.dry_run,
    )
    plan = build_training_plan(
        stage="sft",
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
        metadata=dict(config.metadata),
    )
    return plan, manifest, make_dry_run_report(plan)


def run_sft_training(plan: TrainingPlan, *, backend: Callable[[TrainingPlan], TrainingRunReport] | None = None) -> SFTTrainingReport:
    if plan.dry_run:
        return make_dry_run_report(plan)
    if backend is None:
        raise SFTTrainingError("dry_run=False requires an explicit backend.")
    try:
        report = backend(plan)
    except Exception as exc:
        raise SFTTrainingError("SFT backend failed.") from exc
    try:
        return validate_backend_report(report, expected_plan=plan)
    except TrainingContractError as exc:
        raise SFTTrainingError(str(exc)) from exc


__all__ = ["SFTTrainingConfig", "SFTTrainingError", "SFTTrainingReport", "plan_sft_training", "run_sft_training"]
