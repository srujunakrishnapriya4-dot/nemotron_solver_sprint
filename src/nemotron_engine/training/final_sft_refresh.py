"""Final clean SFT refresh dry-run planning for Pass 7."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Sequence

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.training.lora_config import LoRAConfig, validate_lora_config
from nemotron_engine.training.sft_builder import SFTDatasetRow
from nemotron_engine.training.training_contracts import TrainingContractError, TrainingPlan, build_training_input_manifest, build_training_plan


class FinalSFTRefreshError(TrainingContractError):
    """Raised when final SFT refresh cannot be planned safely."""


@dataclass(frozen=True)
class FinalSFTRefreshConfig:
    lora_config: LoRAConfig
    seed: int = 7
    max_steps: int = 25
    batch_size: int = 1
    learning_rate: float = 5e-5
    gradient_accumulation_steps: int = 1
    precision: str = "bf16"
    dry_run: bool = True
    output_dir: str = "outputs/final_sft_refresh"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FinalSFTRefreshReport:
    plan_hash: str
    row_count: int
    dry_run: bool
    trained: bool = False
    adapter_path: str | None = None
    checkpoint_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    report_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.plan_hash, str) or not self.plan_hash.strip():
            raise FinalSFTRefreshError("plan_hash must be non-empty.")
        if not isinstance(self.row_count, int) or isinstance(self.row_count, bool) or self.row_count < 0:
            raise FinalSFTRefreshError("row_count must be a non-negative integer.")
        if self.dry_run is not True or self.trained is not False or self.adapter_path is not None or self.checkpoint_path is not None:
            raise FinalSFTRefreshError("final SFT refresh is dry-run planning only.")
        object.__setattr__(self, "metadata", dict(self.metadata))
        expected = stable_hash({item.name: getattr(self, item.name) for item in fields(FinalSFTRefreshReport) if item.name != "report_hash"})
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected)
        elif self.report_hash != expected:
            raise FinalSFTRefreshError("report_hash does not match final refresh payload.")


def plan_final_sft_refresh(rows: Sequence[SFTDatasetRow], config: FinalSFTRefreshConfig) -> tuple[TrainingPlan, FinalSFTRefreshReport]:
    try:
        validate_lora_config(config.lora_config)
        manifest = build_training_input_manifest(rows, dataset_type="sft", metadata={"stage": "final_sft_refresh"})
        plan = build_training_plan(
            stage="final_sft_refresh",
            manifest=manifest,
            lora_config=config.lora_config,
            seed=config.seed,
            max_steps=config.max_steps,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            precision=config.precision,
            dry_run=True,
            output_dir=config.output_dir,
            metadata=dict(config.metadata),
        )
    except TrainingContractError as exc:
        raise FinalSFTRefreshError(str(exc)) from exc
    return plan, FinalSFTRefreshReport(plan_hash=plan.plan_hash, row_count=len(tuple(rows)), dry_run=True)


__all__ = ["FinalSFTRefreshConfig", "FinalSFTRefreshError", "FinalSFTRefreshReport", "plan_final_sft_refresh"]
