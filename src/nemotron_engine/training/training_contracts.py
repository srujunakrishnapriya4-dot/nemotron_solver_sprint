"""Shared Pass 7 training dataset contracts and dry-run plans."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Sequence

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.training.dpo_builder import ALLOWED_NEGATIVE_TYPES, DPOPair
from nemotron_engine.training.lora_config import LoRAConfig, LoRAConfigError, validate_adapter_config_json, validate_lora_config
from nemotron_engine.training.sft_builder import SFTDatasetRow


EVAL_ONLY_SPLITS = {"dev", "hard_dev", "private_like", "forbidden_holdout", "stress_only"}


class TrainingContractError(ValueError):
    """Raised when training inputs or plans violate safety contracts."""


@dataclass(frozen=True)
class TrainingDatasetContract:
    dataset_type: str
    item_count: int
    allow_empty_dry_run: bool = False
    mixture: Mapping[str, float] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dataset_type not in {"sft", "dpo"}:
            raise TrainingContractError("dataset_type must be 'sft' or 'dpo'.")
        if not isinstance(self.item_count, int) or isinstance(self.item_count, bool) or self.item_count < 0:
            raise TrainingContractError("item_count must be a non-negative integer.")
        if self.item_count == 0 and self.allow_empty_dry_run is not True:
            raise TrainingContractError("empty dataset requires allow_empty_dry_run=True.")
        if self.mixture is not None:
            total = sum(float(value) for value in self.mixture.values())
            if abs(total - 1.0) > 1e-9:
                raise TrainingContractError("dataset mixture must sum to 1.0.")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class TrainingInputManifest:
    dataset_type: str
    item_ids: tuple[str, ...]
    item_hashes: tuple[str, ...]
    input_count: int
    dataset_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    manifest_hash: str = ""

    def __post_init__(self) -> None:
        if self.dataset_type not in {"sft", "dpo"}:
            raise TrainingContractError("dataset_type must be 'sft' or 'dpo'.")
        object.__setattr__(self, "item_ids", tuple(str(item) for item in self.item_ids))
        object.__setattr__(self, "item_hashes", tuple(str(item) for item in self.item_hashes))
        if self.input_count != len(self.item_ids) or self.input_count != len(self.item_hashes):
            raise TrainingContractError("manifest item counts must match input_count.")
        if len(set(self.item_ids)) != len(self.item_ids):
            raise TrainingContractError("manifest contains duplicate item IDs.")
        if not self.dataset_hash:
            raise TrainingContractError("dataset_hash must be non-empty.")
        expected_dataset_hash = stable_hash(
            {"dataset_type": self.dataset_type, "item_ids": self.item_ids, "item_hashes": self.item_hashes}
        )
        if self.dataset_hash != expected_dataset_hash:
            raise TrainingContractError("dataset_hash does not match manifest items.")
        object.__setattr__(self, "metadata", dict(self.metadata))
        expected = _payload_hash(self, TrainingInputManifest, "manifest_hash")
        if not self.manifest_hash:
            object.__setattr__(self, "manifest_hash", expected)
        elif self.manifest_hash != expected:
            raise TrainingContractError("manifest_hash does not match manifest payload.")


@dataclass(frozen=True)
class TrainingPlan:
    stage: str
    dataset_manifest_hash: str
    lora_config_hash: str
    seed: int
    max_steps: int
    batch_size: int
    learning_rate: float
    gradient_accumulation_steps: int
    precision: str
    dry_run: bool
    output_dir: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    plan_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("stage", "dataset_manifest_hash", "lora_config_hash", "precision", "output_dir"):
            _require_non_empty(getattr(self, name), name)
        for name in ("seed", "max_steps", "batch_size", "gradient_accumulation_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TrainingContractError(f"{name} must be a non-negative integer.")
        if self.max_steps == 0 or self.batch_size == 0 or self.gradient_accumulation_steps == 0:
            raise TrainingContractError("max_steps, batch_size, and gradient_accumulation_steps must be positive.")
        if isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (int, float)) or self.learning_rate <= 0:
            raise TrainingContractError("learning_rate must be positive.")
        if not isinstance(self.dry_run, bool):
            raise TrainingContractError("dry_run must be boolean.")
        object.__setattr__(self, "learning_rate", float(self.learning_rate))
        object.__setattr__(self, "metadata", dict(self.metadata))
        expected = _payload_hash(self, TrainingPlan, "plan_hash")
        if not self.plan_hash:
            object.__setattr__(self, "plan_hash", expected)
        elif self.plan_hash != expected:
            raise TrainingContractError("plan_hash does not match plan payload.")


@dataclass(frozen=True)
class TrainingRunReport:
    run_id: str
    plan_hash: str
    stage: str
    trained: bool
    dry_run: bool
    adapter_path: str | None = None
    checkpoint_path: str | None = None
    backend_name: str | None = None
    errors: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    report_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("run_id", "plan_hash", "stage"):
            _require_non_empty(getattr(self, name), name)
        if not isinstance(self.trained, bool) or not isinstance(self.dry_run, bool):
            raise TrainingContractError("trained and dry_run must be booleans.")
        if self.dry_run:
            if self.trained is not False or self.adapter_path is not None or self.checkpoint_path is not None:
                raise TrainingContractError("dry-run reports cannot claim training artifacts.")
        if self.trained and (self.adapter_path is None and self.checkpoint_path is None):
            raise TrainingContractError("trained reports require real backend artifact paths.")
        object.__setattr__(self, "errors", tuple(str(item) for item in self.errors))
        object.__setattr__(self, "metadata", dict(self.metadata))
        expected = _payload_hash(self, TrainingRunReport, "report_hash")
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected)
        elif self.report_hash != expected:
            raise TrainingContractError("report_hash does not match report payload.")


def validate_sft_rows(rows: Sequence[SFTDatasetRow], *, allow_empty_dry_run: bool = False) -> tuple[SFTDatasetRow, ...]:
    materialized = tuple(rows)
    TrainingDatasetContract("sft", len(materialized), allow_empty_dry_run=allow_empty_dry_run)
    seen: set[str] = set()
    for row in materialized:
        if not isinstance(row, SFTDatasetRow):
            raise TrainingContractError("SFT rows must be SFTDatasetRow instances.")
        if row.row_id in seen:
            raise TrainingContractError("duplicate row_id.")
        seen.add(row.row_id)
        if _row_hash(row) != row.row_hash:
            raise TrainingContractError("bad row_hash.")
        if not row.source_proof_hash or not row.source_program_hash:
            raise TrainingContractError("SFT rows require source proof/program hashes.")
        _reject_bad_metadata(row.metadata)
        _reject_bad_split(row.split, row.metadata)
    return materialized


def validate_dpo_pairs(pairs: Sequence[DPOPair], *, allow_empty_dry_run: bool = False) -> tuple[DPOPair, ...]:
    materialized = tuple(pairs)
    TrainingDatasetContract("dpo", len(materialized), allow_empty_dry_run=allow_empty_dry_run)
    seen: set[str] = set()
    for pair in materialized:
        if not isinstance(pair, DPOPair):
            raise TrainingContractError("DPO pairs must be DPOPair instances.")
        if pair.pair_id in seen:
            raise TrainingContractError("duplicate pair_id.")
        seen.add(pair.pair_id)
        if pair.negative_type not in ALLOWED_NEGATIVE_TYPES:
            raise TrainingContractError("unsupported negative_type.")
        if _pair_hash(pair) != pair.pair_hash:
            raise TrainingContractError("bad pair_hash.")
        _reject_bad_metadata(pair.metadata, allow_negative_type=True)
        _reject_bad_split(None, pair.metadata)
    return materialized


def build_training_input_manifest(
    items: Sequence[SFTDatasetRow] | Sequence[DPOPair],
    *,
    dataset_type: str,
    metadata: dict[str, Any] | None = None,
    allow_empty_dry_run: bool = False,
) -> TrainingInputManifest:
    if dataset_type == "sft":
        rows = validate_sft_rows(items, allow_empty_dry_run=allow_empty_dry_run)  # type: ignore[arg-type]
        item_ids = tuple(row.row_id for row in rows)
        item_hashes = tuple(row.row_hash for row in rows)
    elif dataset_type == "dpo":
        pairs = validate_dpo_pairs(items, allow_empty_dry_run=allow_empty_dry_run)  # type: ignore[arg-type]
        item_ids = tuple(pair.pair_id for pair in pairs)
        item_hashes = tuple(pair.pair_hash for pair in pairs)
    else:
        raise TrainingContractError("dataset_type must be 'sft' or 'dpo'.")
    dataset_hash = stable_hash({"dataset_type": dataset_type, "item_ids": item_ids, "item_hashes": item_hashes})
    return TrainingInputManifest(dataset_type, item_ids, item_hashes, len(item_ids), dataset_hash, dict(metadata or {}))


def build_training_plan(
    *,
    stage: str,
    manifest: TrainingInputManifest,
    lora_config: LoRAConfig,
    seed: int,
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    gradient_accumulation_steps: int,
    precision: str,
    dry_run: bool,
    output_dir: str,
    metadata: dict[str, Any] | None = None,
) -> TrainingPlan:
    validate_lora_config(lora_config)
    return TrainingPlan(
        stage=stage,
        dataset_manifest_hash=manifest.manifest_hash,
        lora_config_hash=lora_config.config_hash,
        seed=seed,
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        precision=precision,
        dry_run=dry_run,
        output_dir=output_dir,
        metadata=dict(metadata or {}),
    )


def validate_training_plan(plan: TrainingPlan) -> TrainingPlan:
    if not isinstance(plan, TrainingPlan):
        raise TrainingContractError("plan must be a TrainingPlan.")
    return plan


def make_dry_run_report(plan: TrainingPlan, *, backend_name: str | None = None, errors: tuple[str, ...] = ()) -> TrainingRunReport:
    return TrainingRunReport(
        run_id="run-" + stable_hash({"stage": plan.stage, "plan_hash": plan.plan_hash, "dry_run": True})[:24],
        plan_hash=plan.plan_hash,
        stage=plan.stage,
        trained=False,
        dry_run=True,
        adapter_path=None,
        checkpoint_path=None,
        backend_name=backend_name,
        errors=errors,
    )


def validate_backend_report(report: Any, *, expected_plan: TrainingPlan) -> TrainingRunReport:
    if not isinstance(report, TrainingRunReport):
        raise TrainingContractError("backend must return a TrainingRunReport.")
    if report.plan_hash != expected_plan.plan_hash:
        raise TrainingContractError("backend report plan_hash mismatch.")
    if report.dry_run:
        raise TrainingContractError("backend report cannot be dry_run for non-dry run execution.")
    if report.trained is not True:
        raise TrainingContractError("backend report did not report trained=True.")
    _validate_backend_adapter_metadata(report.metadata)
    return report


def _validate_backend_adapter_metadata(metadata: Mapping[str, Any]) -> None:
    for key in ("adapter_config", "adapter_config_json"):
        if key in metadata:
            try:
                validate_adapter_config_json(metadata[key])
            except LoRAConfigError as exc:
                raise TrainingContractError("backend adapter_config failed validation.") from exc
    if "peft_config" in metadata:
        try:
            validate_adapter_config_json({"peft_config": metadata["peft_config"]})
        except LoRAConfigError as exc:
            raise TrainingContractError("backend peft_config failed validation.") from exc


def _reject_bad_metadata(metadata: Mapping[str, Any], *, allow_negative_type: bool = False) -> None:
    if metadata.get("contamination_flags") or metadata.get("contaminated") is True:
        raise TrainingContractError("contaminated metadata rejected.")
    if metadata.get("is_negative") is True:
        raise TrainingContractError("negative metadata rejected.")
    if not allow_negative_type and metadata.get("negative_type"):
        raise TrainingContractError("negative metadata rejected.")


def _reject_bad_split(split: str | None, metadata: Mapping[str, Any]) -> None:
    observed = split or metadata.get("split")
    if observed in EVAL_ONLY_SPLITS:
        raise TrainingContractError("eval-only split metadata rejected.")


def _row_hash(row: SFTDatasetRow) -> str:
    return stable_hash({item.name: getattr(row, item.name) for item in fields(SFTDatasetRow) if item.name != "row_hash"})


def _pair_hash(pair: DPOPair) -> str:
    return stable_hash({item.name: getattr(pair, item.name) for item in fields(DPOPair) if item.name != "pair_hash"})


def _payload_hash(instance: Any, cls: type, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(cls) if item.name != hash_field})


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingContractError(f"{field_name} must be a non-empty string.")
    return value


__all__ = [
    "TrainingContractError",
    "TrainingDatasetContract",
    "TrainingInputManifest",
    "TrainingPlan",
    "TrainingRunReport",
    "build_training_input_manifest",
    "build_training_plan",
    "make_dry_run_report",
    "validate_backend_report",
    "validate_dpo_pairs",
    "validate_sft_rows",
    "validate_training_plan",
]
