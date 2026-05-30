"""Tiny deterministic smoke datasets for Pass 12."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash

from .smoke_config import SmokeTrainingConfig, SmokeTrainingConfigError, _reject_unsafe_metadata, validate_smoke_training_config


class SmokeDatasetError(ValueError):
    """Raised when a smoke dataset is invalid."""


@dataclass(frozen=True)
class SmokeDatasetExample:
    example_id: str
    stage: str
    prompt: str
    target: str | None = None
    chosen: str | None = None
    rejected: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    example_hash: str = ""

    def __post_init__(self) -> None:
        _require_non_empty(self.example_id, "example_id")
        if self.stage not in {"sft", "dpo", "final_sft_refresh"}:
            raise SmokeDatasetError("stage must be sft, dpo, or final_sft_refresh.")
        _require_non_empty(self.prompt, "prompt")
        metadata = dict(self.metadata)
        _reject_metadata(metadata)
        if self.stage in {"sft", "final_sft_refresh"}:
            _require_non_empty(self.target, "target")
            if self.chosen is not None or self.rejected is not None:
                raise SmokeDatasetError("SFT smoke examples must not include chosen/rejected.")
        elif self.stage == "dpo":
            _require_non_empty(self.chosen, "chosen")
            _require_non_empty(self.rejected, "rejected")
            if self.chosen == self.rejected:
                raise SmokeDatasetError("DPO chosen and rejected must differ.")
            if self.target is not None:
                raise SmokeDatasetError("DPO smoke examples must not include target.")
        object.__setattr__(self, "metadata", metadata)
        expected = _example_hash(self)
        if not self.example_hash:
            object.__setattr__(self, "example_hash", expected)
        elif self.example_hash != expected:
            raise SmokeDatasetError("example_hash does not match example payload.")


@dataclass(frozen=True)
class SmokeDataset:
    dataset_id: str
    stage: str
    examples: tuple[SmokeDatasetExample, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    dataset_hash: str = ""

    def __post_init__(self) -> None:
        _require_non_empty(self.dataset_id, "dataset_id")
        if self.stage not in {"sft", "dpo", "final_sft_refresh"}:
            raise SmokeDatasetError("stage must be sft, dpo, or final_sft_refresh.")
        examples = tuple(_example(item) for item in self.examples)
        if not examples:
            raise SmokeDatasetError("smoke dataset must contain at least one example.")
        if any(item.stage != self.stage for item in examples):
            raise SmokeDatasetError("all examples must match dataset stage.")
        ids = tuple(item.example_id for item in examples)
        if len(set(ids)) != len(ids):
            raise SmokeDatasetError("duplicate example_id rejected.")
        metadata = dict(self.metadata)
        _reject_metadata(metadata)
        object.__setattr__(self, "examples", examples)
        object.__setattr__(self, "metadata", metadata)
        expected = compute_smoke_dataset_hash(self)
        if not self.dataset_hash:
            object.__setattr__(self, "dataset_hash", expected)
        elif self.dataset_hash != expected:
            raise SmokeDatasetError("dataset_hash does not match dataset payload.")


def build_tiny_sft_smoke_dataset(*, stage: str = "sft", metadata: Mapping[str, Any] | None = None) -> SmokeDataset:
    if stage not in {"sft", "final_sft_refresh"}:
        raise SmokeDatasetError("tiny SFT smoke dataset stage must be sft or final_sft_refresh.")
    example = SmokeDatasetExample(
        example_id=f"smoke-{stage}-0001",
        stage=stage,
        prompt="1 -> 2\n2 -> 3\n4 -> ?",
        target="5",
        metadata={"fixture": "tiny_sft"},
    )
    return SmokeDataset(
        dataset_id=f"smoke-{stage}-dataset-v1",
        stage=stage,
        examples=(example,),
        metadata=dict(metadata or {"source": "pass12_tiny_sft"}),
    )


def build_tiny_dpo_smoke_dataset(*, metadata: Mapping[str, Any] | None = None) -> SmokeDataset:
    example = SmokeDatasetExample(
        example_id="smoke-dpo-0001",
        stage="dpo",
        prompt="Choose the completion that applies the rule 1 -> 2, 2 -> 3, 4 -> ?",
        chosen="The rule adds one, so the answer is 5.",
        rejected="The rule subtracts one, so the answer is 3.",
        metadata={"fixture": "tiny_dpo"},
    )
    return SmokeDataset(
        dataset_id="smoke-dpo-dataset-v1",
        stage="dpo",
        examples=(example,),
        metadata=dict(metadata or {"source": "pass12_tiny_dpo"}),
    )


def validate_smoke_dataset(dataset: SmokeDataset | Mapping[str, Any], config: SmokeTrainingConfig | Mapping[str, Any] | None = None) -> SmokeDataset:
    normalized = _dataset(dataset)
    if normalized.dataset_hash != compute_smoke_dataset_hash(normalized):
        raise SmokeDatasetError("dataset_hash does not match dataset payload.")
    for example in normalized.examples:
        if example.example_hash != _example_hash(example):
            raise SmokeDatasetError("example_hash does not match example payload.")
    if config is not None:
        try:
            cfg = validate_smoke_training_config(config)
        except SmokeTrainingConfigError as exc:
            raise SmokeDatasetError(str(exc)) from exc
        if normalized.stage != cfg.stage:
            raise SmokeDatasetError("dataset stage must match smoke config stage.")
        if len(normalized.examples) > cfg.max_examples:
            raise SmokeDatasetError("dataset size exceeds max_examples.")
    return normalized


def compute_smoke_dataset_hash(dataset: SmokeDataset | Mapping[str, Any]) -> str:
    payload = dict(dataset) if isinstance(dataset, Mapping) else {item.name: getattr(dataset, item.name) for item in fields(SmokeDataset)}
    payload.pop("dataset_hash", None)
    return stable_hash(payload)


def _dataset(value: SmokeDataset | Mapping[str, Any]) -> SmokeDataset:
    if isinstance(value, SmokeDataset):
        return value
    if isinstance(value, Mapping):
        payload = dict(value)
        payload["examples"] = tuple(_example(item) for item in payload.get("examples", ()))
        return SmokeDataset(**payload)
    raise SmokeDatasetError("dataset must be a SmokeDataset or mapping.")


def _example(value: SmokeDatasetExample | Mapping[str, Any]) -> SmokeDatasetExample:
    if isinstance(value, SmokeDatasetExample):
        return value
    if isinstance(value, Mapping):
        return SmokeDatasetExample(**dict(value))
    raise SmokeDatasetError("examples must be SmokeDatasetExample values.")


def _example_hash(example: SmokeDatasetExample) -> str:
    return stable_hash({item.name: getattr(example, item.name) for item in fields(SmokeDatasetExample) if item.name != "example_hash"})


def _reject_metadata(metadata: Mapping[str, Any]) -> None:
    try:
        _reject_unsafe_metadata(metadata, reject_split_claims=True)
    except SmokeTrainingConfigError as exc:
        raise SmokeDatasetError(str(exc)) from exc


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SmokeDatasetError(f"{field_name} must be a non-empty string.")
    return value


__all__ = [
    "SmokeDataset",
    "SmokeDatasetError",
    "SmokeDatasetExample",
    "build_tiny_dpo_smoke_dataset",
    "build_tiny_sft_smoke_dataset",
    "compute_smoke_dataset_hash",
    "validate_smoke_dataset",
]
