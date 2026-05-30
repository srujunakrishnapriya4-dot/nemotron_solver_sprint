"""Deterministic prompt batches for Pass 13 inference evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import re
from typing import Any, Mapping, Sequence

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.evaluation.transfer_harness import TRANSFER_SLICE_NAMES
from nemotron_engine.runtime.serving_config import ServingConfig
from nemotron_engine.scoring.format_policy import AnswerType

from .inference_contracts import _reject_forbidden_metadata


class PromptBatchError(ValueError):
    """Raised when prompt batch evidence is unsafe or inconsistent."""


LEAKAGE_MARKERS = (
    re.compile(r"expected_answer\s*=", flags=re.IGNORECASE),
    re.compile(r"gold_answer\s*=", flags=re.IGNORECASE),
    re.compile(r"target_answer\s*=", flags=re.IGNORECASE),
    re.compile(r"correct\s+answer\s*:", flags=re.IGNORECASE),
)


@dataclass(frozen=True)
class PromptExample:
    problem_id: str
    prompt_text: str
    slice_name: str
    family_id: str
    primitive_family_id: str
    format_family_id: str
    answer_type: str
    split: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    prompt_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("problem_id", "prompt_text", "slice_name", "family_id", "primitive_family_id", "format_family_id", "answer_type", "split"):
            _require_non_empty(getattr(self, name), name)
        _reject_prompt_leakage(self.prompt_text)
        if self.slice_name not in TRANSFER_SLICE_NAMES:
            raise PromptBatchError(f"unsupported slice_name: {self.slice_name!r}.")
        if self.split not in TRANSFER_SLICE_NAMES:
            raise PromptBatchError(f"unsupported split: {self.split!r}.")
        try:
            answer_type = AnswerType(str(self.answer_type)).value
        except ValueError as exc:
            raise PromptBatchError(f"unsupported answer_type: {self.answer_type!r}.") from exc
        metadata = _safe_metadata(self.metadata)
        object.__setattr__(self, "answer_type", answer_type)
        object.__setattr__(self, "metadata", metadata)
        expected = _payload_hash(self, "prompt_hash")
        if not self.prompt_hash:
            object.__setattr__(self, "prompt_hash", expected)
        elif self.prompt_hash != expected:
            raise PromptBatchError("prompt_hash does not match prompt example payload.")


@dataclass(frozen=True)
class PromptBatch:
    batch_id: str
    examples: tuple[PromptExample, ...]
    serving_config_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    batch_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("batch_id", "serving_config_hash"):
            _require_non_empty(getattr(self, name), name)
        examples = tuple(_prompt_example(item) for item in self.examples)
        if not examples:
            raise PromptBatchError("PromptBatch requires at least one example.")
        ids = tuple(item.problem_id for item in examples)
        if len(ids) != len(set(ids)):
            raise PromptBatchError("PromptBatch contains duplicate problem_id values.")
        metadata = _safe_metadata(self.metadata)
        object.__setattr__(self, "examples", examples)
        object.__setattr__(self, "metadata", metadata)
        expected = compute_prompt_batch_hash(self)
        if not self.batch_hash:
            object.__setattr__(self, "batch_hash", expected)
        elif self.batch_hash != expected:
            raise PromptBatchError("batch_hash does not match prompt batch payload.")


def build_prompt_batch_from_transfer_examples(
    records: Sequence[Any],
    *,
    serving_config: ServingConfig | None = None,
    serving_config_hash: str | None = None,
    batch_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PromptBatch:
    materialized = tuple(records)
    if not materialized:
        raise PromptBatchError("records must not be empty.")
    resolved_serving_hash = serving_config_hash
    if serving_config is not None:
        if not isinstance(serving_config, ServingConfig):
            raise PromptBatchError("serving_config must be a ServingConfig.")
        resolved_serving_hash = stable_hash(serving_config.to_dict())
        if len(materialized) > serving_config.batch_size:
            raise PromptBatchError("prompt batch size exceeds serving_config.batch_size.")
    if resolved_serving_hash is None:
        raise PromptBatchError("serving_config_hash is required.")
    examples = tuple(_example_from_record(item) for item in materialized)
    batch_id = batch_id or "prompt-batch-" + stable_hash(
        {"serving_config_hash": resolved_serving_hash, "prompt_hashes": tuple(item.prompt_hash for item in examples)}
    )[:24]
    return PromptBatch(batch_id=batch_id, examples=examples, serving_config_hash=resolved_serving_hash, metadata=dict(metadata or {}))


def validate_prompt_batch(batch: PromptBatch | Mapping[str, Any], *, serving_config: ServingConfig | None = None) -> PromptBatch:
    normalized = batch if isinstance(batch, PromptBatch) else PromptBatch(**dict(batch))
    if normalized.batch_hash != compute_prompt_batch_hash(normalized):
        raise PromptBatchError("batch_hash does not match prompt batch payload.")
    if serving_config is not None:
        serving_hash = stable_hash(serving_config.to_dict())
        if normalized.serving_config_hash != serving_hash:
            raise PromptBatchError("serving_config_hash mismatch.")
        if len(normalized.examples) > serving_config.batch_size:
            raise PromptBatchError("prompt batch size exceeds serving_config.batch_size.")
    return normalized


def compute_prompt_batch_hash(batch: PromptBatch | Mapping[str, Any]) -> str:
    payload = dict(batch) if isinstance(batch, Mapping) else {item.name: getattr(batch, item.name) for item in fields(PromptBatch)}
    payload.pop("batch_hash", None)
    return stable_hash(payload)


def _example_from_record(record: Any) -> PromptExample:
    payload = _record_mapping(record)
    prompt_text = _first(payload, "prompt_text", "prompt", "raw_prompt", "question")
    if prompt_text is None:
        raise PromptBatchError("record requires prompt_text.")
    split = str(_first(payload, "split", "slice_name") or "")
    slice_name = str(_first(payload, "slice_name", "split") or "")
    metadata = dict(payload.get("metadata") or {})
    for forbidden in ("expected_answer", "gold_answer", "target_answer", "label", "correct", "is_correct", "correctness", "completion"):
        metadata.pop(forbidden, None)
    return PromptExample(
        problem_id=str(payload.get("problem_id", "")),
        prompt_text=str(prompt_text),
        slice_name=slice_name,
        family_id=str(payload.get("family_id", "")),
        primitive_family_id=str(payload.get("primitive_family_id", "")),
        format_family_id=str(payload.get("format_family_id", "")),
        answer_type=str(payload.get("answer_type", "")),
        split=split,
        metadata=metadata,
    )


def _record_mapping(record: Any) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    if is_dataclass(record):
        return {item.name: getattr(record, item.name) for item in fields(record)}
    data: dict[str, Any] = {}
    for name in (
        "problem_id",
        "prompt_text",
        "prompt",
        "raw_prompt",
        "question",
        "slice_name",
        "family_id",
        "primitive_family_id",
        "format_family_id",
        "answer_type",
        "split",
        "metadata",
    ):
        if hasattr(record, name):
            data[name] = getattr(record, name)
    return data


def _first(payload: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    return None


def _prompt_example(value: PromptExample | Mapping[str, Any]) -> PromptExample:
    if isinstance(value, PromptExample):
        return value
    if isinstance(value, Mapping):
        return PromptExample(**dict(value))
    raise PromptBatchError("examples must contain PromptExample values.")


def _safe_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PromptBatchError("metadata must be a mapping.")
    metadata = dict(value)
    try:
        _reject_forbidden_metadata(metadata)
    except Exception as exc:
        raise PromptBatchError(str(exc)) from exc
    return metadata


def _reject_prompt_leakage(prompt_text: str) -> None:
    for marker in LEAKAGE_MARKERS:
        if marker.search(prompt_text):
            raise PromptBatchError("prompt_text contains expected-answer leakage marker.")


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PromptBatchError(f"{field_name} must be a non-empty string.")
    return value


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


__all__ = [
    "PromptBatch",
    "PromptBatchError",
    "PromptExample",
    "build_prompt_batch_from_transfer_examples",
    "compute_prompt_batch_hash",
    "validate_prompt_batch",
]
