from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from nemotron_engine.core.schemas import stable_hash


class VexTokenCorpusError(ValueError):
    pass


CORPUS_TYPES = {"direct_raw", "family_tagged", "short_rule_trace", "synthetic_verified", "hard_negative_optional"}
FORBIDDEN_METADATA_KEYS = {"test_id", "test_answer", "public_lb_answer", "private_answer"}


@dataclass(frozen=True)
class VexTokenCorpusRow:
    row_id: str
    source_id: str
    family: str
    corpus_type: str
    text: str
    token_ids: list[int] | None = None
    target_ids: list[int] | None = None
    loss_weights: list[float] | None = None
    prompt_token_count: int | None = None
    supervised_token_count: int | None = None
    answer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    row_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("row_id", "source_id", "family", "text", "answer"):
            if not str(getattr(self, name)).strip():
                raise VexTokenCorpusError(f"{name} must be non-empty")
        if self.corpus_type not in CORPUS_TYPES:
            raise VexTokenCorpusError(f"invalid corpus_type: {self.corpus_type}")
        metadata = dict(self.metadata)
        if FORBIDDEN_METADATA_KEYS & set(metadata):
            raise VexTokenCorpusError("forbidden test/public-private metadata in corpus row")
        object.__setattr__(self, "metadata", metadata)
        if self.token_ids is not None:
            _validate_int_list(self.token_ids, "token_ids")
        if self.target_ids is not None:
            _validate_int_list(self.target_ids, "target_ids")
        if self.loss_weights is not None:
            _validate_float_list(self.loss_weights, "loss_weights")
        if self.token_ids is not None and self.target_ids is not None and len(self.token_ids) != len(self.target_ids):
            raise VexTokenCorpusError("target_ids length mismatch")
        if self.token_ids is not None and self.loss_weights is not None and len(self.token_ids) != len(self.loss_weights):
            raise VexTokenCorpusError("loss_weights length mismatch")
        _set_or_check_hash(self, "row_hash")


def row_to_json(row: VexTokenCorpusRow) -> dict[str, Any]:
    return {field.name: getattr(row, field.name) for field in fields(row)}


def row_from_json(payload: dict[str, Any]) -> VexTokenCorpusRow:
    return VexTokenCorpusRow(**payload)


def _validate_int_list(values: list[int], name: str) -> None:
    if not all(isinstance(value, int) for value in values):
        raise VexTokenCorpusError(f"{name} must contain ints")


def _validate_float_list(values: list[float], name: str) -> None:
    if not all(isinstance(value, (int, float)) for value in values):
        raise VexTokenCorpusError(f"{name} must contain floats")


def _set_or_check_hash(instance: object, hash_field: str) -> None:
    expected = stable_hash({field.name: getattr(instance, field.name) for field in fields(instance) if field.name != hash_field})
    current = getattr(instance, hash_field)
    if not current:
        object.__setattr__(instance, hash_field, expected)
    elif current != expected:
        raise VexTokenCorpusError(f"{hash_field} does not match payload")
