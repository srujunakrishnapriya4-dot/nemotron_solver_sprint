"""Deterministic family-aware split construction for Pass 4."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import json
from pathlib import Path
from typing import Iterable, Mapping

from nemotron_engine.core.schemas import (
    ImmutableProblemRecord,
    RegistryError,
    SchemaValidationError,
    SplitName,
    stable_hash,
    stable_json_dumps,
)


class SplitManifestError(ValueError):
    """Raised when a split manifest or split assignment is invalid."""


@dataclass(frozen=True)
class SplitManifest:
    manifest_id: str
    seed: int
    record_count: int
    split_counts: Mapping[str, int]
    split_record_ids: Mapping[str, tuple[str, ...]]
    family_to_splits: Mapping[str, tuple[str, ...]]
    primitive_family_to_splits: Mapping[str, tuple[str, ...]]
    format_family_to_splits: Mapping[str, tuple[str, ...]]
    record_hashes: Mapping[str, str]
    config_hash: str
    records_hash: str
    manifest_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise SplitManifestError("seed must be an integer.")
        if not isinstance(self.record_count, int) or isinstance(self.record_count, bool) or self.record_count < 0:
            raise SplitManifestError("record_count must be a non-negative integer.")
        object.__setattr__(self, "split_counts", _normalize_int_mapping(self.split_counts, "split_counts"))
        object.__setattr__(self, "split_record_ids", _normalize_tuple_mapping(self.split_record_ids, "split_record_ids"))
        object.__setattr__(self, "family_to_splits", _normalize_tuple_mapping(self.family_to_splits, "family_to_splits"))
        object.__setattr__(
            self,
            "primitive_family_to_splits",
            _normalize_tuple_mapping(self.primitive_family_to_splits, "primitive_family_to_splits"),
        )
        object.__setattr__(
            self,
            "format_family_to_splits",
            _normalize_tuple_mapping(self.format_family_to_splits, "format_family_to_splits"),
        )
        object.__setattr__(self, "record_hashes", {str(key): str(value) for key, value in self.record_hashes.items()})
        flattened_ids = _flatten_split_record_ids(self.split_record_ids)
        if sum(self.split_counts.values()) != self.record_count:
            raise SplitManifestError("split_counts must sum to record_count.")
        if len(flattened_ids) != self.record_count:
            raise SplitManifestError("split_record_ids count must equal record_count.")
        if len(flattened_ids) != len(set(flattened_ids)):
            raise SplitManifestError("split_record_ids cannot contain duplicate problem IDs.")
        for split, ids in self.split_record_ids.items():
            if len(ids) != self.split_counts.get(split, 0):
                raise SplitManifestError(f"split_record_ids count does not match split_counts for {split}.")
        if len(self.record_hashes) != self.record_count:
            raise SplitManifestError("record_hashes count must equal record_count.")
        if set(flattened_ids) != set(self.record_hashes):
            raise SplitManifestError("split_record_ids must match record_hashes keys.")
        if self.records_hash != stable_hash(self.record_hashes):
            raise SplitManifestError("records_hash does not match record_hashes.")
        if not self.manifest_id.strip():
            raise SplitManifestError("manifest_id must be non-empty.")
        if not self.config_hash.strip() or not self.records_hash.strip():
            raise SplitManifestError("config_hash and records_hash must be non-empty.")
        expected = _manifest_payload_hash(self)
        if self.manifest_hash and self.manifest_hash != expected:
            raise SplitManifestError("manifest_hash does not match manifest payload.")
        if not self.manifest_hash:
            object.__setattr__(self, "manifest_hash", expected)


def assign_splits(
    records: Iterable[ImmutableProblemRecord],
    *,
    seed: int,
    train_fraction: float = 0.6,
    dev_fraction: float = 0.2,
    forbidden_holdout_fraction: float = 0.2,
    allow_primitive_train_forbidden_overlap: bool = False,
    allow_format_train_forbidden_overlap: bool = False,
) -> tuple[tuple[ImmutableProblemRecord, ...], SplitManifest]:
    """Assign records to splits by family, deterministically."""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise SplitManifestError("seed must be an integer.")
    _validate_fraction(train_fraction, "train_fraction")
    _validate_fraction(dev_fraction, "dev_fraction")
    _validate_fraction(forbidden_holdout_fraction, "forbidden_holdout_fraction")
    if train_fraction + dev_fraction + forbidden_holdout_fraction > 1.0:
        raise SplitManifestError("split fractions cannot sum above 1.")
    materialized = tuple(records)
    _require_records(materialized)
    families = split_records_by_family(materialized)
    ordered_family_ids = sorted(
        families,
        key=lambda family_id: stable_hash({"seed": seed, "family_id": family_id}),
    )
    assignments: dict[str, SplitName] = {}
    family_count = len(ordered_family_ids)
    for index, family_id in enumerate(ordered_family_ids):
        position = index / max(family_count, 1)
        if position < train_fraction:
            split = SplitName.TRAIN
        elif position < train_fraction + dev_fraction:
            split = SplitName.DEV
        elif position < train_fraction + dev_fraction + forbidden_holdout_fraction:
            split = SplitName.FORBIDDEN_HOLDOUT
        else:
            split = SplitName.HARD_DEV
        assignments[family_id] = split

    assigned: list[ImmutableProblemRecord] = []
    for record in materialized:
        split = assignments[record.family_id]
        assigned.append(_record_with_split_permissions(record, split))

    assigned_tuple = tuple(sorted(assigned, key=lambda record: record.problem_id))
    manifest = _build_manifest(
        assigned_tuple,
        seed=seed,
        config={
            "train_fraction": train_fraction,
            "dev_fraction": dev_fraction,
            "forbidden_holdout_fraction": forbidden_holdout_fraction,
            "allow_primitive_train_forbidden_overlap": allow_primitive_train_forbidden_overlap,
            "allow_format_train_forbidden_overlap": allow_format_train_forbidden_overlap,
        },
    )
    validate_split_manifest(
        manifest,
        assigned_tuple,
        allow_primitive_train_forbidden_overlap=allow_primitive_train_forbidden_overlap,
        allow_format_train_forbidden_overlap=allow_format_train_forbidden_overlap,
    )
    return assigned_tuple, manifest


def split_records_by_family(records: Iterable[ImmutableProblemRecord]) -> dict[str, tuple[ImmutableProblemRecord, ...]]:
    grouped: dict[str, list[ImmutableProblemRecord]] = {}
    for record in records:
        if not isinstance(record, ImmutableProblemRecord):
            raise SplitManifestError("split records must be ImmutableProblemRecord instances.")
        grouped.setdefault(record.family_id, []).append(record)
    return {family_id: tuple(sorted(items, key=lambda record: record.problem_id)) for family_id, items in sorted(grouped.items())}


def write_split_manifest(manifest: SplitManifest, path: str | Path) -> None:
    Path(path).write_text(stable_json_dumps(manifest) + "\n", encoding="utf-8")


def read_split_manifest(path: str | Path) -> SplitManifest:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SplitManifestError(f"Invalid split manifest JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SplitManifestError("Split manifest JSON must be an object.")
    allowed = {item.name for item in fields(SplitManifest)}
    unknown = set(payload) - allowed
    if unknown:
        raise SplitManifestError(f"Split manifest contains unknown fields: {sorted(unknown)}")
    try:
        return SplitManifest(**payload)
    except (TypeError, ValueError, SchemaValidationError) as exc:
        raise SplitManifestError(f"Malformed split manifest: {path}") from exc


def validate_split_manifest(
    manifest: SplitManifest,
    records: Iterable[ImmutableProblemRecord] | None = None,
    *,
    allow_primitive_train_forbidden_overlap: bool = False,
    allow_format_train_forbidden_overlap: bool = False,
) -> bool:
    if not isinstance(manifest, SplitManifest):
        raise SplitManifestError("manifest must be a SplitManifest.")
    if manifest.manifest_hash != _manifest_payload_hash(manifest):
        raise SplitManifestError("manifest_hash does not match manifest payload.")
    if sum(manifest.split_counts.values()) != manifest.record_count:
        raise SplitManifestError("split_counts must sum to record_count.")
    _reject_train_forbidden_overlap(manifest.family_to_splits, "family_id", allow_overlap=False)
    _reject_train_forbidden_overlap(
        manifest.primitive_family_to_splits,
        "primitive_family_id",
        allow_overlap=allow_primitive_train_forbidden_overlap,
    )
    _reject_train_forbidden_overlap(
        manifest.format_family_to_splits,
        "format_family_id",
        allow_overlap=allow_format_train_forbidden_overlap,
    )
    if records is not None:
        materialized = tuple(records)
        if len(materialized) != manifest.record_count:
            raise SplitManifestError("record count does not match manifest.")
        provided_ids = {record.problem_id for record in materialized}
        manifest_hash_ids = set(manifest.record_hashes)
        manifest_split_ids = set(_flatten_split_record_ids(manifest.split_record_ids))
        if provided_ids != manifest_hash_ids:
            raise SplitManifestError("provided record IDs must exactly match manifest.record_hashes keys.")
        if provided_ids != manifest_split_ids:
            raise SplitManifestError("provided record IDs must exactly match manifest split_record_ids.")
        for record in materialized:
            if stable_hash(record) != manifest.record_hashes.get(record.problem_id):
                raise SplitManifestError(f"record hash mismatch for {record.problem_id}.")
            if record.split in {SplitName.FORBIDDEN_HOLDOUT, SplitName.STRESS_ONLY}:
                if record.allowed_for_sft or record.allowed_for_dpo or record.allowed_for_grpo:
                    raise SplitManifestError(f"{record.split.value} cannot allow SFT/DPO/GRPO.")
    return True


def _record_with_split_permissions(record: ImmutableProblemRecord, split: SplitName) -> ImmutableProblemRecord:
    contaminated = bool(record.contamination_flags)
    train_allowed = split == SplitName.TRAIN and not contaminated
    return replace(
        record,
        split=split,
        allowed_for_sft=train_allowed,
        allowed_for_dpo=train_allowed,
        allowed_for_grpo=False,
        allowed_for_eval=True,
    )


def _build_manifest(records: tuple[ImmutableProblemRecord, ...], *, seed: int, config: Mapping[str, object]) -> SplitManifest:
    split_record_ids: dict[str, tuple[str, ...]] = {}
    split_counts: dict[str, int] = {}
    for split in SplitName:
        ids = tuple(record.problem_id for record in records if record.split == split)
        split_record_ids[split.value] = ids
        split_counts[split.value] = len(ids)
    record_hashes = {record.problem_id: stable_hash(record) for record in records}
    config_hash = stable_hash({"seed": seed, **dict(config)})
    records_hash = stable_hash(record_hashes)
    payload = {
        "seed": seed,
        "record_hashes": record_hashes,
        "records_hash": records_hash,
        "config_hash": config_hash,
    }
    return SplitManifest(
        manifest_id=f"split-{stable_hash(payload)[:16]}",
        seed=seed,
        record_count=len(records),
        split_counts=split_counts,
        split_record_ids=split_record_ids,
        family_to_splits=_id_to_splits(records, "family_id"),
        primitive_family_to_splits=_id_to_splits(records, "primitive_family_id"),
        format_family_to_splits=_id_to_splits(records, "format_family_id"),
        record_hashes=record_hashes,
        config_hash=config_hash,
        records_hash=records_hash,
    )


def _id_to_splits(records: tuple[ImmutableProblemRecord, ...], field_name: str) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, set[str]] = {}
    for record in records:
        grouped.setdefault(str(getattr(record, field_name)), set()).add(record.split.value)
    return {key: tuple(sorted(values)) for key, values in sorted(grouped.items())}


def _reject_train_forbidden_overlap(mapping: Mapping[str, tuple[str, ...]], name: str, *, allow_overlap: bool) -> None:
    if allow_overlap:
        return
    for value, splits in mapping.items():
        split_set = set(splits)
        if SplitName.TRAIN.value in split_set and SplitName.FORBIDDEN_HOLDOUT.value in split_set:
            raise SplitManifestError(f"{name} {value!r} appears in train and forbidden_holdout.")


def _manifest_payload_hash(manifest: SplitManifest) -> str:
    return stable_hash({item.name: getattr(manifest, item.name) for item in fields(SplitManifest) if item.name != "manifest_hash"})


def _normalize_tuple_mapping(mapping: Mapping[str, Iterable[str]], field_name: str) -> dict[str, tuple[str, ...]]:
    if not isinstance(mapping, Mapping):
        raise SplitManifestError(f"{field_name} must be a mapping.")
    return {str(key): tuple(str(item) for item in value) for key, value in sorted(mapping.items())}


def _normalize_int_mapping(mapping: Mapping[str, int], field_name: str) -> dict[str, int]:
    if not isinstance(mapping, Mapping):
        raise SplitManifestError(f"{field_name} must be a mapping.")
    normalized: dict[str, int] = {}
    for key, value in sorted(mapping.items()):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise SplitManifestError(f"{field_name} values must be non-negative integers.")
        normalized[str(key)] = value
    return normalized


def _flatten_split_record_ids(mapping: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    flattened: list[str] = []
    for ids in mapping.values():
        flattened.extend(str(item) for item in ids)
    return tuple(flattened)


def _validate_fraction(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0.0 or value > 1.0:
        raise SplitManifestError(f"{field_name} must be a number in [0, 1].")


def _require_records(records: tuple[ImmutableProblemRecord, ...]) -> None:
    if not records:
        raise SplitManifestError("at least one record is required.")
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, ImmutableProblemRecord):
            raise SplitManifestError("records must be ImmutableProblemRecord instances.")
        if record.problem_id in seen:
            raise RegistryError(f"Duplicate problem_id: {record.problem_id}")
        seen.add(record.problem_id)


__all__ = [
    "SplitManifest",
    "SplitManifestError",
    "assign_splits",
    "read_split_manifest",
    "split_records_by_family",
    "validate_split_manifest",
    "write_split_manifest",
]
