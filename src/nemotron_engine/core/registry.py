"""Immutable problem registry helpers."""

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
from typing import Iterable

from .schemas import (
    ImmutableProblemRecord,
    QuarantineRecord,
    RegistryError,
    SchemaValidationError,
    SplitName,
    stable_hash,
    stable_json_dumps,
)


def register_problem(record: ImmutableProblemRecord) -> ImmutableProblemRecord:
    validate_registry([record])
    return record


def load_registry_jsonl(path: Path) -> list[ImmutableProblemRecord]:
    records: list[ImmutableProblemRecord] = []
    allowed = {item.name for item in fields(ImmutableProblemRecord)}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RegistryError(f"Invalid registry JSONL at line {line_no}.") from exc
        if not isinstance(payload, dict):
            raise RegistryError(f"Registry JSONL row {line_no} must be an object.")
        unknown = set(payload) - allowed
        if unknown:
            raise RegistryError(f"Registry JSONL row {line_no} contains unknown fields: {sorted(unknown)}")
        try:
            records.append(ImmutableProblemRecord(**payload))
        except (TypeError, SchemaValidationError, ValueError) as exc:
            raise RegistryError(f"Malformed registry record at line {line_no}.") from exc
    validate_registry(records)
    return records


def write_registry_jsonl(records: Iterable[ImmutableProblemRecord], path: Path) -> None:
    materialized = validate_registry(records)
    text = "".join(stable_json_dumps(record) + "\n" for record in materialized)
    path.write_text(text, encoding="utf-8")


def validate_registry(records: Iterable[ImmutableProblemRecord]) -> list[ImmutableProblemRecord]:
    materialized = list(records)
    seen: set[str] = set()
    for record in materialized:
        if not isinstance(record, ImmutableProblemRecord):
            raise RegistryError("Registry entries must be ImmutableProblemRecord instances.")
        if record.problem_id in seen:
            raise RegistryError(f"Duplicate problem_id: {record.problem_id}")
        seen.add(record.problem_id)
        if record.contamination_flags and (record.allowed_for_sft or record.allowed_for_dpo or record.allowed_for_grpo):
            raise RegistryError("Contaminated records cannot allow SFT/DPO/GRPO.")
        if record.split in {SplitName.FORBIDDEN_HOLDOUT, SplitName.STRESS_ONLY}:
            if record.allowed_for_sft or record.allowed_for_dpo or record.allowed_for_grpo:
                raise RegistryError(f"{record.split.value} cannot allow SFT/DPO/GRPO.")
    return materialized


def quarantine_problem(record: ImmutableProblemRecord, reason: str) -> QuarantineRecord:
    return QuarantineRecord(problem_id=record.problem_id, reason=reason, record_hash=stable_hash(record))


def split_allows_sft(record: ImmutableProblemRecord) -> bool:
    if record.split in {SplitName.FORBIDDEN_HOLDOUT, SplitName.STRESS_ONLY}:
        return False
    if record.contamination_flags:
        return False
    return record.split == SplitName.TRAIN and bool(record.allowed_for_sft)


def split_allows_eval(record: ImmutableProblemRecord) -> bool:
    if record.contamination_flags and record.split == SplitName.TRAIN:
        return False
    return bool(record.allowed_for_eval)


__all__ = [
    "load_registry_jsonl",
    "quarantine_problem",
    "register_problem",
    "split_allows_eval",
    "split_allows_sft",
    "validate_registry",
    "write_registry_jsonl",
]
