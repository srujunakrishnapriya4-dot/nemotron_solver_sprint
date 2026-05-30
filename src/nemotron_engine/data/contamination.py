"""Cross-split contamination checks for Pass 4."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Iterable, Mapping

from nemotron_engine.core.schemas import ImmutableProblemRecord, SplitName, stable_hash


class ContaminationError(ValueError):
    """Raised when contamination checks cannot be completed."""


@dataclass(frozen=True)
class ContaminationReport:
    checked_count: int
    contaminated_count: int
    contaminated_problem_ids: tuple[str, ...] = ()
    flags_by_problem_id: Mapping[str, tuple[str, ...]] = None  # type: ignore[assignment]
    duplicate_problem_ids: tuple[str, ...] = ()
    prompt_hash_collisions: tuple[tuple[str, str, str], ...] = ()
    normalized_prompt_hash_collisions: tuple[tuple[str, str, str], ...] = ()
    family_signature_collisions: tuple[tuple[str, str, str], ...] = ()
    parent_crossings: tuple[tuple[str, str], ...] = ()
    created_from_crossings: tuple[tuple[str, str], ...] = ()
    answer_hash_weak_collisions: tuple[tuple[str, str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.checked_count < 0 or self.contaminated_count < 0:
            raise ContaminationError("counts must be non-negative.")
        if self.checked_count < self.contaminated_count:
            raise ContaminationError("checked_count cannot be smaller than contaminated_count.")
        flags = dict(self.flags_by_problem_id or {})
        normalized = {str(key): tuple(str(item) for item in value) for key, value in sorted(flags.items())}
        contaminated_ids = tuple(sorted(str(item) for item in self.contaminated_problem_ids))
        if self.contaminated_count != len(contaminated_ids):
            raise ContaminationError("contaminated_count must equal len(contaminated_problem_ids).")
        if set(normalized) != set(contaminated_ids):
            raise ContaminationError("flags_by_problem_id keys must match contaminated_problem_ids.")
        for problem_id, problem_flags in normalized.items():
            if not problem_flags or any(not str(flag).strip() for flag in problem_flags):
                raise ContaminationError(f"contaminated problem {problem_id} must have at least one non-empty flag.")
        object.__setattr__(self, "flags_by_problem_id", normalized)
        object.__setattr__(self, "contaminated_problem_ids", contaminated_ids)


def compute_prompt_hash(prompt: str) -> str:
    if not isinstance(prompt, str):
        raise ContaminationError("prompt must be a string.")
    return stable_hash({"prompt": prompt})


def compute_normalized_prompt_hash(prompt: str) -> str:
    if not isinstance(prompt, str):
        raise ContaminationError("prompt must be a string.")
    normalized = re.sub(r"\s+", " ", prompt.strip().lower())
    return stable_hash({"normalized_prompt": normalized})


def compute_family_signature_hash(record: ImmutableProblemRecord) -> str:
    _require_record(record)
    return stable_hash(
        {
            "family_id": record.family_id,
            "primitive_family_id": record.primitive_family_id,
            "format_family_id": record.format_family_id,
            "prompt_wrapper_id": record.prompt_wrapper_id,
        }
    )


def compute_answer_hash(answer: str) -> str:
    if not isinstance(answer, str):
        raise ContaminationError("answer must be a string.")
    return stable_hash({"answer": answer})


def detect_cross_split_contamination(
    records: Iterable[ImmutableProblemRecord],
    *,
    prompts: Mapping[str, str] | None = None,
    answers: Mapping[str, str] | None = None,
) -> ContaminationReport:
    materialized = tuple(records)
    for record in materialized:
        _require_record(record)
    by_id: dict[str, ImmutableProblemRecord] = {}
    duplicates: set[str] = set()
    flags: dict[str, set[str]] = {}
    for record in materialized:
        if record.problem_id in by_id:
            duplicates.add(record.problem_id)
            flags.setdefault(record.problem_id, set()).add("duplicate_problem_id")
        by_id[record.problem_id] = record

    prompt_collisions = _collisions_from_hashes(
        materialized,
        prompts or {},
        compute_prompt_hash,
        "prompt_hash_collision",
        flags,
    )
    normalized_collisions = _collisions_from_hashes(
        materialized,
        prompts or {},
        compute_normalized_prompt_hash,
        "normalized_prompt_hash_collision",
        flags,
    )
    family_collisions = _family_signature_collisions(materialized, flags)
    parent_crossings = _parent_crossings(materialized, by_id, flags)
    created_crossings = _created_from_crossings(materialized, by_id, flags)
    answer_weak = _answer_weak_collisions(materialized, answers or {})

    contaminated_ids = tuple(sorted(flags))
    return ContaminationReport(
        checked_count=len(materialized),
        contaminated_count=len(contaminated_ids),
        contaminated_problem_ids=contaminated_ids,
        flags_by_problem_id={problem_id: tuple(sorted(values)) for problem_id, values in sorted(flags.items())},
        duplicate_problem_ids=tuple(sorted(duplicates)),
        prompt_hash_collisions=tuple(prompt_collisions),
        normalized_prompt_hash_collisions=tuple(normalized_collisions),
        family_signature_collisions=tuple(family_collisions),
        parent_crossings=tuple(parent_crossings),
        created_from_crossings=tuple(created_crossings),
        answer_hash_weak_collisions=tuple(answer_weak),
    )


def mark_contaminated_records(
    records: Iterable[ImmutableProblemRecord],
    report: ContaminationReport,
) -> tuple[ImmutableProblemRecord, ...]:
    flagged = dict(report.flags_by_problem_id)
    updated: list[ImmutableProblemRecord] = []
    for record in records:
        _require_record(record)
        flags = tuple(sorted(set(record.contamination_flags) | set(flagged.get(record.problem_id, ()))))
        if flags:
            updated.append(
                replace(
                    record,
                    contamination_flags=flags,
                    allowed_for_sft=False,
                    allowed_for_dpo=False,
                    allowed_for_grpo=False,
                )
            )
        else:
            updated.append(record)
    return tuple(updated)


def _collisions_from_hashes(
    records: tuple[ImmutableProblemRecord, ...],
    values: Mapping[str, str],
    hash_fn,
    flag: str,
    flags: dict[str, set[str]],
) -> list[tuple[str, str, str]]:
    buckets: dict[str, list[ImmutableProblemRecord]] = {}
    for record in records:
        if record.problem_id not in values:
            continue
        buckets.setdefault(hash_fn(values[record.problem_id]), []).append(record)
    collisions: list[tuple[str, str, str]] = []
    for digest, bucket in buckets.items():
        for left_index, left in enumerate(bucket):
            for right in bucket[left_index + 1 :]:
                if _incompatible(left.split, right.split):
                    collisions.append((left.problem_id, right.problem_id, digest))
                    _flag_train_side(left, right, flag, flags)
    return collisions


def _family_signature_collisions(
    records: tuple[ImmutableProblemRecord, ...],
    flags: dict[str, set[str]],
) -> list[tuple[str, str, str]]:
    buckets: dict[str, list[ImmutableProblemRecord]] = {}
    for record in records:
        buckets.setdefault(compute_family_signature_hash(record), []).append(record)
    collisions: list[tuple[str, str, str]] = []
    for digest, bucket in buckets.items():
        for left_index, left in enumerate(bucket):
            for right in bucket[left_index + 1 :]:
                if {left.split, right.split} == {SplitName.TRAIN, SplitName.FORBIDDEN_HOLDOUT}:
                    collisions.append((left.problem_id, right.problem_id, digest))
                    _flag_train_side(left, right, "family_signature_collision", flags)
    return collisions


def _parent_crossings(
    records: tuple[ImmutableProblemRecord, ...],
    by_id: Mapping[str, ImmutableProblemRecord],
    flags: dict[str, set[str]],
) -> list[tuple[str, str]]:
    crossings: list[tuple[str, str]] = []
    for record in records:
        if record.split != SplitName.TRAIN:
            continue
        for parent_id in record.parent_ids:
            parent = by_id.get(parent_id)
            if parent is not None and _is_eval_or_holdout(parent.split):
                crossings.append((record.problem_id, parent_id))
                flags.setdefault(record.problem_id, set()).add("parent_cross_split")
    return crossings


def _created_from_crossings(
    records: tuple[ImmutableProblemRecord, ...],
    by_id: Mapping[str, ImmutableProblemRecord],
    flags: dict[str, set[str]],
) -> list[tuple[str, str]]:
    crossings: list[tuple[str, str]] = []
    for record in records:
        if record.split != SplitName.TRAIN or not record.created_from:
            continue
        source = by_id.get(record.created_from)
        if source is not None and _is_eval_or_holdout(source.split):
            crossings.append((record.problem_id, record.created_from))
            flags.setdefault(record.problem_id, set()).add("created_from_cross_split")
    return crossings


def _answer_weak_collisions(
    records: tuple[ImmutableProblemRecord, ...],
    answers: Mapping[str, str],
) -> list[tuple[str, str, str]]:
    buckets: dict[str, list[ImmutableProblemRecord]] = {}
    for record in records:
        if record.problem_id in answers:
            buckets.setdefault(compute_answer_hash(answers[record.problem_id]), []).append(record)
    weak: list[tuple[str, str, str]] = []
    for digest, bucket in buckets.items():
        for left_index, left in enumerate(bucket):
            for right in bucket[left_index + 1 :]:
                if _incompatible(left.split, right.split):
                    weak.append((left.problem_id, right.problem_id, digest))
    return weak


def _flag_train_side(
    left: ImmutableProblemRecord,
    right: ImmutableProblemRecord,
    flag: str,
    flags: dict[str, set[str]],
) -> None:
    if left.split == SplitName.TRAIN:
        flags.setdefault(left.problem_id, set()).add(flag)
    if right.split == SplitName.TRAIN:
        flags.setdefault(right.problem_id, set()).add(flag)


def _incompatible(left: SplitName, right: SplitName) -> bool:
    return (left == SplitName.TRAIN and _is_eval_or_holdout(right)) or (right == SplitName.TRAIN and _is_eval_or_holdout(left))


def _is_eval_or_holdout(split: SplitName) -> bool:
    return split in {SplitName.DEV, SplitName.HARD_DEV, SplitName.PRIVATE_LIKE, SplitName.FORBIDDEN_HOLDOUT, SplitName.STRESS_ONLY}


def _require_record(record: ImmutableProblemRecord) -> None:
    if not isinstance(record, ImmutableProblemRecord):
        raise ContaminationError("records must be ImmutableProblemRecord instances.")


__all__ = [
    "ContaminationError",
    "ContaminationReport",
    "compute_answer_hash",
    "compute_family_signature_hash",
    "compute_normalized_prompt_hash",
    "compute_prompt_hash",
    "detect_cross_split_contamination",
    "mark_contaminated_records",
]
