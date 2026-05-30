"""Tiny deterministic synthetic DSL record generator for Pass 4."""

from __future__ import annotations

from typing import Iterable, Mapping

from nemotron_engine.core.registry import validate_registry
from nemotron_engine.core.schemas import ImmutableProblemRecord, ProblemSource, SplitName, stable_hash


class SyntheticGenerationError(ValueError):
    """Raised when synthetic DSL generation is unsafe or invalid."""


SUPPORTED_SYNTHETIC_PRIMITIVES = frozenset({"bit_not", "reverse", "add_const", "affine_small", "digit_sum", "binary_op"})


def generate_synthetic_dsl_records(
    *,
    split: SplitName | str,
    seed: int,
    count: int = 1,
    primitive: str = "bit_not",
    family_id: str | None = None,
    format_family_id: str | None = None,
    prompt_wrapper_id: str = "canonical_arrow",
    parent_ids: Iterable[str] = (),
    existing_records: Iterable[ImmutableProblemRecord] = (),
) -> tuple[ImmutableProblemRecord, ...]:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise SyntheticGenerationError("seed must be an integer.")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise SyntheticGenerationError("count must be a positive integer.")
    if primitive not in SUPPORTED_SYNTHETIC_PRIMITIVES:
        raise SyntheticGenerationError(f"unsupported synthetic primitive: {primitive}")
    split_value = SplitName(split)
    parents = tuple(str(item) for item in parent_ids)
    existing_materialized = tuple(existing_records)
    existing_ids = _existing_ids(existing_materialized)
    if split_value == SplitName.TRAIN and parents:
        _reject_unsafe_train_parent_context(parents, existing_materialized)
    records: list[ImmutableProblemRecord] = []
    for index in range(count):
        record_family = family_id or f"synthetic:{primitive}:{seed}"
        primitive_family_id = f"primitive:{primitive}"
        record_format = format_family_id or _format_family_for_primitive(primitive)
        params = {
            "source": ProblemSource.SYNTHETIC_DSL.value,
            "split": split_value.value,
            "family_id": record_family,
            "primitive_family_id": primitive_family_id,
            "format_family_id": record_format,
            "prompt_wrapper_id": prompt_wrapper_id,
            "parent_ids": parents,
            "seed": seed,
            "primitive": primitive,
            "index": index,
        }
        problem_id = _generated_problem_id(params)
        if problem_id in existing_ids or any(record.problem_id == problem_id for record in records):
            raise SyntheticGenerationError(f"generated problem_id collision: {problem_id}")
        record = ImmutableProblemRecord(
            problem_id=problem_id,
            source=ProblemSource.SYNTHETIC_DSL,
            source_hash=stable_hash({"synthetic_dsl": params}),
            family_id=record_family,
            primitive_family_id=primitive_family_id,
            format_family_id=record_format,
            prompt_wrapper_id=prompt_wrapper_id,
            split=split_value,
            created_from="synthetic_dsl_generator",
            parent_ids=parents,
            contamination_flags=(),
            **_permissions(split_value, contaminated=False),
        )
        records.append(record)
    validate_registry(records)
    return tuple(records)


def _format_family_for_primitive(primitive: str) -> str:
    if primitive in {"bit_not", "reverse"}:
        return "format:bitstring"
    if primitive in {"add_const", "affine_small", "digit_sum", "binary_op"}:
        return "format:integer"
    return "format:unknown"


def _generated_problem_id(params: Mapping[str, object]) -> str:
    return f"syn-{stable_hash(params)[:24]}"


def _existing_ids(existing_records: tuple[ImmutableProblemRecord, ...]) -> set[str]:
    ids: set[str] = set()
    for record in existing_records:
        problem_id = getattr(record, "problem_id", None)
        if not isinstance(problem_id, str) or not problem_id.strip():
            raise SyntheticGenerationError("existing_records must contain records with valid problem_id values.")
        ids.add(problem_id)
    return ids


def _reject_unsafe_train_parent_context(parent_ids: tuple[str, ...], existing_records: tuple[ImmutableProblemRecord, ...]) -> None:
    by_id = {record.problem_id: record for record in existing_records if isinstance(record, ImmutableProblemRecord)}
    for parent_id in parent_ids:
        parent = by_id.get(parent_id)
        if parent is None:
            continue
        if parent.split != SplitName.TRAIN:
            raise SyntheticGenerationError("train synthetic rows cannot use non-train parent context.")
        if parent.contamination_flags:
            raise SyntheticGenerationError("train synthetic rows cannot use contaminated parent context.")


def _permissions(split: SplitName, *, contaminated: bool) -> dict[str, bool]:
    train_allowed = split == SplitName.TRAIN and not contaminated
    return {
        "allowed_for_sft": train_allowed,
        "allowed_for_dpo": train_allowed,
        "allowed_for_grpo": False,
        "allowed_for_eval": True,
    }


__all__ = ["SUPPORTED_SYNTHETIC_PRIMITIVES", "SyntheticGenerationError", "generate_synthetic_dsl_records"]
