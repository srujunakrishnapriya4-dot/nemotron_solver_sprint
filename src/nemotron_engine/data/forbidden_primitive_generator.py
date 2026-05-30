"""Evaluation-only forbidden primitive record generator for Pass 4."""

from __future__ import annotations

from typing import Iterable, Mapping

from nemotron_engine.core.registry import validate_registry
from nemotron_engine.core.schemas import ImmutableProblemRecord, ProblemSource, SplitName, stable_hash
from nemotron_engine.data.split_builder import SplitManifest


class ForbiddenPrimitiveGenerationError(ValueError):
    """Raised when forbidden primitive generation is unsafe."""


def generate_forbidden_primitive_records(
    *,
    seed: int,
    primitive_family_id: str,
    count: int = 1,
    split: SplitName | str = SplitName.FORBIDDEN_HOLDOUT,
    family_id: str | None = None,
    format_family_id: str = "format:forbidden",
    prompt_wrapper_id: str = "forbidden_holdout",
    parent_ids: Iterable[str] = (),
    train_manifest: SplitManifest | None = None,
    existing_records: Iterable[ImmutableProblemRecord] = (),
    generator_params: Mapping[str, object] | None = None,
    **ignored_permissions: object,
) -> tuple[ImmutableProblemRecord, ...]:
    del ignored_permissions
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ForbiddenPrimitiveGenerationError("seed must be an integer.")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ForbiddenPrimitiveGenerationError("count must be a positive integer.")
    split_value = SplitName(split)
    if split_value not in {SplitName.FORBIDDEN_HOLDOUT, SplitName.PRIVATE_LIKE}:
        raise ForbiddenPrimitiveGenerationError("forbidden primitive records must be forbidden_holdout or private_like.")
    if train_manifest is not None:
        train_splits = set(train_manifest.primitive_family_to_splits.get(primitive_family_id, ()))
        if SplitName.TRAIN.value in train_splits:
            raise ForbiddenPrimitiveGenerationError("forbidden primitive family is present in train manifest.")
    parents = tuple(str(item) for item in parent_ids)
    existing_ids = _existing_ids(tuple(existing_records))
    records: list[ImmutableProblemRecord] = []
    record_family = family_id or f"forbidden:{primitive_family_id}:{seed}"
    for index in range(count):
        params = {
            "source": ProblemSource.FORBIDDEN_PRIMITIVE.value,
            "split": split_value.value,
            "family_id": record_family,
            "primitive_family_id": primitive_family_id,
            "format_family_id": format_family_id,
            "prompt_wrapper_id": prompt_wrapper_id,
            "parent_ids": parents,
            "seed": seed,
            "index": index,
            "generator_params": dict(generator_params or {}),
        }
        problem_id = f"forbid-{stable_hash(params)[:22]}"
        if problem_id in existing_ids or any(record.problem_id == problem_id for record in records):
            raise ForbiddenPrimitiveGenerationError(f"generated problem_id collision: {problem_id}")
        records.append(
            ImmutableProblemRecord(
                problem_id=problem_id,
                source=ProblemSource.FORBIDDEN_PRIMITIVE,
                source_hash=stable_hash({"forbidden_primitive": params}),
                family_id=record_family,
                primitive_family_id=primitive_family_id,
                format_family_id=format_family_id,
                prompt_wrapper_id=prompt_wrapper_id,
                split=split_value,
                created_from="forbidden_primitive_generator",
                parent_ids=parents,
                contamination_flags=(),
                allowed_for_sft=False,
                allowed_for_dpo=False,
                allowed_for_grpo=False,
                allowed_for_eval=True,
            )
        )
    validate_registry(records)
    return tuple(records)


def _existing_ids(existing_records: tuple[ImmutableProblemRecord, ...]) -> set[str]:
    ids: set[str] = set()
    for record in existing_records:
        problem_id = getattr(record, "problem_id", None)
        if not isinstance(problem_id, str) or not problem_id.strip():
            raise ForbiddenPrimitiveGenerationError("existing_records must contain records with valid problem_id values.")
        ids.add(problem_id)
    return ids


__all__ = ["ForbiddenPrimitiveGenerationError", "generate_forbidden_primitive_records"]
