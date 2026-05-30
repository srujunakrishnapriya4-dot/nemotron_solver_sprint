"""Controlled deterministic prompt-metadata mutation for Pass 4."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Mapping

from nemotron_engine.core.registry import validate_registry
from nemotron_engine.core.schemas import ImmutableProblemRecord, ProblemSource, SplitName, stable_hash


class MutationGenerationError(ValueError):
    """Raised when a mutation would violate split safety."""


SUPPORTED_MUTATIONS = frozenset(
    {
        "wrapper_text_change",
        "symbol_alphabet_rename",
        "example_order_shuffle",
        "whitespace_formatting",
        "target_label_wording",
    }
)


def mutate_record(
    record: ImmutableProblemRecord,
    *,
    mutation: str,
    seed: int,
    target_split: SplitName | str | None = None,
    existing_records: Iterable[ImmutableProblemRecord] = (),
    generator_params: Mapping[str, object] | None = None,
) -> ImmutableProblemRecord:
    if not isinstance(record, ImmutableProblemRecord):
        raise MutationGenerationError("record must be an ImmutableProblemRecord.")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise MutationGenerationError("seed must be an integer.")
    if mutation not in SUPPORTED_MUTATIONS:
        raise MutationGenerationError(f"unsupported mutation: {mutation}")
    if target_split is not None and SplitName(target_split) != record.split:
        raise MutationGenerationError("mutation must preserve the parent split.")
    if record.split == SplitName.TRAIN and record.contamination_flags:
        raise MutationGenerationError("contaminated train records cannot be mutated for training.")
    if record.split != SplitName.TRAIN and (record.allowed_for_sft or record.allowed_for_dpo or record.allowed_for_grpo):
        raise MutationGenerationError("non-train parent cannot carry training permissions.")

    params = {
        "source": ProblemSource.MUTATION.value,
        "split": record.split.value,
        "family_id": record.family_id,
        "primitive_family_id": record.primitive_family_id,
        "format_family_id": record.format_family_id,
        "prompt_wrapper_id": f"{record.prompt_wrapper_id}:{mutation}",
        "parent_ids": record.parent_ids,
        "seed": seed,
        "mutation": mutation,
        "parent_problem_id": record.problem_id,
        "generator_params": dict(generator_params or {}),
    }
    problem_id = f"mut-{stable_hash(params)[:24]}"
    existing_ids = {item.problem_id for item in existing_records}
    if problem_id in existing_ids:
        raise MutationGenerationError(f"generated problem_id collision: {problem_id}")
    split = record.split
    train_allowed = split == SplitName.TRAIN and not record.contamination_flags
    mutated = replace(
        record,
        problem_id=problem_id,
        source=ProblemSource.MUTATION,
        source_hash=stable_hash({"mutation": params}),
        prompt_wrapper_id=str(params["prompt_wrapper_id"]),
        split=split,
        created_from=record.problem_id,
        parent_ids=record.parent_ids,
        allowed_for_sft=train_allowed,
        allowed_for_dpo=train_allowed,
        allowed_for_grpo=False,
        allowed_for_eval=True,
    )
    validate_registry([mutated])
    return mutated


__all__ = ["MutationGenerationError", "SUPPORTED_MUTATIONS", "mutate_record"]
