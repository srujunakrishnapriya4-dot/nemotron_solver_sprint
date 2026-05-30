from __future__ import annotations

import pytest

from nemotron_engine.core.schemas import ImmutableProblemRecord, ProblemSource, SplitName
from nemotron_engine.data.mutation_generator import MutationGenerationError, mutate_record


def record(problem_id: str, split: SplitName, *, parent_ids: tuple[str, ...] = ("root",)) -> ImmutableProblemRecord:
    return ImmutableProblemRecord(
        problem_id=problem_id,
        source=ProblemSource.MANUAL_FIXTURE,
        source_hash=f"hash-{problem_id}",
        family_id="fam",
        primitive_family_id="prim",
        format_family_id="fmt",
        prompt_wrapper_id="wrapper",
        split=split,
        parent_ids=parent_ids,
        allowed_for_sft=split == SplitName.TRAIN,
        allowed_for_dpo=split == SplitName.TRAIN,
        allowed_for_grpo=False,
        allowed_for_eval=True,
    )


def test_mutation_preserves_split_and_parent_ids() -> None:
    parent = record("parent", SplitName.TRAIN, parent_ids=("seed-parent",))

    mutated = mutate_record(parent, mutation="wrapper_text_change", seed=1)

    assert mutated.split == parent.split
    assert mutated.parent_ids == parent.parent_ids
    assert mutated.created_from == parent.problem_id


def test_mutated_problem_id_is_deterministic() -> None:
    parent = record("parent", SplitName.TRAIN)

    first = mutate_record(parent, mutation="whitespace_formatting", seed=2)
    second = mutate_record(parent, mutation="whitespace_formatting", seed=2)

    assert first.problem_id == second.problem_id


def test_mutation_does_not_silently_overwrite_existing_id() -> None:
    parent = record("parent", SplitName.TRAIN)
    existing = [mutate_record(parent, mutation="target_label_wording", seed=3)]

    with pytest.raises(MutationGenerationError):
        mutate_record(parent, mutation="target_label_wording", seed=3, existing_records=existing)


def test_train_mutation_from_private_like_parent_rejected() -> None:
    parent = record("parent", SplitName.PRIVATE_LIKE)

    with pytest.raises(MutationGenerationError):
        mutate_record(parent, mutation="wrapper_text_change", seed=4, target_split=SplitName.TRAIN)


def test_private_like_mutation_remains_eval_only() -> None:
    parent = record("parent", SplitName.PRIVATE_LIKE)

    mutated = mutate_record(parent, mutation="wrapper_text_change", seed=5)

    assert mutated.split == SplitName.PRIVATE_LIKE
    assert not mutated.allowed_for_sft
    assert not mutated.allowed_for_dpo
    assert not mutated.allowed_for_grpo
    assert mutated.allowed_for_eval


def test_example_order_shuffle_deterministic_and_source_hash_changes() -> None:
    parent = record("parent", SplitName.TRAIN)

    mutated = mutate_record(parent, mutation="example_order_shuffle", seed=6)
    repeated = mutate_record(parent, mutation="example_order_shuffle", seed=6)

    assert mutated == repeated
    assert mutated.source_hash != parent.source_hash


def test_forbidden_parent_remains_eval_only() -> None:
    parent = record("parent", SplitName.FORBIDDEN_HOLDOUT)

    mutated = mutate_record(parent, mutation="symbol_alphabet_rename", seed=7)

    assert mutated.split == SplitName.FORBIDDEN_HOLDOUT
    assert not mutated.allowed_for_sft
    assert mutated.allowed_for_eval

