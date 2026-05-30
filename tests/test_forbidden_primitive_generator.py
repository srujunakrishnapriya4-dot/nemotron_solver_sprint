from __future__ import annotations

import pytest

from nemotron_engine.core.registry import validate_registry
from nemotron_engine.core.schemas import ImmutableProblemRecord, ProblemSource, SplitName
from nemotron_engine.data.forbidden_primitive_generator import (
    ForbiddenPrimitiveGenerationError,
    generate_forbidden_primitive_records,
)
from nemotron_engine.data.split_builder import assign_splits


def train_record() -> ImmutableProblemRecord:
    return ImmutableProblemRecord(
        problem_id="train",
        source=ProblemSource.MANUAL_FIXTURE,
        source_hash="hash-train",
        family_id="fam",
        primitive_family_id="prim-forbidden",
        format_family_id="fmt",
        prompt_wrapper_id="wrapper",
        split=SplitName.TRAIN,
        allowed_for_sft=True,
        allowed_for_dpo=True,
        allowed_for_grpo=False,
        allowed_for_eval=True,
    )


def test_forbidden_rows_are_eval_only() -> None:
    (record,) = generate_forbidden_primitive_records(seed=1, primitive_family_id="heldout-prim")

    assert record.source == ProblemSource.FORBIDDEN_PRIMITIVE
    assert record.split == SplitName.FORBIDDEN_HOLDOUT
    assert not record.allowed_for_sft
    assert not record.allowed_for_dpo
    assert not record.allowed_for_grpo
    assert record.allowed_for_eval


def test_private_like_forbidden_rows_remain_eval_only() -> None:
    (record,) = generate_forbidden_primitive_records(seed=1, primitive_family_id="heldout-prim", split=SplitName.PRIVATE_LIKE)

    assert record.split == SplitName.PRIVATE_LIKE
    assert not record.allowed_for_sft
    assert not record.allowed_for_dpo
    assert not record.allowed_for_grpo
    assert record.allowed_for_eval


def test_caller_cannot_force_training_permissions() -> None:
    (record,) = generate_forbidden_primitive_records(
        seed=2,
        primitive_family_id="heldout-prim",
        allowed_for_sft=True,
        allowed_for_dpo=True,
        allowed_for_grpo=True,
    )

    assert not record.allowed_for_sft
    assert not record.allowed_for_dpo
    assert not record.allowed_for_grpo


def test_forbidden_primitive_family_present_in_train_manifest_rejected() -> None:
    assigned, manifest = assign_splits([train_record()], seed=1, train_fraction=1.0, dev_fraction=0.0, forbidden_holdout_fraction=0.0)
    assert assigned[0].split == SplitName.TRAIN

    with pytest.raises(ForbiddenPrimitiveGenerationError):
        generate_forbidden_primitive_records(seed=3, primitive_family_id="prim-forbidden", train_manifest=manifest)


def test_generated_records_validate() -> None:
    records = generate_forbidden_primitive_records(seed=4, primitive_family_id="heldout-prim", count=2)

    assert validate_registry(records) == list(records)


def test_collision_with_existing_records_rejected() -> None:
    existing = generate_forbidden_primitive_records(seed=5, primitive_family_id="heldout-prim")

    with pytest.raises(ForbiddenPrimitiveGenerationError):
        generate_forbidden_primitive_records(seed=5, primitive_family_id="heldout-prim", existing_records=existing)


def test_malformed_existing_records_raise_explicit_error() -> None:
    with pytest.raises(ForbiddenPrimitiveGenerationError):
        generate_forbidden_primitive_records(seed=6, primitive_family_id="heldout-prim", existing_records=[object()])  # type: ignore[list-item]
