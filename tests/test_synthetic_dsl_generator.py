from __future__ import annotations

import pytest

from nemotron_engine.core.registry import validate_registry
from nemotron_engine.core.schemas import ImmutableProblemRecord, ProblemSource, SplitName
from nemotron_engine.data.synthetic_dsl_generator import SyntheticGenerationError, generate_synthetic_dsl_records


def test_generates_bit_not_train_row_with_required_metadata() -> None:
    (record,) = generate_synthetic_dsl_records(split=SplitName.TRAIN, seed=1, primitive="bit_not")

    assert record.source.value == "synthetic_dsl"
    assert record.split == SplitName.TRAIN
    assert record.primitive_family_id == "primitive:bit_not"
    assert record.allowed_for_sft
    assert record.allowed_for_dpo
    assert record.allowed_for_eval


def test_generates_affine_holdout_row_eval_only() -> None:
    (record,) = generate_synthetic_dsl_records(split=SplitName.FORBIDDEN_HOLDOUT, seed=2, primitive="affine_small")

    assert record.split == SplitName.FORBIDDEN_HOLDOUT
    assert not record.allowed_for_sft
    assert not record.allowed_for_dpo
    assert not record.allowed_for_grpo
    assert record.allowed_for_eval


def test_deterministic_with_seed_and_config() -> None:
    first = generate_synthetic_dsl_records(split=SplitName.TRAIN, seed=3, primitive="digit_sum")
    second = generate_synthetic_dsl_records(split=SplitName.TRAIN, seed=3, primitive="digit_sum")

    assert first == second


def test_generated_problem_id_changes_with_seed_or_config() -> None:
    first = generate_synthetic_dsl_records(split=SplitName.TRAIN, seed=3, primitive="digit_sum")
    second = generate_synthetic_dsl_records(split=SplitName.TRAIN, seed=4, primitive="digit_sum")
    third = generate_synthetic_dsl_records(split=SplitName.TRAIN, seed=3, primitive="reverse")

    assert first[0].problem_id != second[0].problem_id
    assert first[0].problem_id != third[0].problem_id


def test_generated_id_collision_rejected() -> None:
    existing = generate_synthetic_dsl_records(split=SplitName.TRAIN, seed=5, primitive="bit_not")

    with pytest.raises(SyntheticGenerationError):
        generate_synthetic_dsl_records(split=SplitName.TRAIN, seed=5, primitive="bit_not", existing_records=existing)


def test_malformed_existing_records_raise_explicit_error() -> None:
    with pytest.raises(SyntheticGenerationError):
        generate_synthetic_dsl_records(split=SplitName.TRAIN, seed=5, primitive="bit_not", existing_records=[object()])  # type: ignore[list-item]


def test_train_synthetic_row_rejects_contaminated_parent_context() -> None:
    parent = ImmutableProblemRecord(
        problem_id="parent",
        source=ProblemSource.MANUAL_FIXTURE,
        source_hash="hash-parent",
        family_id="fam",
        primitive_family_id="prim",
        format_family_id="fmt",
        prompt_wrapper_id="wrapper",
        split=SplitName.TRAIN,
        contamination_flags=("prompt_hash_collision",),
        allowed_for_sft=False,
        allowed_for_dpo=False,
        allowed_for_grpo=False,
        allowed_for_eval=True,
    )

    with pytest.raises(SyntheticGenerationError):
        generate_synthetic_dsl_records(
            split=SplitName.TRAIN,
            seed=5,
            primitive="bit_not",
            parent_ids=("parent",),
            existing_records=[parent],
        )


def test_train_synthetic_row_rejects_non_train_parent_context() -> None:
    parent = ImmutableProblemRecord(
        problem_id="parent",
        source=ProblemSource.MANUAL_FIXTURE,
        source_hash="hash-parent",
        family_id="fam",
        primitive_family_id="prim",
        format_family_id="fmt",
        prompt_wrapper_id="wrapper",
        split=SplitName.PRIVATE_LIKE,
        allowed_for_eval=True,
    )

    with pytest.raises(SyntheticGenerationError):
        generate_synthetic_dsl_records(
            split=SplitName.TRAIN,
            seed=5,
            primitive="bit_not",
            parent_ids=("parent",),
            existing_records=[parent],
        )


def test_generated_rows_validate_through_registry() -> None:
    records = generate_synthetic_dsl_records(split=SplitName.TRAIN, seed=6, count=2, primitive="add_const")

    assert validate_registry(records) == list(records)


def test_no_external_calls_required() -> None:
    records = generate_synthetic_dsl_records(split=SplitName.PRIVATE_LIKE, seed=7, primitive="binary_op")

    assert len(records) == 1
