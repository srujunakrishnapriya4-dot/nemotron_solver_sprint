from __future__ import annotations

from nemotron_engine.core.schemas import ImmutableProblemRecord, ProblemSource, SplitName
import pytest

from nemotron_engine.data.contamination import ContaminationError, ContaminationReport, detect_cross_split_contamination, mark_contaminated_records


def record(
    problem_id: str,
    split: SplitName,
    *,
    family_id: str = "fam",
    primitive_family_id: str = "prim",
    format_family_id: str = "fmt",
    parent_ids: tuple[str, ...] = (),
    created_from: str = "",
) -> ImmutableProblemRecord:
    return ImmutableProblemRecord(
        problem_id=problem_id,
        source=ProblemSource.MANUAL_FIXTURE,
        source_hash=f"hash-{problem_id}",
        family_id=family_id,
        primitive_family_id=primitive_family_id,
        format_family_id=format_family_id,
        prompt_wrapper_id="wrapper",
        split=split,
        created_from=created_from,
        parent_ids=parent_ids,
        allowed_for_sft=split == SplitName.TRAIN,
        allowed_for_dpo=split == SplitName.TRAIN,
        allowed_for_grpo=False,
        allowed_for_eval=True,
    )


def test_prompt_hash_collision_across_train_private_like_flagged() -> None:
    train = record("train", SplitName.TRAIN)
    private = record("private", SplitName.PRIVATE_LIKE)

    report = detect_cross_split_contamination([train, private], prompts={"train": "A -> B", "private": "A -> B"})

    assert report.contaminated_problem_ids == ("train",)
    assert "prompt_hash_collision" in report.flags_by_problem_id["train"]


def test_normalized_prompt_hash_collision_flagged() -> None:
    train = record("train", SplitName.TRAIN)
    private = record("private", SplitName.PRIVATE_LIKE)

    report = detect_cross_split_contamination([train, private], prompts={"train": " A   -> B ", "private": "a -> b"})

    assert "normalized_prompt_hash_collision" in report.flags_by_problem_id["train"]


def test_parent_crossing_from_holdout_into_train_flagged() -> None:
    holdout = record("holdout", SplitName.FORBIDDEN_HOLDOUT)
    train = record("train", SplitName.TRAIN, parent_ids=("holdout",))

    report = detect_cross_split_contamination([holdout, train])

    assert "parent_cross_split" in report.flags_by_problem_id["train"]


def test_created_from_crossing_from_holdout_into_train_flagged() -> None:
    holdout = record("holdout", SplitName.FORBIDDEN_HOLDOUT)
    train = record("train", SplitName.TRAIN, created_from="holdout")

    report = detect_cross_split_contamination([holdout, train])

    assert "created_from_cross_split" in report.flags_by_problem_id["train"]


def test_duplicate_problem_id_flagged() -> None:
    left = record("dup", SplitName.TRAIN, family_id="a")
    right = record("dup", SplitName.PRIVATE_LIKE, family_id="b")

    report = detect_cross_split_contamination([left, right])

    assert report.duplicate_problem_ids == ("dup",)
    assert "duplicate_problem_id" in report.flags_by_problem_id["dup"]


def test_answer_hash_alone_is_weak_signal_only() -> None:
    train = record("train", SplitName.TRAIN, family_id="train-fam")
    private = record("private", SplitName.PRIVATE_LIKE, family_id="private-fam")

    report = detect_cross_split_contamination([train, private], answers={"train": "42", "private": "42"})

    assert report.answer_hash_weak_collisions
    assert report.contaminated_problem_ids == ()


def test_family_signature_collision_across_forbidden_boundary_flagged() -> None:
    train = record("train", SplitName.TRAIN)
    forbidden = record("forbidden", SplitName.FORBIDDEN_HOLDOUT)

    report = detect_cross_split_contamination([train, forbidden])

    assert "family_signature_collision" in report.flags_by_problem_id["train"]


def test_prompt_hash_plus_answer_hash_flags_contamination() -> None:
    train = record("train", SplitName.TRAIN)
    private = record("private", SplitName.PRIVATE_LIKE)

    report = detect_cross_split_contamination(
        [train, private],
        prompts={"train": "same", "private": "same"},
        answers={"train": "42", "private": "42"},
    )

    assert report.answer_hash_weak_collisions
    assert report.contaminated_problem_ids == ("train",)


def test_contamination_flags_disable_permissions_and_preserve_records() -> None:
    train = record("train", SplitName.TRAIN)
    private = record("private", SplitName.PRIVATE_LIKE)
    report = detect_cross_split_contamination([train, private], prompts={"train": "same", "private": "same"})

    updated = mark_contaminated_records([train, private], report)
    updated_by_id = {item.problem_id: item for item in updated}

    assert set(updated_by_id) == {"train", "private"}
    assert updated_by_id["train"].contamination_flags
    assert not updated_by_id["train"].allowed_for_sft
    assert not updated_by_id["train"].allowed_for_dpo
    assert not updated_by_id["train"].allowed_for_grpo
    assert updated_by_id["private"] == private


def test_forged_inconsistent_contamination_report_rejected() -> None:
    with pytest.raises(ContaminationError):
        ContaminationReport(
            checked_count=1,
            contaminated_count=1,
            contaminated_problem_ids=(),
            flags_by_problem_id={},
        )

    with pytest.raises(ContaminationError):
        ContaminationReport(
            checked_count=1,
            contaminated_count=1,
            contaminated_problem_ids=("p1",),
            flags_by_problem_id={"p1": ()},
        )
