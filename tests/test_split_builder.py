from __future__ import annotations

from dataclasses import replace

import pytest

from nemotron_engine.data import split_builder as split_builder_module
from nemotron_engine.core.schemas import ImmutableProblemRecord, ProblemSource, SplitName
from nemotron_engine.data.split_builder import (
    SplitManifestError,
    assign_splits,
    read_split_manifest,
    split_records_by_family,
    validate_split_manifest,
    write_split_manifest,
)


def record(problem_id: str, family_id: str, *, split: SplitName = SplitName.TRAIN) -> ImmutableProblemRecord:
    return ImmutableProblemRecord(
        problem_id=problem_id,
        source=ProblemSource.MANUAL_FIXTURE,
        source_hash=f"hash-{problem_id}",
        family_id=family_id,
        primitive_family_id=f"prim-{family_id}",
        format_family_id=f"fmt-{family_id}",
        prompt_wrapper_id="wrapper",
        split=split,
        allowed_for_sft=split == SplitName.TRAIN,
        allowed_for_dpo=split == SplitName.TRAIN,
        allowed_for_grpo=False,
        allowed_for_eval=True,
    )


def test_deterministic_split_with_seed() -> None:
    records = [record(f"p{i}", f"fam-{i}") for i in range(6)]
    first, first_manifest = assign_splits(records, seed=7, train_fraction=0.5, dev_fraction=0.0, forbidden_holdout_fraction=0.5)
    second, second_manifest = assign_splits(records, seed=7, train_fraction=0.5, dev_fraction=0.0, forbidden_holdout_fraction=0.5)

    assert [item.problem_id + item.split.value for item in first] == [item.problem_id + item.split.value for item in second]
    assert first_manifest.manifest_hash == second_manifest.manifest_hash


def test_same_family_not_in_train_and_forbidden_holdout() -> None:
    records = [record("p1", "shared"), record("p2", "shared"), record("p3", "other")]
    assigned, manifest = assign_splits(records, seed=1, train_fraction=0.5, dev_fraction=0.0, forbidden_holdout_fraction=0.5)

    grouped = split_records_by_family(assigned)
    assert len({item.split for item in grouped["shared"]}) == 1
    assert validate_split_manifest(manifest, assigned)


def test_forbidden_holdout_cannot_allow_training() -> None:
    records = [record(f"p{i}", f"fam-{i}") for i in range(4)]
    assigned, _ = assign_splits(records, seed=5, train_fraction=0.0, dev_fraction=0.0, forbidden_holdout_fraction=1.0)

    assert all(item.split == SplitName.FORBIDDEN_HOLDOUT for item in assigned)
    assert all(not item.allowed_for_sft and not item.allowed_for_dpo and not item.allowed_for_grpo for item in assigned)


def test_stress_only_cannot_allow_training_in_manifest_validation() -> None:
    stress = record("stress", "stress-family", split=SplitName.STRESS_ONLY)
    assigned, manifest = assign_splits([record("p1", "fam")], seed=1, train_fraction=1.0, dev_fraction=0.0, forbidden_holdout_fraction=0.0)
    assert validate_split_manifest(manifest, assigned)
    assert not stress.allowed_for_sft and not stress.allowed_for_dpo and not stress.allowed_for_grpo


def test_write_read_split_manifest_round_trip(monkeypatch) -> None:
    assigned, manifest = assign_splits([record(f"p{i}", f"fam-{i}") for i in range(3)], seed=11)
    storage: dict[str, str] = {}

    class MemoryPath:
        def __init__(self, path: str) -> None:
            self.path = str(path)

        def write_text(self, text: str, encoding: str) -> None:
            assert encoding == "utf-8"
            storage[self.path] = text

        def read_text(self, encoding: str) -> str:
            assert encoding == "utf-8"
            return storage[self.path]

    monkeypatch.setattr(split_builder_module, "Path", MemoryPath)

    write_split_manifest(manifest, "split_manifest.json")
    loaded = read_split_manifest("split_manifest.json")

    assert loaded == manifest
    assert validate_split_manifest(loaded, assigned)


def test_validate_manifest_rejects_inconsistent_counts() -> None:
    _, manifest = assign_splits([record("p1", "fam")], seed=1)

    with pytest.raises(SplitManifestError):
        replace(manifest, split_counts={SplitName.TRAIN.value: 99})


def test_manifest_rejects_records_hash_mismatch() -> None:
    _, manifest = assign_splits([record("p1", "fam")], seed=1)

    with pytest.raises(SplitManifestError):
        replace(manifest, records_hash="tampered")


def test_manifest_rejects_split_record_ids_count_mismatch() -> None:
    _, manifest = assign_splits([record("p1", "fam")], seed=1)

    with pytest.raises(SplitManifestError):
        replace(manifest, split_record_ids={SplitName.TRAIN.value: ()})


def test_manifest_rejects_duplicate_ids_in_split_record_ids() -> None:
    assigned, manifest = assign_splits([record("p1", "fam1"), record("p2", "fam2")], seed=1, train_fraction=1.0, dev_fraction=0.0, forbidden_holdout_fraction=0.0)

    with pytest.raises(SplitManifestError):
        replace(
            manifest,
            split_record_ids={SplitName.TRAIN.value: (assigned[0].problem_id, assigned[0].problem_id)},
            split_counts={SplitName.TRAIN.value: 2},
        )


def test_validate_rejects_manifest_record_ids_not_matching_records() -> None:
    assigned, _ = assign_splits([record("p1", "fam1")], seed=1)
    _, other_manifest = assign_splits([record("p2", "fam2")], seed=1)

    with pytest.raises(SplitManifestError):
        validate_split_manifest(other_manifest, assigned)


def test_primitive_overlap_across_train_forbidden_rejected_unless_allowed() -> None:
    records = [
        record("p1", "fam1"),
        replace(record("p2", "fam2"), primitive_family_id="prim-fam1"),
    ]

    with pytest.raises(SplitManifestError):
        assign_splits(records, seed=1, train_fraction=0.5, dev_fraction=0.0, forbidden_holdout_fraction=0.5)

    assigned, manifest = assign_splits(
        records,
        seed=1,
        train_fraction=0.5,
        dev_fraction=0.0,
        forbidden_holdout_fraction=0.5,
        allow_primitive_train_forbidden_overlap=True,
    )
    assert validate_split_manifest(manifest, assigned, allow_primitive_train_forbidden_overlap=True)


def test_format_overlap_across_train_forbidden_rejected_unless_allowed() -> None:
    records = [
        record("p1", "fam1"),
        replace(record("p2", "fam2"), format_family_id="fmt-fam1"),
    ]

    with pytest.raises(SplitManifestError):
        assign_splits(records, seed=1, train_fraction=0.5, dev_fraction=0.0, forbidden_holdout_fraction=0.5)

    assigned, manifest = assign_splits(
        records,
        seed=1,
        train_fraction=0.5,
        dev_fraction=0.0,
        forbidden_holdout_fraction=0.5,
        allow_format_train_forbidden_overlap=True,
    )
    assert validate_split_manifest(manifest, assigned, allow_format_train_forbidden_overlap=True)
