from __future__ import annotations

from nemotron_engine.core.registry import split_allows_sft, validate_registry
from nemotron_engine.core.schemas import ImmutableProblemRecord, ProblemSource, SplitName
from nemotron_engine.data.contamination import detect_cross_split_contamination, mark_contaminated_records
from nemotron_engine.data.forbidden_primitive_generator import generate_forbidden_primitive_records
from nemotron_engine.data.split_builder import assign_splits, validate_split_manifest
from nemotron_engine.data.synthetic_dsl_generator import generate_synthetic_dsl_records


def seed_record(problem_id: str, family_id: str) -> ImmutableProblemRecord:
    return ImmutableProblemRecord(
        problem_id=problem_id,
        source=ProblemSource.MANUAL_FIXTURE,
        source_hash=f"hash-{problem_id}",
        family_id=family_id,
        primitive_family_id=f"prim-{family_id}",
        format_family_id=f"fmt-{family_id}",
        prompt_wrapper_id="wrapper",
        split=SplitName.TRAIN,
        allowed_for_sft=True,
        allowed_for_dpo=True,
        allowed_for_grpo=False,
        allowed_for_eval=True,
    )


def test_end_to_end_split_generation_contamination_registry() -> None:
    seeds = [seed_record("seed-a", "fam-a"), seed_record("seed-b", "fam-b")]
    assigned, manifest = assign_splits(seeds, seed=9, train_fraction=0.5, dev_fraction=0.0, forbidden_holdout_fraction=0.5)
    assert validate_split_manifest(manifest, assigned)

    train_rows = generate_synthetic_dsl_records(split=SplitName.TRAIN, seed=10, primitive="bit_not")
    forbidden_rows = generate_forbidden_primitive_records(seed=11, primitive_family_id="heldout-private")
    all_records = tuple(assigned) + train_rows + forbidden_rows

    prompts = {record.problem_id: f"prompt for {record.problem_id}" for record in all_records}
    report = detect_cross_split_contamination(all_records, prompts=prompts)
    clean_records = mark_contaminated_records(all_records, report)

    assert validate_registry(clean_records) == list(clean_records)
    assert all(split_allows_sft(record) for record in clean_records if record.problem_id in {item.problem_id for item in train_rows})
    assert all(not record.allowed_for_sft and record.allowed_for_eval for record in forbidden_rows)


def test_generated_train_row_disabled_after_prompt_collision() -> None:
    train_row = generate_synthetic_dsl_records(split=SplitName.TRAIN, seed=12, primitive="bit_not")[0]
    holdout_row = generate_forbidden_primitive_records(seed=13, primitive_family_id="heldout-private")[0]

    report = detect_cross_split_contamination(
        [train_row, holdout_row],
        prompts={train_row.problem_id: "same prompt", holdout_row.problem_id: "same prompt"},
    )
    clean_records = mark_contaminated_records([train_row, holdout_row], report)
    clean_by_id = {record.problem_id: record for record in clean_records}

    assert clean_by_id[train_row.problem_id].contamination_flags
    assert not split_allows_sft(clean_by_id[train_row.problem_id])
    assert clean_by_id[holdout_row.problem_id] == holdout_row
