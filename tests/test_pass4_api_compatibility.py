from __future__ import annotations


def test_pass4_imports_work() -> None:
    from nemotron_engine.data import (
        SplitManifest,
        assign_splits,
        compute_prompt_hash,
        detect_cross_split_contamination,
        generate_forbidden_primitive_records,
        generate_llm_stress_records,
        generate_synthetic_dsl_records,
        mark_contaminated_records,
        mutate_record,
        read_split_manifest,
        validate_split_manifest,
        write_split_manifest,
    )

    assert SplitManifest
    assert assign_splits
    assert compute_prompt_hash
    assert detect_cross_split_contamination
    assert generate_forbidden_primitive_records
    assert generate_llm_stress_records
    assert generate_synthetic_dsl_records
    assert mark_contaminated_records
    assert mutate_record
    assert read_split_manifest
    assert validate_split_manifest
    assert write_split_manifest


def test_representative_locked_pass_imports_still_work() -> None:
    from nemotron_engine.core import ImmutableProblemRecord, validate_registry
    from nemotron_engine.programs import execute_program, make_program
    from nemotron_engine.scoring import extract_boxed_answer
    from nemotron_engine.solvers import solve_binary, synthesize

    assert ImmutableProblemRecord
    assert validate_registry
    assert execute_program
    assert make_program
    assert extract_boxed_answer
    assert solve_binary
    assert synthesize

