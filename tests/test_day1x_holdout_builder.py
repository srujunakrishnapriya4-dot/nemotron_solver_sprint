from __future__ import annotations

from copy import deepcopy

from kaggle_anti086.training.day1_teacher_trainability_gate import verify_trainable_row
from kaggle_anti086.training.day1_teacher_verifier import count_boxed_answers
from kaggle_anti086.training.day1x_holdout_builder import (
    build_all_holdouts,
    build_composed_holdout,
    build_format_trap_holdout,
    build_hard_p2_holdout,
    build_prompt_style_holdout,
    build_rule_holdout,
    validate_holdout_rows,
)


def test_all_five_holdout_builders_produce_valid_rows() -> None:
    builders = {
        "holdout_rule": build_rule_holdout,
        "holdout_prompt_style": build_prompt_style_holdout,
        "holdout_composed": build_composed_holdout,
        "holdout_format_traps": build_format_trap_holdout,
        "holdout_hard_p2": build_hard_p2_holdout,
    }
    for split, builder in builders.items():
        rows = builder(6, 123)
        assert rows
        assert validate_holdout_rows(rows, split)["status"] == "PASS"
        for row in rows:
            assert row["split"] == split
            assert verify_trainable_row(row).trainable
            assert "ABSTAIN" not in str(row).upper()
            assert count_boxed_answers(row["target_text"]) == 1


def test_holdout_category_requirements_and_deterministic_ids() -> None:
    rule = build_rule_holdout(3, 11)
    assert all(row["holdout_reason"] == "unseen_rule_parameters" for row in rule)
    prompt = build_prompt_style_holdout(8, 11)
    assert all(str(row["prompt_style"]).startswith("public_style_") for row in prompt)
    composed = build_composed_holdout(9, 11)
    assert all(str(row["family"]).startswith("composed_") for row in composed)
    traps = build_format_trap_holdout(5, 11)
    assert all(row["difficulty"] == "trap" or row["holdout_reason"] == "format_trap" for row in traps)
    hard = build_hard_p2_holdout(8, 11)
    assert any(row["family"] in {"equation_operator", "sequence_pattern", "gravity_numeric", "numeric_formula_safe", "composed_sequence_operator", "composed_unit_formula"} for row in hard)
    assert [row["id"] for row in build_all_holdouts(2, 777)["rule"]] == [row["id"] for row in build_all_holdouts(2, 777)["rule"]]


def test_holdout_prompt_duplicates_are_rejected() -> None:
    rows = build_rule_holdout(2, 99)
    dup = deepcopy(rows[0])
    dup["id"] = "different"
    report = validate_holdout_rows([*rows, dup], "holdout_rule")
    assert report["status"] == "FAIL"
    assert report["duplicate_prompt_count"] == 1
