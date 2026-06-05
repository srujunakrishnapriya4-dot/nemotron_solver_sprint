from __future__ import annotations

from kaggle_anti086.training.day10_build_solver_teacher_corpus import (
    _prompt_has_leak,
    _target_format_ok,
    build_teacher_audit,
)


def test_allowed_final_answer_only_targets_pass():
    for answer in ["17", "ABSTAIN", "XLIV", "101101", "BLUE", "42.5", "@&"]:
        assert _target_format_ok(answer), answer


def test_reasoning_markdown_answer_prefix_and_row_ids_are_rejected():
    rejected = [
        "The answer is 17 because the examples are linear.",
        "Answer: 17",
        "First, observe the pattern.\n17",
        "```17```",
        "day10_source_numeric_formula_00001",
        "expected answer is 17",
    ]

    for answer in rejected:
        assert not _target_format_ok(answer), answer


def test_teacher_audit_rejects_prompt_or_source_id_leakage():
    row = {
        "id": "day10_teacher_numeric_formula_deadbeef",
        "source_id": "day10_source_numeric_formula_deadbeef",
        "family": "numeric_formula",
        "subfamily": "linear_offset",
        "prompt": "Solve this row day10_source_numeric_formula_deadbeef",
        "answer": "17",
        "expected_behavior": "answer",
        "solver_name": "solver_ensemble",
        "solver_source": "numeric_formula_solver",
        "solver_confidence": 0.9,
        "verified": True,
        "risk": "low",
        "metadata": {},
    }

    assert _prompt_has_leak(row["prompt"], row)
    audit = build_teacher_audit([row], [])
    assert audit["status"] == "FAIL"
    assert "prompt_leak_detected" in audit["failures"]
