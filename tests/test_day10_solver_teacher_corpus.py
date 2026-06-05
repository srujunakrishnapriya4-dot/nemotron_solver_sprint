from __future__ import annotations

from pathlib import Path

from kaggle_anti086.training.day10_kaggle_commands import COMMAND_PLAN
from kaggle_anti086.training.day10_build_solver_teacher_corpus import (
    REQUIRED_FAMILIES,
    build_day10_source_rows,
    build_solver_teacher_corpus,
    solve_teacher_row,
    write_day10_outputs,
)


REQUIRED_SCHEMA = {
    "id",
    "source_id",
    "family",
    "subfamily",
    "prompt",
    "answer",
    "expected_behavior",
    "solver_name",
    "solver_source",
    "solver_confidence",
    "verified",
    "risk",
    "metadata",
}


def test_teacher_corpus_builds_with_required_schema_and_passes_audits():
    result = build_solver_teacher_corpus(rows=128, seed=1110, eval_paths=[])
    rows = result["rows"]

    assert rows
    assert result["audit"]["status"] == "PASS"
    assert result["overlap"]["status"] == "PASS"
    assert result["learnability"]["status"] == "PASS"
    assert result["manifest"]["status"] == "PASS"
    assert REQUIRED_SCHEMA <= set(rows[0])
    families = {row["family"] for row in rows}
    has_abstain = any(row["answer"] == "ABSTAIN" for row in rows)
    assert REQUIRED_FAMILIES <= (families | ({"abstain/unsupported"} if has_abstain else set()))
    assert all(row["metadata"]["gold_used_for_target"] is False for row in rows)


def test_teacher_target_does_not_come_from_corrupted_gold_answer():
    source_rows = build_day10_source_rows(rows=64, seed=1110)
    numeric = next(row for row in source_rows if row["family"] == "numeric_formula")
    numeric["answer"] = "999999"

    result = solve_teacher_row(numeric)

    assert result["row"] is not None
    assert result["row"]["answer"] != "999999"
    assert result["row"]["metadata"]["gold_used_for_target"] is False


def test_abstain_rows_are_policy_marked_not_default_answer_behavior():
    result = build_solver_teacher_corpus(rows=128, seed=1110, eval_paths=[])
    abstain_rows = [row for row in result["rows"] if row["answer"] == "ABSTAIN"]

    assert abstain_rows
    assert all(row["expected_behavior"] == "abstain" for row in abstain_rows)
    assert all(row["metadata"]["abstain_policy"] == "teacher_verified_abstain_only" for row in abstain_rows)


def test_outputs_include_manifest_hashes_and_audit_paths(tmp_path: Path):
    result = build_solver_teacher_corpus(rows=128, seed=1110, eval_paths=[])
    manifest = write_day10_outputs(
        result,
        out_direct=tmp_path / "direct.jsonl",
        out_manifest=tmp_path / "manifest.json",
        out_audit=tmp_path / "audit.json",
        out_overlap=tmp_path / "overlap.json",
        out_learnability=tmp_path / "learnability.json",
    )

    assert manifest["status"] == "PASS"
    assert manifest["files"]["direct"]["row_count"] == len(result["rows"])
    assert manifest["files"]["direct"]["sha256"]
    assert manifest["files"]["overlap_audit"]["sha256"]
    assert manifest["training_allowed"] is False
    assert manifest["packaging_allowed"] is False
    assert manifest["submission_allowed"] is False


def test_day10_command_plan_has_staged_gates_without_package_or_submit():
    for stage in (
        "Stage A: Day 9 regression",
        "Stage B: teacher corpus",
        "Stage C: audits",
        "Stage D: capacity audit + dry-run train",
        "Stage E: smoke train, Kaggle only",
        "Stage F: full train, gated",
        "Stage G: adapter eval",
        "Stage H: error mining + candidate decision",
        "Stage I: package rehearsal, dry-run only",
    ):
        assert stage in COMMAND_PLAN
    assert "submission.zip" in COMMAND_PLAN
    assert "kaggle competitions submit" not in COMMAND_PLAN.lower()
    assert "package_adapter.py" not in COMMAND_PLAN
