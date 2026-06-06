from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from kaggle_anti086.training.day10_kaggle_commands import COMMAND_PLAN
from kaggle_anti086.training.day10_build_solver_teacher_corpus import (
    REQUIRED_FAMILIES,
    SUPPORTED_DIRECT_FAMILIES,
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


@lru_cache(maxsize=1)
def _repaired_corpus():
    return build_solver_teacher_corpus(
        source_mode="day10_repair",
        target_direct_rows=6144,
        target_abstain_rows=512,
        seed=1110,
        eval_paths=[],
    )


def test_teacher_corpus_builds_with_required_schema_and_passes_audits():
    result = _repaired_corpus()
    rows = result["rows"]

    assert rows
    assert result["audit"]["status"] == "PASS"
    assert result["overlap"]["status"] == "PASS"
    assert result["learnability"]["status"] == "PASS"
    assert result["mix_repair"]["status"] == "PASS"
    assert result["manifest"]["status"] == "PASS"
    assert REQUIRED_SCHEMA <= set(rows[0])
    families = {row["family"] for row in rows}
    assert SUPPORTED_DIRECT_FAMILIES <= families
    assert "abstain/unsupported" in REQUIRED_FAMILIES
    assert all(row["answer"] != "ABSTAIN" for row in rows)
    assert all(row["metadata"]["gold_used_for_target"] is False for row in rows)


def test_teacher_target_does_not_come_from_corrupted_gold_answer():
    source_rows = build_day10_source_rows(rows=64, seed=1110)
    numeric = next(row for row in source_rows if row["family"] == "numeric_formula")
    numeric["answer"] = "999999"

    result = solve_teacher_row(numeric)

    assert result["row"] is not None
    assert result["row"]["answer"] != "999999"
    assert result["row"]["metadata"]["gold_used_for_target"] is False


def test_abstain_rows_are_separated_from_default_direct_sft():
    result = _repaired_corpus()
    direct_abstain_rows = [row for row in result["rows"] if row["answer"] == "ABSTAIN"]
    abstain_rows = result["abstain_rows"]

    assert not direct_abstain_rows
    assert abstain_rows
    assert all(row["expected_behavior"] == "abstain" for row in abstain_rows)
    assert all(row["metadata"]["abstain_policy"] == "teacher_verified_abstain_only" for row in abstain_rows)


def test_outputs_include_manifest_hashes_and_audit_paths(tmp_path: Path):
    result = _repaired_corpus()
    manifest = write_day10_outputs(
        result,
        out_direct=tmp_path / "direct.jsonl",
        out_abstain_policy=tmp_path / "abstain.jsonl",
        out_manifest=tmp_path / "manifest.json",
        out_audit=tmp_path / "audit.json",
        out_overlap=tmp_path / "overlap.json",
        out_learnability=tmp_path / "learnability.json",
        out_mix_repair=tmp_path / "mix.json",
    )

    assert manifest["status"] == "PASS"
    assert manifest["files"]["direct"]["row_count"] == len(result["rows"])
    assert manifest["files"]["abstain_policy"]["row_count"] == len(result["abstain_rows"])
    assert manifest["files"]["direct"]["sha256"]
    assert manifest["files"]["mix_repair"]["sha256"]
    assert manifest["files"]["overlap_audit"]["sha256"]
    assert manifest["training_allowed"] is False
    assert manifest["packaging_allowed"] is False
    assert manifest["submission_allowed"] is False


def test_day10_command_plan_has_staged_gates_without_package_or_submit():
    for stage in (
        "Stage A: Day 9 regression",
        "Stage B: PASS 10C teacher corpus mix repair",
        "Stage C: audits + mix repair checks",
        "Stage D: capacity audit + dry-run train",
        "Stage E: Nemotron LoRA target diagnostics, Kaggle only",
        "Stage F: Day 10 official small smoke train real backend, Kaggle only",
        "Stage G: BF16 q/v smoke-only fallback if 4-bit PEFT shape incompatible",
        "Stage H: smoke adapter package guard + adapter eval",
        "Stage I: full-train gate decision, not implemented in PASS 10E",
        "Stage J: adapter eval, non-smoke manual baseline path",
        "Stage K: error mining + candidate decision",
        "Stage L: package rehearsal, dry-run only",
    ):
        assert stage in COMMAND_PLAN
    assert "submission.zip" in COMMAND_PLAN
    assert "--source-mode day10_repair" in COMMAND_PLAN
    assert "day10_corpus_mix_repair_report.json" in COMMAND_PLAN
    assert "day10_nemotron_lora_target_diagnostics.py" in COMMAND_PLAN
    assert "v4_solver_teacher_lora_smoke_bf16_qv.yaml" in COMMAND_PLAN
    assert "HARD_BLOCKER_GPU_MEMORY_OR_BACKEND" in COMMAND_PLAN
    assert "kaggle competitions submit" not in COMMAND_PLAN.lower()
    assert "package_adapter.py" not in COMMAND_PLAN
