from __future__ import annotations

from functools import lru_cache

from kaggle_anti086.training.day10_build_solver_teacher_corpus import (
    FORMAT_ONLY_MIN_DIRECT_ROWS,
    FULL_TRAINING_MIN_DIRECT_ROWS,
    REQUIRED_DIRECT_FAMILY_MINIMUMS,
    build_solver_teacher_corpus,
    normalized_prompt_hash,
    prompt_hash,
)
from kaggle_anti086.training.day10_teacher_source_generators import build_day10_repair_source_rows
from kaggle_anti086.training.day10_train_solver_teacher_lora import build_training_readiness


@lru_cache(maxsize=1)
def _full_repair():
    return build_solver_teacher_corpus(
        source_mode="day10_repair",
        target_direct_rows=6144,
        target_abstain_rows=512,
        seed=1110,
        eval_paths=[],
    )


def test_day10_repair_source_generation_is_deterministic():
    first = build_day10_repair_source_rows(target_direct_rows=256, target_abstain_rows=32, seed=1110)
    second = build_day10_repair_source_rows(target_direct_rows=256, target_abstain_rows=32, seed=1110)

    assert [row["id"] for row in first["direct_rows"][:50]] == [row["id"] for row in second["direct_rows"][:50]]
    assert first["generation_policy_version"] == "day10_pass10c_v1"
    assert first["source_mode"] == "day10_repair"


def test_full_mix_repair_meets_counts_caps_and_hash_fields():
    result = _full_repair()
    mix = result["mix_repair"]

    assert mix["status"] == "PASS"
    assert mix["direct_answer_rows"] == 6144
    assert mix["abstain_policy_rows"] == 512
    assert mix["target_direct_rows"] == 6144
    assert mix["target_abstain_rows"] == 512
    assert mix["format_only_count"] >= FORMAT_ONLY_MIN_DIRECT_ROWS
    assert mix["full_training_eligible"] is True
    assert mix["smoke_training_eligible"] is True
    assert mix["family_cap_violations"] == {}
    assert mix["subfamily_cap_violations"] == {}
    assert mix["template_dominance_warnings"] == []
    assert mix["near_duplicate_prompt_rate"] <= 0.02
    assert mix["direct_output_sha256"] == ""
    assert mix["abstain_output_sha256"] == ""
    for family, minimum in REQUIRED_DIRECT_FAMILY_MINIMUMS.items():
        assert mix["by_family"][family] >= minimum


def test_repair_direct_rows_have_no_abstain_or_duplicate_inflation():
    rows = _full_repair()["rows"]
    source_ids = [row["source_id"] for row in rows]
    prompt_hashes = [prompt_hash(row["prompt"]) for row in rows]
    normalized_hashes = [normalized_prompt_hash(row["prompt"]) for row in rows]

    assert all(row["answer"] != "ABSTAIN" for row in rows)
    assert len(source_ids) == len(set(source_ids))
    assert len(prompt_hashes) == len(set(prompt_hashes))
    assert len(normalized_hashes) == len(set(normalized_hashes))
    assert all(row["metadata"]["gold_used_for_target"] is False for row in rows)


def test_small_repair_corpus_fails_with_explicit_generation_stats():
    result = build_solver_teacher_corpus(
        source_mode="day10_repair",
        target_direct_rows=512,
        target_abstain_rows=64,
        seed=1110,
        eval_paths=[],
    )
    mix = result["mix_repair"]

    assert mix["status"] == "FAIL"
    assert "direct_rows_below_smoke_threshold" in mix["remaining_blockers"]
    assert "direct_rows_below_full_training_threshold" in mix["remaining_blockers"]
    for stats in mix["generation_stats_by_family"].values():
        for key in (
            "generation_attempted",
            "solver_verified",
            "solver_rejected",
            "duplicate_rejected",
            "overlap_rejected",
            "format_rejected",
            "remaining_needed",
        ):
            assert key in stats


def test_dry_run_gate_uses_mix_repair_for_smoke_and_full_eligibility(tmp_path):
    result = _full_repair()
    teacher_audit = tmp_path / "teacher.json"
    overlap_audit = tmp_path / "overlap.json"
    learnability_audit = tmp_path / "learnability.json"
    mix_repair = tmp_path / "mix.json"
    capacity = tmp_path / "capacity.json"
    abstain_policy = tmp_path / "abstain.jsonl"
    teacher_audit.write_text(__import__("json").dumps(result["audit"]), encoding="utf-8")
    overlap_audit.write_text(__import__("json").dumps(result["overlap"]), encoding="utf-8")
    learnability_audit.write_text(__import__("json").dumps(result["learnability"]), encoding="utf-8")
    mix_repair.write_text(__import__("json").dumps(result["mix_repair"]), encoding="utf-8")
    abstain_policy.write_text("", encoding="utf-8")
    capacity.write_text(
        __import__("json").dumps({"status": "PASS", "configs": {"v4_solver_teacher_lora_small": {"status": "PASS"}}}),
        encoding="utf-8",
    )

    summary = build_training_readiness(
        config_path="kaggle_anti086/training/configs/v4_solver_teacher_lora_small.yaml",
        teacher_audit_path=teacher_audit,
        overlap_audit_path=overlap_audit,
        learnability_audit_path=learnability_audit,
        capacity_audit_path=capacity,
        mix_repair_report_path=mix_repair,
        abstain_policy_path=abstain_policy,
        dry_run=True,
    )

    assert summary["smoke_training_allowed"] is True
    assert summary["full_training_allowed"] is False
    assert summary["mix_repair_status"] == "PASS"
    assert summary["row_count"] >= FULL_TRAINING_MIN_DIRECT_ROWS
    assert summary["full_training_blockers"] == []
