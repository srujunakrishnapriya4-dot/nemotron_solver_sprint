from __future__ import annotations

import json

from kaggle_anti086.training.day10_lora_capacity_audit import audit_config
from kaggle_anti086.training.day10_train_solver_teacher_lora import build_training_readiness
from kaggle_anti086.training.training_config_schema import load_training_config


def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _gate_files(tmp_path):
    teacher = _write_json(tmp_path / "teacher.json", {"status": "PASS", "row_count": 6144, "family_counts": {"format_only": 300}})
    overlap = _write_json(tmp_path / "overlap.json", {"status": "PASS"})
    learnability = _write_json(tmp_path / "learnability.json", {"status": "PASS", "subfamily_balance_warnings": {}})
    mix = _write_json(
        tmp_path / "mix.json",
        {
            "status": "PASS",
            "direct_answer_rows": 6144,
            "format_only_count": 300,
            "by_family": {"format_only": 300, "numeric_formula": 5844},
            "remaining_blockers": [],
            "smoke_training_eligible": True,
            "full_training_eligible": True,
        },
    )
    capacity = _write_json(tmp_path / "capacity.json", {"status": "PASS", "small_config_viable": True})
    abstain = tmp_path / "abstain.jsonl"
    abstain.write_text("", encoding="utf-8")
    corpus = tmp_path / "teacher.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "id": "x",
                "family": "numeric_formula",
                "answer": "7",
                "prompt": "1 -> 3. Now solve: 3",
                "verified": True,
                "metadata": {"source_rule_id": "r", "source_leakage_group": "g"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return teacher, overlap, learnability, mix, capacity, abstain, corpus


def test_smoke_bf16_qv_config_is_smoke_only_and_not_submission_ready():
    config = load_training_config("kaggle_anti086/training/configs/v4_solver_teacher_lora_smoke_bf16_qv.yaml")

    assert config["stage"] == "v4_solver_teacher_lora_smoke_bf16_qv"
    assert config["smoke_bf16_runtime_only"] is True
    assert config["load_in_4bit"] is False
    assert set(config["target_modules"]) == {"q_proj", "v_proj"}
    assert config["full_training_allowed"] is False
    assert config["not_submission_ready"] is True


def test_capacity_audit_allows_bf16_qv_only_with_smoke_flag():
    report = audit_config("kaggle_anti086/training/configs/v4_solver_teacher_lora_smoke_bf16_qv.yaml")

    assert report["status"] == "PASS_TARGETS_UNVERIFIED"
    assert "smoke_bf16_runtime_only_not_submission_ready" in report["warnings"]


def test_official_small_config_still_requires_qvo(tmp_path):
    bad = tmp_path / "bad_small.yaml"
    bad.write_text(
        "\n".join(
            [
                "stage: v4_solver_teacher_lora_small",
                "rank: 32",
                "target_modules: q_proj,v_proj",
                "adapter_size_limit_mb: 1500",
                "assistant_only_loss: true",
                "full_prompt_loss: false",
                "train_on_user: false",
                "abstain_weight: 0.0",
            ]
        ),
        encoding="utf-8",
    )

    report = audit_config(bad)

    assert report["status"] == "FAIL"
    assert "small_targets_must_be_q_v_o" in report["failures"]


def test_smoke_bf16_full_training_is_blocked(tmp_path):
    teacher, overlap, learnability, mix, capacity, abstain, corpus = _gate_files(tmp_path)
    config = tmp_path / "bf16.yaml"
    text = open("kaggle_anti086/training/configs/v4_solver_teacher_lora_smoke_bf16_qv.yaml", encoding="utf-8").read()
    text = text.replace("artifacts/sprint11/day10_solver_teacher_direct.jsonl", corpus.as_posix())
    config.write_text(text, encoding="utf-8")

    summary = build_training_readiness(
        config_path=config,
        teacher_audit_path=teacher,
        overlap_audit_path=overlap,
        learnability_audit_path=learnability,
        mix_repair_report_path=mix,
        capacity_audit_path=capacity,
        abstain_policy_path=abstain,
        dry_run=False,
        smoke_steps=None,
        kaggle_mode=True,
    )

    assert summary["full_training_allowed"] is False
    assert "smoke_bf16_runtime_only_forbids_full_training" in summary["failures"]


def test_smoke_bf16_smoke_gate_allows_only_kaggle_smoke(tmp_path):
    teacher, overlap, learnability, mix, capacity, abstain, corpus = _gate_files(tmp_path)
    config = tmp_path / "bf16.yaml"
    text = open("kaggle_anti086/training/configs/v4_solver_teacher_lora_smoke_bf16_qv.yaml", encoding="utf-8").read()
    text = text.replace("artifacts/sprint11/day10_solver_teacher_direct.jsonl", corpus.as_posix())
    config.write_text(text, encoding="utf-8")

    summary = build_training_readiness(
        config_path=config,
        teacher_audit_path=teacher,
        overlap_audit_path=overlap,
        learnability_audit_path=learnability,
        mix_repair_report_path=mix,
        capacity_audit_path=capacity,
        abstain_policy_path=abstain,
        dry_run=False,
        smoke_steps=5,
        kaggle_mode=True,
    )

    assert summary["smoke_training_allowed"] is True
    assert summary["full_training_allowed"] is False
    assert summary["failures"] == []
