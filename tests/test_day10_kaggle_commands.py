from __future__ import annotations

from kaggle_anti086.training.day10_kaggle_commands import COMMAND_PLAN


def test_day10_command_plan_includes_pass10b_gates_and_no_submit_command():
    for expected in (
        "day10_build_solver_teacher_corpus.py",
        "--source-mode day10_repair",
        "day10_corpus_mix_repair_report.json",
        "day10_solver_teacher_abstain_policy.jsonl",
        "day10_lora_capacity_audit.py",
        "day10_train_solver_teacher_lora.py",
        "day10_adapter_eval.py",
        "day10_adapter_error_mining.py",
        "day10_candidate_decision.py",
        "day10_adapter_package_guard.py",
    ):
        assert expected in COMMAND_PLAN
    assert "kaggle competitions submit" not in COMMAND_PLAN.lower()
    assert "submission.zip" in COMMAND_PLAN
    assert "--dry-run-package" in COMMAND_PLAN
    assert "tests/test_day10_corpus_mix_repair.py" in COMMAND_PLAN
