from __future__ import annotations

from pathlib import Path
import shutil

from kaggle_anti086.training.day11a_kaggle_commands import COMMAND_PLAN, write_command_plan


def test_command_plan_includes_required_stages_a_to_o():
    for stage in [
        "Stage A",
        "Stage B",
        "Stage C",
        "Stage D",
        "Stage E",
        "Stage F",
        "Stage G",
        "Stage H",
        "Stage I",
        "Stage J",
        "Stage K",
        "Stage L",
        "Stage M",
        "Stage N",
        "Stage O",
    ]:
        assert stage in COMMAND_PLAN


def test_command_plan_includes_v2a_train_validation_and_ranking():
    assert "v2a_bf16_qv_50.yaml" in COMMAND_PLAN
    assert "day11a_real_v2_candidate.py" in COMMAND_PLAN
    assert "--kaggle-mode --train --no-submit" in COMMAND_PLAN
    assert "day10_adapter_package_guard.py" in COMMAND_PLAN
    assert "day11a_candidate_ranking.py" in COMMAND_PLAN


def test_command_plan_excludes_submission_and_unproven_training_paths():
    lowered = COMMAND_PLAN.lower()
    assert "submission.zip" not in lowered
    assert "kaggle competitions submit" not in lowered
    assert "load_in_4bit true" not in lowered
    assert "target_modules: q_proj,v_proj,o_proj" not in lowered
    assert "tinker as primary" not in lowered
    assert "repackaged" not in lowered


def test_command_plan_writes_artifact_without_placeholders():
    root = Path("artifacts/test_tmp/day11a_commands")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    out = root / "plan.txt"

    write_command_plan(out)

    text = out.read_text(encoding="utf-8")
    assert "<ADAPTER_DIR>" not in text
    assert "<resolved_base_model_path>" not in text
    assert "submission.zip" not in text
    shutil.rmtree(root, ignore_errors=True)
