from __future__ import annotations

from pathlib import Path

import pytest

from kaggle_anti086.kaggle_path_safety import (
    require_safe_config_output_paths,
    require_writable_output_dir,
    require_writable_output_path,
)


def test_rejects_kaggle_input_output_path() -> None:
    with pytest.raises(SystemExit):
        require_writable_output_path("/kaggle/input/x/out.json", field_name="token_output")


def test_rejects_kaggle_input_output_dir() -> None:
    with pytest.raises(SystemExit):
        require_writable_output_dir("/kaggle/input/datasets/foo", field_name="eval_output_dir")


def test_accepts_kaggle_working_output_path() -> None:
    assert require_writable_output_path("/kaggle/working/x/out.json", field_name="token_output") == Path("/kaggle/working/x/out.json")


def test_accepts_relative_local_artifact_path() -> None:
    assert require_writable_output_path("artifacts/day1/report.json", field_name="candidate_decision_path") == Path("artifacts/day1/report.json")


@pytest.mark.parametrize(
    "field,value",
    [
        ("token_output", "/kaggle/input/bad/train.jsonl"),
        ("eval_output_dir", "/kaggle/input/bad/eval"),
        ("output_adapter_dir", "/kaggle/input/bad/adapter"),
        ("calibrated_eval_path", "/kaggle/input/bad/eval.jsonl"),
    ],
)
def test_configured_output_paths_under_input_are_rejected(field: str, value: str) -> None:
    with pytest.raises(SystemExit):
        require_safe_config_output_paths({field: value}, stage="v1b")


def test_good_output_config_is_accepted() -> None:
    cfg = {
        "token_output": "/kaggle/working/anti086_tokens/v1b/train_tokens.jsonl",
        "eval_output_dir": "/kaggle/working/anti086_eval/v1b",
    }
    assert require_safe_config_output_paths(cfg, stage="v1b") is cfg


def test_tmp_is_rejected_for_training_eval_outputs() -> None:
    with pytest.raises(SystemExit):
        require_writable_output_path("/tmp/ptxas-blackwell", field_name="eval_output_dir")
