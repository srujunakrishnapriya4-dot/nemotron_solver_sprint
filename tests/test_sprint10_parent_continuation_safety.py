from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_anti086"))

from kaggle_prepare_anti086_tokens import load_simple_yaml  # noqa: E402
from kaggle_train_anti086_adapter import validate_train_config  # noqa: E402


def test_v1b_config_is_parent_continuation_safe_shape() -> None:
    cfg = load_simple_yaml(ROOT / "kaggle_anti086" / "anti086_winmode_v1b.yaml")
    assert cfg["stage"] == "v1b"
    assert cfg["parent_adapter_path"] != "auto_or_none"
    assert bool(cfg["direct_answer_only"]) is True
    assert int(cfg["rank"]) <= 16
    assert "lm_head" not in str(cfg["target_modules"]).split(",")
    assert not bool(cfg["full_prompt_loss"])
    assert "submission" not in cfg


def test_parent_adapter_path_auto_or_none_rejected() -> None:
    cfg = {
        "stage": "v1b",
        "rank": 8,
        "target_modules": "q_proj,v_proj",
        "parent_adapter_path": "auto_or_none",
        "full_prompt_loss": False,
    }
    with pytest.raises(SystemExit, match="auto_or_none"):
        validate_train_config(cfg)


def test_lm_head_rejected_for_first_v1b() -> None:
    cfg = {
        "stage": "v1b",
        "rank": 8,
        "target_modules": "q_proj,v_proj,lm_head",
        "parent_adapter_path": "none",
        "full_prompt_loss": False,
    }
    with pytest.raises(SystemExit, match="lm_head"):
        validate_train_config(cfg)
