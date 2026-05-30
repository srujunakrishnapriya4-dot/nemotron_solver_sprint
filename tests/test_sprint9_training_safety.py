from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_anti086"))

from kaggle_train_anti086_adapter import validate_train_config  # noqa: E402
from kaggle_train_stage import validate_training_config  # noqa: E402


def test_v1_training_safety_rejects_rank_lm_head_and_full_prompt_loss() -> None:
    valid = {"stage": "v1", "rank": 8, "target_modules": "q_proj,v_proj", "full_prompt_loss": False, "parent_adapter_path": "none"}
    validate_train_config(valid)
    validate_training_config(valid)
    with pytest.raises(SystemExit):
        validate_train_config({**valid, "rank": 32})
    with pytest.raises(SystemExit):
        validate_train_config({**valid, "target_modules": "q_proj,v_proj,lm_head"})
    with pytest.raises(SystemExit):
        validate_train_config({**valid, "full_prompt_loss": True})


def test_training_scripts_patch_runtime_before_model_import_textually() -> None:
    text = (ROOT / "kaggle_anti086" / "kaggle_train_anti086_adapter.py").read_text(encoding="utf-8")
    assert "apply_runtime_patches()" in text
    assert text.index("apply_runtime_patches()") < text.index("from transformers import AutoModelForCausalLM")
    assert "save_embedding_layers=False" in text
