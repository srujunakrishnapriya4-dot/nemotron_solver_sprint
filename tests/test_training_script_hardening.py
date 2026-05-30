from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle_anti086"))

from kaggle_train_anti086_adapter import load_token_rows, validate_train_config  # noqa: E402


def test_training_script_hardening_rejects_full_prompt_loss_true() -> None:
    validate_train_config({"full_prompt_loss": False, "rank": 32, "target_modules": "q_proj"})
    with pytest.raises(SystemExit):
        validate_train_config({"full_prompt_loss": True, "rank": 32, "target_modules": "q_proj"})


def test_training_script_hardening_uses_token_mask_contract(tmp_path: Path) -> None:
    path = tmp_path / "tokens.jsonl"
    path.write_text(json.dumps({"input_ids": [1, 2], "target_ids": [1, 2], "loss_weights": [0, 1], "prompt_token_count": 1}) + "\n", encoding="utf-8")
    assert len(load_token_rows(path)) == 1
