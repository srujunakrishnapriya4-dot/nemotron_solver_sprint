from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_anti086"))

from kaggle_prepare_anti086_tokens import load_simple_yaml  # noqa: E402


def test_v1_config_is_small_safe_first_experiment() -> None:
    cfg = load_simple_yaml(ROOT / "kaggle_anti086" / "anti086_winmode_v1.yaml")
    assert cfg["stage"] == "v1"
    assert cfg["corpus_file"] == "corpus_anti086_v1.jsonl"
    assert int(cfg["max_rows"]) == 1000
    assert int(cfg["rank"]) <= 16
    assert "lm_head" not in str(cfg["target_modules"]).split(",")
    assert cfg["parent_adapter_path"] != "auto_or_none"
    assert not bool(cfg["full_prompt_loss"])
