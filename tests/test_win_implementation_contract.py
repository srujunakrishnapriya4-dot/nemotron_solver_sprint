from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.win_system_audit.implementation_contract import build_implementation_contract  # noqa: E402


def test_implementation_contract_forbids_full_prompt_loss() -> None:
    contract = build_implementation_contract()
    assert "full_prompt_loss" in contract["forbidden"]
    assert "real_eval_metrics" in contract["required"]
