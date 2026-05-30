from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.training_contract.no_prompt_loss_validator import validate_no_prompt_loss  # noqa: E402


def test_no_prompt_loss_validator_rejects_user_text() -> None:
    with pytest.raises(ValueError):
        validate_no_prompt_loss(["User: solve this"])
    assert validate_no_prompt_loss(["42"])["prompt_leakage_count"] == 0
