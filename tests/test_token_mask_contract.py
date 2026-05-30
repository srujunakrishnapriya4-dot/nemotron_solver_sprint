from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.training_contract.token_mask_contract import TokenMaskContractError, validate_token_mask_rows  # noqa: E402


def test_token_mask_contract_rejects_prompt_loss() -> None:
    with pytest.raises(TokenMaskContractError):
        validate_token_mask_rows([{"input_ids": [1, 2], "target_ids": [1, 2], "loss_weights": [1, 0], "prompt_token_count": 1}])
    assert validate_token_mask_rows([{"input_ids": [1, 2], "target_ids": [1, 2], "loss_weights": [0, 1], "prompt_token_count": 1}])["supervised_tokens"] == 1
