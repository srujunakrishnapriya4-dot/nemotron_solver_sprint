from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.vex_progress_bridge.mask_weight_builder import MaskWeightBuilderError, build_char_level_mask, validate_masked_rows  # noqa: E402


def test_mask_builder_never_supervises_prompt_tokens() -> None:
    masked = build_char_level_mask("User:\nsecret prompt\nAssistant:\n42", "42")
    diag = validate_masked_rows([masked])

    assert diag.zero_supervised_rows == 0
    assert diag.prompt_supervised_leak_count == 0
    assert diag.decoded_supervised_spans == ("42",)


def test_low_supervised_ratio_allowed_when_answer_exists() -> None:
    masked = build_char_level_mask("User:\n" + "x" * 1000 + "\nAssistant:\ny", "y")

    assert validate_masked_rows([masked]).supervised_token_count_min == 1


def test_zero_supervised_rejected() -> None:
    with pytest.raises(MaskWeightBuilderError):
        validate_masked_rows([{"token_ids": [1], "target_ids": [1], "loss_weights": [0.0], "prompt_token_count": 1, "supervised_token_count": 0}])

