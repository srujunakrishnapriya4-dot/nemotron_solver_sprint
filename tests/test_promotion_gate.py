from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.promotion_gate import decide_promotion  # noqa: E402


def test_promotion_gate_rejects_no_eval_model() -> None:
    decision = decide_promotion("submission", {"token_mask_valid": True, "loss_finite": True, "adapter_size_mb": 10, "empty_output_count": 1})

    assert decision["decision"] == "REJECT"
    assert "empty_or_nonsense_outputs" in decision["reasons"]


def test_promotion_gate_rejects_synthetic_only_gain_with_public_collapse() -> None:
    decision = decide_promotion("v2_to_v3", {"token_mask_valid": True, "loss_finite": True, "adapter_size_mb": 10, "adapter_size_limit_mb": 1500, "empty_output_count": 0, "public_like_delta": -0.02, "private_like_delta": 0.03, "synthetic_gain_only": True})

    assert "public_like_collapse" in decision["reasons"]
    assert "synthetic_only_gain" in decision["reasons"]


def test_promotion_gate_accepts_clean_micro() -> None:
    decision = decide_promotion("micro_to_v1", {"token_mask_valid": True, "loss_finite": True, "adapter_size_mb": 10, "adapter_size_limit_mb": 1500, "empty_output_count": 0})

    assert decision["decision"] == "PROMOTE"

