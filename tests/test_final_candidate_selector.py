from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.final_candidate_selector import select_final_candidate  # noqa: E402


def test_final_selector_falls_back_to_parent_if_no_child_beats_parent(tmp_path: Path) -> None:
    decision = select_final_candidate(
        [
            {"name": "parent", "is_parent": True, "rank": 32, "adapter_size_mb": 900, "adapter_size_limit_mb": 1500, "package_clean": True, "known_good_public": True},
            {"name": "child", "rank": 32, "adapter_size_mb": 1000, "adapter_size_limit_mb": 1500, "package_clean": True, "private_like_delta": 0.0, "public_like_delta": 0.01},
        ],
        output_path=tmp_path / "decision.json",
    )

    assert decision["decision"] == "FALLBACK_PARENT"
    assert decision["selected_name"] == "parent"


def test_final_selector_rejects_public_like_only_gain_with_private_regression(tmp_path: Path) -> None:
    decision = select_final_candidate(
        [
            {"name": "parent", "is_parent": True, "rank": 32, "adapter_size_mb": 900, "adapter_size_limit_mb": 1500, "package_clean": True, "known_good_public": True},
            {"name": "child", "rank": 32, "adapter_size_mb": 1000, "adapter_size_limit_mb": 1500, "package_clean": True, "private_like_delta": -0.02, "public_like_delta": 0.02},
        ],
        output_path=tmp_path / "decision.json",
    )

    assert decision["decision"] == "FALLBACK_PARENT"
    assert any("public_like_only_gain_private_regression" in reason for reason in decision["rejected_reasons"])

