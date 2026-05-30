from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.parent_child_eval_protocol import build_parent_child_eval_protocol, validate_eval_report_for_promotion  # noqa: E402


def test_parent_child_protocol_rejects_one_row_eval_as_promotion_evidence(tmp_path: Path) -> None:
    protocol = build_parent_child_eval_protocol(tmp_path / "protocol.json")
    decision = validate_eval_report_for_promotion({"stage": "v1", "rows_evaluated": 1, "answer_format_pass": 1.0, "empty_output_rate": 0.0, "prompt_copy_rate": 0.0, "private_like_delta": 0.1})

    assert protocol["minimum_promotion_rows"] == 60
    assert decision["valid_for_promotion"] is False
    assert "one_row_or_too_small_eval" in decision["reasons"]

