from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.answer_format_policy import render_assistant_answer  # noqa: E402


def test_answer_format_policy_defaults_raw() -> None:
    assert render_assistant_answer("0010") == "0010"
    assert render_assistant_answer("42", style="boxed") == "\\boxed{42}"
