from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_anti086"))

from kaggle_prepare_anti086_tokens import _assistant_has_reasoning, transform_direct_answer_row  # noqa: E402


def test_direct_answer_transform_uses_only_final_answer_target() -> None:
    row = {"id": "r1", "prompt": "Solve 1+1.", "answer": "2", "family": "equation_symbolic"}
    out = transform_direct_answer_row(row)
    assert out["corpus_type"] == "direct_answer_only"
    assert "Respond with only the final answer. No explanation." in out["text"]
    assert out["text"].endswith("Assistant:\n2")
    assert _assistant_has_reasoning(out["text"], "2") is False


def test_direct_answer_transform_rejects_reasoning_assistant_text() -> None:
    text = "User:\nQ\nAssistant:\nBecause it is simple, answer: 2"
    assert _assistant_has_reasoning(text, "2") is True
