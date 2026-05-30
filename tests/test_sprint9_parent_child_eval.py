from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_anti086"))

from kaggle_eval_anti086_vllm import compute_eval_metrics, extract_answer  # noqa: E402


def test_parent_child_eval_metrics_compute_delta_regressions_improvements() -> None:
    parent = [
        {"id": "a", "family": "bit_manipulation", "prediction": "1", "exact_match": True},
        {"id": "b", "family": "unit_conversion", "prediction": "x", "exact_match": False},
    ]
    child = [
        {"id": "a", "family": "bit_manipulation", "prompt": "p", "prediction": "0", "exact_match": False},
        {"id": "b", "family": "unit_conversion", "prompt": "p", "prediction": "2", "exact_match": True},
    ]
    metrics = compute_eval_metrics(child, mode="smoke_16", adapter_path="/child", decoding_params={}, parent_records=parent)
    assert metrics["parent_exact_match"] == 0.5
    assert metrics["child_exact_match"] == 0.5
    assert metrics["child_minus_parent_delta"] == 0.0
    assert metrics["child_regressions"] == ["a"]
    assert metrics["child_improvements"] == ["b"]


def test_answer_extractor_handles_boxed_and_last_line() -> None:
    assert extract_answer("reason\n\\boxed{1010}") == "1010"
    assert extract_answer("blah\nfinal") == "final"
