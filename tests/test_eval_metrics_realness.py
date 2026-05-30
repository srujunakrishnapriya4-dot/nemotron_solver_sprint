from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle_anti086"))

from kaggle_eval_anti086_vllm import compute_eval_metrics  # noqa: E402


def test_eval_metrics_realness_computes_non_hardcoded_rates() -> None:
    records = [
        {"id": "1", "family": "bit_manipulation", "prompt": "abc prompt", "prediction": "", "expected": "1", "exact_match": False},
        {"id": "2", "family": "unit_conversion", "prompt": "long copied answer", "prediction": "long copied answer", "expected": "2", "exact_match": False},
        {"id": "3", "family": "unit_conversion", "prompt": "p", "prediction": "3", "expected": "3", "exact_match": True},
    ]
    metrics = compute_eval_metrics(records, mode="private_like_family_hard", adapter_path="/a", decoding_params={"temperature": 0})
    assert metrics["empty_output_rate"] == 1 / 3
    assert metrics["prompt_copy_rate"] == 1 / 3
    assert metrics["answer_format_pass_rate"] == 2 / 3
    assert "by_family_exact_match" in metrics
