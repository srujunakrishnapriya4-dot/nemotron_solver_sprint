from __future__ import annotations

from kaggle_anti086.training.day10_adapter_eval import build_needs_kaggle_eval_report, evaluate_prediction_rows


class _Args:
    base_model_path = "/kaggle/input/nemotron-model"
    day10_adapter_dir = "/kaggle/working/adapter"
    v2a_adapter_dir = None
    tinker_adapter_dir = None


def test_adapter_eval_metrics_from_precomputed_predictions():
    rows = [
        {"family": "numeric_formula", "expected": "17", "day10_pred": "17", "expected_behavior": "answer"},
        {"family": "word_cipher", "expected": "BLUE", "day10_pred": "The answer is BLUE because", "expected_behavior": "answer"},
        {"family": "unknown", "expected": "ABSTAIN", "day10_pred": "42", "expected_behavior": "abstain"},
    ]

    report = evaluate_prediction_rows(rows)

    assert report["status"] == "PASS"
    assert report["exact_match"] == 1 / 3
    assert report["verbose_output_rate"] == 1 / 3
    assert report["unsafe_answer_rate"] == 1 / 3


def test_adapter_eval_local_mode_is_honest_needs_kaggle():
    report = build_needs_kaggle_eval_report(_Args())

    assert report["status"] == "NEEDS_KAGGLE_MODEL_EVAL"
    assert report["model_eval_completed"] is False
    assert report["tinker_usage"] == "baseline_only_if_locally_available"
    assert report["packaging_allowed"] is False
