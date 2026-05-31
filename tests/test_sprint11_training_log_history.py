import math

from kaggle_anti086.training.training_log_history import build_training_log_history


def test_captures_fake_loss_history():
    report = build_training_log_history({"log_history": [{"step": 1, "loss": 2.0, "learning_rate": 1e-7}, {"step": 2, "loss": 1.5}]})
    assert report["status"] == "PASS"
    assert report["loss_start"] == 2.0
    assert report["loss_end"] == 1.5
    assert report["steps_completed"] == 2


def test_detects_nan_and_inf():
    nan_report = build_training_log_history({"log_history": [{"loss": math.nan}]})
    inf_report = build_training_log_history({"log_history": [{"loss": math.inf}]})
    assert nan_report["status"] == "FAIL"
    assert "nan_loss" in nan_report["failures"]
    assert inf_report["status"] == "FAIL"
    assert "inf_loss" in inf_report["failures"]
