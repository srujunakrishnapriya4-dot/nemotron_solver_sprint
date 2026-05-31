from kaggle_anti086.training.day8_2_smoke_train import build_smoke_summary


def test_smoke_summary_passes_only_with_valid_adapter():
    report = build_smoke_summary({"status": "PASS", "steps_completed": 5, "loss_nan_detected": False}, {"status": "PASS"}, {"status": "PASS"})
    assert report["status"] == "PASS"
    bad = build_smoke_summary({"status": "FAIL", "steps_completed": 0}, {"status": "FAIL"}, {"status": "SKIPPED"})
    assert bad["status"] == "FAIL"
