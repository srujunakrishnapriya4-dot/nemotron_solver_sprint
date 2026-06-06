from __future__ import annotations

from kaggle_anti086.training.day10_candidate_decision import decide_candidate


def test_candidate_decision_defaults_no_submit_without_public_lb_record():
    report = decide_candidate(
        eval_report={"model_eval_completed": True, "exact_match": 0.9, "baselines": {"v2a_exact_match": 0.8, "tinker_exact_match": 0.85}},
        error_mining_report={"status": "PASS"},
        package_guard_report={"status": "PASS"},
        public_lb_calibration=None,
    )

    assert report["status"] == "PASS"
    assert report["submit_recommended"] is False
    assert report["packaging_allowed"] is False
    assert report["submission_allowed"] is False
    assert "public_lb_calibration_missing_submit_recommended_false" in report["warnings"]


def test_candidate_decision_fails_when_eval_missing_or_regressed():
    report = decide_candidate(
        eval_report={"model_eval_completed": True, "exact_match": 0.7, "baselines": {"v2a_exact_match": 0.8}},
        error_mining_report={"status": "PASS"},
        package_guard_report={"status": "PASS"},
    )

    assert report["status"] == "FAIL"
    assert "day10_adapter_does_not_beat_v2a" in report["failures"]
