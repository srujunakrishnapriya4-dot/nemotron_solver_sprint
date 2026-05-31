import json

from kaggle_anti086.training.package_submission_guard import build_package_submission_guard


def test_submission_zip_fails(tmp_path):
    (tmp_path / "submission.zip").write_bytes(b"not allowed")
    report = build_package_submission_guard(paths=[tmp_path])
    assert report["status"] == "FAIL"
    assert report["submission_zip_found"] is True


def test_packaging_flag_fails(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"packaging_allowed": True, "submission_allowed": False}), encoding="utf-8")
    report = build_package_submission_guard(paths=[tmp_path / "empty"], reports=[report_path])
    assert report["status"] == "FAIL"
    assert "packaging_allowed_true" in report["failures"]


def test_clean_guard_passes(tmp_path):
    report = build_package_submission_guard(paths=[tmp_path])
    assert report["status"] == "PASS"
    assert report["package_allowed"] is False
    assert report["submission_allowed"] is False
