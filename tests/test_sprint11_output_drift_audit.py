from kaggle_anti086.training.output_drift_audit import audit_output_drift


def test_identical_outputs_warn_low_change():
    report = audit_output_drift([{"base_pred": "a", "v2a_pred": "a"} for _ in range(10)])
    assert report["status"] == "PASS"
    assert "output_changed_rate_below_0_01" in report["warnings"]


def test_verbose_empty_and_copied_outputs_fail():
    preds = [{"base_pred": "x", "v2a_pred": "because long explanation", "prompt": "p"} for _ in range(20)]
    preds += [{"base_pred": "x", "v2a_pred": "", "prompt": "p"} for _ in range(2)]
    preds += [{"base_pred": "x", "v2a_pred": "copy prompt p", "prompt": "copy prompt p"}]
    report = audit_output_drift(preds)
    assert report["status"] == "FAIL"
    assert "verbose_output_rate_gt_0_10" in report["failures"]
