from __future__ import annotations

from dataclasses import replace
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.evaluation.regression_report import (
    REQUIRED_REGRESSION_METRICS,
    RegressionMetric,
    RegressionReport,
    RegressionReportError,
    build_regression_report,
    compare_reports,
)


def metrics(**updates: float) -> dict[str, float]:
    data = {
        "overall_accuracy": 0.95,
        "private_like_accuracy": 0.95,
        "format_error_rate": 0.0,
        "extraction_error_rate": 0.0,
        "transfer_gap": 0.02,
        "stability_score": 0.99,
        "contamination_rate": 0.0,
    }
    data.update(updates)
    return data


def thresholds(value: float = 0.01) -> dict[str, float]:
    return {name: value for name in REQUIRED_REGRESSION_METRICS}


def test_rejects_nan_inf_and_missing_required_metrics() -> None:
    with pytest.raises(RegressionReportError):
        build_regression_report(metrics(overall_accuracy=math.nan), metrics(), thresholds())
    with pytest.raises(RegressionReportError):
        RegressionMetric("x", 1.0, math.inf, False, 0.0)
    bad = metrics()
    bad.pop("transfer_gap")
    with pytest.raises(RegressionReportError):
        build_regression_report(bad, metrics(), thresholds())


def test_lower_is_better_and_thresholds_work() -> None:
    report = build_regression_report(metrics(format_error_rate=0.03), metrics(format_error_rate=0.0), thresholds(0.01))

    assert report.passed is False
    assert "regression:format_error_rate" in report.failure_reasons


def test_report_hash_deterministic_and_forgery_rejected() -> None:
    first = build_regression_report(metrics(), metrics(), thresholds())
    second = build_regression_report(metrics(), metrics(), thresholds())

    assert first.report_hash == second.report_hash
    with pytest.raises(RegressionReportError):
        replace(first, report_hash="forged")


def test_report_rejects_passed_true_with_failing_metric() -> None:
    metric = RegressionMetric("overall_accuracy", 1.0, 0.8, False, 0.01)

    with pytest.raises(RegressionReportError):
        RegressionReport((metric,), True)


def test_compare_reports_deterministic() -> None:
    baseline = build_regression_report(metrics(), metrics(), thresholds())
    candidate = build_regression_report(metrics(overall_accuracy=0.94), metrics(), thresholds(0.02))

    first = compare_reports(baseline, candidate, thresholds=thresholds(0.02))
    second = compare_reports(baseline, candidate, thresholds=thresholds(0.02))

    assert first.report_hash == second.report_hash
