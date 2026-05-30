"""Deterministic regression reports for Pass 8 promotion gates."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
from typing import Mapping

from nemotron_engine.core.schemas import stable_hash


REQUIRED_REGRESSION_METRICS = (
    "overall_accuracy",
    "private_like_accuracy",
    "format_error_rate",
    "extraction_error_rate",
    "transfer_gap",
    "stability_score",
    "contamination_rate",
)

LOWER_IS_BETTER_METRICS = frozenset(
    {
        "format_error_rate",
        "extraction_error_rate",
        "transfer_gap",
        "contamination_rate",
    }
)


class RegressionReportError(ValueError):
    """Raised when regression metrics or reports are inconsistent."""


@dataclass(frozen=True)
class RegressionMetric:
    name: str
    baseline: float
    current: float
    lower_is_better: bool
    max_regression: float
    delta: float | None = None
    regression: float | None = None
    passed: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise RegressionReportError("metric name must be non-empty.")
        _require_finite(self.baseline, "baseline")
        _require_finite(self.current, "current")
        _require_finite(self.max_regression, "max_regression")
        if self.max_regression < 0:
            raise RegressionReportError("max_regression must be non-negative.")
        if not isinstance(self.lower_is_better, bool):
            raise RegressionReportError("lower_is_better must be boolean.")
        delta = float(self.current) - float(self.baseline)
        regression = delta if self.lower_is_better else -delta
        passed = regression <= float(self.max_regression) + 1e-12
        if self.delta is not None and abs(float(self.delta) - delta) > 1e-12:
            raise RegressionReportError("delta does not match baseline/current.")
        if self.regression is not None and abs(float(self.regression) - regression) > 1e-12:
            raise RegressionReportError("regression does not match metric direction.")
        if self.passed is True and not passed:
            raise RegressionReportError("metric cannot pass while exceeding regression threshold.")
        if self.passed is False and passed:
            raise RegressionReportError("metric cannot fail while within regression threshold.")
        object.__setattr__(self, "baseline", float(self.baseline))
        object.__setattr__(self, "current", float(self.current))
        object.__setattr__(self, "max_regression", float(self.max_regression))
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "regression", regression)
        object.__setattr__(self, "passed", passed)


@dataclass(frozen=True)
class RegressionReport:
    metrics: tuple[RegressionMetric, ...]
    passed: bool
    failure_reasons: tuple[str, ...] = ()
    report_hash: str = ""

    def __post_init__(self) -> None:
        metrics = tuple(self.metrics)
        if not metrics:
            raise RegressionReportError("RegressionReport requires at least one metric.")
        if len({metric.name for metric in metrics}) != len(metrics):
            raise RegressionReportError("RegressionReport metric names must be unique.")
        sorted_metrics = tuple(sorted(metrics, key=lambda item: item.name))
        failures = tuple(str(item) for item in self.failure_reasons)
        failed_names = tuple(metric.name for metric in sorted_metrics if not metric.passed)
        if self.passed is True and failed_names:
            raise RegressionReportError("passed=True while at least one metric failed.")
        expected_failures = failures or tuple(f"regression:{name}" for name in failed_names)
        passed = not failed_names
        if self.passed is True and expected_failures:
            raise RegressionReportError("passed=True cannot include failure_reasons.")
        object.__setattr__(self, "metrics", sorted_metrics)
        object.__setattr__(self, "passed", passed)
        object.__setattr__(self, "failure_reasons", () if passed else expected_failures)
        expected_hash = _payload_hash(self, "report_hash")
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected_hash)
        elif self.report_hash != expected_hash:
            raise RegressionReportError("report_hash does not match regression report payload.")


def build_regression_report(
    current_metrics: Mapping[str, float],
    baseline_metrics: Mapping[str, float],
    thresholds: Mapping[str, float],
    *,
    lower_is_better: Mapping[str, bool] | None = None,
    required_metrics: tuple[str, ...] = REQUIRED_REGRESSION_METRICS,
) -> RegressionReport:
    """Build a deterministic pass/fail report from current and baseline metrics."""

    if not isinstance(current_metrics, Mapping) or not isinstance(baseline_metrics, Mapping):
        raise RegressionReportError("current_metrics and baseline_metrics must be mappings.")
    missing = tuple(name for name in required_metrics if name not in current_metrics or name not in baseline_metrics)
    if missing:
        raise RegressionReportError(f"missing required metrics: {missing}")
    directions = dict(lower_is_better or {})
    metrics: list[RegressionMetric] = []
    for name in sorted(required_metrics):
        is_lower_better = bool(directions.get(name, name in LOWER_IS_BETTER_METRICS))
        threshold = float(thresholds.get(name, 0.0))
        metrics.append(
            RegressionMetric(
                name=name,
                baseline=float(baseline_metrics[name]),
                current=float(current_metrics[name]),
                lower_is_better=is_lower_better,
                max_regression=threshold,
            )
        )
    failures = tuple(f"regression:{metric.name}" for metric in metrics if not metric.passed)
    return RegressionReport(metrics=tuple(metrics), passed=not failures, failure_reasons=failures)


def compare_reports(
    baseline_report: RegressionReport,
    candidate_report: RegressionReport,
    *,
    thresholds: Mapping[str, float] | None = None,
) -> RegressionReport:
    """Compare two reports by using their current metric values as baseline/candidate."""

    _validate_report_hash(baseline_report)
    _validate_report_hash(candidate_report)
    baseline = {metric.name: metric.current for metric in baseline_report.metrics}
    current = {metric.name: metric.current for metric in candidate_report.metrics}
    directions = {metric.name: metric.lower_is_better for metric in candidate_report.metrics}
    chosen_thresholds = thresholds or {metric.name: metric.max_regression for metric in candidate_report.metrics}
    return build_regression_report(
        current,
        baseline,
        chosen_thresholds,
        lower_is_better=directions,
        required_metrics=tuple(sorted(current)),
    )


def _validate_report_hash(report: RegressionReport) -> None:
    expected = _payload_hash(report, "report_hash")
    if report.report_hash != expected:
        raise RegressionReportError("report_hash does not match regression report payload.")


def _require_finite(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(float(value)) or math.isinf(float(value)):
        raise RegressionReportError(f"{field_name} must be a finite number.")


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


__all__ = [
    "LOWER_IS_BETTER_METRICS",
    "REQUIRED_REGRESSION_METRICS",
    "RegressionMetric",
    "RegressionReport",
    "RegressionReportError",
    "build_regression_report",
    "compare_reports",
]
