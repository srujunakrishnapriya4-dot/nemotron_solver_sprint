"""Evidence-only smoke handoff to Pass 11 manifests."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import re
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash

from .smoke_report import SmokeTrainingReport, SmokeTrainingReportError, validate_smoke_training_report


class SmokeHandoffError(ValueError):
    """Raised when smoke handoff evidence is invalid."""


@dataclass(frozen=True)
class SmokeHandoffReport:
    training_run_manifest_hash: str
    adapter_hash: str
    backend_result_hash: str
    adapter_validation_hash: str
    training_log_hash: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    report_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("training_run_manifest_hash", "adapter_hash", "backend_result_hash", "adapter_validation_hash"):
            _require_non_empty(getattr(self, name), name)
        if self.training_log_hash is not None:
            _require_non_empty(self.training_log_hash, "training_log_hash")
        metadata = dict(self.metadata)
        _reject_handoff_metadata(metadata)
        if metadata.get("handoff_only") is not True:
            raise SmokeHandoffError("handoff metadata must include handoff_only=True.")
        require_log_validation = metadata.get("require_log_validation") is not False
        if require_log_validation and not self.training_log_hash:
            raise SmokeHandoffError("training_log_hash required when log validation was required.")
        object.__setattr__(self, "metadata", metadata)
        expected = _payload_hash(self)
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected)
        elif self.report_hash != expected:
            raise SmokeHandoffError("report_hash does not match smoke handoff payload.")


def build_smoke_handoff_to_pass11(
    smoke_report: SmokeTrainingReport,
    *,
    adapter_hash: str,
    require_log_validation: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> SmokeHandoffReport:
    try:
        report = validate_smoke_training_report(smoke_report)
    except SmokeTrainingReportError as exc:
        raise SmokeHandoffError(str(exc)) from exc
    if report.passed is not True or report.trained is not True:
        raise SmokeHandoffError("handoff requires passed and trained smoke report.")
    if not report.training_run_manifest_hash or not report.backend_result_hash or not report.adapter_validation_hash:
        raise SmokeHandoffError("handoff requires Pass 11 manifest, backend, and adapter evidence.")
    source_requires_log = report.metadata.get("require_log_validation") is not False
    effective_require_log = require_log_validation or source_requires_log
    if effective_require_log and not report.training_log_hash:
        raise SmokeHandoffError("training_log_hash required when log validation was required.")
    handoff_metadata = {**dict(metadata or {}), "handoff_only": True, "require_log_validation": effective_require_log}
    return SmokeHandoffReport(
        training_run_manifest_hash=report.training_run_manifest_hash,
        adapter_hash=adapter_hash,
        backend_result_hash=report.backend_result_hash,
        adapter_validation_hash=report.adapter_validation_hash,
        training_log_hash=report.training_log_hash,
        metadata=handoff_metadata,
    )


def validate_smoke_handoff_report(report: SmokeHandoffReport | Mapping[str, Any]) -> SmokeHandoffReport:
    normalized = report if isinstance(report, SmokeHandoffReport) else SmokeHandoffReport(**dict(report))
    if normalized.report_hash != _payload_hash(normalized):
        raise SmokeHandoffError("report_hash does not match smoke handoff payload.")
    return normalized


def _payload_hash(report: SmokeHandoffReport) -> str:
    return stable_hash({item.name: getattr(report, item.name) for item in fields(SmokeHandoffReport) if item.name != "report_hash"})


def _reject_handoff_metadata(metadata: Mapping[str, Any]) -> None:
    for key, value in _flatten(metadata):
        raw = f"{key} {value}"
        text = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+]+", " ", raw.lower())).strip()
        if any(
            word in text
            for word in (
                "promote",
                "promotion",
                "package",
                "packaging",
                "submit",
                "submission",
                "leaderboard",
                "kaggle",
                "winner",
            )
        ):
            raise SmokeHandoffError("handoff metadata cannot claim promotion, packaging, submission, leaderboard, or Kaggle status.")
        if "competition ready" in text:
            raise SmokeHandoffError("handoff metadata cannot claim competition readiness.")
        if (
            "95+" in raw.lower()
            or "95 plus" in text
            or "public score" in text
            or "private score" in text
            or ("score" in text and "guarante" in text)
        ):
            raise SmokeHandoffError("handoff metadata cannot claim score guarantees.")


def _flatten(value: Any, prefix: str = "") -> tuple[tuple[str, Any], ...]:
    if isinstance(value, Mapping):
        items: list[tuple[str, Any]] = []
        for key, child in sorted(value.items(), key=lambda item: str(item[0])):
            name = f"{prefix}.{key}" if prefix else str(key)
            items.extend(_flatten(child, name))
        return tuple(items)
    if isinstance(value, (tuple, list, set)):
        items = []
        for index, child in enumerate(value):
            items.extend(_flatten(child, f"{prefix}.{index}"))
        return tuple(items)
    return ((prefix, value),)


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SmokeHandoffError(f"{field_name} must be non-empty.")
    return value


__all__ = [
    "SmokeHandoffError",
    "SmokeHandoffReport",
    "build_smoke_handoff_to_pass11",
    "validate_smoke_handoff_report",
]
