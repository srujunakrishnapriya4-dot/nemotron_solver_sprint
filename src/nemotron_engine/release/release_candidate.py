"""Final release candidate go/no-go reports."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
import re
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash

from .system_audit import SystemAuditReport, validate_system_audit_report


class ReleaseCandidateError(ValueError):
    """Raised when release candidate reports are dishonest or inconsistent."""


RELEASE_LEVELS = ("internal_candidate", "training_ready_candidate", "packaging_ready_candidate", "submission_dry_run_candidate")


@dataclass(frozen=True)
class ReleaseCandidateConfig:
    system_audit_report: SystemAuditReport
    repository_root: str | Path | None = None
    release_level: str = "internal_candidate"
    scoped_suite_result_summary: str | None = None
    accepted_suite_summaries: tuple[str, ...] = ("597 passed, 1 skipped", "595 passed, 1 skipped", "559 passed, 1 skipped", "494 passed, 1 skipped")
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    required_next_actions: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.system_audit_report, SystemAuditReport):
            raise ReleaseCandidateError("system_audit_report must be a SystemAuditReport.")
        if self.release_level not in RELEASE_LEVELS:
            raise ReleaseCandidateError("unsupported release_level.")
        if not isinstance(self.accepted_suite_summaries, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.accepted_suite_summaries
        ):
            raise ReleaseCandidateError("accepted_suite_summaries must be a tuple of non-empty strings.")
        object.__setattr__(self, "repository_root", None if self.repository_root is None else str(self.repository_root))
        object.__setattr__(self, "blockers", tuple(str(item) for item in self.blockers))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(self, "required_next_actions", tuple(str(item) for item in self.required_next_actions))
        object.__setattr__(self, "metadata", dict(self.metadata))
        _reject_unsafe_claims(self.metadata)
        expected = _payload_hash(self, "config_hash")
        if not self.config_hash:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise ReleaseCandidateError("config_hash does not match release candidate config payload.")


@dataclass(frozen=True)
class ReleaseCandidateReport:
    candidate_id: str
    repository_root: str | None
    locked_registry_hash: str
    system_audit_hash: str
    scoped_suite_result_summary: str | None
    ready: bool
    release_level: str
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    required_next_actions: tuple[str, ...] = ()
    system_audit_passed: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    report_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("candidate_id", "locked_registry_hash", "system_audit_hash", "release_level"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ReleaseCandidateError(f"{name} must be non-empty.")
        if self.release_level not in RELEASE_LEVELS:
            raise ReleaseCandidateError("unsupported release_level.")
        if self.ready and self.blockers:
            raise ReleaseCandidateError("ready=True cannot include blockers.")
        if self.ready and not self.system_audit_passed:
            raise ReleaseCandidateError("ready=True requires a passed system audit.")
        if self.repository_root is not None and not isinstance(self.repository_root, str):
            raise ReleaseCandidateError("repository_root must be a string or None.")
        if self.scoped_suite_result_summary is not None and not isinstance(self.scoped_suite_result_summary, str):
            raise ReleaseCandidateError("scoped_suite_result_summary must be a string or None.")
        object.__setattr__(self, "blockers", tuple(str(item) for item in self.blockers))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(self, "required_next_actions", tuple(str(item) for item in self.required_next_actions))
        object.__setattr__(self, "metadata", dict(self.metadata))
        _reject_unsafe_claims(self.metadata)
        expected = _payload_hash(self, "report_hash")
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected)
        elif self.report_hash != expected:
            raise ReleaseCandidateError("report_hash does not match release candidate report payload.")


def build_release_candidate_report(config: ReleaseCandidateConfig) -> ReleaseCandidateReport:
    if not isinstance(config, ReleaseCandidateConfig):
        raise ReleaseCandidateError("config must be a ReleaseCandidateConfig.")
    validate_system_audit_report(config.system_audit_report)
    suite_summary = config.scoped_suite_result_summary or config.system_audit_report.scoped_suite_result_summary
    blockers = list(config.blockers)
    if not config.system_audit_report.passed:
        blockers.append("system_audit_failed")
    if suite_summary not in config.accepted_suite_summaries:
        blockers.append("suite_summary_not_accepted")
    ready = not blockers
    candidate_id = stable_hash(
        {
            "repository_root": config.repository_root,
            "system_audit_hash": config.system_audit_report.report_hash,
            "suite_summary": suite_summary,
            "release_level": config.release_level,
            "metadata": config.metadata,
        }
    )
    return ReleaseCandidateReport(
        candidate_id=candidate_id,
        repository_root=config.repository_root,
        locked_registry_hash=config.system_audit_report.locked_registry_hash,
        system_audit_hash=config.system_audit_report.report_hash,
        scoped_suite_result_summary=suite_summary,
        ready=ready,
        release_level=config.release_level,
        blockers=tuple(sorted(set(blockers))),
        warnings=tuple(sorted(set(config.warnings + config.system_audit_report.warnings))),
        required_next_actions=config.required_next_actions,
        system_audit_passed=config.system_audit_report.passed,
        metadata=dict(config.metadata),
    )


def evaluate_release_candidate(config: ReleaseCandidateConfig) -> ReleaseCandidateReport:
    return build_release_candidate_report(config)


def _reject_unsafe_claims(metadata: Mapping[str, Any]) -> None:
    flattened = tuple(_flatten_metadata(metadata))
    has_adapter_evidence = _truthy(metadata.get("validated_adapter_hash")) and _truthy(metadata.get("preflight_gate_hash"))
    for key, value in flattened:
        text = _normalized_text(f"{key} {value}")
        truthy = value is True or str(value).strip().lower() in {"true", "yes", "ready", "success", "guaranteed"}
        if ("kaggle" in text or "leaderboard" in text or "submission" in text) and any(
            word in text for word in ("success", "ready", "readiness", "submitted")
        ):
            raise ReleaseCandidateError("release candidate cannot claim Kaggle or leaderboard success/readiness.")
        if "public" in text and "leaderboard" in text:
            raise ReleaseCandidateError("release candidate cannot claim public leaderboard readiness.")
        if "private" in text and ("leaderboard" in text or "score" in text):
            raise ReleaseCandidateError("release candidate cannot claim private leaderboard/score readiness.")
        if "95" in text and "guarante" in text:
            raise ReleaseCandidateError("release candidate cannot claim guaranteed scores.")
        if "score" in text and "guarante" in text:
            raise ReleaseCandidateError("release candidate cannot claim guaranteed scores.")
        if "guaranteed" in text:
            raise ReleaseCandidateError("release candidate cannot claim guaranteed scores.")
        adapter_claim = (
            ("trained" in text and "adapter" in text)
            or ("adapter" in text and any(term in text for term in ("available", "success", "ready")))
        )
        if (truthy or adapter_claim) and adapter_claim and not has_adapter_evidence:
            raise ReleaseCandidateError("release candidate cannot claim trained adapter without validated evidence.")


def _flatten_metadata(metadata: Mapping[str, Any], prefix: str = "") -> tuple[tuple[str, Any], ...]:
    items: list[tuple[str, Any]] = []
    for key, value in sorted(metadata.items(), key=lambda item: str(item[0])):
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            items.extend(_flatten_metadata(value, full_key))
        else:
            items.append((full_key, value))
    return tuple(items)


def _truthy(value: Any) -> bool:
    return value is True or (isinstance(value, str) and bool(value.strip()))


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+]+", " ", value.lower())).strip()


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


__all__ = [
    "RELEASE_LEVELS",
    "ReleaseCandidateConfig",
    "ReleaseCandidateError",
    "ReleaseCandidateReport",
    "build_release_candidate_report",
    "evaluate_release_candidate",
]
