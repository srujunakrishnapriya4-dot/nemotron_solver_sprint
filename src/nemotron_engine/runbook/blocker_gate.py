"""Fail-closed blocker gate for the final human-operated runbook."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.release import ReleaseCandidateReport
from nemotron_engine.rehearsal import FinalRehearsalManifest, validate_final_rehearsal_manifest

from .command_manifest import CommandManifest, validate_command_manifest
from .operator_checklist import OperatorChecklist, validate_operator_checklist
from .runbook_config import RunbookConfig, reject_forbidden_runbook_claims, validate_runbook_config


class BlockerGateError(ValueError):
    """Raised when blocker gate evidence is inconsistent."""


@dataclass(frozen=True)
class BlockerGateConfig:
    require_release_candidate: bool = False
    allow_warnings: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("require_release_candidate", "allow_warnings"):
            if not isinstance(getattr(self, name), bool):
                raise BlockerGateError(f"{name} must be boolean.")
        metadata = dict(self.metadata)
        reject_forbidden_runbook_claims(metadata, error_cls=BlockerGateError)
        object.__setattr__(self, "metadata", metadata)
        expected = _payload_hash(self, "config_hash")
        if not self.config_hash:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise BlockerGateError("config_hash does not match blocker gate config payload.")


@dataclass(frozen=True)
class BlockerGateReport:
    runbook_config_hash: str
    command_manifest_hash: str
    checklist_hash: str
    final_rehearsal_manifest_hash: str | None
    release_candidate_hash: str | None
    blocked: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    report_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("runbook_config_hash", "command_manifest_hash", "checklist_hash"):
            _require_non_empty(getattr(self, name), name)
        for name in ("final_rehearsal_manifest_hash", "release_candidate_hash"):
            value = getattr(self, name)
            if value is not None:
                _require_non_empty(value, name)
        if not isinstance(self.blocked, bool):
            raise BlockerGateError("blocked must be boolean.")
        blockers = tuple(sorted(str(item) for item in self.blockers))
        warnings = tuple(sorted(str(item) for item in self.warnings))
        metadata = dict(self.metadata)
        reject_forbidden_runbook_claims(metadata, error_cls=BlockerGateError)
        if not self.blocked and blockers:
            raise BlockerGateError("blocked=False cannot include blockers.")
        if self.blocked and not blockers and not metadata.get("reason"):
            raise BlockerGateError("blocked=True requires blockers unless metadata reason exists.")
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "metadata", metadata)
        expected = compute_blocker_gate_report_hash(self)
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected)
        elif self.report_hash != expected:
            raise BlockerGateError("report_hash does not match blocker gate report payload.")


def evaluate_blocker_gate(
    runbook_config: RunbookConfig | Mapping[str, Any],
    command_manifest: CommandManifest | Mapping[str, Any],
    operator_checklist: OperatorChecklist | Mapping[str, Any],
    final_rehearsal_manifest: FinalRehearsalManifest | Mapping[str, Any] | None = None,
    release_candidate_report: ReleaseCandidateReport | None = None,
    metadata: Mapping[str, Any] | None = None,
    gate_config: BlockerGateConfig | Mapping[str, Any] | None = None,
) -> BlockerGateReport:
    cfg = validate_runbook_config(runbook_config)
    gate_cfg = gate_config if isinstance(gate_config, BlockerGateConfig) else BlockerGateConfig(**dict(gate_config or {}))
    commands = validate_command_manifest(command_manifest, cfg)
    checklist = validate_operator_checklist(operator_checklist, cfg)
    meta = dict(metadata or {})
    reject_forbidden_runbook_claims(meta, error_cls=BlockerGateError)
    blockers: list[str] = []
    warnings: list[str] = []
    blockers.extend(checklist.blockers)
    if not checklist.all_required_checked:
        blockers.append("operator_checklist_incomplete")
    final_hash = None
    if cfg.require_final_rehearsal or cfg.require_pass14_lock:
        if final_rehearsal_manifest is None:
            blockers.append("final_rehearsal_manifest_missing")
        else:
            final = validate_final_rehearsal_manifest(final_rehearsal_manifest)
            final_hash = final.manifest_hash
            if final.manifest_hash != cfg.final_rehearsal_manifest_hash:
                blockers.append("final_rehearsal_manifest_hash_mismatch")
            if final.ready_for_external_submission is not True:
                blockers.append("final_rehearsal_not_ready_for_manual_consideration")
            if final.submitted:
                blockers.append("final_rehearsal_cannot_be_submitted")
            warnings.extend(final.warnings)
    elif final_rehearsal_manifest is not None:
        final_hash = validate_final_rehearsal_manifest(final_rehearsal_manifest).manifest_hash
    release_hash = None
    require_release = bool(gate_cfg.require_release_candidate or meta.get("require_release_candidate") is True)
    if release_candidate_report is not None:
        _validate_report_hash(release_candidate_report, "report_hash")
        release_hash = release_candidate_report.report_hash
        if release_candidate_report.ready is not True:
            blockers.append("release_candidate_not_ready")
    elif require_release:
        blockers.append("release_candidate_missing")
    failed_commands = tuple(str(item) for item in meta.get("failed_blocker_commands", ()))
    blocker_command_ids = {command.command_id for command in commands.commands if command.blocker_if_fails}
    for command_id in failed_commands:
        if command_id in blocker_command_ids:
            blockers.append(f"command_failed:{command_id}")
    if warnings and not gate_cfg.allow_warnings:
        blockers.append("warnings_not_allowed")
    unique_blockers = tuple(sorted(set(blockers)))
    return BlockerGateReport(
        runbook_config_hash=cfg.config_hash,
        command_manifest_hash=commands.manifest_hash,
        checklist_hash=checklist.checklist_hash,
        final_rehearsal_manifest_hash=final_hash,
        release_candidate_hash=release_hash,
        blocked=bool(unique_blockers),
        blockers=unique_blockers,
        warnings=tuple(sorted(set(warnings))),
        metadata={**meta, "meaning": "manual consideration only; not external-result evidence"},
    )


def validate_blocker_gate_report(report: BlockerGateReport | Mapping[str, Any]) -> BlockerGateReport:
    normalized = report if isinstance(report, BlockerGateReport) else BlockerGateReport(**dict(report))
    if normalized.report_hash != compute_blocker_gate_report_hash(normalized):
        raise BlockerGateError("report_hash does not match blocker gate report payload.")
    return normalized


def compute_blocker_gate_report_hash(report: BlockerGateReport | Mapping[str, Any]) -> str:
    payload = dict(report) if isinstance(report, Mapping) else {item.name: getattr(report, item.name) for item in fields(BlockerGateReport)}
    payload.pop("report_hash", None)
    return stable_hash(payload)


def _validate_report_hash(report: object, hash_field: str) -> None:
    try:
        observed = getattr(report, hash_field)
        expected = stable_hash({item.name: getattr(report, item.name) for item in fields(report) if item.name != hash_field})
    except Exception as exc:
        raise BlockerGateError("could not validate input report hash.") from exc
    if observed != expected:
        raise BlockerGateError("input report hash mismatch.")


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BlockerGateError(f"{field_name} must be non-empty.")
    return value.strip()


__all__ = [
    "BlockerGateConfig",
    "BlockerGateError",
    "BlockerGateReport",
    "compute_blocker_gate_report_hash",
    "evaluate_blocker_gate",
    "validate_blocker_gate_report",
]
