"""Top-level Pass 15 final operations manifest."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash

from .blocker_gate import BlockerGateReport, validate_blocker_gate_report
from .command_manifest import CommandManifest, validate_command_manifest
from .failure_recovery import FailureRecoveryPlan, validate_failure_recovery_plan
from .human_approval import HumanApprovalStatement, validate_human_approval_statement
from .operator_checklist import OperatorChecklist, validate_operator_checklist
from .reproducibility_manifest import ReproducibilityManifest, validate_reproducibility_manifest
from .runbook_config import RunbookConfig, reject_forbidden_runbook_claims, validate_runbook_config
from .submission_record import SubmissionRecord, validate_submission_record


class FinalOperationsManifestError(ValueError):
    """Raised when final operations state is inconsistent."""


@dataclass(frozen=True)
class FinalOperationsManifest:
    operations_id: str
    runbook_config_hash: str
    command_manifest_hash: str
    checklist_hash: str
    blocker_gate_hash: str
    reproducibility_manifest_hash: str
    human_approval_hash: str | None
    submission_record_hash: str | None
    failure_recovery_hash: str
    ready_for_manual_submission: bool
    manual_submission_recorded: bool
    blocked: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    operations_hash: str = ""

    def __post_init__(self) -> None:
        for name in (
            "operations_id",
            "runbook_config_hash",
            "command_manifest_hash",
            "checklist_hash",
            "blocker_gate_hash",
            "reproducibility_manifest_hash",
            "failure_recovery_hash",
        ):
            _require_non_empty(getattr(self, name), name)
        for name in ("human_approval_hash", "submission_record_hash"):
            value = getattr(self, name)
            if value is not None:
                _require_non_empty(value, name)
        for name in ("ready_for_manual_submission", "manual_submission_recorded", "blocked"):
            if not isinstance(getattr(self, name), bool):
                raise FinalOperationsManifestError(f"{name} must be boolean.")
        blockers = tuple(sorted(str(item) for item in self.blockers))
        warnings = tuple(sorted(str(item) for item in self.warnings))
        metadata = dict(self.metadata)
        reject_forbidden_runbook_claims(metadata, error_cls=FinalOperationsManifestError)
        if self.ready_for_manual_submission and metadata.get("builder_validated_evidence") is not True:
            raise FinalOperationsManifestError("ready_for_manual_submission requires builder-validated evidence.")
        if self.ready_for_manual_submission and self.human_approval_hash is None:
            raise FinalOperationsManifestError("ready_for_manual_submission requires approved human approval.")
        if self.ready_for_manual_submission and self.blocked:
            raise FinalOperationsManifestError("ready_for_manual_submission cannot be true when blocked.")
        if self.ready_for_manual_submission and blockers:
            raise FinalOperationsManifestError("ready_for_manual_submission cannot include blockers.")
        if self.manual_submission_recorded and self.submission_record_hash is None:
            raise FinalOperationsManifestError("manual_submission_recorded requires a supplied manual submission record.")
        if self.blocked != bool(blockers):
            raise FinalOperationsManifestError("blocked state must match blockers.")
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "metadata", metadata)
        expected_id = compute_final_operations_id(self)
        if self.operations_id != expected_id:
            raise FinalOperationsManifestError("operations_id does not match final operations identity payload.")
        expected_hash = compute_final_operations_manifest_hash(self)
        if not self.operations_hash:
            object.__setattr__(self, "operations_hash", expected_hash)
        elif self.operations_hash != expected_hash:
            raise FinalOperationsManifestError("operations_hash does not match final operations payload.")


def build_final_operations_manifest(
    runbook_config: RunbookConfig | Mapping[str, Any],
    command_manifest: CommandManifest | Mapping[str, Any],
    operator_checklist: OperatorChecklist | Mapping[str, Any],
    blocker_gate_report: BlockerGateReport | Mapping[str, Any],
    reproducibility_manifest: ReproducibilityManifest | Mapping[str, Any],
    failure_recovery_plan: FailureRecoveryPlan | Mapping[str, Any],
    *,
    human_approval: HumanApprovalStatement | Mapping[str, Any] | None = None,
    submission_record: SubmissionRecord | Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FinalOperationsManifest:
    cfg = validate_runbook_config(runbook_config)
    commands = validate_command_manifest(command_manifest, cfg)
    checklist = validate_operator_checklist(operator_checklist, cfg)
    gate = validate_blocker_gate_report(blocker_gate_report)
    repro = validate_reproducibility_manifest(reproducibility_manifest)
    recovery = validate_failure_recovery_plan(failure_recovery_plan, cfg)
    approval = validate_human_approval_statement(human_approval) if human_approval is not None else None
    record = validate_submission_record(submission_record) if submission_record is not None else None
    _validate_component_bindings(
        cfg=cfg,
        commands=commands,
        checklist=checklist,
        gate=gate,
        repro=repro,
        recovery=recovery,
        approval=approval,
        record=record,
    )
    blockers = tuple(gate.blockers)
    ready = bool(gate.blocked is False and approval is not None and approval.approved and not blockers)
    manual_recorded = bool(record is not None and record.manual_submission_performed)
    payload = {
        "runbook_config_hash": cfg.config_hash,
        "command_manifest_hash": commands.manifest_hash,
        "checklist_hash": checklist.checklist_hash,
        "blocker_gate_hash": gate.report_hash,
        "reproducibility_manifest_hash": repro.manifest_hash,
        "human_approval_hash": approval.approval_hash if approval is not None else None,
        "submission_record_hash": record.record_hash if record is not None else None,
        "failure_recovery_hash": recovery.plan_hash,
        "ready_for_manual_submission": ready,
        "manual_submission_recorded": manual_recorded,
        "blocked": bool(blockers),
        "blockers": blockers,
    }
    return FinalOperationsManifest(
        operations_id=stable_hash(payload),
        runbook_config_hash=cfg.config_hash,
        command_manifest_hash=commands.manifest_hash,
        checklist_hash=checklist.checklist_hash,
        blocker_gate_hash=gate.report_hash,
        reproducibility_manifest_hash=repro.manifest_hash,
        human_approval_hash=approval.approval_hash if approval is not None else None,
        submission_record_hash=record.record_hash if record is not None else None,
        failure_recovery_hash=recovery.plan_hash,
        ready_for_manual_submission=ready,
        manual_submission_recorded=manual_recorded,
        blocked=bool(blockers),
        blockers=blockers,
        warnings=gate.warnings,
        metadata={
            **dict(metadata or {}),
            "meaning": "human may manually consider external action; record is not external-result proof",
            "builder_validated_evidence": ready,
        },
    )


def validate_final_operations_manifest(manifest: FinalOperationsManifest | Mapping[str, Any]) -> FinalOperationsManifest:
    normalized = manifest if isinstance(manifest, FinalOperationsManifest) else FinalOperationsManifest(**dict(manifest))
    if normalized.operations_id != compute_final_operations_id(normalized):
        raise FinalOperationsManifestError("operations_id does not match final operations identity payload.")
    if normalized.operations_hash != compute_final_operations_manifest_hash(normalized):
        raise FinalOperationsManifestError("operations_hash does not match final operations payload.")
    return normalized


def compute_final_operations_manifest_hash(manifest: FinalOperationsManifest | Mapping[str, Any]) -> str:
    payload = dict(manifest) if isinstance(manifest, Mapping) else {item.name: getattr(manifest, item.name) for item in fields(FinalOperationsManifest)}
    payload.pop("operations_hash", None)
    return stable_hash(payload)


def compute_final_operations_id(manifest: FinalOperationsManifest | Mapping[str, Any]) -> str:
    payload = dict(manifest) if isinstance(manifest, Mapping) else {item.name: getattr(manifest, item.name) for item in fields(FinalOperationsManifest)}
    return stable_hash(
        {
            "runbook_config_hash": payload.get("runbook_config_hash"),
            "command_manifest_hash": payload.get("command_manifest_hash"),
            "checklist_hash": payload.get("checklist_hash"),
            "blocker_gate_hash": payload.get("blocker_gate_hash"),
            "reproducibility_manifest_hash": payload.get("reproducibility_manifest_hash"),
            "human_approval_hash": payload.get("human_approval_hash"),
            "submission_record_hash": payload.get("submission_record_hash"),
            "failure_recovery_hash": payload.get("failure_recovery_hash"),
            "ready_for_manual_submission": payload.get("ready_for_manual_submission"),
            "manual_submission_recorded": payload.get("manual_submission_recorded"),
            "blocked": payload.get("blocked"),
            "blockers": payload.get("blockers"),
        }
    )


def _validate_component_bindings(
    *,
    cfg: RunbookConfig,
    commands: CommandManifest,
    checklist: OperatorChecklist,
    gate: BlockerGateReport,
    repro: ReproducibilityManifest,
    recovery: FailureRecoveryPlan,
    approval: HumanApprovalStatement | None,
    record: SubmissionRecord | None,
) -> None:
    if commands.runbook_config_hash != cfg.config_hash:
        raise FinalOperationsManifestError("command manifest runbook_config_hash mismatch.")
    if checklist.runbook_config_hash != cfg.config_hash:
        raise FinalOperationsManifestError("operator checklist runbook_config_hash mismatch.")
    if gate.runbook_config_hash != cfg.config_hash:
        raise FinalOperationsManifestError("blocker gate runbook_config_hash mismatch.")
    if gate.command_manifest_hash != commands.manifest_hash:
        raise FinalOperationsManifestError("blocker gate command_manifest_hash mismatch.")
    if gate.checklist_hash != checklist.checklist_hash:
        raise FinalOperationsManifestError("blocker gate checklist_hash mismatch.")
    if repro.runbook_config_hash != cfg.config_hash:
        raise FinalOperationsManifestError("reproducibility manifest runbook_config_hash mismatch.")
    if repro.command_manifest_hash != commands.manifest_hash:
        raise FinalOperationsManifestError("reproducibility manifest command_manifest_hash mismatch.")
    if repro.checklist_hash != checklist.checklist_hash:
        raise FinalOperationsManifestError("reproducibility manifest checklist_hash mismatch.")
    if repro.blocker_gate_hash != gate.report_hash:
        raise FinalOperationsManifestError("reproducibility manifest blocker_gate_hash mismatch.")
    if recovery.runbook_config_hash != cfg.config_hash:
        raise FinalOperationsManifestError("failure recovery runbook_config_hash mismatch.")
    if approval is not None:
        if approval.runbook_config_hash != cfg.config_hash:
            raise FinalOperationsManifestError("human approval runbook_config_hash mismatch.")
        if approval.blocker_gate_hash != gate.report_hash:
            raise FinalOperationsManifestError("human approval blocker_gate_hash mismatch.")
        if approval.reproducibility_manifest_hash != repro.manifest_hash:
            raise FinalOperationsManifestError("human approval reproducibility_manifest_hash mismatch.")
    if record is not None:
        if approval is None:
            raise FinalOperationsManifestError("submission record requires supplied human approval.")
        if record.human_approval_hash != approval.approval_hash:
            raise FinalOperationsManifestError("submission record human_approval_hash mismatch.")
        if record.reproducibility_manifest_hash != repro.manifest_hash:
            raise FinalOperationsManifestError("submission record reproducibility_manifest_hash mismatch.")


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FinalOperationsManifestError(f"{field_name} must be non-empty.")
    return value.strip()


__all__ = [
    "FinalOperationsManifest",
    "FinalOperationsManifestError",
    "build_final_operations_manifest",
    "compute_final_operations_id",
    "compute_final_operations_manifest_hash",
    "validate_final_operations_manifest",
]
