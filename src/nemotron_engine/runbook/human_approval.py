"""Explicit human approval statement for Pass 15."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash

from .blocker_gate import BlockerGateReport, validate_blocker_gate_report
from .reproducibility_manifest import ReproducibilityManifest, validate_reproducibility_manifest
from .runbook_config import RunbookConfig, reject_forbidden_runbook_claims, validate_runbook_config


class HumanApprovalError(ValueError):
    """Raised when human approval is incomplete or dishonest."""


REQUIRED_APPROVAL_PHRASE = "I understand this does not mean Kaggle submission success."


@dataclass(frozen=True)
class HumanApprovalStatement:
    approver: str
    approval_text: str
    acknowledged_no_automation_submission: bool
    acknowledged_dry_run_not_score: bool
    acknowledged_manual_external_submission_only: bool
    runbook_config_hash: str
    blocker_gate_hash: str
    reproducibility_manifest_hash: str
    approved: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    approval_hash: str = ""

    def __post_init__(self) -> None:
        if self.approved:
            _require_non_empty(self.approver, "approver")
        approval_text = _require_non_empty(self.approval_text, "approval_text")
        for name in (
            "acknowledged_no_automation_submission",
            "acknowledged_dry_run_not_score",
            "acknowledged_manual_external_submission_only",
            "approved",
        ):
            if not isinstance(getattr(self, name), bool):
                raise HumanApprovalError(f"{name} must be boolean.")
        for name in ("runbook_config_hash", "blocker_gate_hash", "reproducibility_manifest_hash"):
            _require_non_empty(getattr(self, name), name)
        acknowledgements = (
            self.acknowledged_no_automation_submission,
            self.acknowledged_dry_run_not_score,
            self.acknowledged_manual_external_submission_only,
        )
        if self.approved and not all(acknowledgements):
            raise HumanApprovalError("approved=True requires all acknowledgement booleans.")
        if self.approved and REQUIRED_APPROVAL_PHRASE not in approval_text:
            raise HumanApprovalError("approval_text is missing the required acknowledgement phrase.")
        metadata = dict(self.metadata)
        reject_forbidden_runbook_claims(metadata, error_cls=HumanApprovalError)
        object.__setattr__(self, "approval_text", approval_text)
        object.__setattr__(self, "metadata", metadata)
        expected = compute_human_approval_hash(self)
        if not self.approval_hash:
            object.__setattr__(self, "approval_hash", expected)
        elif self.approval_hash != expected:
            raise HumanApprovalError("approval_hash does not match human approval payload.")


def build_human_approval_statement(
    *,
    approver: str,
    approval_text: str,
    runbook_config: RunbookConfig | Mapping[str, Any],
    blocker_gate_report: BlockerGateReport | Mapping[str, Any],
    reproducibility_manifest: ReproducibilityManifest | Mapping[str, Any],
    acknowledged_no_automation_submission: bool,
    acknowledged_dry_run_not_score: bool,
    acknowledged_manual_external_submission_only: bool,
    metadata: Mapping[str, Any] | None = None,
) -> HumanApprovalStatement:
    cfg = validate_runbook_config(runbook_config)
    gate = validate_blocker_gate_report(blocker_gate_report)
    repro = validate_reproducibility_manifest(reproducibility_manifest)
    approved = bool(
        approver
        and REQUIRED_APPROVAL_PHRASE in approval_text
        and acknowledged_no_automation_submission
        and acknowledged_dry_run_not_score
        and acknowledged_manual_external_submission_only
        and gate.blocked is False
    )
    return HumanApprovalStatement(
        approver=approver,
        approval_text=approval_text,
        acknowledged_no_automation_submission=acknowledged_no_automation_submission,
        acknowledged_dry_run_not_score=acknowledged_dry_run_not_score,
        acknowledged_manual_external_submission_only=acknowledged_manual_external_submission_only,
        runbook_config_hash=cfg.config_hash,
        blocker_gate_hash=gate.report_hash,
        reproducibility_manifest_hash=repro.manifest_hash,
        approved=approved,
        metadata=dict(metadata or {}),
    )


def validate_human_approval_statement(statement: HumanApprovalStatement | Mapping[str, Any]) -> HumanApprovalStatement:
    normalized = statement if isinstance(statement, HumanApprovalStatement) else HumanApprovalStatement(**dict(statement))
    if normalized.approval_hash != compute_human_approval_hash(normalized):
        raise HumanApprovalError("approval_hash does not match human approval payload.")
    return normalized


def compute_human_approval_hash(statement: HumanApprovalStatement | Mapping[str, Any]) -> str:
    payload = dict(statement) if isinstance(statement, Mapping) else {item.name: getattr(statement, item.name) for item in fields(HumanApprovalStatement)}
    payload.pop("approval_hash", None)
    return stable_hash(payload)


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HumanApprovalError(f"{field_name} must be non-empty.")
    return value.strip()


__all__ = [
    "HumanApprovalError",
    "HumanApprovalStatement",
    "REQUIRED_APPROVAL_PHRASE",
    "build_human_approval_statement",
    "compute_human_approval_hash",
    "validate_human_approval_statement",
]
