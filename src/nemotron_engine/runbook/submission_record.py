"""Human-supplied external submission recordkeeping for Pass 15."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import re
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash

from .human_approval import HumanApprovalStatement, validate_human_approval_statement
from .reproducibility_manifest import ReproducibilityManifest, validate_reproducibility_manifest
from .runbook_config import reject_forbidden_runbook_claims


class SubmissionRecordError(ValueError):
    """Raised when submission recordkeeping is dishonest or inconsistent."""


_FORBIDDEN_RECORD_PHRASES = (
    "automatic submission",
    "auto submitted",
    "submitted by automation",
    "kaggle success",
    "leaderboard ready",
    "leaderboard success",
    "95+",
    "guaranteed",
    "score guaranteed",
    "private score guaranteed",
    "public score guaranteed",
    "winning score",
    "verified score",
)


@dataclass(frozen=True)
class SubmissionRecord:
    record_id: str
    manual_submission_performed: bool
    submitted_by: str | None
    external_platform: str | None
    artifact_hash: str | None
    submission_reference: str | None
    submitted_at_text: str | None
    public_score_text: str | None
    private_score_text: str | None
    notes: str | None
    human_approval_hash: str
    reproducibility_manifest_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    record_hash: str = ""

    def __post_init__(self) -> None:
        _require_non_empty(self.record_id, "record_id")
        if not isinstance(self.manual_submission_performed, bool):
            raise SubmissionRecordError("manual_submission_performed must be boolean.")
        for name in ("human_approval_hash", "reproducibility_manifest_hash"):
            _require_non_empty(getattr(self, name), name)
        for name in (
            "submitted_by",
            "external_platform",
            "artifact_hash",
            "submission_reference",
            "submitted_at_text",
            "public_score_text",
            "private_score_text",
            "notes",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_non_empty(value, name)
        metadata = dict(self.metadata)
        reject_forbidden_runbook_claims(metadata, error_cls=SubmissionRecordError)
        if "automatic" in str(metadata).lower() and "submission" in str(metadata).lower():
            raise SubmissionRecordError("submission record cannot claim automatic submission.")
        _reject_forbidden_record_text(
            notes=self.notes,
            external_platform=self.external_platform,
            public_score_text=self.public_score_text,
            private_score_text=self.private_score_text,
            submission_reference=self.submission_reference,
            submitted_at_text=self.submitted_at_text,
        )
        if not self.manual_submission_performed:
            forbidden = (
                self.submitted_by,
                self.external_platform,
                self.submission_reference,
                self.submitted_at_text,
                self.public_score_text,
                self.private_score_text,
            )
            if any(value is not None for value in forbidden):
                raise SubmissionRecordError("no-submission record cannot include submission details or scores.")
        else:
            if not self.submitted_by or not self.submission_reference or not self.artifact_hash:
                raise SubmissionRecordError("manual submission record requires submitted_by, submission_reference, and artifact_hash.")
        if (self.public_score_text is not None or self.private_score_text is not None) and metadata.get("human_recorded") is not True:
            raise SubmissionRecordError("score text requires metadata['human_recorded'] is True.")
        expected_id = compute_submission_record_id(self)
        if self.record_id != expected_id:
            raise SubmissionRecordError("record_id does not match submission record identity payload.")
        object.__setattr__(self, "metadata", metadata)
        expected = compute_submission_record_hash(self)
        if not self.record_hash:
            object.__setattr__(self, "record_hash", expected)
        elif self.record_hash != expected:
            raise SubmissionRecordError("record_hash does not match submission record payload.")


def build_manual_submission_record(
    *,
    manual_submission_performed: bool,
    human_approval: HumanApprovalStatement | Mapping[str, Any],
    reproducibility_manifest: ReproducibilityManifest | Mapping[str, Any],
    submitted_by: str | None = None,
    external_platform: str | None = None,
    artifact_hash: str | None = None,
    submission_reference: str | None = None,
    submitted_at_text: str | None = None,
    public_score_text: str | None = None,
    private_score_text: str | None = None,
    notes: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SubmissionRecord:
    approval = validate_human_approval_statement(human_approval)
    repro = validate_reproducibility_manifest(reproducibility_manifest)
    payload = {
        "manual_submission_performed": manual_submission_performed,
        "human_approval_hash": approval.approval_hash,
        "reproducibility_manifest_hash": repro.manifest_hash,
        "submission_reference": submission_reference,
        "artifact_hash": artifact_hash,
    }
    return SubmissionRecord(
        record_id=stable_hash(payload),
        manual_submission_performed=manual_submission_performed,
        submitted_by=submitted_by,
        external_platform=external_platform,
        artifact_hash=artifact_hash,
        submission_reference=submission_reference,
        submitted_at_text=submitted_at_text,
        public_score_text=public_score_text,
        private_score_text=private_score_text,
        notes=notes,
        human_approval_hash=approval.approval_hash,
        reproducibility_manifest_hash=repro.manifest_hash,
        metadata=dict(metadata or {}),
    )


def validate_submission_record(record: SubmissionRecord | Mapping[str, Any]) -> SubmissionRecord:
    normalized = record if isinstance(record, SubmissionRecord) else SubmissionRecord(**dict(record))
    if normalized.record_hash != compute_submission_record_hash(normalized):
        raise SubmissionRecordError("record_hash does not match submission record payload.")
    return normalized


def compute_submission_record_hash(record: SubmissionRecord | Mapping[str, Any]) -> str:
    payload = dict(record) if isinstance(record, Mapping) else {item.name: getattr(record, item.name) for item in fields(SubmissionRecord)}
    payload.pop("record_hash", None)
    return stable_hash(payload)


def compute_submission_record_id(record: SubmissionRecord | Mapping[str, Any]) -> str:
    payload = dict(record) if isinstance(record, Mapping) else {item.name: getattr(record, item.name) for item in fields(SubmissionRecord)}
    return stable_hash(
        {
            "manual_submission_performed": payload.get("manual_submission_performed"),
            "human_approval_hash": payload.get("human_approval_hash"),
            "reproducibility_manifest_hash": payload.get("reproducibility_manifest_hash"),
            "submission_reference": payload.get("submission_reference"),
            "artifact_hash": payload.get("artifact_hash"),
        }
    )


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SubmissionRecordError(f"{field_name} must be non-empty.")
    return value.strip()


def _reject_forbidden_record_text(**fields_to_scan: str | None) -> None:
    for field_name, value in fields_to_scan.items():
        if value is None:
            continue
        text = _normalize_text(value)
        raw = value.lower()
        for phrase in _FORBIDDEN_RECORD_PHRASES:
            if phrase == "95+":
                if "95+" in raw:
                    raise SubmissionRecordError(f"{field_name} cannot include fake score or submission claim.")
                continue
            if phrase in text:
                raise SubmissionRecordError(f"{field_name} cannot include fake score or submission claim.")


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+]+", " ", value.lower())).strip()


__all__ = [
    "SubmissionRecord",
    "SubmissionRecordError",
    "build_manual_submission_record",
    "compute_submission_record_hash",
    "compute_submission_record_id",
    "validate_submission_record",
]
