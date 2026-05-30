"""Deterministic reproducibility manifest for Pass 15 operations."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.packaging.submission_manifest import compute_file_sha256

from .blocker_gate import BlockerGateReport, validate_blocker_gate_report
from .command_manifest import CommandManifest, validate_command_manifest
from .operator_checklist import OperatorChecklist, validate_operator_checklist
from .runbook_config import RunbookConfig, reject_forbidden_runbook_claims, validate_runbook_config


class ReproducibilityManifestError(ValueError):
    """Raised when reproducibility evidence is incomplete or dishonest."""


PASS_LOCK_KEYS = tuple(f"pass_{index}" for index in range(1, 15))


@dataclass(frozen=True)
class ReproducibilityManifest:
    runbook_config_hash: str
    checkpoint_zip_path: str | None
    checkpoint_zip_hash: str | None
    scoped_suite_summary: str
    pass_lock_summaries: dict[str, str]
    final_rehearsal_manifest_hash: str | None
    release_candidate_hash: str | None
    environment_summary: dict[str, str]
    command_manifest_hash: str
    checklist_hash: str
    blocker_gate_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    manifest_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("runbook_config_hash", "scoped_suite_summary", "command_manifest_hash", "checklist_hash", "blocker_gate_hash"):
            _require_non_empty(getattr(self, name), name)
        for name in ("checkpoint_zip_path", "checkpoint_zip_hash", "final_rehearsal_manifest_hash", "release_candidate_hash"):
            value = getattr(self, name)
            if value is not None:
                _require_non_empty(value, name)
        pass_locks = {str(key): str(value) for key, value in sorted(self.pass_lock_summaries.items())}
        missing = tuple(key for key in PASS_LOCK_KEYS if not pass_locks.get(key))
        if missing:
            raise ReproducibilityManifestError(f"missing pass lock summaries: {missing}")
        env = {str(key): str(value) for key, value in sorted(self.environment_summary.items())}
        metadata = dict(self.metadata)
        reject_forbidden_runbook_claims(metadata, error_cls=ReproducibilityManifestError)
        reject_forbidden_runbook_claims(env, error_cls=ReproducibilityManifestError)
        if self.checkpoint_zip_path is not None and self.checkpoint_zip_hash is None:
            raise ReproducibilityManifestError("checkpoint_zip_hash is required when checkpoint_zip_path is supplied.")
        object.__setattr__(self, "pass_lock_summaries", pass_locks)
        object.__setattr__(self, "environment_summary", env)
        object.__setattr__(self, "metadata", metadata)
        expected = compute_reproducibility_manifest_hash(self)
        if not self.manifest_hash:
            object.__setattr__(self, "manifest_hash", expected)
        elif self.manifest_hash != expected:
            raise ReproducibilityManifestError("manifest_hash does not match reproducibility manifest payload.")


def build_reproducibility_manifest(
    runbook_config: RunbookConfig | Mapping[str, Any],
    command_manifest: CommandManifest | Mapping[str, Any],
    operator_checklist: OperatorChecklist | Mapping[str, Any],
    blocker_gate_report: BlockerGateReport | Mapping[str, Any],
    *,
    pass_lock_summaries: Mapping[str, str],
    environment_summary: Mapping[str, str] | None = None,
    checkpoint_zip_hash: str | None = None,
    release_candidate_hash: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ReproducibilityManifest:
    cfg = validate_runbook_config(runbook_config)
    commands = validate_command_manifest(command_manifest, cfg)
    checklist = validate_operator_checklist(operator_checklist, cfg)
    gate = validate_blocker_gate_report(blocker_gate_report)
    if gate.runbook_config_hash != cfg.config_hash:
        raise ReproducibilityManifestError("blocker gate runbook_config_hash mismatch.")
    if gate.command_manifest_hash != commands.manifest_hash:
        raise ReproducibilityManifestError("blocker gate command_manifest_hash mismatch.")
    if gate.checklist_hash != checklist.checklist_hash:
        raise ReproducibilityManifestError("blocker gate checklist_hash mismatch.")
    artifact_hash = checkpoint_zip_hash
    if cfg.checkpoint_zip_path is not None and artifact_hash is None:
        path = Path(cfg.checkpoint_zip_path)
        if path.exists() and path.is_file():
            artifact_hash = compute_file_sha256(path)
    return ReproducibilityManifest(
        runbook_config_hash=cfg.config_hash,
        checkpoint_zip_path=cfg.checkpoint_zip_path,
        checkpoint_zip_hash=artifact_hash,
        scoped_suite_summary=cfg.accepted_suite_summary,
        pass_lock_summaries=dict(pass_lock_summaries),
        final_rehearsal_manifest_hash=cfg.final_rehearsal_manifest_hash,
        release_candidate_hash=release_candidate_hash or gate.release_candidate_hash,
        environment_summary=dict(environment_summary or {}),
        command_manifest_hash=commands.manifest_hash,
        checklist_hash=checklist.checklist_hash,
        blocker_gate_hash=gate.report_hash,
        metadata=dict(metadata or {}),
    )


def validate_reproducibility_manifest(manifest: ReproducibilityManifest | Mapping[str, Any]) -> ReproducibilityManifest:
    normalized = manifest if isinstance(manifest, ReproducibilityManifest) else ReproducibilityManifest(**dict(manifest))
    if normalized.manifest_hash != compute_reproducibility_manifest_hash(normalized):
        raise ReproducibilityManifestError("manifest_hash does not match reproducibility manifest payload.")
    return normalized


def compute_reproducibility_manifest_hash(manifest: ReproducibilityManifest | Mapping[str, Any]) -> str:
    payload = dict(manifest) if isinstance(manifest, Mapping) else {item.name: getattr(manifest, item.name) for item in fields(ReproducibilityManifest)}
    payload.pop("manifest_hash", None)
    return stable_hash(payload)


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReproducibilityManifestError(f"{field_name} must be non-empty.")
    return value.strip()


__all__ = [
    "PASS_LOCK_KEYS",
    "ReproducibilityManifest",
    "ReproducibilityManifestError",
    "build_reproducibility_manifest",
    "compute_reproducibility_manifest_hash",
    "validate_reproducibility_manifest",
]
