"""Static command manifest contracts for Pass 15 runbooks."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import re
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash

from .runbook_config import RunbookConfig, reject_forbidden_runbook_claims, validate_runbook_config


class CommandManifestError(ValueError):
    """Raised when runbook command documentation is unsafe."""


COMMAND_PHASES = (
    "verify_checkpoint",
    "run_tests",
    "run_release_audit",
    "run_rehearsal_audit",
    "package_dry_run",
    "external_submission_manual",
    "post_submission_record",
    "recovery",
)
_DESTRUCTIVE_PATTERNS = ("rm -rf", "git reset --hard", "git clean", "del /s", "remove-item -recurse -force", "format")
_SUBMIT_PATTERNS = ("kaggle competitions submit", "kaggle.api", "submit(")


@dataclass(frozen=True)
class RunbookCommand:
    command_id: str
    description: str
    command_text: str
    phase: str
    manual_only: bool
    destructive: bool
    expected_output: str | None
    blocker_if_fails: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    command_hash: str = ""

    def __post_init__(self) -> None:
        command_id = _require_non_empty(self.command_id, "command_id")
        description = _require_non_empty(self.description, "description")
        command_text = _require_non_empty(self.command_text, "command_text")
        if self.phase not in COMMAND_PHASES:
            raise CommandManifestError("unsupported command phase.")
        for name in ("manual_only", "destructive", "blocker_if_fails"):
            if not isinstance(getattr(self, name), bool):
                raise CommandManifestError(f"{name} must be boolean.")
        if self.destructive:
            raise CommandManifestError("destructive commands are forbidden in Pass 15 runbooks.")
        if self.expected_output is not None:
            _require_non_empty(self.expected_output, "expected_output")
        metadata = dict(self.metadata)
        reject_forbidden_runbook_claims(metadata, error_cls=CommandManifestError)
        _validate_command_text(command_text, self.phase, self.manual_only, metadata)
        object.__setattr__(self, "command_id", command_id)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "command_text", command_text)
        object.__setattr__(self, "metadata", metadata)
        expected = compute_runbook_command_hash(self)
        if not self.command_hash:
            object.__setattr__(self, "command_hash", expected)
        elif self.command_hash != expected:
            raise CommandManifestError("command_hash does not match command payload.")


@dataclass(frozen=True)
class CommandManifest:
    commands: tuple[RunbookCommand, ...]
    runbook_config_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    manifest_hash: str = ""

    def __post_init__(self) -> None:
        commands = tuple(_command(item) for item in self.commands)
        if not commands:
            raise CommandManifestError("command manifest requires commands.")
        ids = tuple(item.command_id for item in commands)
        if len(ids) != len(set(ids)):
            raise CommandManifestError("command ids must be unique.")
        _require_non_empty(self.runbook_config_hash, "runbook_config_hash")
        for command in commands:
            if command.command_hash != compute_runbook_command_hash(command):
                raise CommandManifestError("command_hash does not match command payload.")
        metadata = dict(self.metadata)
        reject_forbidden_runbook_claims(metadata, error_cls=CommandManifestError)
        object.__setattr__(self, "commands", tuple(sorted(commands, key=lambda item: item.command_id)))
        object.__setattr__(self, "metadata", metadata)
        expected = compute_command_manifest_hash(self)
        if not self.manifest_hash:
            object.__setattr__(self, "manifest_hash", expected)
        elif self.manifest_hash != expected:
            raise CommandManifestError("manifest_hash does not match command manifest payload.")


def build_default_command_manifest(config: RunbookConfig | Mapping[str, Any], metadata: Mapping[str, Any] | None = None) -> CommandManifest:
    cfg = validate_runbook_config(config)
    commands = [
        RunbookCommand(
            command_id="verify-pass14-evidence",
            description="Verify Pass 14 final rehearsal evidence and recorded hashes.",
            command_text="Human reads the Pass 14 final rehearsal manifest and confirms the manifest hash matches the runbook.",
            phase="verify_checkpoint",
            manual_only=True,
            destructive=False,
            expected_output="Recorded final rehearsal manifest hash matches.",
            blocker_if_fails=True,
        ),
        RunbookCommand(
            command_id="run-scoped-tests",
            description="Run the locked scoped pytest suite.",
            command_text="python -m pytest tests -q -p no:cacheprovider",
            phase="run_tests",
            manual_only=True,
            destructive=False,
            expected_output="891 passed, 2 skipped",
            blocker_if_fails=True,
        ),
        RunbookCommand(
            command_id="run-release-audit",
            description="Confirm Pass 10 release candidate and artifact safety evidence.",
            command_text="Human reviews the existing release audit reports and confirms no forbidden artifacts are present.",
            phase="run_release_audit",
            manual_only=True,
            destructive=False,
            expected_output="Release audit remains locked and clean.",
            blocker_if_fails=True,
        ),
        RunbookCommand(
            command_id="run-rehearsal-audit",
            description="Confirm final rehearsal remains a dry-run record only.",
            command_text="Human verifies package, runtime, and submission rehearsal reports are dry-run records and no external API was used.",
            phase="run_rehearsal_audit",
            manual_only=True,
            destructive=False,
            expected_output="Dry-run evidence is recorded without external submission.",
            blocker_if_fails=True,
        ),
        RunbookCommand(
            command_id="package-dry-run-review",
            description="Review package dry-run evidence without writing a package.",
            command_text="Human confirms package dry-run report records required artifact hashes without creating a new archive.",
            phase="package_dry_run",
            manual_only=True,
            destructive=False,
            expected_output="Dry-run package evidence is hash-complete.",
            blocker_if_fails=True,
        ),
        RunbookCommand(
            command_id="post-submission-record-template",
            description="Record human-supplied external submission reference after manual action, if any.",
            command_text="Human enters submission reference, artifact hash, and optional score text into a recordkeeping object.",
            phase="post_submission_record",
            manual_only=True,
            destructive=False,
            expected_output="Recordkeeping payload is complete or explicitly no-submission.",
            blocker_if_fails=False,
        ),
    ]
    if cfg.allow_external_submission_instructions:
        commands.append(
            RunbookCommand(
                command_id="external-submission-ui-only",
                description="Manual external submission instruction through UI only.",
                command_text="Manual only: use the Kaggle web UI to upload the already-audited artifact, then record the human-visible reference.",
                phase="external_submission_manual",
                manual_only=True,
                destructive=False,
                expected_output="Human-supplied external reference is available for recordkeeping.",
                blocker_if_fails=False,
                metadata={"manual_external_instruction_allowed": True},
            )
        )
    return CommandManifest(commands=tuple(commands), runbook_config_hash=cfg.config_hash, metadata=dict(metadata or {}))


def validate_command_manifest(manifest: CommandManifest | Mapping[str, Any], config: RunbookConfig | Mapping[str, Any] | None = None) -> CommandManifest:
    cfg = validate_runbook_config(config) if config is not None else None
    normalized = manifest if isinstance(manifest, CommandManifest) else CommandManifest(**dict(manifest))
    if normalized.manifest_hash != compute_command_manifest_hash(normalized):
        raise CommandManifestError("manifest_hash does not match command manifest payload.")
    if cfg is not None and normalized.runbook_config_hash != cfg.config_hash:
        raise CommandManifestError("command manifest runbook_config_hash mismatch.")
    for command in normalized.commands:
        if command.command_hash != compute_runbook_command_hash(command):
            raise CommandManifestError("command_hash does not match command payload.")
        if _mentions_kaggle(command.command_text):
            allowed = (
                cfg is not None
                and cfg.allow_external_submission_instructions
                and command.phase == "external_submission_manual"
                and command.manual_only
            )
            if not allowed:
                raise CommandManifestError("manual Kaggle instructions require explicit config permission.")
    return normalized


def compute_command_manifest_hash(manifest: CommandManifest | Mapping[str, Any]) -> str:
    payload = dict(manifest) if isinstance(manifest, Mapping) else {item.name: getattr(manifest, item.name) for item in fields(CommandManifest)}
    payload.pop("manifest_hash", None)
    return stable_hash(payload)


def compute_runbook_command_hash(command: RunbookCommand | Mapping[str, Any]) -> str:
    payload = dict(command) if isinstance(command, Mapping) else {item.name: getattr(command, item.name) for item in fields(RunbookCommand)}
    payload.pop("command_hash", None)
    return stable_hash(payload)


def _command(value: RunbookCommand | Mapping[str, Any]) -> RunbookCommand:
    return value if isinstance(value, RunbookCommand) else RunbookCommand(**dict(value))


def _validate_command_text(text: str, phase: str, manual_only: bool, metadata: Mapping[str, Any]) -> None:
    lower = text.lower()
    normalized = re.sub(r"\s+", " ", lower)
    if any(pattern in normalized for pattern in _DESTRUCTIVE_PATTERNS):
        raise CommandManifestError("command_text contains destructive command text.")
    if any(pattern in normalized for pattern in _SUBMIT_PATTERNS):
        raise CommandManifestError("command_text contains automatic submission text.")
    reject_forbidden_runbook_claims({"command_text": text}, error_cls=CommandManifestError)
    if _mentions_kaggle(text):
        allowed = phase == "external_submission_manual" and manual_only and metadata.get("manual_external_instruction_allowed") is True
        if not allowed:
            raise CommandManifestError("manual Kaggle instructions require explicit permission.")


def _mentions_kaggle(text: str) -> bool:
    return "kaggle" in text.lower()


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CommandManifestError(f"{field_name} must be non-empty.")
    return value.strip()


__all__ = [
    "COMMAND_PHASES",
    "CommandManifest",
    "CommandManifestError",
    "RunbookCommand",
    "build_default_command_manifest",
    "compute_command_manifest_hash",
    "compute_runbook_command_hash",
    "validate_command_manifest",
]
