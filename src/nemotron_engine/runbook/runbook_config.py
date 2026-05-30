"""Pass 15 human-operated final runbook configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import ntpath
import re
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash


class RunbookConfigError(ValueError):
    """Raised when final runbook configuration is unsafe."""


BASELINE_SUITE_SUMMARY = "891 passed, 2 skipped"
RUNBOOK_MODES = ("dry_run_only", "external_submission_runbook", "post_submission_recordkeeping")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")


@dataclass(frozen=True)
class RunbookConfig:
    runbook_id: str
    mode: str
    require_pass14_lock: bool
    require_final_rehearsal: bool
    require_human_approval: bool
    allow_external_submission_instructions: bool
    allow_post_submission_record: bool
    accepted_suite_summary: str
    checkpoint_zip_path: str | None
    final_rehearsal_manifest_hash: str | None
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        runbook_id = _validate_safe_id(self.runbook_id, "runbook_id", RunbookConfigError)
        if self.mode not in RUNBOOK_MODES:
            raise RunbookConfigError("unsupported runbook mode.")
        for name in (
            "require_pass14_lock",
            "require_final_rehearsal",
            "require_human_approval",
            "allow_external_submission_instructions",
            "allow_post_submission_record",
        ):
            _require_bool(getattr(self, name), name, RunbookConfigError)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise RunbookConfigError("seed must be an integer, not bool.")
        summary = _require_non_empty(self.accepted_suite_summary, "accepted_suite_summary", RunbookConfigError)
        metadata = _normalize_metadata(self.metadata, RunbookConfigError)
        reject_forbidden_runbook_claims(metadata, error_cls=RunbookConfigError)
        if BASELINE_SUITE_SUMMARY not in summary and metadata.get("allow_suite_summary_override") is not True:
            raise RunbookConfigError("accepted_suite_summary must include locked Pass 14 baseline.")
        if self.require_final_rehearsal and not _non_empty(self.final_rehearsal_manifest_hash):
            raise RunbookConfigError("final_rehearsal_manifest_hash is required when final rehearsal is required.")
        if self.final_rehearsal_manifest_hash is not None:
            _require_non_empty(self.final_rehearsal_manifest_hash, "final_rehearsal_manifest_hash", RunbookConfigError)
        checkpoint = _validate_optional_path(self.checkpoint_zip_path, "checkpoint_zip_path", RunbookConfigError)
        if checkpoint is not None:
            reject_forbidden_runbook_claims({"checkpoint_zip_path": checkpoint}, error_cls=RunbookConfigError)
        if (
            self.mode == "dry_run_only"
            and self.allow_external_submission_instructions
            and metadata.get("allow_dry_run_external_submission_instructions") is not True
        ):
            raise RunbookConfigError("dry_run_only cannot include external submission instructions without explicit override.")
        object.__setattr__(self, "runbook_id", runbook_id)
        object.__setattr__(self, "accepted_suite_summary", summary)
        object.__setattr__(self, "checkpoint_zip_path", checkpoint)
        object.__setattr__(self, "metadata", metadata)
        expected = compute_runbook_config_hash(self)
        if not self.config_hash:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise RunbookConfigError("config_hash does not match runbook config payload.")


def validate_runbook_config(config: RunbookConfig | Mapping[str, Any]) -> RunbookConfig:
    normalized = config if isinstance(config, RunbookConfig) else RunbookConfig(**dict(config))
    if normalized.config_hash != compute_runbook_config_hash(normalized):
        raise RunbookConfigError("config_hash does not match runbook config payload.")
    return normalized


def compute_runbook_config_hash(config: RunbookConfig | Mapping[str, Any]) -> str:
    payload = dict(config) if isinstance(config, Mapping) else {item.name: getattr(config, item.name) for item in fields(RunbookConfig)}
    payload.pop("config_hash", None)
    return stable_hash(payload)


def reject_forbidden_runbook_claims(metadata: Mapping[str, Any], *, error_cls: type[ValueError]) -> None:
    for key, value in _flatten(metadata):
        raw = f"{key} {value}"
        text = _normalize_text(raw)
        if "already submitted" in text or "already_submitted" in text:
            raise error_cls("metadata cannot claim already submitted.")
        if "95+" in raw.lower() or "95 plus" in text:
            raise error_cls("metadata cannot claim 95+ score.")
        if _has_token(text, "score") and _has_any_token(text, ("guarantee", "guaranteed", "verified", "ready")):
            raise error_cls("metadata cannot claim score guarantees or readiness.")
        if _has_token(text, "public") and _has_token(text, "score"):
            raise error_cls("metadata cannot claim public score.")
        if _has_token(text, "private") and _has_token(text, "score"):
            raise error_cls("metadata cannot claim private score.")
        if _has_token(text, "leaderboard") and _has_any_token(text, ("ready", "readiness", "success", "score", "public", "private")):
            raise error_cls("metadata cannot claim leaderboard success/readiness.")
        if _has_token(text, "kaggle") and _has_any_token(text, ("success", "ready", "readiness", "submitted", "submission")):
            raise error_cls("metadata cannot claim Kaggle success/readiness/submission.")
        if _has_token(text, "submission") and _has_any_token(text, ("success", "ready", "submitted", "complete", "completed", "automatic")):
            raise error_cls("metadata cannot claim submission success/readiness.")
        if _has_token(text, "training") and _has_any_token(text, ("success", "complete", "completed")):
            raise error_cls("metadata cannot claim training success.")


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


def _normalize_metadata(metadata: Mapping[str, Any], error_cls: type[ValueError]) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise error_cls("metadata must be a mapping.")
    return dict(metadata)


def _validate_safe_id(value: Any, field_name: str, error_cls: type[ValueError]) -> str:
    text = _require_non_empty(value, field_name, error_cls)
    if not _SAFE_ID_RE.fullmatch(text):
        raise error_cls(f"{field_name} contains unsafe characters.")
    return text


def _validate_optional_path(value: str | None, field_name: str, error_cls: type[ValueError]) -> str | None:
    if value is None:
        return None
    text = _require_non_empty(value, field_name, error_cls)
    raw = text.replace("\\", "/")
    drive, _ = ntpath.splitdrive(raw)
    if raw.startswith("/") or drive or any(part in {"", ".", ".."} for part in raw.split("/")):
        raise error_cls(f"{field_name} must be a safe relative path string.")
    return text


def _require_bool(value: Any, field_name: str, error_cls: type[ValueError]) -> bool:
    if not isinstance(value, bool):
        raise error_cls(f"{field_name} must be boolean.")
    return value


def _require_non_empty(value: Any, field_name: str, error_cls: type[ValueError]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error_cls(f"{field_name} must be non-empty.")
    return value.strip()


def _non_empty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


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


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+]+", " ", value.lower())).strip()


def _has_token(text: str, token: str) -> bool:
    return token in text.split()


def _has_any_token(text: str, tokens: tuple[str, ...]) -> bool:
    parts = set(text.split())
    return any(token in parts for token in tokens)


__all__ = [
    "BASELINE_SUITE_SUMMARY",
    "RUNBOOK_MODES",
    "RunbookConfig",
    "RunbookConfigError",
    "compute_runbook_config_hash",
    "validate_runbook_config",
]
