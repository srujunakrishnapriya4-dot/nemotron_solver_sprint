"""Pass 14 final dry-run rehearsal configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import ntpath
import re
from pathlib import Path
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash


class RehearsalConfigError(ValueError):
    """Raised when rehearsal configuration is unsafe or inconsistent."""


_SAFE_PACKAGE_RE = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class RehearsalConfig:
    rehearsal_id: str
    dry_run: bool
    allow_zip_build: bool
    allow_runtime_unloaded: bool
    allow_warnings: bool
    require_smoke_evidence: bool
    require_completion_evaluation: bool
    require_private_like: bool
    require_release_candidate: bool
    output_dir: str | None
    package_name: str
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        _require_non_empty(self.rehearsal_id, "rehearsal_id")
        for name in (
            "dry_run",
            "allow_zip_build",
            "allow_runtime_unloaded",
            "allow_warnings",
            "require_smoke_evidence",
            "require_completion_evaluation",
            "require_private_like",
            "require_release_candidate",
        ):
            _require_bool(getattr(self, name), name)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise RehearsalConfigError("seed must be an integer, not bool.")
        package_name = _validate_package_name(self.package_name, allow_zip_build=self.allow_zip_build, output_dir=self.output_dir)
        output_dir = None if self.output_dir is None else str(self.output_dir).strip()
        if self.allow_zip_build and not output_dir:
            raise RehearsalConfigError("output_dir is required when allow_zip_build=True.")
        metadata = _normalize_metadata(self.metadata)
        reject_unsafe_metadata_claims(metadata, allow_adapter_evidence=False)
        object.__setattr__(self, "package_name", package_name)
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "metadata", metadata)
        expected = compute_rehearsal_config_hash(self)
        if not self.config_hash:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise RehearsalConfigError("config_hash does not match rehearsal config payload.")


def validate_rehearsal_config(config: RehearsalConfig | Mapping[str, Any]) -> RehearsalConfig:
    normalized = config if isinstance(config, RehearsalConfig) else RehearsalConfig(**dict(config))
    if normalized.config_hash != compute_rehearsal_config_hash(normalized):
        raise RehearsalConfigError("config_hash does not match rehearsal config payload.")
    return normalized


def compute_rehearsal_config_hash(config: RehearsalConfig | Mapping[str, Any]) -> str:
    payload = dict(config) if isinstance(config, Mapping) else {item.name: getattr(config, item.name) for item in fields(RehearsalConfig)}
    payload.pop("config_hash", None)
    return stable_hash(payload)


def reject_unsafe_metadata_claims(metadata: Mapping[str, Any], *, allow_adapter_evidence: bool = False) -> None:
    for key, value in _flatten(metadata):
        raw = f"{key} {value}"
        text = _normalize_text(raw)
        truthy = value is True or (isinstance(value, str) and value.strip().lower() in {"true", "yes", "ready", "success", "submitted", "guaranteed"})
        if "kaggle" in text and any(word in text for word in ("success", "ready", "readiness", "submitted", "submit")):
            raise RehearsalConfigError("metadata cannot claim Kaggle success/readiness/submission.")
        if "leaderboard" in text and any(word in text for word in ("success", "ready", "readiness", "public", "private", "score")):
            raise RehearsalConfigError("metadata cannot claim leaderboard readiness or score status.")
        if "submission" in text and any(word in text for word in ("success", "ready", "submitted", "complete")):
            raise RehearsalConfigError("metadata cannot claim submission success/readiness.")
        if "submit" in text and any(word in text for word in ("success", "submitted", "kaggle")):
            raise RehearsalConfigError("metadata cannot claim submission success.")
        if "95+" in raw.lower() or "95 plus" in text:
            raise RehearsalConfigError("metadata cannot claim 95+ score.")
        if "score" in text and any(word in text for word in ("guarantee", "guaranteed", "guarantees")):
            raise RehearsalConfigError("metadata cannot claim score guarantees.")
        adapter_claim = "adapter" in text and any(word in text for word in ("trained", "available", "ready", "success"))
        if (truthy or adapter_claim) and adapter_claim and not allow_adapter_evidence:
            raise RehearsalConfigError("metadata cannot claim trained adapter availability without evidence.")


def _validate_report_hash(report: object, hash_field: str, error_cls: type[ValueError] = RehearsalConfigError) -> None:
    try:
        observed = getattr(report, hash_field)
        expected = stable_hash({item.name: getattr(report, item.name) for item in fields(report) if item.name != hash_field})
    except Exception as exc:
        raise error_cls("could not validate input report hash.") from exc
    if observed != expected:
        raise error_cls("input report hash mismatch.")


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


def _normalize_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise RehearsalConfigError("metadata must be a mapping.")
    return dict(metadata)


def _validate_package_name(value: str, *, allow_zip_build: bool, output_dir: str | None) -> str:
    name = _require_non_empty(value, "package_name").strip()
    raw = name.replace("\\", "/")
    drive, _ = ntpath.splitdrive(raw)
    if drive or raw.startswith("/") or "/" in raw or any(part in {"", ".", ".."} for part in raw.split("/")):
        raise RehearsalConfigError("package_name must be a safe simple filename.")
    if not _SAFE_PACKAGE_RE.fullmatch(name):
        raise RehearsalConfigError("package_name contains unsafe characters.")
    if name.lower() == "submission.zip" and not (allow_zip_build and output_dir):
        raise RehearsalConfigError("submission.zip package_name is only allowed under explicit safe zip output.")
    return name


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise RehearsalConfigError(f"{field_name} must be boolean.")
    return value


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RehearsalConfigError(f"{field_name} must be non-empty.")
    return value


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


def _is_under_directory(path: str | Path, parent: str | Path) -> bool:
    target = Path(path).resolve()
    root = Path(parent).resolve()
    return target == root or root in target.parents


__all__ = [
    "RehearsalConfig",
    "RehearsalConfigError",
    "compute_rehearsal_config_hash",
    "reject_unsafe_metadata_claims",
    "validate_rehearsal_config",
]
