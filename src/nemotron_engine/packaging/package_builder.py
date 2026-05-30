"""Deterministic optional package builder for Pass 9."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import json
import os
from pathlib import Path
from typing import Any, Mapping
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from nemotron_engine.core.schemas import stable_hash, stable_json_dumps
from nemotron_engine.runtime.serving_config import ServingConfigError, load_serving_config, validate_serving_config
from nemotron_engine.training.lora_config import LoRAConfigError, validate_adapter_config_json

from .artifact_audit import detect_forbidden_artifacts
from .submission_manifest import (
    SubmissionManifest,
    SubmissionManifestError,
    validate_submission_manifest,
    normalize_package_path,
)


class PackageBuilderError(ValueError):
    """Raised when a package build would be unsafe or nondeterministic."""


FORBIDDEN_ROLES = frozenset({"checkpoint", "model_checkpoint", "training_state", "secret", "cache", "leaderboard", "dataset_export"})


@dataclass(frozen=True)
class PackageBuildConfig:
    root_path: str | Path
    manifest: SubmissionManifest
    output_path: str | Path | None = None
    allowed_output_parent: str | Path | None = None
    build_zip: bool = False
    dry_run: bool = True
    allow_symlinks: bool = False
    manifest_filename: str = "submission_manifest.json"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "root_path", Path(self.root_path))
        if self.output_path is not None:
            object.__setattr__(self, "output_path", Path(self.output_path))
        if self.allowed_output_parent is not None:
            object.__setattr__(self, "allowed_output_parent", Path(self.allowed_output_parent))
        if not isinstance(self.manifest, SubmissionManifest):
            raise PackageBuilderError("manifest must be a SubmissionManifest.")
        if not all(isinstance(value, bool) for value in (self.build_zip, self.dry_run, self.allow_symlinks)):
            raise PackageBuilderError("build_zip, dry_run, and allow_symlinks must be booleans.")
        object.__setattr__(self, "manifest_filename", normalize_package_path(self.manifest_filename))
        if self.manifest_filename != "submission_manifest.json":
            raise PackageBuilderError("manifest_filename must be submission_manifest.json.")
        object.__setattr__(self, "metadata", dict(self.metadata))
        expected = _payload_hash(self, "config_hash")
        if not self.config_hash:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise PackageBuilderError("config_hash does not match package build config payload.")


@dataclass(frozen=True)
class PackageBuildReport:
    package_path: str | None
    manifest_hash: str
    package_hash: str
    file_count: int
    built: bool
    dry_run: bool
    passed: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    report_hash: str = ""

    def __post_init__(self) -> None:
        if self.package_path is not None and not isinstance(self.package_path, str):
            raise PackageBuilderError("package_path must be a string or None.")
        if not isinstance(self.manifest_hash, str) or not self.manifest_hash.strip():
            raise PackageBuilderError("manifest_hash must be non-empty.")
        if not isinstance(self.package_hash, str) or not self.package_hash.strip():
            raise PackageBuilderError("package_hash must be non-empty.")
        if not isinstance(self.file_count, int) or isinstance(self.file_count, bool) or self.file_count < 0:
            raise PackageBuilderError("file_count must be a non-negative integer.")
        if not all(isinstance(value, bool) for value in (self.built, self.dry_run, self.passed)):
            raise PackageBuilderError("built, dry_run, and passed must be booleans.")
        if self.dry_run and self.built:
            raise PackageBuilderError("dry_run package report cannot claim built=True.")
        errors = tuple(str(item) for item in self.errors)
        warnings = tuple(str(item) for item in self.warnings)
        if self.passed and errors:
            raise PackageBuilderError("passed=True cannot include errors.")
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", warnings)
        expected = _payload_hash(self, "report_hash")
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected)
        elif self.report_hash != expected:
            raise PackageBuilderError("report_hash does not match package build report payload.")


def list_package_files(root_path: str | Path) -> tuple[str, ...]:
    root = Path(root_path)
    files: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() or item.is_symlink()):
        files.append(normalize_package_path(path.relative_to(root).as_posix()))
    return tuple(files)


def validate_package_layout(config: PackageBuildConfig) -> tuple[str, ...]:
    if not isinstance(config, PackageBuildConfig):
        raise PackageBuilderError("config must be a PackageBuildConfig.")
    errors: list[str] = []
    try:
        validate_submission_manifest(config.manifest, config.root_path)
    except SubmissionManifestError as exc:
        errors.append(str(exc))
    seen: set[str] = set()
    for entry in config.manifest.files:
        try:
            rel = normalize_package_path(entry.path)
        except SubmissionManifestError as exc:
            errors.append(str(exc))
            continue
        if rel in seen:
            errors.append(f"duplicate package path: {rel}")
        seen.add(rel)
        target = _resolve_under_root(config.root_path, rel)
        if entry.required and not target.exists():
            errors.append(f"missing required file: {rel}")
        if target.exists() and not config.allow_symlinks and (target.is_symlink() or _is_reparse_point(target)):
            errors.append(f"symlink or reparse point forbidden: {rel}")
        if entry.role in FORBIDDEN_ROLES:
            errors.append(f"forbidden artifact role: {entry.role}:{rel}")
        if rel.endswith("adapter_config.json") and target.exists() and not target.is_symlink():
            try:
                payload = json.loads(target.read_text(encoding="utf-8"))
                validate_adapter_config_json(payload)
            except (json.JSONDecodeError, OSError, LoRAConfigError) as exc:
                errors.append(f"invalid adapter_config.json: {rel}: {exc}")
        if rel.endswith("serving_config.json") and target.exists() and not target.is_symlink():
            try:
                validate_serving_config(load_serving_config(target))
            except (OSError, ServingConfigError, TypeError, ValueError) as exc:
                errors.append(f"unsafe serving_config.json: {rel}: {exc}")
        errors.extend(f"{rel}: {message}" for message in _adapter_rank_metadata_errors(entry.metadata))
    forbidden = detect_forbidden_artifacts(config.manifest.files, allow_dataset_exports=False)
    errors.extend(f"forbidden artifact: {path}" for path in forbidden)
    if config.build_zip and not config.dry_run and config.output_path is None:
        errors.append("output_path is required when build_zip=True and dry_run=False.")
    if config.output_path is not None:
        errors.extend(_validate_output_path(config))
    return tuple(sorted(set(errors)))


def build_submission_package(config: PackageBuildConfig) -> PackageBuildReport:
    if not isinstance(config, PackageBuildConfig):
        raise PackageBuilderError("config must be a PackageBuildConfig.")
    errors = validate_package_layout(config)
    built = False
    if not errors and config.build_zip and not config.dry_run:
        assert config.output_path is not None
        output_path = _resolve_output_path(config)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_deterministic_zip(output_path, config)
        built = True
    return PackageBuildReport(
        package_path=None if config.output_path is None else str(config.output_path),
        manifest_hash=config.manifest.manifest_hash,
        package_hash=config.manifest.package_hash,
        file_count=len(config.manifest.files),
        built=built,
        dry_run=config.dry_run,
        passed=not errors,
        errors=errors,
        warnings=(),
    )


def _write_deterministic_zip(path: Path, config: PackageBuildConfig) -> None:
    manifest_bytes = (stable_json_dumps(config.manifest) + "\n").encode("utf-8")
    with ZipFile(path, "w", compression=ZIP_STORED) as archive:
        _write_zip_bytes(archive, "submission_manifest.json", manifest_bytes)
        for entry in config.manifest.files:
            target = _resolve_under_root(config.root_path, entry.path)
            _write_zip_bytes(archive, entry.path, target.read_bytes())


def _write_zip_bytes(archive: ZipFile, arcname: str, payload: bytes) -> None:
    info = ZipInfo(normalize_package_path(arcname), date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_STORED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload)


def _resolve_under_root(root: Path, rel_path: str) -> Path:
    safe = normalize_package_path(rel_path)
    base = Path(root).resolve()
    target = (base / Path(*safe.split("/"))).resolve()
    if target != base and base not in target.parents:
        raise PackageBuilderError("resolved path escapes package root.")
    return target


def _resolve_output_path(config: PackageBuildConfig) -> Path:
    if config.output_path is None:
        raise PackageBuilderError("output_path is required.")
    output = Path(config.output_path)
    if not output.is_absolute():
        output = Path(config.root_path) / output
    return output.resolve()


def _validate_output_path(config: PackageBuildConfig) -> tuple[str, ...]:
    errors: list[str] = []
    assert config.output_path is not None
    raw = str(config.output_path).replace("\\", "/")
    if any(part == ".." for part in raw.split("/")):
        errors.append("output_path must not contain path traversal.")
    root = Path(config.root_path).resolve()
    allowed_parent = Path(config.allowed_output_parent).resolve() if config.allowed_output_parent is not None else root
    output = _resolve_output_path(config)
    if output != allowed_parent and allowed_parent not in output.parents:
        errors.append("output_path resolves outside the allowed output parent.")
    if output != root and root not in output.parents and config.allowed_output_parent is None:
        errors.append("output_path resolves outside package root.")
    for entry in config.manifest.files:
        try:
            if output == _resolve_under_root(config.root_path, entry.path):
                errors.append(f"output_path must not overwrite package entry: {entry.path}")
        except PackageBuilderError as exc:
            errors.append(str(exc))
    return tuple(errors)


def _adapter_rank_metadata_errors(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if not isinstance(metadata, Mapping):
        return ("metadata must be a mapping.",)

    def check_value(value: Any, label: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(f"{label} must be an integer.")
            return
        if value < 1 or value > 32:
            errors.append(f"{label} must be in [1, 32].")

    for key in ("adapter_rank", "rank", "lora_rank"):
        if key in metadata:
            check_value(metadata[key], f"metadata.{key}")
    adapter_config = metadata.get("adapter_config")
    if adapter_config is not None:
        if not isinstance(adapter_config, Mapping):
            errors.append("metadata.adapter_config must be a mapping.")
        else:
            for key in ("r", "rank", "lora_rank"):
                if key in adapter_config:
                    check_value(adapter_config[key], f"metadata.adapter_config.{key}")
            peft_config = adapter_config.get("peft_config")
            if peft_config is not None:
                if not isinstance(peft_config, Mapping):
                    errors.append("metadata.adapter_config.peft_config must be a mapping.")
                elif "r" in peft_config:
                    check_value(peft_config["r"], "metadata.adapter_config.peft_config.r")
    peft_config = metadata.get("peft_config")
    if peft_config is not None:
        if not isinstance(peft_config, Mapping):
            errors.append("metadata.peft_config must be a mapping.")
        elif "r" in peft_config:
            check_value(peft_config["r"], "metadata.peft_config.r")
    return tuple(errors)


def _is_reparse_point(path: Path) -> bool:
    if os.name != "nt":
        return False
    try:
        return bool(path.lstat().st_file_attributes & 0x400)
    except (AttributeError, OSError):
        return False


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


__all__ = [
    "FORBIDDEN_ROLES",
    "PackageBuildConfig",
    "PackageBuilderError",
    "PackageBuildReport",
    "build_submission_package",
    "list_package_files",
    "validate_package_layout",
]
