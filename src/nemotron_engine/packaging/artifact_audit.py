"""Offline artifact safety audits for Pass 9 packages."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import fnmatch
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from nemotron_engine.core.schemas import stable_hash

from .submission_manifest import FileEntry, SubmissionManifestError, compute_file_sha256, normalize_package_path


class ArtifactAuditError(ValueError):
    """Raised when artifact audit contracts are violated."""


FORBIDDEN_PATTERNS = (
    "*.bin",
    "*.safetensors",
    "pytorch_model*",
    "model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "rng_state*.pth",
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
    ".env",
    "kaggle.json",
    "credentials*",
    "token*",
    "api_key*",
)

DATASET_PATTERNS = ("sft*.jsonl", "dpo*.jsonl", "train*.jsonl")


@dataclass(frozen=True)
class ArtifactAuditConfig:
    root_path: str | Path | None = None
    file_entries: tuple[FileEntry, ...] = ()
    allow_symlinks: bool = False
    allow_dataset_exports: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        if self.root_path is not None:
            object.__setattr__(self, "root_path", Path(self.root_path))
        if not isinstance(self.allow_symlinks, bool) or not isinstance(self.allow_dataset_exports, bool):
            raise ArtifactAuditError("allow flags must be booleans.")
        object.__setattr__(self, "file_entries", tuple(self.file_entries))
        object.__setattr__(self, "metadata", dict(self.metadata))
        expected = _payload_hash(self, "config_hash")
        if not self.config_hash:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise ArtifactAuditError("config_hash does not match artifact audit config payload.")


@dataclass(frozen=True)
class ArtifactAuditReport:
    passed: bool
    checked_count: int
    forbidden_paths: tuple[str, ...] = ()
    hash_mismatches: tuple[str, ...] = ()
    missing_hashes: tuple[str, ...] = ()
    duplicate_paths: tuple[str, ...] = ()
    symlink_paths: tuple[str, ...] = ()
    path_errors: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    report_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.checked_count, int) or isinstance(self.checked_count, bool) or self.checked_count < 0:
            raise ArtifactAuditError("checked_count must be a non-negative integer.")
        for name in (
            "forbidden_paths",
            "hash_mismatches",
            "missing_hashes",
            "duplicate_paths",
            "symlink_paths",
            "path_errors",
            "errors",
            "warnings",
        ):
            object.__setattr__(self, name, tuple(str(item) for item in getattr(self, name)))
        all_errors = (
            self.forbidden_paths
            + self.hash_mismatches
            + self.missing_hashes
            + self.duplicate_paths
            + self.symlink_paths
            + self.path_errors
            + self.errors
        )
        if self.passed and all_errors:
            raise ArtifactAuditError("passed=True cannot include audit failures.")
        expected = _payload_hash(self, "report_hash")
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected)
        elif self.report_hash != expected:
            raise ArtifactAuditError("report_hash does not match artifact audit report payload.")


def audit_artifacts(config: ArtifactAuditConfig) -> ArtifactAuditReport:
    if not isinstance(config, ArtifactAuditConfig):
        raise ArtifactAuditError("config must be an ArtifactAuditConfig.")
    entries = _entries_from_config(config)
    path_errors: list[str] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        try:
            path = normalize_package_path(entry.path)
        except SubmissionManifestError:
            path_errors.append(str(entry.path))
            continue
        if path in seen:
            duplicates.append(path)
        seen.add(path)
    forbidden = detect_forbidden_artifacts(entries, allow_dataset_exports=config.allow_dataset_exports)
    if config.root_path is not None:
        forbidden = tuple(sorted(set(forbidden + _detect_forbidden_directories(config.root_path))))
    symlinks = _detect_symlinks(config.root_path, entries) if not config.allow_symlinks else ()
    missing_hashes, mismatches = validate_artifact_hashes(config.root_path, entries) if config.root_path is not None else (_missing_hashes(entries), ())
    errors = tuple()
    passed = not (forbidden or symlinks or missing_hashes or mismatches or duplicates or path_errors or errors)
    return ArtifactAuditReport(
        passed=passed,
        checked_count=len(entries),
        forbidden_paths=tuple(sorted(forbidden)),
        hash_mismatches=tuple(sorted(mismatches)),
        missing_hashes=tuple(sorted(missing_hashes)),
        duplicate_paths=tuple(sorted(set(duplicates))),
        symlink_paths=tuple(sorted(symlinks)),
        path_errors=tuple(sorted(path_errors)),
        errors=errors,
        warnings=(),
    )


def detect_forbidden_artifacts(
    entries: Iterable[FileEntry | str],
    *,
    allow_dataset_exports: bool = False,
) -> tuple[str, ...]:
    forbidden: list[str] = []
    for item in entries:
        raw_path = item.path if isinstance(item, FileEntry) else str(item)
        path = normalize_package_path(raw_path)
        parts = tuple(part.lower() for part in path.split("/"))
        basename = parts[-1]
        patterns = FORBIDDEN_PATTERNS if allow_dataset_exports else FORBIDDEN_PATTERNS + DATASET_PATTERNS
        if any(fnmatch.fnmatch(basename, pattern) for pattern in patterns):
            forbidden.append(path)
            continue
        if any(part in {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"} for part in parts):
            forbidden.append(path)
    return tuple(sorted(set(forbidden)))


def validate_artifact_hashes(
    root_path: str | Path | None,
    entries: Iterable[FileEntry],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    missing: list[str] = []
    mismatch: list[str] = []
    root = Path(root_path) if root_path is not None else None
    for entry in entries:
        path = normalize_package_path(entry.path)
        if not entry.sha256:
            missing.append(path)
            continue
        if root is None:
            continue
        target = _resolve_under_root(root, path)
        if not target.exists():
            mismatch.append(path)
            continue
        if compute_file_sha256(target) != entry.sha256:
            mismatch.append(path)
    return tuple(sorted(missing)), tuple(sorted(mismatch))


def _entries_from_config(config: ArtifactAuditConfig) -> tuple[FileEntry, ...]:
    if config.file_entries:
        return tuple(config.file_entries)
    if config.root_path is None:
        return ()
    root = Path(config.root_path)
    entries: list[FileEntry] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() or item.is_symlink()):
        rel = path.relative_to(root).as_posix()
        sha = "0" * 64 if path.is_symlink() or not path.exists() else compute_file_sha256(path)
        size = 0 if path.is_symlink() or not path.exists() else path.stat().st_size
        entries.append(FileEntry(path=rel, size_bytes=size, sha256=sha, required=True, role="artifact", metadata={}))
    return tuple(entries)


def _detect_forbidden_directories(root_path: str | Path) -> tuple[str, ...]:
    root = Path(root_path)
    forbidden_names = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
    found: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_dir() or item.is_symlink()):
        rel = normalize_package_path(path.relative_to(root).as_posix())
        parts = tuple(part.lower() for part in rel.split("/"))
        if any(part in forbidden_names for part in parts):
            found.append(rel)
    return tuple(sorted(set(found)))


def _detect_symlinks(root_path: str | Path | None, entries: Iterable[FileEntry]) -> tuple[str, ...]:
    if root_path is None:
        return ()
    root = Path(root_path)
    bad: list[str] = []
    for entry in entries:
        target = _resolve_under_root(root, entry.path)
        if target.is_symlink() or _is_reparse_point(target):
            bad.append(entry.path)
    return tuple(sorted(set(bad)))


def _missing_hashes(entries: Iterable[FileEntry]) -> tuple[str, ...]:
    return tuple(sorted(entry.path for entry in entries if not entry.sha256))


def _resolve_under_root(root: Path, rel_path: str) -> Path:
    safe = normalize_package_path(rel_path)
    base = root.resolve()
    target = (base / Path(*safe.split("/"))).resolve()
    if target != base and base not in target.parents:
        raise ArtifactAuditError("resolved path escapes audit root.")
    return target


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
    "ArtifactAuditConfig",
    "ArtifactAuditError",
    "ArtifactAuditReport",
    "audit_artifacts",
    "detect_forbidden_artifacts",
    "validate_artifact_hashes",
]
