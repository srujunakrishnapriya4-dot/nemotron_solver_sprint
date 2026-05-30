"""Deterministic submission manifests for Pass 9 packaging."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import ntpath
from pathlib import Path
from typing import Any, Iterable, Mapping

from nemotron_engine.core.schemas import stable_hash, stable_json_dumps


class SubmissionManifestError(ValueError):
    """Raised when a submission manifest or file entry is unsafe."""


@dataclass(frozen=True)
class FileEntry:
    path: str
    size_bytes: int
    sha256: str
    required: bool
    role: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        safe_path = normalize_package_path(self.path)
        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise SubmissionManifestError("size_bytes must be a non-negative integer.")
        if not _is_sha256(self.sha256):
            raise SubmissionManifestError("sha256 must be a lowercase 64-character hex digest.")
        if not isinstance(self.required, bool):
            raise SubmissionManifestError("required must be a boolean.")
        if not isinstance(self.role, str) or not self.role.strip():
            raise SubmissionManifestError("role must be a non-empty string.")
        if not isinstance(self.metadata, Mapping):
            raise SubmissionManifestError("metadata must be a mapping.")
        object.__setattr__(self, "path", safe_path)
        object.__setattr__(self, "role", self.role.strip())
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class SubmissionManifest:
    manifest_id: str
    package_name: str
    created_by: str
    files: tuple[FileEntry, ...]
    serving_config_hash: str
    lora_config_hash: str | None
    adapter_hash: str | None
    promotion_decision_hash: str
    training_plan_hash: str
    dataset_manifest_hash: str
    trace_manifest_hash: str
    package_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    manifest_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("manifest_id", "package_name", "created_by"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise SubmissionManifestError(f"{name} must be non-empty.")
        required_hash_names = (
            "serving_config_hash",
            "promotion_decision_hash",
            "training_plan_hash",
            "dataset_manifest_hash",
            "trace_manifest_hash",
        )
        for name in required_hash_names:
            if not _non_empty(getattr(self, name)):
                raise SubmissionManifestError(f"{name} must be non-empty.")
        files_tuple = tuple(self.files)
        if not files_tuple:
            raise SubmissionManifestError("manifest requires at least one file.")
        paths = tuple(item.path for item in files_tuple)
        if len(paths) != len(set(paths)):
            raise SubmissionManifestError("manifest file paths must be unique.")
        sorted_files = tuple(sorted(files_tuple, key=lambda item: item.path))
        object.__setattr__(self, "files", sorted_files)
        object.__setattr__(self, "metadata", dict(self.metadata))
        expected_package_hash = compute_package_hash(sorted_files)
        if self.package_hash != expected_package_hash:
            raise SubmissionManifestError("package_hash does not match manifest file entries.")
        expected_manifest_hash = compute_manifest_hash(self)
        if not self.manifest_hash:
            object.__setattr__(self, "manifest_hash", expected_manifest_hash)
        elif self.manifest_hash != expected_manifest_hash:
            raise SubmissionManifestError("manifest_hash does not match manifest payload.")


def normalize_package_path(path: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise SubmissionManifestError("package path must be a non-empty string.")
    raw = path.replace("\\", "/")
    drive, _ = ntpath.splitdrive(raw)
    if drive:
        raise SubmissionManifestError("package path must not contain a drive letter.")
    if raw.startswith("/"):
        raise SubmissionManifestError("package path must not be absolute.")
    parts = tuple(part for part in raw.split("/") if part)
    if not parts or any(part == ".." for part in parts):
        raise SubmissionManifestError("package path must not contain path traversal.")
    if any(part in {".", ""} for part in parts):
        raise SubmissionManifestError("package path contains an invalid component.")
    return "/".join(parts)


def compute_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_package_hash(files: Iterable[FileEntry]) -> str:
    return stable_hash({"files": tuple(sorted(files, key=lambda item: item.path))})


def compute_manifest_hash(manifest: SubmissionManifest | Mapping[str, Any]) -> str:
    if isinstance(manifest, Mapping):
        payload = dict(manifest)
    else:
        payload = {item.name: getattr(manifest, item.name) for item in fields(SubmissionManifest)}
    payload.pop("manifest_hash", None)
    return stable_hash(payload)


def build_submission_manifest(
    *,
    root_path: str | Path,
    package_name: str,
    created_by: str,
    files: Iterable[str | FileEntry | Mapping[str, Any]],
    serving_config_hash: str,
    promotion_decision_hash: str,
    training_plan_hash: str,
    dataset_manifest_hash: str,
    trace_manifest_hash: str,
    lora_config_hash: str | None = None,
    adapter_hash: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SubmissionManifest:
    root = Path(root_path)
    entries: list[FileEntry] = []
    for item in files:
        if isinstance(item, FileEntry):
            entry = item
        else:
            if isinstance(item, Mapping):
                rel = normalize_package_path(str(item["path"]))
                required = bool(item.get("required", True))
                role = str(item.get("role", "artifact"))
                entry_metadata = dict(item.get("metadata", {}))
            else:
                rel = normalize_package_path(str(item))
                required = True
                role = "artifact"
                entry_metadata = {}
            path = _resolve_under_root(root, rel)
            if not path.exists():
                if required:
                    raise SubmissionManifestError(f"required file is missing: {rel}")
                continue
            if not path.is_file():
                raise SubmissionManifestError(f"manifest entry is not a file: {rel}")
            entry = FileEntry(
                path=rel,
                size_bytes=path.stat().st_size,
                sha256=compute_file_sha256(path),
                required=required,
                role=role,
                metadata=entry_metadata,
            )
        entries.append(entry)
    sorted_entries = tuple(sorted(entries, key=lambda entry: entry.path))
    package_hash = compute_package_hash(sorted_entries)
    manifest_id = stable_hash(
        {
            "package_name": package_name,
            "created_by": created_by,
            "package_hash": package_hash,
            "serving_config_hash": serving_config_hash,
            "promotion_decision_hash": promotion_decision_hash,
            "training_plan_hash": training_plan_hash,
            "dataset_manifest_hash": dataset_manifest_hash,
            "trace_manifest_hash": trace_manifest_hash,
            "lora_config_hash": lora_config_hash,
            "adapter_hash": adapter_hash,
        }
    )
    return SubmissionManifest(
        manifest_id=manifest_id,
        package_name=package_name,
        created_by=created_by,
        files=sorted_entries,
        serving_config_hash=serving_config_hash,
        lora_config_hash=lora_config_hash,
        adapter_hash=adapter_hash,
        promotion_decision_hash=promotion_decision_hash,
        training_plan_hash=training_plan_hash,
        dataset_manifest_hash=dataset_manifest_hash,
        trace_manifest_hash=trace_manifest_hash,
        package_hash=package_hash,
        metadata=dict(metadata or {}),
    )


def validate_submission_manifest(manifest: SubmissionManifest, root_path: str | Path | None = None) -> SubmissionManifest:
    if not isinstance(manifest, SubmissionManifest):
        raise SubmissionManifestError("manifest must be a SubmissionManifest.")
    if manifest.package_hash != compute_package_hash(manifest.files):
        raise SubmissionManifestError("package_hash does not match manifest file entries.")
    if manifest.manifest_hash != compute_manifest_hash(manifest):
        raise SubmissionManifestError("manifest_hash does not match manifest payload.")
    if root_path is not None:
        root = Path(root_path)
        for entry in manifest.files:
            path = _resolve_under_root(root, entry.path)
            if entry.required and not path.exists():
                raise SubmissionManifestError(f"required file is missing: {entry.path}")
            if path.exists():
                if not path.is_file():
                    raise SubmissionManifestError(f"manifest entry is not a file: {entry.path}")
                if path.stat().st_size != entry.size_bytes:
                    raise SubmissionManifestError(f"file size mismatch: {entry.path}")
                if compute_file_sha256(path) != entry.sha256:
                    raise SubmissionManifestError(f"file sha256 mismatch: {entry.path}")
    return manifest


def write_submission_manifest(manifest: SubmissionManifest, path: str | Path) -> None:
    validate_submission_manifest(manifest)
    Path(path).write_text(stable_json_dumps(manifest) + "\n", encoding="utf-8")


def read_submission_manifest(path: str | Path) -> SubmissionManifest:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SubmissionManifestError("could not read submission manifest JSON.") from exc
    if not isinstance(payload, Mapping):
        raise SubmissionManifestError("manifest JSON must contain an object.")
    files_payload = payload.get("files")
    if not isinstance(files_payload, list):
        raise SubmissionManifestError("manifest files must be a list.")
    data = dict(payload)
    data["files"] = tuple(FileEntry(**item) for item in files_payload)
    return SubmissionManifest(**data)


def _resolve_under_root(root: Path, rel_path: str) -> Path:
    safe_rel = normalize_package_path(rel_path)
    base = root.resolve()
    target = (base / Path(*safe_rel.split("/"))).resolve()
    if target != base and base not in target.parents:
        raise SubmissionManifestError("resolved path escapes package root.")
    return target


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


def _non_empty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


__all__ = [
    "FileEntry",
    "SubmissionManifest",
    "SubmissionManifestError",
    "build_submission_manifest",
    "compute_file_sha256",
    "compute_manifest_hash",
    "compute_package_hash",
    "normalize_package_path",
    "read_submission_manifest",
    "validate_submission_manifest",
    "write_submission_manifest",
]
