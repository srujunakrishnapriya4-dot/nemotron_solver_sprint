"""Artifact manifests for Pass 2 proof-spine stages."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schemas import ArtifactManifest, ManifestError, SchemaValidationError, stable_hash, stable_json_dumps


def hash_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def hash_file(path: str | Path) -> str:
    target = Path(path)
    if not target.is_file():
        raise ManifestError(f"Cannot hash missing file: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_json_obj(obj: Any) -> str:
    return stable_hash(obj)


def create_manifest(
    *,
    stage: str,
    row_count: int,
    accepted_count: int,
    rejected_count: int,
    rejection_summary: Mapping[str, int] | None = None,
    input_manifest_hash: str | None = None,
    schema_version: str = "pass2.v1",
    code_git_hash: str = "",
    config_hash: str = "",
    created_at: str = "1970-01-01T00:00:00Z",
) -> ArtifactManifest:
    payload = {
        "stage": stage,
        "input_manifest_hash": input_manifest_hash,
        "schema_version": schema_version,
        "row_count": row_count,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "rejection_summary": dict(rejection_summary or {}),
        "code_git_hash": code_git_hash,
        "config_hash": config_hash,
        "created_at": created_at,
    }
    output_manifest_hash = stable_hash(payload)
    return ArtifactManifest(output_manifest_hash=output_manifest_hash, **payload)


def write_manifest(manifest: ArtifactManifest, path: str | Path) -> None:
    Path(path).write_text(stable_json_dumps(manifest) + "\n", encoding="utf-8")


def read_manifest(path: str | Path) -> ArtifactManifest:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Invalid manifest JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ManifestError(f"Manifest JSON must be an object: {path}")
    allowed = {item.name for item in fields(ArtifactManifest)}
    unknown = set(payload) - allowed
    if unknown:
        raise ManifestError(f"Manifest contains unknown fields: {sorted(unknown)}")
    try:
        return ArtifactManifest(**payload)
    except (TypeError, SchemaValidationError, ValueError) as exc:
        raise ManifestError(f"Malformed manifest object: {path}") from exc


def validate_manifest_chain(manifests: Iterable[ArtifactManifest]) -> bool:
    materialized = list(manifests)
    for index, manifest in enumerate(materialized):
        expected_hash = _manifest_payload_hash(manifest)
        if manifest.output_manifest_hash != expected_hash:
            raise ManifestError(
                f"Manifest hash mismatch at index {index}: expected {expected_hash}, "
                f"got {manifest.output_manifest_hash}."
            )
        if index == 0:
            continue
        previous = materialized[index - 1]
        if manifest.input_manifest_hash != previous.output_manifest_hash:
            raise ManifestError(
                f"Manifest chain break at index {index}: expected input hash {previous.output_manifest_hash}, "
                f"got {manifest.input_manifest_hash}."
            )
    return True


def _manifest_payload_hash(manifest: ArtifactManifest) -> str:
    return stable_hash({item.name: getattr(manifest, item.name) for item in fields(ArtifactManifest) if item.name != "output_manifest_hash"})


__all__ = [
    "create_manifest",
    "hash_file",
    "hash_json_obj",
    "hash_text",
    "read_manifest",
    "stable_hash",
    "validate_manifest_chain",
    "write_manifest",
]
