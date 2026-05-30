from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.core.manifest import create_manifest, read_manifest, stable_hash, stable_json_dumps, validate_manifest_chain, write_manifest  # noqa: E402
from nemotron_engine.core.schemas import ArtifactManifest, ManifestError, SchemaValidationError  # noqa: E402


_TMP = Path(__file__).resolve().parent / ".tmp_pass2"


def _tmp_dir() -> Path:
    path = _TMP / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_stable_hash_deterministic() -> None:
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})


def test_manifest_rejects_bad_counts() -> None:
    with pytest.raises(SchemaValidationError):
        create_manifest(stage="s", row_count=-1, accepted_count=0, rejected_count=0)
    with pytest.raises(SchemaValidationError):
        create_manifest(stage="s", row_count=1, accepted_count=1, rejected_count=1)


def test_write_read_manifest_round_trip_and_chain() -> None:
    root = _tmp_dir()
    try:
        first = create_manifest(stage="parse", row_count=2, accepted_count=1, rejected_count=1)
        second = create_manifest(
            stage="verify",
            row_count=1,
            accepted_count=1,
            rejected_count=0,
            input_manifest_hash=first.output_manifest_hash,
        )
        path = root / "manifest.json"
        write_manifest(first, path)

        assert read_manifest(path) == first
        assert validate_manifest_chain([first, second]) is True
        with pytest.raises(ManifestError):
            validate_manifest_chain([first, create_manifest(stage="bad", row_count=1, accepted_count=1, rejected_count=0)])
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)


def test_read_manifest_rejects_unknown_fields_non_object_and_malformed_objects() -> None:
    root = _tmp_dir()
    try:
        unknown = root / "unknown.json"
        manifest = create_manifest(stage="parse", row_count=1, accepted_count=1, rejected_count=0)
        unknown.write_text(stable_json_dumps({**manifest.__dict__, "extra": "bad"}), encoding="utf-8")
        with pytest.raises(ManifestError):
            read_manifest(unknown)

        non_object = root / "non_object.json"
        non_object.write_text("[]", encoding="utf-8")
        with pytest.raises(ManifestError):
            read_manifest(non_object)

        malformed = root / "malformed.json"
        malformed.write_text(stable_json_dumps({"stage": "parse"}), encoding="utf-8")
        with pytest.raises(ManifestError):
            read_manifest(malformed)
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)


def test_validate_manifest_chain_rejects_tampered_output_manifest_hash() -> None:
    first = create_manifest(stage="parse", row_count=2, accepted_count=1, rejected_count=1)
    tampered = ArtifactManifest(
        stage=first.stage,
        input_manifest_hash=first.input_manifest_hash,
        output_manifest_hash="0" * 64,
        schema_version=first.schema_version,
        row_count=first.row_count,
        accepted_count=first.accepted_count,
        rejected_count=first.rejected_count,
        rejection_summary=first.rejection_summary,
        code_git_hash=first.code_git_hash,
        config_hash=first.config_hash,
        created_at=first.created_at,
    )

    with pytest.raises(ManifestError):
        validate_manifest_chain([tampered])
