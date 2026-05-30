from __future__ import annotations

from dataclasses import replace
from contextlib import contextmanager
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.packaging.submission_manifest import (
    FileEntry,
    SubmissionManifest,
    SubmissionManifestError,
    build_submission_manifest,
    compute_file_sha256,
    compute_package_hash,
    read_submission_manifest,
    validate_submission_manifest,
    write_submission_manifest,
)


@contextmanager
def temp_root(name: str):
    root = Path(__file__).resolve().parent / ".tmp_pass9_submission_manifest" / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def write_file(root: Path, rel: str, text: str = "payload") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def manifest(tmp_path: Path) -> SubmissionManifest:
    write_file(tmp_path, "run.py", "print('offline')\n")
    return build_submission_manifest(
        root_path=tmp_path,
        package_name="pkg",
        created_by="pass9",
        files=[{"path": "run.py", "role": "entrypoint", "required": True}],
        serving_config_hash="serving-hash",
        lora_config_hash="lora-hash",
        adapter_hash="adapter-hash",
        promotion_decision_hash="decision-hash",
        training_plan_hash="training-hash",
        dataset_manifest_hash="dataset-hash",
        trace_manifest_hash="trace-hash",
    )


def test_builds_deterministic_manifest() -> None:
    with temp_root("deterministic") as tmp_path:
        first = manifest(tmp_path)
        second = manifest(tmp_path)
        assert first.manifest_id == second.manifest_id
        assert first.package_hash == second.package_hash
        assert first.manifest_hash == second.manifest_hash
        assert validate_submission_manifest(first, tmp_path) is first


@pytest.mark.parametrize("bad_path", ["/abs/file.txt", "C:/abs/file.txt", "../escape.txt", "safe/../escape.txt"])
def test_file_entry_rejects_unsafe_paths(bad_path: str) -> None:
    with pytest.raises(SubmissionManifestError):
        FileEntry(path=bad_path, size_bytes=1, sha256="0" * 64, required=True, role="artifact")


def test_manifest_rejects_duplicate_file_paths() -> None:
    with temp_root("duplicate") as tmp_path:
        path = write_file(tmp_path, "a.txt")
        sha = compute_file_sha256(path)
        entry = FileEntry(path="a.txt", size_bytes=path.stat().st_size, sha256=sha, required=True, role="artifact")
        with pytest.raises(SubmissionManifestError):
            SubmissionManifest(
                manifest_id="id",
                package_name="pkg",
                created_by="pass9",
                files=(entry, entry),
                serving_config_hash="serving",
                lora_config_hash="lora",
                adapter_hash="adapter",
                promotion_decision_hash="decision",
                training_plan_hash="training",
                dataset_manifest_hash="dataset",
                trace_manifest_hash="trace",
                package_hash=compute_package_hash((entry, entry)),
            )


def test_rejects_missing_required_file() -> None:
    with temp_root("missing") as tmp_path:
        with pytest.raises(SubmissionManifestError):
            build_submission_manifest(
                root_path=tmp_path,
                package_name="pkg",
                created_by="pass9",
                files=["missing.py"],
                serving_config_hash="serving",
                promotion_decision_hash="decision",
                training_plan_hash="training",
                dataset_manifest_hash="dataset",
                trace_manifest_hash="trace",
            )


def test_validate_rejects_hash_mismatch_when_root_supplied() -> None:
    with temp_root("hash_mismatch") as tmp_path:
        report = manifest(tmp_path)
        (tmp_path / "run.py").write_text("changed\n", encoding="utf-8")
        with pytest.raises(SubmissionManifestError):
            validate_submission_manifest(report, tmp_path)


def test_rejects_forged_package_hash() -> None:
    with temp_root("forged_package") as tmp_path:
        report = manifest(tmp_path)
        with pytest.raises(SubmissionManifestError):
            replace(report, package_hash="f" * 64)


def test_rejects_forged_manifest_hash() -> None:
    with temp_root("forged_manifest") as tmp_path:
        report = manifest(tmp_path)
        with pytest.raises(SubmissionManifestError):
            replace(report, manifest_hash="f" * 64)


def test_write_read_round_trip_is_deterministic() -> None:
    with temp_root("roundtrip") as tmp_path:
        report = manifest(tmp_path)
        path = tmp_path / "manifest.json"
        write_submission_manifest(report, path)
        loaded = read_submission_manifest(path)
        assert loaded == report
        assert loaded.manifest_hash == report.manifest_hash
