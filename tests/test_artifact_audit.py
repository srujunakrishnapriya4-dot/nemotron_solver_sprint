from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.packaging.artifact_audit import (
    ArtifactAuditConfig,
    ArtifactAuditError,
    ArtifactAuditReport,
    audit_artifacts,
)
from nemotron_engine.packaging.submission_manifest import FileEntry, compute_file_sha256


@contextmanager
def temp_root(name: str):
    root = Path(__file__).resolve().parent / ".tmp_pass9_artifact_audit" / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def entry(path: str, sha: str = "0" * 64) -> FileEntry:
    return FileEntry(path=path, size_bytes=1, sha256=sha, required=True, role="artifact")


def write_file(root: Path, rel: str, text: str = "payload") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("path", ["model.safetensors", "pytorch_model-00001.bin"])
def test_forbidden_model_checkpoint_rejected(path: str) -> None:
    report = audit_artifacts(ArtifactAuditConfig(file_entries=(entry(path),)))
    assert not report.passed
    assert path in report.forbidden_paths


@pytest.mark.parametrize("path", ["optimizer.pt", "trainer_state.json", "rng_state0.pth"])
def test_training_state_rejected(path: str) -> None:
    assert not audit_artifacts(ArtifactAuditConfig(file_entries=(entry(path),))).passed


@pytest.mark.parametrize("path", [".env", "kaggle.json", "credentials-prod.json", "token.txt", "api_key.json"])
def test_secret_file_rejected(path: str) -> None:
    assert not audit_artifacts(ArtifactAuditConfig(file_entries=(entry(path),))).passed


def test_cache_dir_rejected() -> None:
    report = audit_artifacts(ArtifactAuditConfig(file_entries=(entry("__pycache__/module.pyc"),)))
    assert not report.passed
    assert "__pycache__/module.pyc" in report.forbidden_paths


@pytest.mark.parametrize("dirname", ["__pycache__", ".pytest_cache", ".ipynb_checkpoints"])
def test_empty_forbidden_directory_rejected_under_root(dirname: str) -> None:
    with temp_root(f"empty_{dirname.replace('.', '_')}") as tmp_path:
        (tmp_path / dirname).mkdir()
        report = audit_artifacts(ArtifactAuditConfig(root_path=tmp_path))
        assert not report.passed
        assert dirname in report.forbidden_paths


def test_dataset_jsonl_rejected_unless_allowed() -> None:
    strict = audit_artifacts(ArtifactAuditConfig(file_entries=(entry("sft_train.jsonl"),)))
    allowed = audit_artifacts(ArtifactAuditConfig(file_entries=(entry("sft_train.jsonl"),), allow_dataset_exports=True))
    assert not strict.passed
    assert allowed.passed


def test_hash_mismatch_rejected() -> None:
    with temp_root("hash_mismatch") as tmp_path:
        path = write_file(tmp_path, "run.py", "a")
        report = audit_artifacts(ArtifactAuditConfig(root_path=tmp_path, file_entries=(entry("run.py", "f" * 64),)))
        assert not report.passed
        assert "run.py" in report.hash_mismatches
        assert compute_file_sha256(path) != "f" * 64


def test_duplicate_file_paths_rejected() -> None:
    report = audit_artifacts(ArtifactAuditConfig(file_entries=(entry("run.py"), entry("run.py"))))
    assert not report.passed
    assert "run.py" in report.duplicate_paths


def test_passed_true_with_forbidden_artifacts_rejected() -> None:
    with pytest.raises(ArtifactAuditError):
        ArtifactAuditReport(passed=True, checked_count=1, forbidden_paths=("model.safetensors",))


def test_passed_true_with_symlink_paths_rejected() -> None:
    with pytest.raises(ArtifactAuditError):
        ArtifactAuditReport(passed=True, checked_count=1, symlink_paths=("link.txt",))


def test_forged_audit_report_hash_rejected() -> None:
    report = audit_artifacts(ArtifactAuditConfig(file_entries=(entry("run.py"),)))
    with pytest.raises(ArtifactAuditError):
        replace(report, report_hash="forged")
