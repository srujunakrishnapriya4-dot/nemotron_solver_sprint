from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.release.artifact_safety import (
    ArtifactSafetyConfig,
    ArtifactSafetyError,
    ArtifactSafetyFinding,
    ArtifactSafetyReport,
    scan_release_artifacts,
)


@contextmanager
def temp_tree(name: str):
    root = Path(__file__).resolve().parent / ".tmp_pass10_artifact_safety" / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def write_file(root: Path, rel: str, text: str = "x") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_clean_temp_tree_passes() -> None:
    with temp_tree("clean") as root:
        write_file(root, "src/nemotron_engine/app.py")
        assert scan_release_artifacts(ArtifactSafetyConfig(root_path=root)).passed


@pytest.mark.parametrize("path", ["model.safetensors", "pytorch_model-00001.bin"])
def test_checkpoint_file_rejected(path: str) -> None:
    with temp_tree("checkpoint") as root:
        write_file(root, path)
        assert not scan_release_artifacts(ArtifactSafetyConfig(root_path=root)).passed


@pytest.mark.parametrize("path", ["sft_data.jsonl", "dpo_pairs.jsonl", "train_rows.jsonl"])
def test_dataset_jsonl_rejected(path: str) -> None:
    with temp_tree("dataset") as root:
        write_file(root, path)
        assert not scan_release_artifacts(ArtifactSafetyConfig(root_path=root)).passed


def test_submission_zip_rejected_unless_allowed() -> None:
    with temp_tree("submission") as root:
        write_file(root, "submission.zip")
        assert not scan_release_artifacts(ArtifactSafetyConfig(root_path=root)).passed
        assert scan_release_artifacts(ArtifactSafetyConfig(root_path=root, allow_submission_zip=True)).passed


@pytest.mark.parametrize("dirname", ["__pycache__", ".pytest_cache", ".ipynb_checkpoints", "pytest-cache-files-a1b2c3"])
def test_cache_directory_rejected(dirname: str) -> None:
    with temp_tree(f"cache_{dirname.replace('.', '_')}") as root:
        (root / dirname).mkdir()
        assert not scan_release_artifacts(ArtifactSafetyConfig(root_path=root)).passed


def test_pyc_rejected() -> None:
    with temp_tree("pyc") as root:
        write_file(root, "module.pyc")
        assert not scan_release_artifacts(ArtifactSafetyConfig(root_path=root)).passed


@pytest.mark.parametrize("path", [".env", "kaggle.json", "credentials.json", "token.txt", "api_key.txt"])
def test_secret_files_rejected(path: str) -> None:
    with temp_tree("secret") as root:
        write_file(root, path)
        assert not scan_release_artifacts(ArtifactSafetyConfig(root_path=root)).passed


def test_notebook_rejected_unless_allowed() -> None:
    with temp_tree("notebook") as root:
        write_file(root, "analysis.ipynb", "{}")
        assert not scan_release_artifacts(ArtifactSafetyConfig(root_path=root)).passed
        assert scan_release_artifacts(ArtifactSafetyConfig(root_path=root, allow_notebooks=True)).passed


def test_symlink_report_path_rejected() -> None:
    with pytest.raises(ArtifactSafetyError):
        ArtifactSafetyReport(passed=True, scanned_paths=("link",), symlink_paths=("link",))


def test_forged_artifact_safety_report_hash_rejected() -> None:
    with temp_tree("forged") as root:
        report = scan_release_artifacts(ArtifactSafetyConfig(root_path=root))
        with pytest.raises(ArtifactSafetyError):
            replace(report, report_hash="forged")


def test_passed_true_with_forbidden_artifacts_rejected() -> None:
    finding = ArtifactSafetyFinding(path="model.safetensors", artifact_type="checkpoint", reason="bad")
    with pytest.raises(ArtifactSafetyError):
        ArtifactSafetyReport(passed=True, scanned_paths=("model.safetensors",), forbidden_artifacts=(finding,))


def test_configured_ignore_path_cannot_hide_critical_source() -> None:
    with temp_tree("ignore_critical") as root:
        (root / "src/nemotron_engine/release").mkdir(parents=True)
        with pytest.raises(ArtifactSafetyError):
            ArtifactSafetyConfig(root_path=root, ignore_paths=("src/nemotron_engine/release",))


@pytest.mark.parametrize("critical_path", ["src/nemotron_engine/scoring", "src/nemotron_engine/release", "tests/test_local_scorer.py"])
def test_configured_ignore_path_cannot_hide_locked_source_or_tests(critical_path: str) -> None:
    with temp_tree(f"ignore_{critical_path.replace('/', '_')}") as root:
        (root / critical_path).parent.mkdir(parents=True, exist_ok=True)
        if critical_path.endswith(".py"):
            (root / critical_path).write_text("x = 1\n", encoding="utf-8")
        else:
            (root / critical_path).mkdir(exist_ok=True)
        with pytest.raises(ArtifactSafetyError):
            ArtifactSafetyConfig(root_path=root, ignore_paths=(critical_path,))
