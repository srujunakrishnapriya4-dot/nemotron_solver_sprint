from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.packaging.package_builder import (
    PackageBuildConfig,
    PackageBuilderError,
    PackageBuildReport,
    build_submission_package,
    validate_package_layout,
)
from nemotron_engine.packaging.submission_manifest import (
    FileEntry,
    SubmissionManifestError,
    build_submission_manifest,
)


@contextmanager
def temp_root(name: str):
    root = Path(__file__).resolve().parent / ".tmp_pass9_package_builder" / name
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


def manifest(root: Path, files: list[object] | None = None):
    write_file(root, "run.py", "print('offline')\n")
    return build_submission_manifest(
        root_path=root,
        package_name="pkg",
        created_by="pass9",
        files=files or [{"path": "run.py", "role": "entrypoint", "required": True}],
        serving_config_hash="serving-hash",
        lora_config_hash="lora-hash",
        adapter_hash="adapter-hash",
        promotion_decision_hash="decision-hash",
        training_plan_hash="training-hash",
        dataset_manifest_hash="dataset-hash",
        trace_manifest_hash="trace-hash",
    )


def test_dry_run_writes_no_zip_and_built_false() -> None:
    with temp_root("dry_run") as tmp_path:
        output = tmp_path / "package.zip"
        report = build_submission_package(
            PackageBuildConfig(root_path=tmp_path, manifest=manifest(tmp_path), output_path=output, build_zip=True, dry_run=True)
        )
        assert report.passed
        assert report.dry_run
        assert not report.built
        assert not output.exists()


def test_package_build_report_rejects_built_true_with_dry_run() -> None:
    with pytest.raises(PackageBuilderError):
        PackageBuildReport(
            package_path="pkg.zip",
            manifest_hash="manifest",
            package_hash="package",
            file_count=1,
            built=True,
            dry_run=True,
            passed=True,
        )


def test_build_zip_only_when_explicit_and_includes_manifest() -> None:
    with temp_root("zip") as tmp_path:
        output = tmp_path / "package.zip"
        report = build_submission_package(
            PackageBuildConfig(root_path=tmp_path, manifest=manifest(tmp_path), output_path=output, build_zip=True, dry_run=False)
        )
        assert report.passed
        assert report.built
        assert output.exists()
        with ZipFile(output, "r") as archive:
            assert "submission_manifest.json" in archive.namelist()
        assert "run.py" in archive.namelist()


def test_package_builder_rejects_noncanonical_manifest_filename() -> None:
    with temp_root("noncanonical_manifest") as tmp_path:
        with pytest.raises(PackageBuilderError):
            PackageBuildConfig(root_path=tmp_path, manifest=manifest(tmp_path), manifest_filename="manifest.json")


def test_forged_package_build_report_hash_rejected() -> None:
    with temp_root("forged") as tmp_path:
        report = build_submission_package(PackageBuildConfig(root_path=tmp_path, manifest=manifest(tmp_path)))
        with pytest.raises(PackageBuilderError):
            replace(report, report_hash="forged")


def test_path_traversal_rejected() -> None:
    with pytest.raises(SubmissionManifestError):
        FileEntry(path="../escape.txt", size_bytes=1, sha256="0" * 64, required=True, role="artifact")


def test_symlink_reparse_point_rejected_if_feasible() -> None:
    with temp_root("symlink") as tmp_path:
        target = write_file(tmp_path, "target.txt", "x")
        link = tmp_path / "link.txt"
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation is not available in this environment")
        report = manifest(tmp_path, [{"path": "link.txt", "role": "artifact", "required": True}])
        errors = validate_package_layout(PackageBuildConfig(root_path=tmp_path, manifest=report))
        assert any("symlink" in error for error in errors)


def test_adapter_config_rank_above_32_rejected() -> None:
    with temp_root("rank") as tmp_path:
        write_file(tmp_path, "adapter_config.json", '{"r": 64}\n')
        report = manifest(tmp_path, [{"path": "adapter_config.json", "role": "adapter_config", "required": True}])
        errors = validate_package_layout(PackageBuildConfig(root_path=tmp_path, manifest=report))
        assert any("adapter_config" in error for error in errors)


@pytest.mark.parametrize(
    "metadata",
    [
        {"adapter_config": {"r": 64}},
        {"adapter_config": {"peft_config": {"r": 64}}},
    ],
)
def test_nested_manifest_metadata_adapter_rank_above_32_rejected(metadata: dict[str, object]) -> None:
    with temp_root("nested_rank") as tmp_path:
        write_file(tmp_path, "adapter.txt", "x")
        report = manifest(
            tmp_path,
            [{"path": "adapter.txt", "role": "adapter", "required": True, "metadata": metadata}],
        )
        errors = validate_package_layout(PackageBuildConfig(root_path=tmp_path, manifest=report))
        assert any("must be in [1, 32]" in error for error in errors)


@pytest.mark.parametrize("metadata", [{"adapter_config": {"r": "64"}}, {"adapter_config": {"r": True}}])
def test_malformed_rank_metadata_rejected_deterministically(metadata: dict[str, object]) -> None:
    with temp_root("malformed_rank") as tmp_path:
        write_file(tmp_path, "adapter.txt", "x")
        report = manifest(
            tmp_path,
            [{"path": "adapter.txt", "role": "adapter", "required": True, "metadata": metadata}],
        )
        errors = validate_package_layout(PackageBuildConfig(root_path=tmp_path, manifest=report))
        assert any("must be an integer" in error for error in errors)


def test_package_builder_rejects_output_path_outside_root() -> None:
    with temp_root("outside_output") as tmp_path:
        report = build_submission_package(
            PackageBuildConfig(
                root_path=tmp_path,
                manifest=manifest(tmp_path),
                output_path=tmp_path.parent / "outside.zip",
                build_zip=True,
                dry_run=True,
            )
        )
        assert not report.passed
        assert any("outside" in error for error in report.errors)


def test_package_builder_rejects_output_path_equal_to_package_entry() -> None:
    with temp_root("overwrite_entry") as tmp_path:
        report = build_submission_package(
            PackageBuildConfig(
                root_path=tmp_path,
                manifest=manifest(tmp_path),
                output_path=tmp_path / "run.py",
                build_zip=True,
                dry_run=True,
            )
        )
        assert not report.passed
        assert any("overwrite package entry" in error for error in report.errors)


def test_forbidden_artifact_role_rejected() -> None:
    with temp_root("forbidden_role") as tmp_path:
        write_file(tmp_path, "weights.txt", "not real weights")
        report = manifest(tmp_path, [{"path": "weights.txt", "role": "checkpoint", "required": True}])
        errors = validate_package_layout(PackageBuildConfig(root_path=tmp_path, manifest=report))
        assert any("forbidden artifact role" in error for error in errors)


def test_unsafe_serving_config_rejected() -> None:
    with temp_root("serving_config") as tmp_path:
        write_file(
            tmp_path,
            "serving_config.json",
            '{"temperature": 0.7, "top_p": 1.0, "num_samples": 1, "majority_vote": false, "max_tokens": 512}\n',
        )
        report = manifest(tmp_path, [{"path": "serving_config.json", "role": "serving_config", "required": True}])
        errors = validate_package_layout(PackageBuildConfig(root_path=tmp_path, manifest=report))
        assert any("unsafe serving_config" in error for error in errors)
