from __future__ import annotations

import dataclasses

import pytest

from nemotron_engine.packaging import PackageBuildReport, SubmissionManifest
from nemotron_engine.rehearsal import (
    PackageRehearsalConfig,
    PackageRehearsalError,
    PackageRehearsalReport,
    run_package_rehearsal,
)
from test_pass14_end_to_end_rehearsal import (
    make_adapter_bundle,
    make_bridge,
    make_config,
    make_package_report,
    make_submission_manifest,
)


def test_dry_run_no_zip_build_passes_with_built_false() -> None:
    config = make_config()
    bundle = make_adapter_bundle()
    bridge = make_bridge()
    report = run_package_rehearsal(
        rehearsal_config=config,
        adapter_evidence_bundle=bundle,
        promotion_bridge_report=bridge,
        submission_manifest=make_submission_manifest(bundle, bridge),
    )
    assert report.passed is True
    assert report.built is False


def test_run_package_rehearsal_rejects_missing_manifest_evidence() -> None:
    config = make_config()
    bundle = make_adapter_bundle()
    bridge = make_bridge()
    report = run_package_rehearsal(rehearsal_config=config, adapter_evidence_bundle=bundle, promotion_bridge_report=bridge)
    assert report.passed is False
    assert "submission_manifest_missing" in report.errors


def test_allow_zip_build_false_cannot_write_package() -> None:
    config = make_config()
    bundle = make_adapter_bundle()
    bridge = make_bridge()
    build = PackageBuildReport(
        package_path="C:/pass14/out/pkg.zip",
        manifest_hash="manifest",
        package_hash="package",
        file_count=1,
        built=True,
        dry_run=False,
        passed=True,
    )
    report = run_package_rehearsal(
        rehearsal_config=config,
        adapter_evidence_bundle=bundle,
        promotion_bridge_report=bridge,
        package_build_report=build,
    )
    assert "zip_build_not_allowed" in report.errors


def test_allow_zip_build_true_requires_output_dir() -> None:
    with pytest.raises(Exception):
        make_config(allow_zip_build=True, output_dir=None)


def test_supplied_package_build_report_failure_rejected() -> None:
    config = make_config()
    bundle = make_adapter_bundle()
    build = PackageBuildReport(package_path=None, manifest_hash="m", package_hash="p", file_count=1, built=False, dry_run=True, passed=False, errors=("bad",))
    report = run_package_rehearsal(rehearsal_config=config, adapter_evidence_bundle=bundle, promotion_bridge_report=make_bridge(), package_build_report=build)
    assert "package_build_failed" in report.errors


def test_supplied_package_build_report_forged_rejected() -> None:
    build = PackageBuildReport(package_path=None, manifest_hash="m", package_hash="p", file_count=1, built=False, dry_run=True, passed=True)
    data = dataclasses.asdict(build)
    data["report_hash"] = "forged"
    with pytest.raises(Exception):
        PackageBuildReport(**data)


def test_supplied_submission_manifest_forged_rejected() -> None:
    bundle = make_adapter_bundle()
    bridge = make_bridge()
    manifest = make_submission_manifest(bundle, bridge)
    data = dataclasses.asdict(manifest)
    data["manifest_hash"] = "forged"
    with pytest.raises(Exception):
        SubmissionManifest(**data)


def test_promotion_bridge_not_allowed_blocks() -> None:
    config = make_config()
    bundle = make_adapter_bundle()
    bridge = dataclasses.replace(make_bridge(), allowed=False, errors=("blocked",), report_hash="")
    report = run_package_rehearsal(rehearsal_config=config, adapter_evidence_bundle=bundle, promotion_bridge_report=bridge)
    assert "promotion_bridge_not_allowed" in report.errors


def test_adapter_evidence_incomplete_blocks() -> None:
    config = make_config()
    bundle = dataclasses.replace(make_adapter_bundle(), evidence_complete=False, evidence_hash="")
    report = run_package_rehearsal(rehearsal_config=config, adapter_evidence_bundle=bundle, promotion_bridge_report=make_bridge())
    assert "adapter_evidence_incomplete" in report.errors


def test_package_path_outside_output_dir_rejected() -> None:
    config = make_config(allow_zip_build=True, output_dir="C:/pass14/out")
    bundle = make_adapter_bundle()
    bridge = make_bridge()
    build = PackageBuildReport(package_path="C:/pass14/elsewhere/pkg.zip", manifest_hash="m", package_hash="p", file_count=1, built=True, dry_run=False, passed=True)
    report = run_package_rehearsal(rehearsal_config=config, adapter_evidence_bundle=bundle, promotion_bridge_report=bridge, package_build_report=build)
    assert "package_path_outside_output_dir" in report.errors


def test_report_built_true_during_dry_run_without_zip_permission_rejected() -> None:
    base = make_package_report(make_config(), make_adapter_bundle(), make_bridge())
    data = dataclasses.asdict(base)
    data["built"] = True
    data["package_path"] = "C:/pass14/out/pkg.zip"
    data["report_hash"] = ""
    with pytest.raises(PackageRehearsalError):
        PackageRehearsalReport(**data)


def test_passed_true_empty_required_hashes_rejected() -> None:
    base = make_package_report(make_config(), make_adapter_bundle(), make_bridge())
    data = dataclasses.asdict(base)
    data["required_hashes"] = {}
    data["report_hash"] = ""
    with pytest.raises(PackageRehearsalError):
        PackageRehearsalReport(**data)


def test_passed_true_missing_submission_manifest_hash_rejected() -> None:
    base = make_package_report(make_config(), make_adapter_bundle(), make_bridge())
    data = dataclasses.asdict(base)
    data["submission_manifest_hash"] = None
    data["report_hash"] = ""
    with pytest.raises(PackageRehearsalError):
        PackageRehearsalReport(**data)


def test_passed_true_missing_required_package_build_hash_rejected() -> None:
    base = make_package_report(make_config(), make_adapter_bundle(), make_bridge())
    data = dataclasses.asdict(base)
    data["package_build_evidence_required"] = True
    data["package_build_report_hash"] = None
    data["report_hash"] = ""
    with pytest.raises(PackageRehearsalError):
        PackageRehearsalReport(**data)


def test_forged_report_hash_rejected() -> None:
    report = make_package_report(make_config(), make_adapter_bundle(), make_bridge())
    data = dataclasses.asdict(report)
    data["report_hash"] = "forged"
    with pytest.raises(PackageRehearsalError):
        PackageRehearsalReport(**data)


def test_fake_kaggle_submission_metadata_rejected() -> None:
    with pytest.raises(PackageRehearsalError):
        run_package_rehearsal(
            rehearsal_config=make_config(),
            adapter_evidence_bundle=make_adapter_bundle(),
            promotion_bridge_report=make_bridge(),
            metadata={"submission_success": True},
        )
