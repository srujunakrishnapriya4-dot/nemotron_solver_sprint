from __future__ import annotations

import dataclasses

import pytest

from nemotron_engine.rehearsal import (
    FinalRehearsalManifest,
    FinalRehearsalManifestError,
    build_final_rehearsal_manifest,
    validate_final_rehearsal_manifest,
)
from test_pass14_end_to_end_rehearsal import (
    make_adapter_bundle,
    make_bridge,
    make_config,
    make_package_report,
    make_runtime_report,
)
from nemotron_engine.rehearsal import run_submission_rehearsal


def make_manifest_base(*, config=None):
    config = config or make_config()
    bundle = make_adapter_bundle()
    bridge = make_bridge()
    package = make_package_report(config, bundle, bridge)
    runtime = make_runtime_report(config, bundle, package)
    submission = run_submission_rehearsal(
        rehearsal_config=config,
        promotion_bridge_report=bridge,
        package_rehearsal_report=package,
        runtime_rehearsal_report=runtime,
    )
    return build_final_rehearsal_manifest(
        rehearsal_config=config,
        adapter_evidence_bundle=bundle,
        promotion_bridge_report=bridge,
        package_rehearsal_report=package,
        runtime_rehearsal_report=runtime,
        submission_rehearsal_report=submission,
    )


def test_valid_final_rehearsal_manifest_passes() -> None:
    manifest = make_manifest_base()
    assert validate_final_rehearsal_manifest(manifest).ready_for_external_submission is True


def test_direct_ready_true_without_component_status_rejected() -> None:
    data = dataclasses.asdict(make_manifest_base())
    data["package_passed"] = False
    data["runtime_passed"] = False
    data["submission_passed"] = False
    data["manifest_hash"] = ""
    with pytest.raises(FinalRehearsalManifestError):
        FinalRehearsalManifest(**data)


def test_forged_manifest_id_rejected() -> None:
    data = dataclasses.asdict(make_manifest_base())
    data["manifest_id"] = "forged"
    data["manifest_hash"] = ""
    with pytest.raises(FinalRehearsalManifestError):
        FinalRehearsalManifest(**data)


def test_forged_manifest_hash_rejected() -> None:
    data = dataclasses.asdict(make_manifest_base())
    data["manifest_hash"] = "forged"
    with pytest.raises(FinalRehearsalManifestError):
        FinalRehearsalManifest(**data)


def test_submitted_true_rejected() -> None:
    data = dataclasses.asdict(make_manifest_base())
    data["submitted"] = True
    data["ready_for_external_submission"] = False
    data["manifest_hash"] = ""
    with pytest.raises(FinalRehearsalManifestError):
        FinalRehearsalManifest(**data)


def test_ready_true_with_errors_rejected() -> None:
    data = dataclasses.asdict(make_manifest_base())
    data["errors"] = ("bad",)
    data["manifest_hash"] = ""
    with pytest.raises(FinalRehearsalManifestError):
        FinalRehearsalManifest(**data)


def test_ready_true_with_warnings_when_not_allowed_rejected() -> None:
    data = dataclasses.asdict(make_manifest_base())
    data["warnings"] = ("warn",)
    data["allow_warnings"] = False
    data["metadata"] = {"allow_warnings": False}
    data["manifest_hash"] = ""
    with pytest.raises(FinalRehearsalManifestError):
        FinalRehearsalManifest(**data)


def test_ready_true_with_runtime_failed_rejected_by_builder() -> None:
    config = make_config()
    bundle = make_adapter_bundle()
    bridge = make_bridge()
    package = make_package_report(config, bundle, bridge)
    runtime = dataclasses.replace(make_runtime_report(config, bundle, package), passed=False, errors=("bad",), report_hash="")
    submission = run_submission_rehearsal(rehearsal_config=config, promotion_bridge_report=bridge, package_rehearsal_report=package, runtime_rehearsal_report=runtime)
    manifest = build_final_rehearsal_manifest(
        rehearsal_config=config,
        adapter_evidence_bundle=bundle,
        promotion_bridge_report=bridge,
        package_rehearsal_report=package,
        runtime_rehearsal_report=runtime,
        submission_rehearsal_report=submission,
    )
    assert manifest.ready_for_external_submission is False
    assert "runtime_rehearsal_failed" in manifest.errors


def test_ready_true_with_package_passed_false_rejected() -> None:
    data = dataclasses.asdict(make_manifest_base())
    data["package_passed"] = False
    data["manifest_hash"] = ""
    with pytest.raises(FinalRehearsalManifestError):
        FinalRehearsalManifest(**data)


def test_ready_true_with_runtime_passed_false_without_unloaded_allowance_rejected() -> None:
    data = dataclasses.asdict(make_manifest_base())
    data["runtime_passed"] = False
    data["runtime_unloaded_dry_run"] = False
    data["unloaded_runtime_dry_run_allowed"] = False
    data["manifest_hash"] = ""
    with pytest.raises(FinalRehearsalManifestError):
        FinalRehearsalManifest(**data)


def test_ready_true_with_submission_passed_false_rejected() -> None:
    data = dataclasses.asdict(make_manifest_base())
    data["submission_passed"] = False
    data["manifest_hash"] = ""
    with pytest.raises(FinalRehearsalManifestError):
        FinalRehearsalManifest(**data)


def test_ready_true_with_dry_run_false_rejected() -> None:
    data = dataclasses.asdict(make_manifest_base())
    data["dry_run"] = False
    data["manifest_id"] = "x"
    data["manifest_hash"] = ""
    with pytest.raises(FinalRehearsalManifestError):
        FinalRehearsalManifest(**data)


def test_ready_meaning_is_local_handoff_not_leaderboard() -> None:
    manifest = make_manifest_base()
    assert manifest.ready_for_external_submission is True
    assert "local dry-run rehearsal evidence" in manifest.metadata["meaning"]


def test_fake_leaderboard_kaggle_metadata_rejected() -> None:
    data = dataclasses.asdict(make_manifest_base())
    data["metadata"] = {"leaderboard_ready": True}
    data["manifest_hash"] = ""
    with pytest.raises(FinalRehearsalManifestError):
        FinalRehearsalManifest(**data)
