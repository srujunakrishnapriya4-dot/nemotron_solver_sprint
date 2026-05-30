from __future__ import annotations

import dataclasses

import pytest

from nemotron_engine.rehearsal import SubmissionRehearsalConfig, SubmissionRehearsalError, SubmissionRehearsalReport, run_submission_rehearsal
from test_pass14_end_to_end_rehearsal import (
    make_adapter_bundle,
    make_bridge,
    make_config,
    make_package_report,
    make_preflight,
    make_release,
    make_runtime_report,
)


def make_submission_base(*, config=None):
    config = config or make_config()
    bundle = make_adapter_bundle()
    bridge = make_bridge()
    package = make_package_report(config, bundle, bridge)
    runtime = make_runtime_report(config, bundle, package)
    return run_submission_rehearsal(
        rehearsal_config=config,
        promotion_bridge_report=bridge,
        package_rehearsal_report=package,
        runtime_rehearsal_report=runtime,
    )


def test_all_green_dry_run_rehearsal_passes() -> None:
    assert make_submission_base().passed is True


def test_direct_passed_true_without_component_status_rejected() -> None:
    base = make_submission_base()
    data = dataclasses.asdict(base)
    data["promotion_allowed"] = False
    data["package_passed"] = False
    data["runtime_passed"] = False
    data["report_hash"] = ""
    with pytest.raises(SubmissionRehearsalError):
        SubmissionRehearsalReport(**data)


def test_promotion_bridge_failure_blocks() -> None:
    config = make_config()
    bundle = make_adapter_bundle()
    bridge = dataclasses.replace(make_bridge(), allowed=False, errors=("blocked",), report_hash="")
    package = make_package_report(config, bundle, make_bridge())
    runtime = make_runtime_report(config, bundle, package)
    report = run_submission_rehearsal(rehearsal_config=config, promotion_bridge_report=bridge, package_rehearsal_report=package, runtime_rehearsal_report=runtime)
    assert "promotion_bridge_failed" in report.errors


def test_package_rehearsal_failure_blocks() -> None:
    config = make_config()
    bundle = make_adapter_bundle()
    bridge = make_bridge()
    package = dataclasses.replace(make_package_report(config, bundle, bridge), passed=False, errors=("bad",), report_hash="")
    runtime = make_runtime_report(config, bundle, make_package_report(config, bundle, bridge))
    report = run_submission_rehearsal(rehearsal_config=config, promotion_bridge_report=bridge, package_rehearsal_report=package, runtime_rehearsal_report=runtime)
    assert "package_rehearsal_failed" in report.errors


def test_runtime_rehearsal_failure_blocks() -> None:
    config = make_config()
    bundle = make_adapter_bundle()
    bridge = make_bridge()
    package = make_package_report(config, bundle, bridge)
    runtime = dataclasses.replace(make_runtime_report(config, bundle, package), passed=False, errors=("bad",), report_hash="")
    report = run_submission_rehearsal(rehearsal_config=config, promotion_bridge_report=bridge, package_rehearsal_report=package, runtime_rehearsal_report=runtime)
    assert "runtime_rehearsal_failed" in report.errors


def test_preflight_failure_blocks() -> None:
    config = make_config()
    bundle = make_adapter_bundle()
    bridge = make_bridge()
    package = make_package_report(config, bundle, bridge)
    runtime = make_runtime_report(config, bundle, package)
    report = run_submission_rehearsal(
        rehearsal_config=config,
        promotion_bridge_report=bridge,
        package_rehearsal_report=package,
        runtime_rehearsal_report=runtime,
        preflight_gate_report=make_preflight(False),
    )
    assert "preflight_gate_failed" in report.errors


def test_preflight_required_but_missing_blocks() -> None:
    config = make_config()
    bundle = make_adapter_bundle()
    bridge = make_bridge()
    package = make_package_report(config, bundle, bridge)
    runtime = make_runtime_report(config, bundle, package)
    report = run_submission_rehearsal(
        rehearsal_config=config,
        promotion_bridge_report=bridge,
        package_rehearsal_report=package,
        runtime_rehearsal_report=runtime,
        config=SubmissionRehearsalConfig(require_preflight=True),
    )
    assert "preflight_gate_missing" in report.errors


def test_release_candidate_not_ready_blocks_when_required() -> None:
    config = make_config(require_release_candidate=True)
    bundle = make_adapter_bundle()
    bridge = make_bridge()
    package = make_package_report(config, bundle, bridge)
    runtime = make_runtime_report(config, bundle, package)
    report = run_submission_rehearsal(
        rehearsal_config=config,
        promotion_bridge_report=bridge,
        package_rehearsal_report=package,
        runtime_rehearsal_report=runtime,
        release_candidate_report=make_release(False),
    )
    assert "release_candidate_not_ready" in report.errors


def test_release_candidate_required_but_missing_rejected_directly() -> None:
    data = dataclasses.asdict(make_submission_base())
    data["release_candidate_required"] = True
    data["release_candidate_hash"] = None
    data["release_candidate_ready"] = None
    data["report_hash"] = ""
    with pytest.raises(SubmissionRehearsalError):
        SubmissionRehearsalReport(**data)


def test_submitted_true_impossible_rejected() -> None:
    data = dataclasses.asdict(make_submission_base())
    data["submitted"] = True
    data["report_hash"] = ""
    with pytest.raises(SubmissionRehearsalError):
        SubmissionRehearsalReport(**data)


def test_fake_kaggle_submission_metadata_rejected() -> None:
    data = dataclasses.asdict(make_submission_base())
    data["metadata"] = {"kaggle_success": True}
    data["report_hash"] = ""
    with pytest.raises(SubmissionRehearsalError):
        SubmissionRehearsalReport(**data)


def test_forged_report_hash_rejected() -> None:
    data = dataclasses.asdict(make_submission_base())
    data["report_hash"] = "forged"
    with pytest.raises(SubmissionRehearsalError):
        SubmissionRehearsalReport(**data)
