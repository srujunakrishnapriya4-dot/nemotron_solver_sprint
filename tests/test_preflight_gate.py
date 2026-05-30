from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields, replace
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.evaluation.promotion_gates import PromotionDecision, PromotionGateReport
from nemotron_engine.evaluation.submission_exact import (
    SubmissionExactConfig,
    SubmissionExactReport,
    validate_submission_exact_config,
)
from nemotron_engine.packaging.artifact_audit import ArtifactAuditConfig, audit_artifacts
from nemotron_engine.packaging.package_builder import PackageBuildConfig, PackageBuildReport, build_submission_package
from nemotron_engine.packaging.preflight_gate import (
    PreflightGateConfig,
    PreflightGateError,
    PreflightGateReport,
    evaluate_preflight_gate,
)
from nemotron_engine.packaging.runtime_validator import RuntimeValidationConfig, validate_offline_runtime
from nemotron_engine.packaging.submission_manifest import build_submission_manifest
from nemotron_engine.runtime.serving_config import ServingConfig


HASHES = {
    "serving_config_hash": "serving-hash",
    "lora_config_hash": "lora-hash",
    "adapter_hash": "adapter-hash",
    "promotion_decision_hash": "decision-hash",
    "training_plan_hash": "training-hash",
    "dataset_manifest_hash": "dataset-hash",
    "trace_manifest_hash": "trace-hash",
}


@contextmanager
def temp_root(name: str):
    root = Path(__file__).resolve().parent / ".tmp_pass9_preflight" / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def serving(**updates: object) -> ServingConfig:
    data = {
        "temperature": 0.0,
        "top_p": 1.0,
        "num_samples": 1,
        "majority_vote": False,
        "max_tokens": 512,
        "prompt_template_hash": "prompt-hash",
        "tokenizer_hash": "tokenizer-hash",
        "model_hash": "model-hash",
        "adapter_hash": "adapter-hash",
    }
    data.update(updates)
    return ServingConfig(**data)


def write_file(root: Path, rel: str, text: str = "payload") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def manifest(root: Path, **updates: object):
    write_file(root, "run.py", "print('offline')\n")
    data = dict(HASHES)
    data["serving_config_hash"] = submission_exact().serving_config_hash
    data["promotion_decision_hash"] = promotion().decision_hash
    data.update(updates)
    return build_submission_manifest(
        root_path=root,
        package_name="pkg",
        created_by="pass9",
        files=[{"path": "run.py", "role": "entrypoint", "required": True}],
        **data,
    )


def promotion(decision: PromotionDecision = PromotionDecision.ALLOW) -> PromotionGateReport:
    failures = () if decision is PromotionDecision.ALLOW else ("promotion_not_allow",)
    return PromotionGateReport(
        decision=decision,
        passed=decision is PromotionDecision.ALLOW,
        failure_reasons=failures,
        input_report_hashes={"submission": "sub-hash"},
        required_hashes={
            "training_plan_hash": "training-hash",
            "dataset_manifest_hash": "dataset-hash",
            "trace_manifest_hash": "trace-hash",
            "lora_config_hash": "lora-hash",
            "adapter_hash": "adapter-hash",
        },
    )


def submission_exact(passed: bool = True) -> SubmissionExactReport:
    if passed:
        return validate_submission_exact_config(
            SubmissionExactConfig(serving_config=serving(), adapter_config={"r": 8}, dry_run=True)
        )
    return SubmissionExactReport(
        config_hash="config",
        serving_config_hash="serving",
        passed=False,
        dry_run=True,
        runtime_success=False,
        required_hashes_present=False,
        adapter_checked=False,
        critical_serving_fields={},
        errors=("bad",),
    )


def inputs(tmp_path: Path):
    man = manifest(tmp_path)
    package = build_submission_package(PackageBuildConfig(root_path=tmp_path, manifest=man, dry_run=True))
    runtime = validate_offline_runtime(
        RuntimeValidationConfig(serving_config=serving(), dry_run=True, dry_run_allow_unloaded=True)
    )
    audit = audit_artifacts(ArtifactAuditConfig(root_path=tmp_path, file_entries=man.files))
    return man, package, runtime, audit, promotion(), submission_exact()


def preflight(tmp_path: Path, **overrides):
    names = ("submission_manifest", "package_build_report", "runtime_validation_report", "artifact_audit_report", "promotion_gate_report", "submission_exact_report")
    values = dict(zip(names, inputs(tmp_path)))
    values.update(overrides)
    return evaluate_preflight_gate(
        **values,
        config=PreflightGateConfig(allow_package_dry_run=True, allow_runtime_dry_run_unloaded=True, allow_warnings=True),
    )


def test_all_green_dry_run_passes_only_with_allowances() -> None:
    with temp_root("green") as tmp_path:
        man, package, runtime, audit, promo, sub = inputs(tmp_path)
        blocked = evaluate_preflight_gate(
            submission_manifest=man,
            package_build_report=package,
            runtime_validation_report=runtime,
            artifact_audit_report=audit,
            promotion_gate_report=promo,
            submission_exact_report=sub,
            config=PreflightGateConfig(),
        )
        allowed = preflight(tmp_path)
        assert not blocked.passed
        assert allowed.passed


def test_preflight_rejects_promotion_reject() -> None:
    with temp_root("promotion_reject") as tmp_path:
        report = preflight(tmp_path, promotion_gate_report=promotion(PromotionDecision.REJECT))
        assert not report.passed
        assert "promotion_not_allow" in report.failure_reasons


def test_preflight_rejects_promotion_decision_hash_mismatch() -> None:
    with temp_root("promotion_hash_mismatch") as tmp_path:
        bad_manifest = manifest(tmp_path, promotion_decision_hash="wrong-decision")
        package = build_submission_package(PackageBuildConfig(root_path=tmp_path, manifest=bad_manifest, dry_run=True))
        runtime = validate_offline_runtime(RuntimeValidationConfig(serving_config=serving(), dry_run=True, dry_run_allow_unloaded=True))
        audit = audit_artifacts(ArtifactAuditConfig(root_path=tmp_path, file_entries=bad_manifest.files))
        report = evaluate_preflight_gate(
            submission_manifest=bad_manifest,
            package_build_report=package,
            runtime_validation_report=runtime,
            artifact_audit_report=audit,
            promotion_gate_report=promotion(),
            submission_exact_report=submission_exact(),
            config=PreflightGateConfig(allow_package_dry_run=True, allow_runtime_dry_run_unloaded=True, allow_warnings=True),
        )
        assert not report.passed
        assert "promotion_decision_hash_mismatch" in report.failure_reasons


def test_preflight_rejects_submission_exact_failure() -> None:
    with temp_root("submission_failure") as tmp_path:
        report = preflight(tmp_path, submission_exact_report=submission_exact(False))
        assert not report.passed
        assert "submission_exact_failed" in report.failure_reasons


def test_preflight_rejects_serving_config_hash_mismatch() -> None:
    with temp_root("serving_hash_mismatch") as tmp_path:
        bad_manifest = manifest(tmp_path, serving_config_hash="wrong-serving")
        package = build_submission_package(PackageBuildConfig(root_path=tmp_path, manifest=bad_manifest, dry_run=True))
        runtime = validate_offline_runtime(RuntimeValidationConfig(serving_config=serving(), dry_run=True, dry_run_allow_unloaded=True))
        audit = audit_artifacts(ArtifactAuditConfig(root_path=tmp_path, file_entries=bad_manifest.files))
        report = evaluate_preflight_gate(
            submission_manifest=bad_manifest,
            package_build_report=package,
            runtime_validation_report=runtime,
            artifact_audit_report=audit,
            promotion_gate_report=promotion(),
            submission_exact_report=submission_exact(),
            config=PreflightGateConfig(allow_package_dry_run=True, allow_runtime_dry_run_unloaded=True, allow_warnings=True),
        )
        assert not report.passed
        assert "submission_serving_hash_mismatch" in report.failure_reasons


def test_preflight_rejects_runtime_fake_success() -> None:
    with temp_root("fake_runtime") as tmp_path:
        _, _, runtime, _, _, _ = inputs(tmp_path)
        object.__setattr__(runtime, "runtime_loaded", True)
        object.__setattr__(runtime, "model_loaded", True)
        object.__setattr__(runtime, "report_hash", stable_hash({item.name: getattr(runtime, item.name) for item in fields(runtime) if item.name != "report_hash"}))
        report = preflight(tmp_path, runtime_validation_report=runtime)
        assert not report.passed
        assert "runtime_fake_success" in report.failure_reasons


def test_preflight_rejects_runtime_serving_hash_mismatch() -> None:
    with temp_root("runtime_serving_mismatch") as tmp_path:
        _, _, runtime, _, _, _ = inputs(tmp_path)
        object.__setattr__(runtime, "serving_config_hash", "wrong-runtime-serving")
        object.__setattr__(runtime, "report_hash", stable_hash({item.name: getattr(runtime, item.name) for item in fields(runtime) if item.name != "report_hash"}))
        report = preflight(tmp_path, runtime_validation_report=runtime)
        assert not report.passed
        assert "runtime_serving_hash_mismatch" in report.failure_reasons


def test_preflight_rejects_artifact_audit_failure() -> None:
    with temp_root("audit_failure") as tmp_path:
        man, package, runtime, _, promo, sub = inputs(tmp_path)
        bad_audit = audit_artifacts(ArtifactAuditConfig(file_entries=man.files + (man.files[0],)))
        report = evaluate_preflight_gate(
            submission_manifest=man,
            package_build_report=package,
            runtime_validation_report=runtime,
            artifact_audit_report=bad_audit,
            promotion_gate_report=promo,
            submission_exact_report=sub,
            config=PreflightGateConfig(allow_package_dry_run=True, allow_runtime_dry_run_unloaded=True, allow_warnings=True),
        )
        assert not report.passed
        assert "artifact_audit_failed" in report.failure_reasons


def test_preflight_rejects_missing_required_hash() -> None:
    with temp_root("missing_hash") as tmp_path:
        bad_manifest = manifest(tmp_path, adapter_hash=None)
        package = build_submission_package(PackageBuildConfig(root_path=tmp_path, manifest=bad_manifest))
        runtime = validate_offline_runtime(RuntimeValidationConfig(serving_config=serving(), dry_run=True, dry_run_allow_unloaded=True))
        audit = audit_artifacts(ArtifactAuditConfig(root_path=tmp_path, file_entries=bad_manifest.files))
        report = evaluate_preflight_gate(
            submission_manifest=bad_manifest,
            package_build_report=package,
            runtime_validation_report=runtime,
            artifact_audit_report=audit,
            promotion_gate_report=promotion(),
            submission_exact_report=submission_exact(),
            config=PreflightGateConfig(allow_package_dry_run=True, allow_runtime_dry_run_unloaded=True, allow_warnings=True),
        )
        assert not report.passed
        assert "missing_hash:adapter_hash" in report.failure_reasons


def test_preflight_rejects_warning_when_disallowed() -> None:
    with temp_root("warning") as tmp_path:
        report = evaluate_preflight_gate(
            **dict(zip(("submission_manifest", "package_build_report", "runtime_validation_report", "artifact_audit_report", "promotion_gate_report", "submission_exact_report"), inputs(tmp_path))),
            config=PreflightGateConfig(allow_package_dry_run=True, allow_runtime_dry_run_unloaded=True, allow_warnings=False),
        )
        assert not report.passed
        assert "warnings_not_allowed" in report.failure_reasons


def test_forged_preflight_gate_hash_rejected() -> None:
    with temp_root("forged_gate") as tmp_path:
        report = preflight(tmp_path)
        with pytest.raises(PreflightGateError):
            replace(report, gate_hash="forged")


def test_preflight_report_rejects_passed_true_with_missing_required_hashes() -> None:
    with pytest.raises(PreflightGateError):
        PreflightGateReport(
            passed=True,
            failure_reasons=(),
            input_report_hashes={"input": "hash"},
            required_hashes={"adapter_hash": ""},
            warnings_allowed=True,
        )


def test_preflight_report_rejects_passed_true_with_unallowed_warnings() -> None:
    with pytest.raises(PreflightGateError):
        PreflightGateReport(
            passed=True,
            failure_reasons=(),
            input_report_hashes={"input": "hash"},
            required_hashes={"adapter_hash": "adapter"},
            warnings=("warning",),
            warnings_allowed=False,
        )


def test_preflight_report_rejects_passed_true_with_failure_reasons() -> None:
    with pytest.raises(PreflightGateError):
        PreflightGateReport(
            passed=True,
            failure_reasons=("failure",),
            input_report_hashes={"input": "hash"},
            required_hashes={"adapter_hash": "adapter"},
            warnings_allowed=True,
        )


def test_passed_true_with_failed_inputs_rejected() -> None:
    with temp_root("failed_inputs") as tmp_path:
        package = replace(inputs(tmp_path)[1], passed=False, errors=("bad",), report_hash="")
        report = preflight(tmp_path, package_build_report=package)
        assert not report.passed
        assert "package_build_failed" in report.failure_reasons
