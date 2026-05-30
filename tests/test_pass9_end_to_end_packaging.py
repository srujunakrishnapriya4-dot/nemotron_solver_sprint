from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.evaluation.promotion_gates import PromotionDecision, PromotionGateReport
from nemotron_engine.evaluation.submission_exact import SubmissionExactConfig, validate_submission_exact_config
from nemotron_engine.packaging.artifact_audit import ArtifactAuditConfig, audit_artifacts
from nemotron_engine.packaging.package_builder import PackageBuildConfig, build_submission_package
from nemotron_engine.packaging.preflight_gate import PreflightGateConfig, evaluate_preflight_gate
from nemotron_engine.packaging.runtime_validator import RuntimeValidationConfig, validate_offline_runtime
from nemotron_engine.packaging.submission_manifest import build_submission_manifest
from nemotron_engine.runtime.serving_config import ServingConfig


@contextmanager
def temp_root(name: str):
    root = Path(__file__).resolve().parent / ".tmp_pass9_end_to_end" / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def serving() -> ServingConfig:
    return ServingConfig(
        temperature=0.0,
        top_p=1.0,
        num_samples=1,
        majority_vote=False,
        max_tokens=512,
        prompt_template_hash="prompt-hash",
        tokenizer_hash="tokenizer-hash",
        model_hash="model-hash",
        adapter_hash="adapter-hash",
    )


def promotion() -> PromotionGateReport:
    return PromotionGateReport(
        decision=PromotionDecision.ALLOW,
        passed=True,
        failure_reasons=(),
        input_report_hashes={"pass8": "report-hash"},
        required_hashes={
            "training_plan_hash": "training-hash",
            "dataset_manifest_hash": "dataset-hash",
            "trace_manifest_hash": "trace-hash",
            "lora_config_hash": "lora-hash",
            "adapter_hash": "adapter-hash",
        },
    )


def test_end_to_end_dry_run_preflight_and_optional_zip_are_deterministic() -> None:
    with temp_root("green") as tmp_path:
        (tmp_path / "run.py").write_text("print('offline')\n", encoding="utf-8")
        exact = validate_submission_exact_config(
            SubmissionExactConfig(serving_config=serving(), adapter_config={"r": 8}, dry_run=True)
        )
        manifest = build_submission_manifest(
            root_path=tmp_path,
            package_name="pkg",
            created_by="pass9",
            files=[{"path": "run.py", "role": "entrypoint", "required": True}],
            serving_config_hash=exact.serving_config_hash,
            lora_config_hash="lora-hash",
            adapter_hash="adapter-hash",
            promotion_decision_hash=promotion().decision_hash,
            training_plan_hash="training-hash",
            dataset_manifest_hash="dataset-hash",
            trace_manifest_hash="trace-hash",
        )
        package = build_submission_package(PackageBuildConfig(root_path=tmp_path, manifest=manifest, dry_run=True))
        runtime = validate_offline_runtime(
            RuntimeValidationConfig(serving_config=serving(), dry_run=True, dry_run_allow_unloaded=True)
        )
        audit = audit_artifacts(ArtifactAuditConfig(root_path=tmp_path, file_entries=manifest.files))
        gate = evaluate_preflight_gate(
            submission_manifest=manifest,
            package_build_report=package,
            runtime_validation_report=runtime,
            artifact_audit_report=audit,
            promotion_gate_report=promotion(),
            submission_exact_report=exact,
            config=PreflightGateConfig(
                allow_package_dry_run=True,
                allow_runtime_dry_run_unloaded=True,
                allow_warnings=True,
            ),
        )
        again = evaluate_preflight_gate(
            submission_manifest=manifest,
            package_build_report=package,
            runtime_validation_report=runtime,
            artifact_audit_report=audit,
            promotion_gate_report=promotion(),
            submission_exact_report=exact,
            config=PreflightGateConfig(
                allow_package_dry_run=True,
                allow_runtime_dry_run_unloaded=True,
                allow_warnings=True,
            ),
        )
        assert gate.passed
        assert gate.gate_hash == again.gate_hash
        assert not runtime.runtime_loaded and not runtime.model_loaded and not runtime.adapter_loaded
        output = tmp_path / "package.zip"
        built = build_submission_package(
            PackageBuildConfig(root_path=tmp_path, manifest=manifest, output_path=output, build_zip=True, dry_run=False)
        )
        assert built.built
        with ZipFile(output, "r") as archive:
            assert "submission_manifest.json" in archive.namelist()
            assert "run.py" in archive.namelist()


def _green_inputs(tmp_path: Path):
    (tmp_path / "run.py").write_text("print('offline')\n", encoding="utf-8")
    exact = validate_submission_exact_config(
        SubmissionExactConfig(serving_config=serving(), adapter_config={"r": 8}, dry_run=True)
    )
    manifest = build_submission_manifest(
        root_path=tmp_path,
        package_name="pkg",
        created_by="pass9",
        files=[{"path": "run.py", "role": "entrypoint", "required": True}],
        serving_config_hash=exact.serving_config_hash,
        lora_config_hash="lora-hash",
        adapter_hash="adapter-hash",
        promotion_decision_hash=promotion().decision_hash,
        training_plan_hash="training-hash",
        dataset_manifest_hash="dataset-hash",
        trace_manifest_hash="trace-hash",
    )
    package = build_submission_package(PackageBuildConfig(root_path=tmp_path, manifest=manifest, dry_run=True))
    runtime = validate_offline_runtime(
        RuntimeValidationConfig(serving_config=serving(), dry_run=True, dry_run_allow_unloaded=True)
    )
    audit = audit_artifacts(ArtifactAuditConfig(root_path=tmp_path, file_entries=manifest.files))
    return manifest, package, runtime, audit, promotion(), exact


def test_end_to_end_rejects_manifest_promotion_hash_mismatch() -> None:
    with temp_root("promotion_mismatch") as tmp_path:
        manifest, _, runtime, audit, promo, exact = _green_inputs(tmp_path)
        bad_manifest = build_submission_manifest(
            root_path=tmp_path,
            package_name="pkg",
            created_by="pass9",
            files=[{"path": "run.py", "role": "entrypoint", "required": True}],
            serving_config_hash=exact.serving_config_hash,
            lora_config_hash="lora-hash",
            adapter_hash="adapter-hash",
            promotion_decision_hash="wrong-decision",
            training_plan_hash="training-hash",
            dataset_manifest_hash="dataset-hash",
            trace_manifest_hash="trace-hash",
        )
        package = build_submission_package(PackageBuildConfig(root_path=tmp_path, manifest=bad_manifest, dry_run=True))
        gate = evaluate_preflight_gate(
            submission_manifest=bad_manifest,
            package_build_report=package,
            runtime_validation_report=runtime,
            artifact_audit_report=audit,
            promotion_gate_report=promo,
            submission_exact_report=exact,
            config=PreflightGateConfig(allow_package_dry_run=True, allow_runtime_dry_run_unloaded=True, allow_warnings=True),
        )
        assert not gate.passed
        assert "promotion_decision_hash_mismatch" in gate.failure_reasons


def test_end_to_end_rejects_runtime_serving_hash_mismatch() -> None:
    with temp_root("runtime_mismatch") as tmp_path:
        manifest, package, runtime, audit, promo, exact = _green_inputs(tmp_path)
        object.__setattr__(runtime, "serving_config_hash", "wrong-runtime-serving")
        object.__setattr__(runtime, "report_hash", stable_hash({item.name: getattr(runtime, item.name) for item in fields(runtime) if item.name != "report_hash"}))
        gate = evaluate_preflight_gate(
            submission_manifest=manifest,
            package_build_report=package,
            runtime_validation_report=runtime,
            artifact_audit_report=audit,
            promotion_gate_report=promo,
            submission_exact_report=exact,
            config=PreflightGateConfig(allow_package_dry_run=True, allow_runtime_dry_run_unloaded=True, allow_warnings=True),
        )
        assert not gate.passed
        assert "runtime_serving_hash_mismatch" in gate.failure_reasons
from nemotron_engine.core.schemas import stable_hash
