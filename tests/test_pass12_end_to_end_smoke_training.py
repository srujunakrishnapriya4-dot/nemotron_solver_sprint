from __future__ import annotations

import json

from nemotron_engine.parsing import canonicalize_prompt
from nemotron_engine.smoke_training import (
    SmokeBackendResult,
    SmokeRunConfig,
    SmokeTrainingConfig,
    build_smoke_handoff_to_pass11,
    build_smoke_training_report,
    build_tiny_sft_smoke_dataset,
    run_smoke_training,
)
from nemotron_engine.solvers import solve_numeric
from nemotron_engine.traces import compile_trace_from_solver_result
from nemotron_engine.training import build_sft_row
from nemotron_engine.training.lora_config import LoRAConfig
from nemotron_engine.training.train_sft import SFTTrainingConfig, plan_sft_training
from nemotron_engine.training_backend import (
    AdapterValidationConfig,
    build_invocation_from_sft_plan,
    validate_adapter_output,
    validate_training_log,
)


SHA = "e" * 64


def write_adapter(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 8}), encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"x")
    return adapter


def backend_payload(inv):
    return {
        "result_id": "res",
        "invocation_hash": inv.invocation_hash,
        "backend_kind": inv.backend_kind,
        "stage": inv.stage,
        "status": "success",
        "trained": True,
        "adapter_path": "adapter",
        "checkpoint_path": None,
        "artifacts": [
            {"path": "adapter/adapter_model.safetensors", "role": "adapter_model", "sha256": SHA, "size_bytes": 1, "required": True}
        ],
        "metrics": [{"name": "loss", "value": 1.0, "step": 0, "split": None, "metadata": {}}],
        "started": True,
        "finished": True,
        "error": None,
        "metadata": {},
    }


def build_pass7_invocation() -> object:
    problem = canonicalize_prompt("p", "1 -> 2\n2 -> 3\n4 -> ?")
    trace = compile_trace_from_solver_result(solve_numeric(problem), prompt=problem.raw_prompt)
    row = build_sft_row(trace)
    lora = LoRAConfig("base", ("q_proj",), 8, 16, 0.0)
    plan, manifest, _ = plan_sft_training(
        [row],
        SFTTrainingConfig(lora, max_steps=2, dry_run=False, output_dir="out/sft"),
    )
    return build_invocation_from_sft_plan(plan, manifest)


def test_end_to_end_smoke_training_boundary_without_packaging_submission_or_promotion(tmp_path) -> None:
    smoke_dataset = build_tiny_sft_smoke_dataset()
    invocation = build_pass7_invocation()
    adapter = write_adapter(tmp_path)
    log = [{"step": 0, "loss": 1.0, "grad_norm": 1.0}]
    adapter_report = validate_adapter_output(adapter)
    log_report = validate_training_log(log)
    assert adapter_report.passed
    assert log_report.passed

    run_config = SmokeRunConfig(
        smoke_config=SmokeTrainingConfig(
            stage="sft",
            max_examples=1,
            max_steps=2,
            max_runtime_seconds=60,
            dry_run=False,
            output_dir="out/sft",
            require_log_validation=True,
        ),
        smoke_dataset=smoke_dataset,
        invocation=invocation,
        adapter_validation_config=AdapterValidationConfig(adapter),
    )

    def caller_backend(inv):
        return SmokeBackendResult(backend_payload(inv), adapter_dir=str(adapter), training_log=log)

    smoke_run = run_smoke_training(run_config, caller_backend)
    assert smoke_run.passed
    assert smoke_run.trained

    smoke_report = build_smoke_training_report(
        smoke_config_hash=smoke_run.smoke_config_hash,
        smoke_dataset_hash=smoke_run.smoke_dataset_hash,
        backend_invocation_hash=smoke_run.backend_invocation_hash,
        backend_result_hash=smoke_run.backend_result_hash,
        adapter_validation_hash=smoke_run.adapter_validation_hash,
        training_log_hash=smoke_run.training_log_hash,
        training_run_manifest_hash=smoke_run.training_run_manifest_hash,
        trained=smoke_run.trained,
        passed=smoke_run.passed,
        errors=smoke_run.errors,
        warnings=smoke_run.warnings,
        metadata={"require_log_validation": True},
    )
    handoff = build_smoke_handoff_to_pass11(smoke_report, adapter_hash=adapter_report.adapter_hash or "", require_log_validation=True)
    assert smoke_run.report_hash == run_smoke_training(run_config, caller_backend).report_hash
    assert smoke_report.report_hash == build_smoke_training_report(
        smoke_config_hash=smoke_run.smoke_config_hash,
        smoke_dataset_hash=smoke_run.smoke_dataset_hash,
        backend_invocation_hash=smoke_run.backend_invocation_hash,
        backend_result_hash=smoke_run.backend_result_hash,
        adapter_validation_hash=smoke_run.adapter_validation_hash,
        training_log_hash=smoke_run.training_log_hash,
        training_run_manifest_hash=smoke_run.training_run_manifest_hash,
        trained=smoke_run.trained,
        passed=smoke_run.passed,
        errors=smoke_run.errors,
        warnings=smoke_run.warnings,
        metadata={"require_log_validation": True},
    ).report_hash
    assert handoff.training_run_manifest_hash == smoke_run.training_run_manifest_hash
    assert "promote" not in str(handoff.metadata).lower()
    assert "package" not in str(handoff.metadata).lower()
    assert "submit" not in str(handoff.metadata).lower()
    assert not (tmp_path / "submission.zip").exists()


def test_fake_success_without_adapter_fails(tmp_path) -> None:
    smoke_dataset = build_tiny_sft_smoke_dataset()
    invocation = build_pass7_invocation()
    adapter = write_adapter(tmp_path)
    run_config = SmokeRunConfig(
        smoke_config=SmokeTrainingConfig(
            stage="sft",
            max_examples=1,
            max_steps=2,
            max_runtime_seconds=60,
            dry_run=False,
            output_dir="out/sft",
            require_log_validation=True,
        ),
        smoke_dataset=smoke_dataset,
        invocation=invocation,
        adapter_validation_config=AdapterValidationConfig(adapter),
    )

    def fake_backend(inv):
        payload = backend_payload(inv)
        payload["adapter_path"] = None
        payload["artifacts"] = []
        return {"backend_result": payload, "training_log": [{"step": 0, "loss": 1.0}]}

    report = run_smoke_training(run_config, fake_backend)
    assert not report.passed
    assert not report.trained
    assert report.training_run_manifest_hash is None
