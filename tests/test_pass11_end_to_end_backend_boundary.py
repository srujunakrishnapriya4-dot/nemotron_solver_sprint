from __future__ import annotations

import json

import pytest

from nemotron_engine.parsing import canonicalize_prompt
from nemotron_engine.solvers import solve_numeric
from nemotron_engine.traces import compile_trace_from_solver_result
from nemotron_engine.training import build_sft_row
from nemotron_engine.training.lora_config import LoRAConfig
from nemotron_engine.training.train_sft import SFTTrainingConfig, plan_sft_training
from nemotron_engine.training_backend.adapter_validator import AdapterValidationConfig, validate_adapter_output
from nemotron_engine.training_backend.backend_bridge import build_invocation_from_sft_plan, build_run_manifest_from_backend_outputs
from nemotron_engine.training_backend.backend_runner import invoke_training_backend
from nemotron_engine.training_backend.training_log_validator import validate_training_log


SHA = "e" * 64


def test_end_to_end_backend_boundary_without_training_or_kaggle(tmp_path) -> None:
    problem = canonicalize_prompt("p", "1 -> 2\n2 -> 3\n4 -> ?")
    trace = compile_trace_from_solver_result(solve_numeric(problem), prompt=problem.raw_prompt)
    row = build_sft_row(trace)
    lora = LoRAConfig("base", ("q_proj",), 8, 16, 0.0)
    plan, manifest, _ = plan_sft_training([row], SFTTrainingConfig(lora, dry_run=False, output_dir="out/sft"))
    invocation = build_invocation_from_sft_plan(plan, manifest)

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 8}), encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"x")

    called = {"count": 0}

    def external_backend(inv):
        called["count"] += 1
        return {
            "result_id": "res",
            "invocation_hash": inv.invocation_hash,
            "backend_kind": inv.backend_kind,
            "stage": inv.stage,
            "status": "success",
            "trained": True,
            "adapter_path": "adapter",
            "checkpoint_path": None,
            "artifacts": [{"path": "adapter/adapter_model.safetensors", "role": "adapter_model", "sha256": SHA, "size_bytes": 1, "required": True}],
            "metrics": [{"name": "loss", "value": 1.0, "step": 0, "split": None, "metadata": {}}],
            "started": True,
            "finished": True,
            "error": None,
            "metadata": {},
        }

    log = [{"step": 0, "loss": 1.0, "grad_norm": 1.0}]
    run_report = invoke_training_backend(
        invocation,
        external_backend,
        adapter_validation_config=AdapterValidationConfig(adapter),
        training_log=log,
    )
    assert called["count"] == 1
    assert run_report.trained
    adapter_report = validate_adapter_output(adapter)
    log_report = validate_training_log(log)
    manifest11 = build_run_manifest_from_backend_outputs(
        invocation=invocation,
        backend_result=run_report.backend_result,
        adapter_report=adapter_report,
        log_report=log_report,
    )
    assert manifest11.manifest_hash == build_run_manifest_from_backend_outputs(
        invocation=invocation,
        backend_result=run_report.backend_result,
        adapter_report=adapter_report,
        log_report=log_report,
    ).manifest_hash

    def fake_backend(inv):
        payload = external_backend(inv)
        payload["adapter_path"] = None
        payload["artifacts"] = []
        return payload

    with pytest.raises(Exception):
        invoke_training_backend(invocation, fake_backend, adapter_validation_config=AdapterValidationConfig(adapter))

    def kaggle_backend(inv):
        payload = external_backend(inv)
        payload["metadata"] = {"kaggle_success": True}
        return payload

    with pytest.raises(Exception):
        invoke_training_backend(invocation, kaggle_backend, adapter_validation_config=AdapterValidationConfig(adapter))
