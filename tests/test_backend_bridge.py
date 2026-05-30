from __future__ import annotations

from dataclasses import replace
import json

import pytest

from nemotron_engine.parsing import canonicalize_prompt
from nemotron_engine.solvers import solve_numeric
from nemotron_engine.traces import compile_trace_from_solver_result
from nemotron_engine.training import build_sft_row
from nemotron_engine.training.dpo_builder import build_dpo_pair
from nemotron_engine.training.lora_config import LoRAConfig
from nemotron_engine.training.negative_generator import generate_negative_for_trace
from nemotron_engine.training.reward_audit import audit_reward_model_signals
from nemotron_engine.training.training_contracts import TrainingInputManifest, TrainingPlan
from nemotron_engine.training.train_dpo import DPOTrainingConfig, plan_dpo_training
from nemotron_engine.training.train_grpo import GRPOTrainingConfig, plan_grpo_training
from nemotron_engine.training.train_sft import SFTTrainingConfig, plan_sft_training
from nemotron_engine.training_backend.adapter_validator import validate_adapter_output
from nemotron_engine.training_backend.backend_bridge import (
    BackendBridgeError,
    build_invocation_from_dpo_plan,
    build_invocation_from_grpo_plan,
    build_invocation_from_sft_plan,
    build_run_manifest_from_backend_outputs,
)
from nemotron_engine.training_backend.backend_contracts import BackendArtifactRef, BackendResult
from nemotron_engine.training_backend.training_log_validator import validate_training_log


SHA = "d" * 64


def row_pair():
    problem = canonicalize_prompt("p", "1 -> 2\n2 -> 3\n4 -> ?")
    trace = compile_trace_from_solver_result(solve_numeric(problem), prompt=problem.raw_prompt)
    row = build_sft_row(trace)
    neg = generate_negative_for_trace(trace, "false_rule").trace
    pair = build_dpo_pair(prompt=problem.raw_prompt, chosen_trace=trace, rejected_trace=neg, negative_type="false_rule")
    return row, pair


def lora(rank=8) -> LoRAConfig:
    return LoRAConfig("base", ("q_proj",), rank, 16, 0.0)


def test_builds_sft_invocation_from_pass7_plan() -> None:
    row, _ = row_pair()
    plan, manifest, _ = plan_sft_training([row], SFTTrainingConfig(lora(), dry_run=False))
    inv = build_invocation_from_sft_plan(plan, manifest)
    assert inv.stage == "sft"
    assert inv.input_row_count == 1
    assert inv.training_plan_hash == plan.plan_hash


def test_builds_dpo_invocation_from_pass7_plan() -> None:
    _, pair = row_pair()
    plan, manifest, _ = plan_dpo_training([pair], DPOTrainingConfig(lora(), dry_run=False))
    inv = build_invocation_from_dpo_plan(plan, manifest)
    assert inv.stage == "dpo"
    assert inv.input_pair_count == 1


def test_grpo_invocation_requires_accepted_reward_audit() -> None:
    rejected = audit_reward_model_signals(
        sample_count=1,
        false_positive_rate=0.0,
        format_failure_rate=0.0,
        leakage_reward_rate=0.0,
        length_gaming_rate=0.0,
        heldout_regression_rate=0.0,
    )
    with pytest.raises(Exception):
        plan_grpo_training(rejected, GRPOTrainingConfig(lora(), enabled=True))
    accepted = audit_reward_model_signals(
        sample_count=100,
        false_positive_rate=0.0,
        format_failure_rate=0.0,
        leakage_reward_rate=0.0,
        length_gaming_rate=0.0,
        heldout_regression_rate=0.0,
    )
    plan, _ = plan_grpo_training(accepted, GRPOTrainingConfig(lora(), enabled=True, dry_run=False))
    inv = build_invocation_from_grpo_plan(plan, accepted)
    assert inv.stage == "grpo"
    with pytest.raises(BackendBridgeError):
        build_invocation_from_grpo_plan(plan, rejected)


def test_rejects_raw_dict_plans_and_rank_over_32() -> None:
    with pytest.raises(BackendBridgeError):
        build_invocation_from_sft_plan({"stage": "sft"})  # type: ignore[arg-type]
    with pytest.raises(Exception):
        lora(33)


def forged_dataclass(instance, field_name: str, value: object):
    clone = object.__new__(type(instance))
    for name in type(instance).__dataclass_fields__:
        object.__setattr__(clone, name, getattr(instance, name))
    object.__setattr__(clone, field_name, value)
    return clone


def test_forged_training_plan_and_manifest_hashes_rejected() -> None:
    row, _ = row_pair()
    plan, manifest, _ = plan_sft_training([row], SFTTrainingConfig(lora(), dry_run=False))
    with pytest.raises(BackendBridgeError, match="plan_hash"):
        build_invocation_from_sft_plan(forged_dataclass(plan, "plan_hash", "forged"), manifest)
    with pytest.raises(BackendBridgeError, match="manifest_hash"):
        build_invocation_from_sft_plan(plan, forged_dataclass(manifest, "manifest_hash", "forged"))


def test_forged_reward_audit_hash_rejected_for_grpo() -> None:
    accepted = audit_reward_model_signals(
        sample_count=100,
        false_positive_rate=0.0,
        format_failure_rate=0.0,
        leakage_reward_rate=0.0,
        length_gaming_rate=0.0,
        heldout_regression_rate=0.0,
    )
    plan, _ = plan_grpo_training(accepted, GRPOTrainingConfig(lora(), enabled=True, dry_run=False))
    with pytest.raises(BackendBridgeError, match="reward audit"):
        build_invocation_from_grpo_plan(plan, forged_dataclass(accepted, "report_hash", "forged"))


def test_invocation_hashes_deterministic() -> None:
    row, _ = row_pair()
    plan, manifest, _ = plan_sft_training([row], SFTTrainingConfig(lora(), dry_run=False))
    assert build_invocation_from_sft_plan(plan, manifest).invocation_hash == build_invocation_from_sft_plan(plan, manifest).invocation_hash


def test_run_manifest_built_only_from_validated_outputs(tmp_path) -> None:
    row, _ = row_pair()
    plan, manifest, _ = plan_sft_training([row], SFTTrainingConfig(lora(), dry_run=False))
    inv = build_invocation_from_sft_plan(plan, manifest)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 8}), encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"x")
    adapter_report = validate_adapter_output(adapter)
    log_report = validate_training_log([{"step": 0, "loss": 1.0}])
    result = BackendResult(
        "res",
        inv.invocation_hash,
        inv.backend_kind,
        inv.stage,
        "success",
        True,
        "adapter",
        None,
        (BackendArtifactRef("adapter/adapter_model.safetensors", "adapter_model", SHA, 1, True),),
        (),
        True,
        True,
        None,
        {},
    )
    manifest11 = build_run_manifest_from_backend_outputs(invocation=inv, backend_result=result, adapter_report=adapter_report, log_report=log_report)
    assert manifest11.trained
    with pytest.raises(Exception):
        build_run_manifest_from_backend_outputs(invocation=inv, backend_result=result, adapter_report=None, log_report=log_report)


def test_run_manifest_bridge_rejects_failed_adapter_report_and_invocation_mismatch(tmp_path) -> None:
    row, _ = row_pair()
    plan, manifest, _ = plan_sft_training([row], SFTTrainingConfig(lora(), dry_run=False))
    inv = build_invocation_from_sft_plan(plan, manifest)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 64}), encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"x")
    failed_adapter_report = validate_adapter_output(adapter)
    result = BackendResult(
        "res",
        inv.invocation_hash,
        inv.backend_kind,
        inv.stage,
        "success",
        True,
        "adapter",
        None,
        (BackendArtifactRef("adapter/adapter_model.safetensors", "adapter_model", SHA, 1, True),),
        (),
        True,
        True,
        None,
        {},
    )
    with pytest.raises(BackendBridgeError, match="adapter report"):
        build_run_manifest_from_backend_outputs(invocation=inv, backend_result=result, adapter_report=failed_adapter_report)
    other_inv = build_invocation_from_sft_plan(plan, manifest, invocation_id="other")
    with pytest.raises(Exception, match="invocation"):
        build_run_manifest_from_backend_outputs(invocation=other_inv, backend_result=result, adapter_report=failed_adapter_report)
