from __future__ import annotations

from dataclasses import replace

import pytest

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.parsing import canonicalize_prompt
from nemotron_engine.solvers import solve_numeric
from nemotron_engine.traces import compile_trace_from_solver_result
from nemotron_engine.training import build_sft_row
from nemotron_engine.training.dpo_builder import build_dpo_pair
from nemotron_engine.training.lora_config import LoRAConfig
from nemotron_engine.training.negative_generator import generate_negative_for_trace
from nemotron_engine.training.training_contracts import (
    TrainingContractError,
    TrainingDatasetContract,
    TrainingInputManifest,
    TrainingPlan,
    TrainingRunReport,
    build_training_input_manifest,
    build_training_plan,
    validate_dpo_pairs,
    validate_sft_rows,
)


def lora() -> LoRAConfig:
    return LoRAConfig("base", ("q_proj",), 8, 16, 0.0)


def sft_row(**metadata):
    problem = canonicalize_prompt("p", "1 -> 2\n2 -> 3\n4 -> ?")
    trace = compile_trace_from_solver_result(solve_numeric(problem), prompt=problem.raw_prompt)
    return build_sft_row(trace, metadata=metadata)


def dpo_pair():
    row_trace = compile_trace_from_solver_result(solve_numeric(canonicalize_prompt("p", "1 -> 2\n2 -> 3\n4 -> ?")), prompt="prompt")
    neg = generate_negative_for_trace(row_trace, "false_rule").trace
    return build_dpo_pair(prompt="prompt", chosen_trace=row_trace, rejected_trace=neg, negative_type="false_rule")


def test_validates_sft_rows_and_rejects_duplicates() -> None:
    row = sft_row()
    assert validate_sft_rows([row]) == (row,)
    with pytest.raises(TrainingContractError, match="duplicate row_id"):
        validate_sft_rows([row, row])


def test_rejects_contaminated_negative_and_eval_sft_metadata() -> None:
    with pytest.raises(TrainingContractError, match="contaminated"):
        validate_sft_rows([sft_row(contamination_flags=("x",))])
    with pytest.raises(TrainingContractError, match="negative"):
        validate_sft_rows([sft_row(is_negative=True)])
    with pytest.raises(TrainingContractError, match="eval-only"):
        validate_sft_rows([sft_row(split="private_like")])


def test_validates_dpo_pairs_and_rejects_duplicate_or_unsupported_negative_type() -> None:
    pair = dpo_pair()
    assert validate_dpo_pairs([pair]) == (pair,)
    with pytest.raises(TrainingContractError, match="duplicate pair_id"):
        validate_dpo_pairs([pair, pair])
    bad = object.__new__(type(pair))
    for name in type(pair).__dataclass_fields__:
        object.__setattr__(bad, name, getattr(pair, name))
    object.__setattr__(bad, "negative_type", "unknown")
    with pytest.raises(TrainingContractError, match="unsupported"):
        validate_dpo_pairs([bad])


def test_rejects_bad_row_and_pair_hashes() -> None:
    row = object.__new__(type(sft_row()))
    source_row = sft_row()
    for name in type(source_row).__dataclass_fields__:
        object.__setattr__(row, name, getattr(source_row, name))
    object.__setattr__(row, "row_hash", "bad")
    with pytest.raises(TrainingContractError, match="bad row_hash"):
        validate_sft_rows([row])

    source_pair = dpo_pair()
    pair = object.__new__(type(source_pair))
    for name in type(source_pair).__dataclass_fields__:
        object.__setattr__(pair, name, getattr(source_pair, name))
    object.__setattr__(pair, "pair_hash", "bad")
    with pytest.raises(TrainingContractError, match="bad pair_hash"):
        validate_dpo_pairs([pair])


def test_mixture_and_empty_dataset_gates() -> None:
    with pytest.raises(TrainingContractError, match="sum"):
        TrainingDatasetContract("sft", 1, mixture={"a": 0.4, "b": 0.5})
    with pytest.raises(TrainingContractError, match="empty"):
        validate_sft_rows([])
    assert validate_sft_rows([], allow_empty_dry_run=True) == ()


def test_manifest_plan_and_run_report_hash_integrity() -> None:
    manifest = build_training_input_manifest([sft_row()], dataset_type="sft")
    with pytest.raises(TrainingContractError, match="dataset_hash"):
        replace(manifest, dataset_hash="forged")
    with pytest.raises(TrainingContractError, match="manifest_hash"):
        replace(manifest, manifest_hash="forged")

    plan = build_training_plan(
        stage="sft",
        manifest=manifest,
        lora_config=lora(),
        seed=1,
        max_steps=1,
        batch_size=1,
        learning_rate=1e-4,
        gradient_accumulation_steps=1,
        precision="bf16",
        dry_run=True,
        output_dir="out",
    )
    with pytest.raises(TrainingContractError, match="plan_hash"):
        replace(plan, plan_hash="forged")

    report = TrainingRunReport("run", plan.plan_hash, "sft", False, True)
    with pytest.raises(TrainingContractError, match="report_hash"):
        replace(report, report_hash="forged")


def test_shared_backend_report_adapter_config_rank_rejected() -> None:
    manifest = build_training_input_manifest([sft_row()], dataset_type="sft")
    plan = build_training_plan(
        stage="sft",
        manifest=manifest,
        lora_config=lora(),
        seed=1,
        max_steps=1,
        batch_size=1,
        learning_rate=1e-4,
        gradient_accumulation_steps=1,
        precision="bf16",
        dry_run=False,
        output_dir="out",
    )
    report = TrainingRunReport("run", plan.plan_hash, "sft", True, False, adapter_path="real", metadata={"adapter_config": {"r": 64}})
    from nemotron_engine.training.training_contracts import validate_backend_report

    with pytest.raises(TrainingContractError, match="adapter_config"):
        validate_backend_report(report, expected_plan=plan)
