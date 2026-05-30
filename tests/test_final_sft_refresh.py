from __future__ import annotations

from dataclasses import replace

import pytest

from nemotron_engine.parsing import canonicalize_prompt
from nemotron_engine.solvers import solve_numeric
from nemotron_engine.traces import compile_trace_from_solver_result
from nemotron_engine.training import build_sft_row
from nemotron_engine.training.final_sft_refresh import FinalSFTRefreshConfig, FinalSFTRefreshError, plan_final_sft_refresh
from nemotron_engine.training.lora_config import LoRAConfig


def row(**metadata):
    problem = canonicalize_prompt("p", "1 -> 2\n2 -> 3\n4 -> ?")
    trace = compile_trace_from_solver_result(solve_numeric(problem), prompt=problem.raw_prompt)
    return build_sft_row(trace, metadata=metadata)


def lora(rank: int = 8) -> LoRAConfig:
    return LoRAConfig("base", ("q_proj",), rank, 16, 0.0)


def test_clean_rows_produce_refresh_plan_and_hash_deterministic() -> None:
    first_plan, first = plan_final_sft_refresh([row()], FinalSFTRefreshConfig(lora()))
    second_plan, second = plan_final_sft_refresh([row()], FinalSFTRefreshConfig(lora()))

    assert first_plan.plan_hash == second_plan.plan_hash
    assert first.report_hash == second.report_hash
    assert not first.trained
    assert first.adapter_path is None


def test_contaminated_or_eval_rows_and_rank_over_32_rejected() -> None:
    with pytest.raises(FinalSFTRefreshError):
        plan_final_sft_refresh([row(contamination_flags=("x",))], FinalSFTRefreshConfig(lora()))
    with pytest.raises(FinalSFTRefreshError):
        plan_final_sft_refresh([row(split="private_like")], FinalSFTRefreshConfig(lora()))
    with pytest.raises(Exception):
        lora(33)


def test_forged_refresh_hash_rejected() -> None:
    _, report = plan_final_sft_refresh([row()], FinalSFTRefreshConfig(lora()))
    with pytest.raises(FinalSFTRefreshError, match="report_hash"):
        replace(report, report_hash="forged")
