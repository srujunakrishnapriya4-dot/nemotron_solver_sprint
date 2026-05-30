from __future__ import annotations

import pytest

from nemotron_engine.parsing import canonicalize_prompt
from nemotron_engine.solvers import solve_binary, solve_numeric
from nemotron_engine.traces import audit_trace, compile_trace_from_solver_result
from nemotron_engine.training.dpo_builder import build_dpo_dataset
from nemotron_engine.training.negative_generator import ALLOWED_NEGATIVE_TYPES, generate_negative_for_trace, generate_negative_traces
from nemotron_engine.training.sft_builder import SFTBuildError, build_sft_row


def test_end_to_end_sft_and_dpo_builders() -> None:
    problem = canonicalize_prompt("p", "1 -> 2\n2 -> 3\n4 -> ?")
    result = solve_numeric(problem)
    trace = compile_trace_from_solver_result(result, prompt=problem.raw_prompt)
    audit = audit_trace(trace, proof=result.best_attempt.proof)

    row = build_sft_row(trace, audit=audit)
    negatives, negative_report = generate_negative_traces([trace], negative_types=ALLOWED_NEGATIVE_TYPES, max_negatives_per_trace=7)
    pairs, dpo_report = build_dpo_dataset([trace], [item.trace for item in negatives], prompts={"p": problem.raw_prompt})
    pairs_again, _ = build_dpo_dataset([trace], [item.trace for item in negatives], prompts={"p": problem.raw_prompt})

    assert row.row_id
    assert negative_report.generated_count == len(ALLOWED_NEGATIVE_TYPES)
    assert pairs
    assert dpo_report.rejected_pair_count == sum(dpo_report.rejection_reasons.values())
    assert [pair.pair_id for pair in pairs] == [pair.pair_id for pair in pairs_again]
    assert [pair.pair_hash for pair in pairs] == [pair.pair_hash for pair in pairs_again]
    assert dpo_report.accepted_pair_count == len(pairs)


def test_generated_target_leak_and_all_negatives_cannot_become_sft() -> None:
    problem = canonicalize_prompt("p", "1 -> 2\n2 -> 3\n4 -> ?")
    trace = compile_trace_from_solver_result(solve_numeric(problem), prompt=problem.raw_prompt)

    target_leak = generate_negative_for_trace(trace, "target_leak")
    with pytest.raises(SFTBuildError):
        build_sft_row(target_leak.trace)

    negatives, _ = generate_negative_traces([trace], negative_types=ALLOWED_NEGATIVE_TYPES, max_negatives_per_trace=7)
    for candidate in negatives:
        with pytest.raises(SFTBuildError):
            build_sft_row(candidate.trace)


def test_invalid_or_ambiguous_solver_result_cannot_become_sft() -> None:
    problem = canonicalize_prompt("amb", "00 -> 00\n01 -> ?")
    result = solve_binary(problem)
    trace = compile_trace_from_solver_result(result, prompt=problem.raw_prompt)

    with pytest.raises(SFTBuildError):
        build_sft_row(trace)
