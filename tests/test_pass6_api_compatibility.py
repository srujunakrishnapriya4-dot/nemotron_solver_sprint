from __future__ import annotations


def test_pass6_imports() -> None:
    from nemotron_engine.training import (  # noqa: F401
        DPOPair,
        NegativeTraceCandidate,
        SFTDatasetRow,
        build_dpo_dataset,
        build_dpo_pair,
        build_sft_dataset,
        build_sft_row,
        generate_negative_traces,
    )


def test_representative_locked_pass_imports_still_work() -> None:
    from nemotron_engine.scoring.answer_extractor import extract_boxed_answer  # noqa: F401
    from nemotron_engine.core.schemas import ImmutableProblemRecord  # noqa: F401
    from nemotron_engine.solvers import solve_numeric  # noqa: F401
    from nemotron_engine.data import assign_splits  # noqa: F401
    from nemotron_engine.traces import compile_trace_from_solver_result  # noqa: F401
