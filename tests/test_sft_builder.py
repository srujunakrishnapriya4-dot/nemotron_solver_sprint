from __future__ import annotations

import pytest

from nemotron_engine.core.schemas import ImmutableProblemRecord, ProblemSource, SplitName
from nemotron_engine.parsing import canonicalize_prompt
from nemotron_engine.solvers import solve_numeric
from nemotron_engine.traces import TraceAuditReport, TraceRecord, audit_trace, compile_rejection_trace, compile_trace_from_solver_result
from nemotron_engine.training.sft_builder import SFTBuildError, SFTBuildReport, SFTDatasetRow, build_sft_dataset, build_sft_row


def valid_trace() -> TraceRecord:
    problem = canonicalize_prompt("p", "1 -> 2\n2 -> 3\n4 -> ?")
    result = solve_numeric(problem)
    return compile_trace_from_solver_result(result, prompt=problem.raw_prompt)


def record(**updates: object) -> ImmutableProblemRecord:
    data = {
        "problem_id": "p",
        "source": ProblemSource.MANUAL_FIXTURE,
        "source_hash": "hash",
        "family_id": "fam",
        "primitive_family_id": "prim",
        "format_family_id": "fmt",
        "prompt_wrapper_id": "wrap",
        "split": SplitName.TRAIN,
        "allowed_for_sft": True,
        "allowed_for_dpo": True,
        "allowed_for_eval": True,
    }
    data.update(updates)
    return ImmutableProblemRecord(**data)


def forged_trace(base: TraceRecord, **updates: object) -> TraceRecord:
    forged = object.__new__(TraceRecord)
    for name in TraceRecord.__dataclass_fields__:
        object.__setattr__(forged, name, getattr(base, name))
    for key, value in updates.items():
        object.__setattr__(forged, key, value)
    return forged


def test_builds_sft_row_from_valid_positive_audited_trace() -> None:
    trace = valid_trace()
    row = build_sft_row(trace, audit=audit_trace(trace), problem_record=record())

    assert row.problem_id == "p"
    assert row.prompt
    assert row.source_proof_hash == trace.source_proof_hash


def test_rejects_negative_or_rejection_trace() -> None:
    trace = compile_rejection_trace("p", "no solution", prompt="prompt")

    with pytest.raises(SFTBuildError, match="trace_not_positive"):
        build_sft_row(trace)


def test_rejects_valid_looking_completion_with_invalid_audit() -> None:
    trace = valid_trace()
    invalid = TraceAuditReport(trace.trace_id, valid=False, errors=("forced_invalid",))

    with pytest.raises(SFTBuildError, match="invalid_audit"):
        build_sft_row(trace, audit=invalid)


def test_rejects_trace_with_boxed_answer_target_mismatch() -> None:
    trace = forged_trace(valid_trace(), target_output="different")

    with pytest.raises(SFTBuildError, match="boxed_answer_target_mismatch"):
        build_sft_row(trace)


def test_rejects_missing_source_hashes() -> None:
    trace = valid_trace()

    with pytest.raises(SFTBuildError, match="missing_source_proof_hash"):
        build_sft_row(forged_trace(trace, source_proof_hash=None))
    with pytest.raises(SFTBuildError, match="missing_source_program_hash"):
        build_sft_row(forged_trace(trace, source_program_hash=None))


def test_rejects_trace_with_no_prompt() -> None:
    trace = forged_trace(valid_trace(), prompt=None)

    with pytest.raises(SFTBuildError, match="missing_prompt"):
        build_sft_row(trace)


def test_rejects_contaminated_and_non_train_problem_records() -> None:
    trace = valid_trace()

    with pytest.raises(SFTBuildError, match="problem_record_contaminated"):
        build_sft_row(trace, problem_record=record(contamination_flags=("prompt_collision",)))
    for split in (SplitName.DEV, SplitName.HARD_DEV, SplitName.PRIVATE_LIKE):
        with pytest.raises(SFTBuildError, match="problem_record_not_train"):
            build_sft_row(trace, problem_record=record(split=split))
    for split in (SplitName.FORBIDDEN_HOLDOUT, SplitName.STRESS_ONLY):
        with pytest.raises(SFTBuildError):
            build_sft_row(trace, problem_record=record(split=split, allowed_for_sft=False, allowed_for_dpo=False))


def test_rejects_negative_metadata_even_if_trace_shape_is_positive() -> None:
    trace = forged_trace(valid_trace(), metadata={"negative_type": "target_leak", "is_negative": True})

    with pytest.raises(SFTBuildError, match="trace_marked_negative"):
        build_sft_row(trace)


def test_row_id_and_hash_are_deterministic() -> None:
    trace = valid_trace()

    first = build_sft_row(trace, problem_record=record(), metadata={"b": 2, "a": 1})
    second = build_sft_row(trace, problem_record=record(), metadata={"a": 1, "b": 2})

    assert first.row_id == second.row_id
    assert first.row_hash == second.row_hash


def test_sft_row_rejects_forged_row_id_and_hash() -> None:
    row = build_sft_row(valid_trace(), problem_record=record())

    with pytest.raises(SFTBuildError, match="row_id"):
        SFTDatasetRow(**{**row.__dict__, "row_id": "sft-forged"})
    with pytest.raises(SFTBuildError, match="row_hash"):
        SFTDatasetRow(**{**row.__dict__, "row_hash": "forged"})


def test_build_sft_dataset_reports_exact_rejections() -> None:
    trace = valid_trace()
    bad = compile_rejection_trace("p", "bad", prompt="prompt")

    rows, report = build_sft_dataset([trace, bad], problem_records={"p": record()})

    assert len(rows) == 1
    assert report.input_count == 2
    assert report.accepted_count == 1
    assert report.rejected_count == 1
    assert report.rejection_reasons["trace_not_positive"] == 1


def test_sft_build_report_rejects_forged_count_mismatches() -> None:
    with pytest.raises(SFTBuildError, match="accepted_count"):
        SFTBuildReport(input_count=2, accepted_count=1, rejected_count=0)
    with pytest.raises(SFTBuildError, match="rejected_trace_ids"):
        SFTBuildReport(input_count=1, accepted_count=0, rejected_count=1, rejection_reasons={"bad": 1})
    with pytest.raises(SFTBuildError, match="rejection reason totals"):
        SFTBuildReport(input_count=1, accepted_count=0, rejected_count=1, rejection_reasons={"bad": 2}, rejected_trace_ids=("t",))
