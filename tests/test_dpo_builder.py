from __future__ import annotations

import pytest

from nemotron_engine.core.schemas import ImmutableProblemRecord, ProblemSource, SplitName
from nemotron_engine.parsing import canonicalize_prompt
from nemotron_engine.solvers import solve_numeric
from nemotron_engine.traces import TraceAuditReport, TraceRecord, audit_trace, compile_trace_from_solver_result
from nemotron_engine.training.dpo_builder import DPOBuildError, DPOBuildReport, DPOPair, build_dpo_dataset, build_dpo_pair
from nemotron_engine.training.negative_generator import ALLOWED_NEGATIVE_TYPES, generate_negative_for_trace, generate_negative_traces


def valid_trace(problem_id: str = "p") -> TraceRecord:
    problem = canonicalize_prompt(problem_id, "1 -> 2\n2 -> 3\n4 -> ?")
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


def test_builds_dpo_pair_from_valid_chosen_and_meaningful_rejected_trace() -> None:
    chosen = valid_trace()
    rejected = generate_negative_for_trace(chosen, "target_leak").trace

    pair = build_dpo_pair(prompt=chosen.prompt or "prompt", chosen_trace=chosen, rejected_trace=rejected, negative_type="target_leak")

    assert pair.problem_id == chosen.problem_id
    assert pair.negative_type == "target_leak"
    assert pair.verifier_delta > 0


def test_rejects_invalid_chosen_trace() -> None:
    chosen = forged_trace(valid_trace(), metadata={"negative_type": "false_rule", "is_negative": True})
    rejected = generate_negative_for_trace(valid_trace(), "false_rule").trace

    with pytest.raises(DPOBuildError, match="chosen_marked_negative"):
        build_dpo_pair(prompt="prompt", chosen_trace=chosen, rejected_trace=rejected, negative_type="false_rule")


def test_rejects_rejected_trace_that_is_valid_positive() -> None:
    chosen = valid_trace()
    rejected = forged_trace(
        valid_trace(),
        completion="Apply to target.\n\\boxed{3}",
        boxed_answer="3",
        target_output="3",
        metadata={},
    )

    with pytest.raises(DPOBuildError, match="rejected_valid_positive"):
        build_dpo_pair(prompt="prompt", chosen_trace=chosen, rejected_trace=rejected, negative_type="false_rule")


def test_rejects_metadata_negative_otherwise_valid_positive() -> None:
    chosen = valid_trace()
    rejected = forged_trace(
        valid_trace(),
        completion="Apply to target.\n\\boxed{3}",
        boxed_answer="3",
        target_output="3",
        metadata={"negative_type": "false_rule", "is_negative": True},
    )

    with pytest.raises(DPOBuildError, match="metadata_negative_otherwise_valid_positive"):
        build_dpo_pair(prompt="prompt", chosen_trace=chosen, rejected_trace=rejected, negative_type="false_rule")


def test_rejects_missing_or_unsupported_negative_type_and_empty_prompt() -> None:
    chosen = valid_trace()
    rejected = generate_negative_for_trace(chosen, "false_rule").trace

    with pytest.raises(DPOBuildError, match="empty_prompt"):
        build_dpo_pair(prompt="", chosen_trace=chosen, rejected_trace=rejected, negative_type="false_rule")
    with pytest.raises(DPOBuildError, match="missing_negative_type"):
        build_dpo_pair(prompt="prompt", chosen_trace=chosen, rejected_trace=rejected, negative_type="")
    with pytest.raises(DPOBuildError, match="unsupported_negative_type"):
        build_dpo_pair(prompt="prompt", chosen_trace=chosen, rejected_trace=rejected, negative_type="unknown")


def test_rejects_mismatched_problem_id_and_whitespace_identical_completion() -> None:
    chosen = valid_trace("p")
    other = generate_negative_for_trace(valid_trace("q"), "false_rule").trace

    with pytest.raises(DPOBuildError, match="problem_id_mismatch"):
        build_dpo_pair(prompt="prompt", chosen_trace=chosen, rejected_trace=other, negative_type="false_rule")
    identical = forged_trace(generate_negative_for_trace(chosen, "false_rule").trace, completion=f" {chosen.completion} ")
    with pytest.raises(DPOBuildError, match="chosen_rejected_identical"):
        build_dpo_pair(prompt="prompt", chosen_trace=chosen, rejected_trace=identical, negative_type="false_rule")


def test_rejects_length_ratio_too_small_or_too_large() -> None:
    chosen = valid_trace()
    short = forged_trace(generate_negative_for_trace(chosen, "false_rule").trace, completion="bad")
    answer = chosen.boxed_answer
    long_completion = "leaked answer first: " + answer + "\n" + "\n".join(["filler"] * 80) + f"\nApply to target.\n\\boxed{{{answer}}}"
    long = forged_trace(generate_negative_for_trace(chosen, "target_leak").trace, completion=long_completion)

    with pytest.raises(DPOBuildError, match="length_ratio_uncontrolled"):
        build_dpo_pair(prompt="prompt", chosen_trace=chosen, rejected_trace=short, negative_type="false_rule")
    with pytest.raises(DPOBuildError, match="length_ratio_uncontrolled"):
        build_dpo_pair(prompt="prompt", chosen_trace=chosen, rejected_trace=long, negative_type="target_leak")


def test_manual_pair_rejects_bad_scores() -> None:
    chosen = valid_trace()
    rejected = generate_negative_for_trace(chosen, "false_rule").trace
    pair = build_dpo_pair(prompt="prompt", chosen_trace=chosen, rejected_trace=rejected, negative_type="false_rule")

    with pytest.raises(DPOBuildError, match="difficulty_match_score_out_of_range"):
        DPOPair(**{**pair.__dict__, "difficulty_match_score": 2.0})
    with pytest.raises(DPOBuildError, match="verifier_delta_not_meaningful"):
        DPOPair(**{**pair.__dict__, "verifier_delta": 0.0})


def test_pair_rejects_forged_pair_id_and_hash() -> None:
    chosen = valid_trace()
    rejected = generate_negative_for_trace(chosen, "false_rule").trace
    pair = build_dpo_pair(prompt="prompt", chosen_trace=chosen, rejected_trace=rejected, negative_type="false_rule")

    with pytest.raises(DPOBuildError, match="pair_id"):
        DPOPair(**{**pair.__dict__, "pair_id": "dpo-forged"})
    with pytest.raises(DPOBuildError, match="pair_hash"):
        DPOPair(**{**pair.__dict__, "pair_hash": "forged"})


def test_dataset_rejects_contaminated_or_eval_only_record() -> None:
    chosen = valid_trace()
    rejected = generate_negative_for_trace(chosen, "false_rule").trace

    rows, report = build_dpo_dataset([chosen], [rejected], problem_records={"p": record(contamination_flags=("x",))})
    assert rows == ()
    assert report.rejection_reasons["problem_record_contaminated"] == 1

    rows, report = build_dpo_dataset([chosen], [rejected], problem_records={"p": record(split=SplitName.DEV)})
    assert rows == ()
    assert report.rejection_reasons["problem_record_not_train"] == 1

    rows, report = build_dpo_dataset([chosen], [rejected], problem_records={"p": record(allowed_for_dpo=False)})
    assert rows == ()
    assert report.rejection_reasons["problem_record_dpo_not_allowed"] == 1


def test_dataset_counts_invalid_record_for_each_candidate_opportunity() -> None:
    chosen = valid_trace()
    negatives, _ = generate_negative_traces([chosen], negative_types=("false_rule", "target_leak"), max_negatives_per_trace=2)

    rows, report = build_dpo_dataset([chosen], [item.trace for item in negatives], problem_records={"p": record(contamination_flags=("x",))})

    assert rows == ()
    assert report.rejection_reasons["problem_record_contaminated"] == len(negatives)


def test_dataset_reports_no_matching_negative() -> None:
    chosen = valid_trace()

    rows, report = build_dpo_dataset([chosen], [])

    assert rows == ()
    assert report.rejection_reasons["no_matching_negative"] == 1


def test_fake_verification_negative_can_form_dpo_pair_with_meaningful_delta() -> None:
    chosen = valid_trace()
    rejected = generate_negative_for_trace(chosen, "fake_verification").trace

    pair = build_dpo_pair(prompt="prompt", chosen_trace=chosen, rejected_trace=rejected, negative_type="fake_verification")

    assert pair.verifier_delta > 0
    assert pair.negative_type == "fake_verification"


def test_pair_ids_hashes_and_max_pairs_are_deterministic() -> None:
    chosen = valid_trace()
    negatives, _ = generate_negative_traces(chosen and [chosen], negative_types=ALLOWED_NEGATIVE_TYPES, max_negatives_per_trace=7)

    first, first_report = build_dpo_dataset([chosen], [item.trace for item in negatives], prompts={"p": "prompt"}, max_pairs_per_problem=2)
    second, second_report = build_dpo_dataset([chosen], [item.trace for item in negatives], prompts={"p": "prompt"}, max_pairs_per_problem=2)

    assert len(first) == 2
    assert [pair.pair_id for pair in first] == [pair.pair_id for pair in second]
    assert [pair.pair_hash for pair in first] == [pair.pair_hash for pair in second]
    assert first_report.negative_type_counts == second_report.negative_type_counts
    assert first_report.rejection_reasons["max_pairs_per_problem"] >= 1


def test_chosen_trace_with_proofless_verification_warning_rejected_if_audit_invalid() -> None:
    chosen = valid_trace()
    rejected = generate_negative_for_trace(chosen, "false_rule").trace
    invalid = TraceAuditReport(chosen.trace_id, valid=False, errors=("proofless_warning_promoted",), warnings=("verification_claim_without_proof",))

    with pytest.raises(DPOBuildError, match="chosen_audit_invalid"):
        build_dpo_pair(prompt="prompt", chosen_trace=chosen, rejected_trace=rejected, negative_type="false_rule", chosen_audit=invalid)


def test_dpo_build_report_rejects_forged_count_mismatches() -> None:
    with pytest.raises(DPOBuildError, match="pair_ids"):
        DPOBuildReport(input_positive_count=1, input_negative_count=1, accepted_pair_count=1, rejected_pair_count=0, negative_type_counts={"false_rule": 1})
    with pytest.raises(DPOBuildError, match="rejection reason totals"):
        DPOBuildReport(input_positive_count=1, input_negative_count=1, accepted_pair_count=0, rejected_pair_count=1, rejection_reasons={"bad": 2})
    with pytest.raises(DPOBuildError, match="negative type totals"):
        DPOBuildReport(input_positive_count=1, input_negative_count=1, accepted_pair_count=1, rejected_pair_count=0, negative_type_counts={"false_rule": 2}, pair_ids=("p",))
