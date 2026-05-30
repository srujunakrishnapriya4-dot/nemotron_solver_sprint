from __future__ import annotations

from dataclasses import replace
import math

import pytest

from nemotron_engine.inference_eval.inference_contracts import (
    InferenceBackendInvocation,
    InferenceBackendResult,
    InferenceCompletion,
    InferenceContractError,
    InferenceMetric,
    validate_inference_invocation,
    validate_inference_result,
)


def invocation(**updates: object) -> InferenceBackendInvocation:
    data = {
        "invocation_id": "inv-1",
        "backend_kind": "external_mock_for_tests",
        "serving_config_hash": "serving",
        "prompt_batch_hash": "batch",
        "model_hash": "model",
        "adapter_hash": None,
        "dry_run": False,
        "expected_completion_count": 1,
        "metadata": {},
    }
    data.update(updates)
    return InferenceBackendInvocation(**data)


def completion(**updates: object) -> InferenceCompletion:
    data = {
        "problem_id": "p1",
        "prompt_hash": "prompt",
        "completion_text": r"raw reasoning \boxed{42}",
        "finish_reason": "stop",
        "token_count": 5,
        "latency_ms": 1.5,
        "metadata": {},
    }
    data.update(updates)
    return InferenceCompletion(**data)


def result(inv: InferenceBackendInvocation, **updates: object) -> InferenceBackendResult:
    data = {
        "result_id": "res-1",
        "invocation_hash": inv.invocation_hash,
        "backend_kind": inv.backend_kind,
        "status": "success",
        "completions": (completion(),),
        "metrics": (InferenceMetric("tokens_per_second", 1.0),),
        "started": True,
        "finished": True,
        "error": None,
        "metadata": {},
    }
    data.update(updates)
    return InferenceBackendResult(**data)


def test_valid_invocation_and_result_validate() -> None:
    inv = invocation()
    res = result(inv)

    assert validate_inference_invocation(inv) == inv
    assert validate_inference_result(res, invocation=inv) == res
    assert inv.invocation_hash == invocation().invocation_hash
    assert res.result_hash == result(inv).result_hash


def test_forged_hashes_rejected() -> None:
    inv = invocation()
    comp = completion()
    res = result(inv)

    with pytest.raises(InferenceContractError, match="invocation_hash"):
        replace(inv, invocation_hash="forged")
    with pytest.raises(InferenceContractError, match="completion_hash"):
        replace(comp, completion_hash="forged")
    with pytest.raises(InferenceContractError, match="result_hash"):
        replace(res, result_hash="forged")


def test_invocation_rejects_bad_fields_and_forbidden_metadata() -> None:
    with pytest.raises(InferenceContractError, match="dry_run"):
        invocation(dry_run="false")
    with pytest.raises(InferenceContractError, match="expected_completion_count"):
        invocation(expected_completion_count=-1)
    with pytest.raises(InferenceContractError, match="serving_config_hash"):
        invocation(serving_config_hash="")
    with pytest.raises(InferenceContractError, match="prompt_batch_hash"):
        invocation(prompt_batch_hash="")
    for metadata in (
        {"expected_answer": "42"},
        {"gold_answer": "42"},
        {"correctness": True},
        {"claim": "95+ guaranteed score"},
        {"leaderboard_success": True},
    ):
        with pytest.raises(InferenceContractError):
            invocation(metadata=metadata)


def test_completion_rejects_invalid_values_and_forbidden_metadata() -> None:
    with pytest.raises(InferenceContractError, match="completion_text"):
        completion(completion_text="")
    with pytest.raises(InferenceContractError, match="token_count"):
        completion(token_count=-1)
    with pytest.raises(InferenceContractError, match="latency_ms"):
        completion(latency_ms=-0.1)
    with pytest.raises(InferenceContractError, match="latency_ms"):
        completion(latency_ms=math.inf)
    for metadata in (
        {"normalized_answer": "42"},
        {"final_answer": "42"},
        {"is_correct": True},
        {"target_answer": "42"},
        {"score": 1.0},
        {"nested": {"public_score": 0.9}},
        {"nested": [{"accuracy_claim": "high"}]},
    ):
        with pytest.raises(InferenceContractError):
            completion(metadata=metadata)


def test_result_rejects_lifecycle_count_dry_run_and_metadata_violations() -> None:
    inv = invocation()
    with pytest.raises(InferenceContractError, match="started"):
        result(inv, started=False)
    with pytest.raises(InferenceContractError, match="error"):
        result(inv, error="boom")
    with pytest.raises(InferenceContractError, match="completion count"):
        validate_inference_result(result(inv, completions=(completion(), completion(problem_id="p2"))), invocation=inv)
    dry_inv = invocation(dry_run=True)
    with pytest.raises(InferenceContractError, match="dry_run"):
        validate_inference_result(result(dry_inv, invocation_hash=dry_inv.invocation_hash), invocation=dry_inv)
    with pytest.raises(InferenceContractError):
        completion(metadata={"correctness": True})
    with pytest.raises(InferenceContractError):
        result(inv, metadata={"submission_success": True})
    with pytest.raises(InferenceContractError):
        result(inv, metadata={"nested": {"promotion_success": True}})


def test_metric_rejects_nan_inf_and_bad_metadata() -> None:
    with pytest.raises(InferenceContractError, match="finite"):
        InferenceMetric("loss", math.nan)
    with pytest.raises(InferenceContractError, match="finite"):
        InferenceMetric("loss", math.inf)
    with pytest.raises(InferenceContractError):
        InferenceMetric("score", 1.0, metadata={"score_guarantee": "guaranteed"})


def test_result_count_mismatch_requires_source_invocation_to_enforce() -> None:
    inv = invocation(expected_completion_count=1)
    two_completion_result = result(inv, completions=(completion(), completion(problem_id="p2", prompt_hash="prompt-2")))

    assert validate_inference_result(two_completion_result) == two_completion_result
    with pytest.raises(InferenceContractError, match="completion count"):
        validate_inference_result(two_completion_result, invocation=inv)
