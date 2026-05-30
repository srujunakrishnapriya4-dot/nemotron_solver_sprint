from __future__ import annotations

from dataclasses import replace

import pytest

from nemotron_engine.inference_eval.prompt_batch import (
    PromptBatch,
    PromptBatchError,
    PromptExample,
    build_prompt_batch_from_transfer_examples,
)
from nemotron_engine.runtime.serving_config import ServingConfig


def serving() -> ServingConfig:
    return ServingConfig(prompt_template_hash="prompt", tokenizer_hash="tokenizer", model_hash="model", batch_size=4)


def example(**updates: object) -> PromptExample:
    data = {
        "problem_id": "p1",
        "prompt_text": "Solve the problem and put the final answer in one box.",
        "slice_name": "private_like",
        "family_id": "fam",
        "primitive_family_id": "prim",
        "format_family_id": "fmt",
        "answer_type": "integer",
        "split": "private_like",
        "metadata": {},
    }
    data.update(updates)
    return PromptExample(**data)


def test_prompt_example_rejects_leakage_markers_and_forbidden_metadata() -> None:
    with pytest.raises(PromptBatchError, match="leakage marker"):
        example(prompt_text="expected_answer=42")
    with pytest.raises(PromptBatchError, match="leakage marker"):
        example(prompt_text="expected_answer = 42")
    with pytest.raises(PromptBatchError, match="leakage marker"):
        example(prompt_text="gold_answer = 42")
    with pytest.raises(PromptBatchError, match="leakage marker"):
        example(prompt_text="target_answer = 42")
    with pytest.raises(PromptBatchError, match="leakage marker"):
        example(prompt_text="Correct answer: 42")
    with pytest.raises(PromptBatchError, match="leakage marker"):
        example(prompt_text="correct answer : 42")
    with pytest.raises(PromptBatchError, match="leakage marker"):
        example(prompt_text="CORRECT ANSWER : 42")
    for metadata in (
        {"expected_answer": "42"},
        {"gold_answer": "42"},
        {"target_answer": "42"},
        {"correctness": True},
        {"leakage": True},
        {"leaderboard_success": True},
    ):
        with pytest.raises(PromptBatchError):
            example(metadata=metadata)


def test_prompt_example_rejects_bad_core_fields_and_forged_hash() -> None:
    with pytest.raises(PromptBatchError, match="prompt_text"):
        example(prompt_text="")
    with pytest.raises(PromptBatchError, match="problem_id"):
        example(problem_id="")
    with pytest.raises(PromptBatchError, match="answer_type"):
        example(answer_type="not_supported")
    with pytest.raises(PromptBatchError, match="slice_name"):
        example(slice_name="not_a_slice")
    with pytest.raises(PromptBatchError, match="split"):
        example(split="not_a_slice")
    with pytest.raises(PromptBatchError, match="prompt_hash"):
        replace(example(), prompt_hash="forged")


def test_prompt_batch_rejects_duplicate_and_forged_batch_hash() -> None:
    first = example()
    second = example()
    with pytest.raises(PromptBatchError, match="duplicate"):
        PromptBatch("batch", (first, second), "serving")

    batch = PromptBatch("batch", (first,), "serving")
    with pytest.raises(PromptBatchError, match="batch_hash"):
        replace(batch, batch_hash="forged")


def test_build_prompt_batch_is_deterministic_and_excludes_expected_answer() -> None:
    record = {
        "problem_id": "p1",
        "prompt_text": "Question?",
        "slice_name": "private_like",
        "split": "private_like",
        "family_id": "fam",
        "primitive_family_id": "prim",
        "format_family_id": "fmt",
        "answer_type": "integer",
        "expected_answer": "42",
        "metadata": {"answer_min": 0, "answer_max": 99},
    }
    first = build_prompt_batch_from_transfer_examples([record], serving_config=serving())
    second = build_prompt_batch_from_transfer_examples([record], serving_config=serving())

    assert first.batch_hash == second.batch_hash
    assert "expected_answer" not in first.examples[0].metadata
    assert not hasattr(first.examples[0], "expected_answer")
    assert first.examples[0].metadata["answer_min"] == 0


def test_builder_rejects_unsupported_answer_type_slice_and_batch_size() -> None:
    base = {
        "problem_id": "p1",
        "prompt_text": "Question?",
        "slice_name": "private_like",
        "split": "private_like",
        "family_id": "fam",
        "primitive_family_id": "prim",
        "format_family_id": "fmt",
        "answer_type": "integer",
    }
    with pytest.raises(PromptBatchError):
        build_prompt_batch_from_transfer_examples([{**base, "answer_type": "bad"}], serving_config=serving())
    with pytest.raises(PromptBatchError):
        build_prompt_batch_from_transfer_examples([{**base, "slice_name": "bad"}], serving_config=serving())
    small = ServingConfig(prompt_template_hash="prompt", tokenizer_hash="tokenizer", model_hash="model", batch_size=1)
    with pytest.raises(PromptBatchError, match="batch_size"):
        build_prompt_batch_from_transfer_examples([base, {**base, "problem_id": "p2"}], serving_config=small)
