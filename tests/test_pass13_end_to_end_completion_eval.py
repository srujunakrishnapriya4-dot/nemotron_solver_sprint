from __future__ import annotations

from nemotron_engine.evaluation.transfer_harness import TransferHarnessConfig
from nemotron_engine.inference_eval import (
    CompletionTransferConfig,
    InferenceBackendInvocation,
    InferenceBackendResult,
    InferenceCompletion,
    InferenceRunnerConfig,
    build_inference_evaluation_manifest,
    build_prompt_batch_from_transfer_examples,
    capture_completions,
    evaluate_completions_with_transfer_harness,
    evaluate_private_like_from_completions,
    invoke_inference_backend,
)
from nemotron_engine.runtime.serving_config import ServingConfig


def serving(**updates: object) -> ServingConfig:
    data = {"prompt_template_hash": "prompt", "tokenizer_hash": "tokenizer", "model_hash": "model", "batch_size": 4}
    data.update(updates)
    return ServingConfig(**data)


def records() -> list[dict[str, object]]:
    return [
        {
            "problem_id": "p1",
            "prompt_text": "Solve private-like problem 1.",
            "slice_name": "private_like",
            "split": "private_like",
            "family_id": "fam1",
            "primitive_family_id": "prim",
            "format_family_id": "fmt",
            "answer_type": "integer",
        },
        {
            "problem_id": "p2",
            "prompt_text": "Solve private-like problem 2.",
            "slice_name": "private_like",
            "split": "private_like",
            "family_id": "fam2",
            "primitive_family_id": "prim",
            "format_family_id": "fmt",
            "answer_type": "integer",
        },
    ]


class FakeBackend:
    def __init__(self, completions: dict[str, str]) -> None:
        self.completions = completions
        self.received_expected_answers = False

    def run(self, invocation: InferenceBackendInvocation, prompt_batch):
        for example in prompt_batch.examples:
            if hasattr(example, "expected_answer") or "expected_answer" in example.metadata:
                self.received_expected_answers = True
        return InferenceBackendResult(
            "res",
            invocation.invocation_hash,
            invocation.backend_kind,
            "success",
            tuple(
                InferenceCompletion(example.problem_id, example.prompt_hash, self.completions[example.problem_id])
                for example in prompt_batch.examples
            ),
            (),
            True,
            True,
            None,
            {},
        )


def run_pipeline(completions: dict[str, str], *, config: ServingConfig | None = None, non_exact_allowed: bool = False):
    cfg = config or serving()
    batch = build_prompt_batch_from_transfer_examples(records(), serving_config=cfg)
    invocation = InferenceBackendInvocation(
        "inv",
        "external_mock_for_tests",
        batch.serving_config_hash,
        batch.batch_hash,
        "model",
        None,
        False,
        len(batch.examples),
        {},
    )
    backend = FakeBackend(completions)
    run_report = invoke_inference_backend(
        invocation,
        batch,
        backend,
        config=InferenceRunnerConfig(cfg, allow_non_submission_exact_eval=non_exact_allowed),
    )
    assert backend.received_expected_answers is False
    capture = capture_completions(batch, invocation, run_report.backend_result)
    transfer = evaluate_completions_with_transfer_harness(
        batch,
        capture,
        {"p1": "42", "p2": "42"},
        config=CompletionTransferConfig(
            TransferHarnessConfig(required_slices=("private_like",), allow_small_slices=True),
            non_submission_exact=run_report.non_submission_exact,
        ),
    )
    private = evaluate_private_like_from_completions(transfer)
    manifest = build_inference_evaluation_manifest(
        prompt_batch=batch,
        invocation=invocation,
        result=run_report.backend_result,
        capture_report=capture,
        transfer_report=transfer,
        private_like_report=private,
    )
    return batch, run_report, capture, transfer, private, manifest


def test_end_to_end_completion_eval_passes_with_raw_boxed_completions_and_deterministic_hashes() -> None:
    first = run_pipeline({"p1": "reasoning stays raw\n\\boxed{42}", "p2": "different raw text \\boxed{42}"})
    second = run_pipeline({"p1": "reasoning stays raw\n\\boxed{42}", "p2": "different raw text \\boxed{42}"})

    assert first[-1].passed is True
    assert first[2].records[0].completion_text == "reasoning stays raw\n\\boxed{42}"
    assert first[0].batch_hash == second[0].batch_hash
    assert first[2].report_hash == second[2].report_hash
    assert first[3].report_hash == second[3].report_hash
    assert first[4].report_hash == second[4].report_hash
    assert first[5].manifest_hash == second[5].manifest_hash


def test_no_box_fake_completion_fails_without_package_submission_or_promotion() -> None:
    _, _, _, transfer, private, manifest = run_pipeline({"p1": "no box", "p2": r"\boxed{42}"})

    assert transfer.passed is False
    assert private.passed is False
    assert manifest.passed is False
    assert manifest.transfer_report_hash == transfer.report_hash
    assert "private_like" in " ".join(manifest.errors) or "transfer" in " ".join(manifest.errors)


def test_non_submission_exact_eval_cannot_pass() -> None:
    cfg = serving(temperature=0.7, strict_submission_mode=False)
    _, run_report, _, transfer, private, manifest = run_pipeline(
        {"p1": r"\boxed{42}", "p2": r"\boxed{42}"},
        config=cfg,
        non_exact_allowed=True,
    )

    assert run_report.non_submission_exact is True
    assert transfer.passed is False
    assert private.passed is False
    assert manifest.passed is False
    assert "non_submission_exact" in manifest.errors
