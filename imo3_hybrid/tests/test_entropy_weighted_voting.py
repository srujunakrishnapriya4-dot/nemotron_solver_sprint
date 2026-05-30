from __future__ import annotations

from src.common.schemas import AttemptEntropySource, AttemptRecord, Difficulty, ParsedProblem, ProblemDomain, RouteDecision
from src.online.inference_engine import InferenceEngine
from src.online.kaggle_runner import CompetitionAttemptGenerator, MinimalPromptProfile
from src.serving.vllm_server import GenerationOutput, GenerationParameters, GenerationRequest, GenerationResponse, RequestStatus, RuntimeState


def _problem() -> ParsedProblem:
    return ParsedProblem(
        problem_id="entropy-problem",
        raw_text="Find the required non-negative integer.",
        knowns=["n is a non-negative integer"],
        unknowns=["n"],
        constraints=["n is a non-negative integer"],
        domain=ProblemDomain.NUMBER_THEORY,
        target="final_answer",
        answer_type="non_negative_integer",
        difficulty_seed=0.30,
        likely_archetypes=["modular"],
        parse_quality={"confidence": 0.90},
    )


def _route() -> RouteDecision:
    return RouteDecision(
        problem_id="entropy-problem",
        difficulty=Difficulty.EASY,
        difficulty_score=0.25,
        route_uncertainty=0.10,
        problem_type_probs={"number_theory": 0.82, "algebra": 0.18},
        archetype_probs={"modular": 0.72, "case_work": 0.28},
        compute_signals={"proof_burden": 0.10, "retrieval_need": 0.08, "repair_need": 0.05},
        route_rationale=["top_domain=number_theory"],
    )


class _FakeVLLMClient:
    def __init__(self, outputs: list[GenerationOutput]) -> None:
        self.outputs = outputs
        self.calls = 0
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        output = self.outputs[self.calls]
        self.calls += 1
        return GenerationResponse(
            request_id=request.request_id,
            status=RequestStatus.OK,
            model_name="fake-model",
            runtime_state=RuntimeState.RUNNING,
            outputs=(output,),
            latency_ms=25,
            backend_name="fake",
        )


def _prompt_profile() -> MinimalPromptProfile:
    return MinimalPromptProfile(
        name="minimal_default",
        system_prompt="Solve the following math problem step by step.\nUse Python code for calculations, verification, or brute-force search.\nThe answer is a non-negative integer in [0, 99999].\nPut your final answer in \\boxed{}.",
        tool_prompt="Execute Python code in a stateful Jupyter notebook.",
        preference_prompt="Use sympy for exact symbolic computation.",
    )


def test_competition_attempt_generator_stops_at_four_agreements() -> None:
    client = _FakeVLLMClient(
        [
            GenerationOutput(text="work\n\\boxed{42}", finish_reason="stop", token_count=20, mean_token_entropy=0.30),
            GenerationOutput(text="work\n\\boxed{42}", finish_reason="stop", token_count=21, mean_token_entropy=0.28),
            GenerationOutput(text="work\n\\boxed{42}", finish_reason="stop", token_count=22, mean_token_entropy=0.31),
            GenerationOutput(text="work\n\\boxed{42}", finish_reason="stop", token_count=19, mean_token_entropy=0.29),
            GenerationOutput(text="work\n\\boxed{17}", finish_reason="stop", token_count=20, mean_token_entropy=0.40),
        ]
    )
    generator = CompetitionAttemptGenerator(
        client=client,
        prompt_profile=_prompt_profile(),
        deterministic_seed=42,
        max_tokens=512,
        temperature=1.0,
        top_p=0.95,
        min_p=0.02,
        request_logprobs=5,
    )

    batch = generator.run_minimal_attempts(
        problem=_problem(),
        route=_route(),
        attempt_count=8,
        early_stop_threshold=4,
    )

    assert batch.stopped_reason == "early_stop_consensus"
    assert len(batch.attempts) == 4
    assert client.calls == 4
    assert batch.metadata["prompt_profile"] == "minimal_default"
    assert batch.metadata["python_tool_mode"] == "bounded_stateful"
    assert batch.attempts[-1].stopped_reason == "early_stop_consensus"
    assert all(item.entropy_source is AttemptEntropySource.TRUE_LOGPROBS for item in batch.attempts)
    assert all(item.tool_call_count == 0 for item in batch.attempts)
    assert all(item.tool_error_count == 0 for item in batch.attempts)
    assert all(request.parameters.logprobs == 5 for request in client.requests)


def test_invalid_answers_do_not_contribute_votes() -> None:
    engine = InferenceEngine()
    selection = engine._select_from_attempt_records(
        problem_id="entropy-problem",
        attempts=(
            AttemptRecord(attempt_id="a0", seed=0, raw_text="\\boxed{42}", extracted_answer="42", valid_answer=True, mean_token_entropy=0.20, entropy_source=AttemptEntropySource.TRUE_LOGPROBS, runtime_sec=0.1),
            AttemptRecord(attempt_id="a1", seed=1, raw_text="No boxed answer", extracted_answer=None, valid_answer=False, mean_token_entropy=0.10, entropy_source=AttemptEntropySource.TRUE_LOGPROBS, runtime_sec=0.1),
            AttemptRecord(attempt_id="a2", seed=2, raw_text="\\boxed{42}", extracted_answer="42", valid_answer=True, mean_token_entropy=0.22, entropy_source=AttemptEntropySource.TRUE_LOGPROBS, runtime_sec=0.1),
            AttemptRecord(attempt_id="a3", seed=3, raw_text="\\boxed{17}", extracted_answer="17", valid_answer=True, mean_token_entropy=0.23, entropy_source=AttemptEntropySource.TRUE_LOGPROBS, runtime_sec=0.1),
        ),
        stopped_reason="attempt_budget_exhausted",
    )

    assert selection.submission_answer == 42
    assert selection.metadata["valid_attempt_count"] == 3


def test_entropy_weighting_breaks_vote_ties_in_favor_of_lower_entropy_attempts() -> None:
    engine = InferenceEngine()
    selection = engine._select_from_attempt_records(
        problem_id="entropy-problem",
        attempts=(
            AttemptRecord(attempt_id="a0", seed=0, raw_text="\\boxed{42}", extracted_answer="42", valid_answer=True, mean_token_entropy=0.20, entropy_source=AttemptEntropySource.TRUE_LOGPROBS, runtime_sec=0.1),
            AttemptRecord(attempt_id="a1", seed=1, raw_text="\\boxed{42}", extracted_answer="42", valid_answer=True, mean_token_entropy=0.25, entropy_source=AttemptEntropySource.TRUE_LOGPROBS, runtime_sec=0.1),
            AttemptRecord(attempt_id="a2", seed=2, raw_text="\\boxed{17}", extracted_answer="17", valid_answer=True, mean_token_entropy=1.30, entropy_source=AttemptEntropySource.TRUE_LOGPROBS, runtime_sec=0.1),
            AttemptRecord(attempt_id="a3", seed=3, raw_text="\\boxed{17}", extracted_answer="17", valid_answer=True, mean_token_entropy=1.45, entropy_source=AttemptEntropySource.TRUE_LOGPROBS, runtime_sec=0.1),
        ),
        stopped_reason="attempt_budget_exhausted",
    )

    assert selection.submission_answer == 42
    assert selection.prediction.method_used == "minimal_entropy_weighted_voting"
    assert selection.metadata["attempt_weighting_mode"] == "true_entropy"


def test_proxy_and_unavailable_weighting_modes_are_explicitly_labeled() -> None:
    engine = InferenceEngine()
    proxy_selection = engine._select_from_attempt_records(
        problem_id="entropy-problem",
        attempts=(
            AttemptRecord(attempt_id="p0", seed=0, raw_text="\\boxed{42}", extracted_answer="42", valid_answer=True, mean_token_entropy=0.15, entropy_source=AttemptEntropySource.PROXY_CONFIDENCE, runtime_sec=0.1),
            AttemptRecord(attempt_id="p1", seed=1, raw_text="\\boxed{17}", extracted_answer="17", valid_answer=True, mean_token_entropy=0.60, entropy_source=AttemptEntropySource.PROXY_CONFIDENCE, runtime_sec=0.1),
        ),
        stopped_reason="attempt_budget_exhausted",
    )
    unavailable_selection = engine._select_from_attempt_records(
        problem_id="entropy-problem",
        attempts=(
            AttemptRecord(attempt_id="u0", seed=0, raw_text="\\boxed{42}", extracted_answer="42", valid_answer=True, mean_token_entropy=None, entropy_source=AttemptEntropySource.UNAVAILABLE, runtime_sec=0.1),
            AttemptRecord(attempt_id="u1", seed=1, raw_text="\\boxed{17}", extracted_answer="17", valid_answer=True, mean_token_entropy=None, entropy_source=AttemptEntropySource.UNAVAILABLE, runtime_sec=0.1),
        ),
        stopped_reason="attempt_budget_exhausted",
    )

    assert proxy_selection.metadata["attempt_weighting_mode"] == "proxy_uncertainty"
    assert proxy_selection.prediction.method_used == "minimal_proxy_weighted_voting"
    assert unavailable_selection.metadata["attempt_weighting_mode"] == "equal_weight"
    assert unavailable_selection.prediction.method_used == "minimal_equal_weight_voting"


def test_true_entropy_mode_does_not_proxy_weight_non_true_attempts() -> None:
    engine = InferenceEngine()
    selection = engine._select_from_attempt_records(
        problem_id="entropy-problem",
        attempts=(
            AttemptRecord(attempt_id="t0", seed=0, raw_text="\\boxed{42}", extracted_answer="42", valid_answer=True, mean_token_entropy=0.20, entropy_source=AttemptEntropySource.TRUE_LOGPROBS, runtime_sec=0.1),
            AttemptRecord(attempt_id="p0", seed=1, raw_text="\\boxed{17}", extracted_answer="17", valid_answer=True, mean_token_entropy=0.01, entropy_source=AttemptEntropySource.PROXY_CONFIDENCE, runtime_sec=0.1),
        ),
        stopped_reason="attempt_budget_exhausted",
    )

    assert selection.submission_answer == 42
    assert selection.metadata["attempt_weighting_mode"] == "true_entropy"
    assert selection.metadata["attempt_weighting_fallback"] == "equal_for_non_true_entropy_attempts"
