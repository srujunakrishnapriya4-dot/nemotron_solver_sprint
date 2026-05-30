from __future__ import annotations

from src.common.schemas import AttemptEntropySource, Difficulty, ParsedProblem, ProblemDomain, RouteDecision
from src.online.kaggle_runner import CompetitionAttemptGenerator, MinimalPromptProfile
from src.online.python_tool_runtime import PythonToolConfig, PythonToolRequest, PythonToolSession
from src.serving.vllm_server import (
    GenerationOutput,
    GenerationRequest,
    GenerationResponse,
    RequestStatus,
    RuntimeState,
)


def _problem() -> ParsedProblem:
    return ParsedProblem(
        problem_id="python-tool-problem",
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
        problem_id="python-tool-problem",
        difficulty=Difficulty.EASY,
        difficulty_score=0.20,
        route_uncertainty=0.08,
        problem_type_probs={"number_theory": 0.80, "algebra": 0.20},
        archetype_probs={"modular": 0.70, "case_work": 0.30},
        compute_signals={"proof_burden": 0.10, "retrieval_need": 0.05, "repair_need": 0.05},
        route_rationale=["top_domain=number_theory"],
    )


def _prompt_profile() -> MinimalPromptProfile:
    return MinimalPromptProfile(
        name="minimal_default",
        system_prompt="Solve the following math problem step by step.\nUse Python code for calculations, verification, or brute-force search.\nThe answer is a non-negative integer in [0, 99999].\nPut your final answer in \\boxed{}.",
        tool_prompt="Execute Python code in a stateful Jupyter notebook. When you need Python, emit exactly <python_tool>...</python_tool> containing only the code, then stop.",
        preference_prompt="Use sympy for exact symbolic computation.",
    )


class _FakeTurnClient:
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
            latency_ms=30,
            backend_name="fake",
        )


def test_python_tool_session_executes_and_persists_state() -> None:
    session = PythonToolSession(
        attempt_id="attempt::stateful",
        config=PythonToolConfig(max_execution_time_sec=5.0),
    )
    first = session.execute(PythonToolRequest(code="x = 7", source="text_protocol_adapter"))
    second = session.execute(PythonToolRequest(code="print(x + 1)", source="text_protocol_adapter"))

    assert first.success
    assert second.success
    assert second.stdout.strip() == "8"
    assert session.tool_call_count == 2
    assert session.tool_error_count == 0


def test_python_tool_session_timeout_is_captured() -> None:
    session = PythonToolSession(
        attempt_id="attempt::timeout",
        config=PythonToolConfig(max_execution_time_sec=0.2),
    )
    result = session.execute(
        PythonToolRequest(
            code="while True:\n    pass",
            source="text_protocol_adapter",
        )
    )

    assert result.success is False
    assert result.timeout is True
    assert result.error_type == "ToolTimeout"
    assert session.tool_error_count == 1


def test_python_tool_session_error_is_captured() -> None:
    session = PythonToolSession(
        attempt_id="attempt::error",
        config=PythonToolConfig(max_execution_time_sec=5.0),
    )
    result = session.execute(
        PythonToolRequest(
            code="raise ValueError('boom')",
            source="text_protocol_adapter",
        )
    )

    assert result.success is False
    assert result.timeout is False
    assert result.error_type == "ValueError"
    assert "boom" in str(result.error_message)
    assert session.tool_error_count == 1


def test_competition_attempt_generator_executes_python_tool_calls() -> None:
    client = _FakeTurnClient(
        [
            GenerationOutput(
                text="<python_tool>\nprint(2 + 2)\n</python_tool>",
                finish_reason="stop",
                token_count=12,
                mean_token_entropy=0.44,
            ),
            GenerationOutput(
                text="The computation confirms the value. \\boxed{4}",
                finish_reason="stop",
                token_count=18,
                mean_token_entropy=0.28,
            ),
        ]
    )
    generator = CompetitionAttemptGenerator(
        client=client,
        prompt_profile=_prompt_profile(),
        deterministic_seed=7,
        max_tokens=512,
        temperature=1.0,
        top_p=0.95,
        min_p=0.02,
        request_logprobs=5,
        tool_config=PythonToolConfig(max_turns_per_attempt=4, max_execution_time_sec=5.0),
    )

    batch = generator.run_minimal_attempts(
        problem=_problem(),
        route=_route(),
        attempt_count=1,
        early_stop_threshold=4,
    )
    attempt = batch.attempts[0]

    assert attempt.valid_answer is True
    assert attempt.extracted_answer == "4"
    assert attempt.entropy_source is AttemptEntropySource.TRUE_LOGPROBS
    assert attempt.tool_call_count == 1
    assert attempt.tool_error_count == 0
    assert attempt.metadata["python_tool_mode"] == "bounded_stateful"
    assert attempt.metadata["python_tool_protocol"] == "text_protocol_adapter"
    assert attempt.metadata["turn_count"] == 2
    assert attempt.metadata["tool_history_cell_count"] == 1
    assert "stdout:\n4" in client.requests[1].prompt
