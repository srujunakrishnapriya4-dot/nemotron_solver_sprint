from __future__ import annotations

from nemotron_engine.competition_sprint.competition_answer_policy import validate_competition_answer


def normalize_training_answer(answer) -> str:
    return validate_competition_answer(answer)


def render_assistant_answer(answer: str, *, style: str = "raw") -> str:
    if style == "boxed":
        return f"\\boxed{{{answer}}}"
    if style != "raw":
        raise ValueError(f"unsupported answer style: {style}")
    return answer
