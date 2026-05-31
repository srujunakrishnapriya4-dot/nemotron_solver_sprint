from __future__ import annotations


SYSTEM_PROMPT = "You are a reasoning model. Respond with only the final answer."


def render_system_prompt() -> str:
    return SYSTEM_PROMPT


def render_training_example(prompt: str, answer: str) -> dict[str, str]:
    return {
        "system": SYSTEM_PROMPT,
        "user": str(prompt),
        "assistant": str(answer),
        "text": render_chat_like_text(prompt, answer),
    }


def render_inference_prompt(prompt: str) -> str:
    return render_chat_like_text(prompt, None)


def render_chat_like_text(prompt: str, answer: str | None = None) -> str:
    base = f"SYSTEM:\n{SYSTEM_PROMPT}\n\nUSER:\n{prompt}\n\nASSISTANT:\n"
    if answer is None:
        return base
    return base + str(answer)
