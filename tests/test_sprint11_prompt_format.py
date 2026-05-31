from kaggle_anti086.runtime.prompt_format import (
    render_chat_like_text,
    render_inference_prompt,
    render_system_prompt,
    render_training_example,
)


def test_training_render_includes_final_answer_only_target():
    rendered = render_training_example("Input: 1010\nOutput: ?", "0101")
    assert rendered["assistant"] == "0101"
    assert rendered["text"].endswith("ASSISTANT:\n0101")
    assert "Answer:" not in rendered["text"]
    assert "explain" not in rendered["text"].lower()


def test_inference_render_excludes_answer_and_matches_prefix():
    prompt = "Input: @&\nOutput: ?"
    train_text = render_chat_like_text(prompt, "@@")
    infer_text = render_inference_prompt(prompt)
    assert infer_text.endswith("ASSISTANT:\n")
    assert "@@" not in infer_text
    assert train_text.startswith(infer_text)
    assert render_system_prompt() in infer_text
