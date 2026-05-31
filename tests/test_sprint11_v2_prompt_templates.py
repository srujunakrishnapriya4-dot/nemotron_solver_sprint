from kaggle_anti086.data.v2_prompt_templates import build_direct_answer_messages, validate_training_messages


def test_messages_contain_user_and_assistant_only():
    row = {"prompt": "1 -> I. Input: 4"}
    messages = build_direct_answer_messages(row, "IV")
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[1]["content"] == "IV"
    assert validate_training_messages(messages, answer="IV")["status"] == "PASS"


def test_assistant_explanation_rejected():
    messages = [{"role": "user", "content": "Input: 4"}, {"role": "assistant", "content": "IV because 4 is IV"}]
    report = validate_training_messages(messages, answer="IV")
    assert report["status"] == "FAIL"
    assert "assistant_verbose_or_markdown" in report["failures"]


def test_answer_prefix_rejected():
    messages = [{"role": "user", "content": "Input: 4"}, {"role": "assistant", "content": "Answer: IV"}]
    assert validate_training_messages(messages, answer="IV")["status"] == "FAIL"


def test_assistant_does_not_copy_prompt():
    messages = [{"role": "user", "content": "Input: 4"}, {"role": "assistant", "content": "Input: 4"}]
    assert "assistant_copies_full_prompt" in validate_training_messages(messages)["failures"]


def test_query_gold_answer_leakage_rejected():
    messages = [{"role": "user", "content": "Input: 4\nExpected answer: IV"}, {"role": "assistant", "content": "IV"}]
    assert "query_gold_answer_leaked_in_user_prompt" in validate_training_messages(messages, answer="IV")["failures"]


def test_example_answers_inside_prompt_are_allowed():
    row = {"prompt": "1 -> I; 5 -> V. Input: 4"}
    messages = build_direct_answer_messages(row, "IV")
    assert validate_training_messages(messages, answer="IV")["status"] == "PASS"
