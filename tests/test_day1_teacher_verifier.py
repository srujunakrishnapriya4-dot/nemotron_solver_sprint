from __future__ import annotations

from kaggle_anti086.training.day1_teacher_verifier import (
    count_boxed_answers,
    extract_boxed_answer,
    normalize_answer_for_teacher,
    validate_target_text,
)


def test_extract_boxed_answer_requires_exactly_one_final_box() -> None:
    assert extract_boxed_answer("work\n\\boxed{42}") == "42"
    assert extract_boxed_answer("\\boxed{1} and \\boxed{2}") is None
    assert extract_boxed_answer("work\n\\boxed{42} trailing") is None
    assert count_boxed_answers("\\boxed{1} and \\boxed{2}") == 2


def test_validate_target_text_rejects_bad_formats() -> None:
    assert validate_target_text("reason\n\\boxed{abc}", "abc") == (True, None)
    assert not validate_target_text("reason\n\\boxed{}", "")[0]
    assert not validate_target_text("reason\n\\boxed{abc} after", "abc")[0]
    assert not validate_target_text("reason\n\\boxed{abc}\n\\boxed{abc}", "abc")[0]
    assert not validate_target_text("ABSTAIN\n\\boxed{abc}", "abc")[0]


def test_normalize_answer_for_teacher_is_conservative() -> None:
    assert normalize_answer_for_teacher(" 007 ") == "7"
    assert normalize_answer_for_teacher(" AbC-! ") == "AbC-!"

