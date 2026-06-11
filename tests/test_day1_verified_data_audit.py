from __future__ import annotations

from copy import deepcopy

from kaggle_anti086.training.day1_verified_data_audit import (
    audit_dedup_leakage,
    audit_family_balance,
    audit_format,
    audit_prompt_diversity,
    audit_teacher_verification,
)
from kaggle_anti086.training.day1_teacher_families import available_teachers
from kaggle_anti086.training.day1_verified_data_schema import stable_record_id


def _row(family: str = "unit_conversion", prompt: str = "Convert 1 m to cm.") -> dict[str, object]:
    target = "1 m = 100 cm.\n\\boxed{100}"
    return {
        "record_id": stable_record_id(family, prompt, "100", "unit_exact", 1),
        "family": family,
        "rule_id": "unit_exact",
        "difficulty": "medium",
        "prompt_style": "direct",
        "prompt": prompt,
        "answer": "100",
        "trace": target,
        "target_text": target,
        "source": "deterministic_teacher",
        "verification_status": "PASS",
        "ambiguity_count": 0,
        "trainable": True,
        "split": "train",
        "seed": 1,
        "teacher_version": "test",
        "metadata": {},
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ],
    }


def test_format_audit_catches_multiple_boxes_text_after_box_and_mismatch() -> None:
    multiple = _row()
    multiple["target_text"] = "\\boxed{1}\\boxed{2}"
    multiple["trace"] = multiple["target_text"]
    multiple["messages"][1]["content"] = multiple["target_text"]
    trailing = _row(prompt="Convert 2 m to cm.")
    trailing["target_text"] = "\\boxed{100} trailing"
    trailing["trace"] = trailing["target_text"]
    trailing["messages"][1]["content"] = trailing["target_text"]
    mismatch = _row(prompt="Convert 3 m to cm.")
    mismatch["target_text"] = "\\boxed{101}"
    mismatch["trace"] = mismatch["target_text"]
    mismatch["messages"][1]["content"] = mismatch["target_text"]
    result = audit_format([multiple, trailing, mismatch])
    assert result["status"] == "FAIL"
    assert result["counts"]["multiple_boxes"] == 1
    assert result["counts"]["text_after_box"] == 1
    assert result["counts"]["answer_mismatch"] == 1


def test_dedup_audit_catches_duplicate_prompt_and_split_leakage() -> None:
    row = _row()
    duplicate = deepcopy(row)
    duplicate["record_id"] = "different"
    result = audit_dedup_leakage([row, duplicate], [deepcopy(row)], [], [])
    assert result["status"] == "FAIL"
    assert result["counts"]["train_duplicate_prompts"] == 1
    assert result["counts"]["train_eval_prompt_overlap"] == 1


def test_family_balance_audit_catches_missing_family() -> None:
    result = audit_family_balance([_row()], {family: 1 for family in available_teachers()})
    assert result["status"] == "FAIL"
    assert any("missing" in failure for failure in result["failures"])


def test_prompt_diversity_audit_catches_one_style_collapse() -> None:
    rows = [_row(prompt=f"p{i}") for i in range(10)]
    result = audit_prompt_diversity(rows)
    assert result["status"] == "FAIL"
    assert any("prompt_style_dominates" in failure for failure in result["failures"])


def test_teacher_verification_audit_catches_ambiguity_and_trainable_false() -> None:
    ambiguous = _row()
    ambiguous["ambiguity_count"] = 1
    blocked = _row(prompt="Convert 4 m to cm.")
    blocked["trainable"] = False
    result = audit_teacher_verification([ambiguous, blocked])
    assert result["status"] == "FAIL"
    assert result["counts"]["ambiguity_nonzero"] >= 1
    assert result["counts"]["trainable_not_true"] >= 1
