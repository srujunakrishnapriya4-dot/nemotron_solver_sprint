from __future__ import annotations

from copy import deepcopy

from kaggle_anti086.training.day1_teacher_trainability_gate import verify_trainable_row
from kaggle_anti086.training.day1x_dpo_hard_negative_expander import (
    build_dpo_audit_report,
    extract_boxed_answer,
    generate_dpo_pairs,
    make_chosen,
    make_rejected,
    validate_dpo_pair,
)


def _row(index: int = 0, family: str = "unit_conversion") -> dict[str, object]:
    answer = str(100 + index)
    prompt = f"Convert {index + 1} m to cm. Case {index}."
    target = f"Use the exact conversion.\nThe value is {answer}.\n\\boxed{{{answer}}}"
    return {
        "schema_version": 1,
        "id": f"id-{family}-{index}",
        "record_id": f"record-{family}-{index}",
        "split": "train",
        "family": family,
        "rule_id": "rule",
        "difficulty": "medium",
        "prompt_style": "direct",
        "prompt": prompt,
        "answer": answer,
        "trace": target,
        "target_text": target,
        "source": "original_day1",
        "generator": "test_generator",
        "verification_status": "PASS",
        "ambiguity_count": 0,
        "trainable": True,
        "metadata": {"mixture_source": "original_day1", "output_type": "integer"},
    }


def _pair(row: dict[str, object] | None = None, negative_type: str = "near_miss_answer") -> dict[str, object]:
    row = _row() if row is None else row
    rejected, meta = make_rejected(row, negative_type, 123)
    return {
        "schema_version": 1,
        "id": f"pair-{row['record_id']}-{negative_type}",
        "source_row_id": row["record_id"],
        "split": "dpo_train",
        "family": row["family"],
        "prompt_style": row["prompt_style"],
        "difficulty": row["difficulty"],
        "prompt": row["prompt"],
        "chosen": make_chosen(row),
        "rejected": rejected,
        "answer": str(row["answer"]),
        "chosen_boxed_answer": extract_boxed_answer(make_chosen(row)),
        "rejected_boxed_answer": extract_boxed_answer(rejected),
        "negative_type": negative_type,
        "reason_rejected": meta["reason_rejected"],
        "rejected_expected_failure": True,
        "verification_status": "PASS",
        "chosen_trainability_status": "PASS",
        "rejected_trainability_status": "FAIL",
        "metadata": {"source_family": row["family"], "source_generator": "test", "source_prompt_style": row["prompt_style"], "is_composed": str(row["family"]).startswith("composed_"), "subfamilies": []},
    }


def test_extract_boxed_answer_works() -> None:
    assert extract_boxed_answer("work\n\\boxed{42}") == "42"
    assert extract_boxed_answer("no box") is None


def test_chosen_passes_trainability_gate() -> None:
    row = _row()
    check = verify_trainable_row({**row, "target_text": make_chosen(row), "trace": make_chosen(row)})
    assert check.trainable


def test_answer_level_rejected_differs_from_chosen() -> None:
    for negative_type in ("near_miss_answer", "off_by_one", "wrong_arithmetic"):
        pair = _pair(negative_type=negative_type)
        assert pair["chosen"] != pair["rejected"]
        assert validate_dpo_pair(pair)[0]


def test_format_negatives_have_expected_shapes() -> None:
    cases = {
        "format_no_box": lambda text: "\\boxed{" not in text,
        "format_multiple_boxes": lambda text: text.count("\\boxed{") >= 2,
        "format_text_after_box": lambda text: text.rstrip().endswith("trailing text"),
        "format_abstain": lambda text: "ABSTAIN" in text,
    }
    for negative_type, predicate in cases.items():
        rejected, meta = make_rejected(_row(), negative_type, 123)
        assert meta["negative_type"] == negative_type
        assert predicate(rejected)


def test_validate_dpo_pair_rejects_malformed_chosen_equals_and_accidental_trainable() -> None:
    malformed = _pair()
    malformed["chosen"] = "no box"
    ok, failures = validate_dpo_pair(malformed)
    assert not ok
    assert any("chosen" in failure for failure in failures)

    same = _pair()
    same["rejected"] = same["chosen"]
    ok, failures = validate_dpo_pair(same)
    assert not ok
    assert "chosen_equals_rejected" in failures

    trainable = _pair()
    trainable["rejected"] = "Correct.\n\\boxed{100}"
    trainable["rejected_boxed_answer"] = "100"
    ok, failures = validate_dpo_pair(trainable)
    assert not ok
    assert "rejected_accidentally_trainable" in failures


def test_right_answer_wrong_format_allowed_only_when_format_invalid() -> None:
    pair = _pair(negative_type="right_answer_wrong_format")
    assert validate_dpo_pair(pair)[0]
    bad = deepcopy(pair)
    bad["rejected"] = "Correct.\n\\boxed{100}"
    bad["rejected_boxed_answer"] = "100"
    ok, failures = validate_dpo_pair(bad)
    assert not ok
    assert "rejected_accidentally_trainable" in failures


def test_composed_row_can_produce_wrong_substep_negative() -> None:
    row = _row(0, "composed_unit_formula")
    pair = _pair(row, "wrong_substep_1")
    assert pair["negative_type"] == "wrong_substep_1"
    assert validate_dpo_pair(pair)[0]


def test_generated_pairs_have_unique_ids_no_prompt_overlap_and_are_deterministic() -> None:
    train_rows = [_row(i) for i in range(40)]
    eval_rows = [_row(i + 100) for i in range(20)]
    first, rejected_first = generate_dpo_pairs(train_rows, 20, 11, "dpo_train")
    second, _ = generate_dpo_pairs(train_rows, 20, 11, "dpo_train")
    eval_pairs, _ = generate_dpo_pairs(eval_rows, 10, 12, "dpo_eval")
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len({row["id"] for row in first}) == len(first)
    assert not ({row["prompt"] for row in first} & {row["prompt"] for row in eval_pairs})
    audit = build_dpo_audit_report(first, eval_pairs, rejected_first)
    assert len(audit["negative_type_counts"]) >= 5
