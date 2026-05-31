import json

from kaggle_anti086.data.schema import validate_rows
from kaggle_anti086.eval.day5_eval_factory import build_answerable_rows, build_eval_rows


def test_answerable_rows_are_verified_answer_only() -> None:
    rows = build_answerable_rows("private_like", 128, 1151)
    assert len(rows) == 128
    assert validate_rows(rows, context="answerable")["failure_count"] == 0
    assert len({row["id"] for row in rows}) == len(rows)
    assert len({row["rule_id"] for row in rows}) == len(rows)
    assert len({row["leakage_group"] for row in rows}) == len(rows)
    assert all(row["metadata"]["expected_solver_behavior"] == "answer" for row in rows)
    assert all(row["metadata"]["eval_purpose"] == "answer_accuracy_eval" for row in rows)
    assert all(row["verification_status"] == "verified" for row in rows)
    assert all(row["answer"] and row["answer"] != "ABSTAIN" for row in rows)
    assert not {"equation_operator", "sequence_pattern", "permutation_sorting", "format_only"} & {row["family"] for row in rows}


def test_answerable_builder_does_not_mutate_mixed_eval() -> None:
    mixed = build_eval_rows("family_hard", 128, 1107)
    before = json.dumps(mixed, sort_keys=True)
    _ = build_answerable_rows("family_hard", 128, 1153)
    assert json.dumps(mixed, sort_keys=True) == before
