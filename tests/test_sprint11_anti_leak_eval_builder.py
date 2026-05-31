from kaggle_anti086.eval.day5_eval_factory import build_eval_rows


VALID_PROBES = {
    "surface_template_reuse",
    "same_style_different_rule",
    "distractor_rule",
    "conflicting_examples",
    "near_duplicate_structure",
    "answer_format_trap",
}


def test_anti_leak_metadata_and_no_duplicate_leakage_group() -> None:
    rows = build_eval_rows("anti_leak", 256, 1108)
    assert len({row["leakage_group"] for row in rows}) == len(rows)
    assert all(row["metadata"]["anti_leak"] is True for row in rows)
    assert {row["metadata"]["leakage_probe_type"] for row in rows} <= VALID_PROBES
    conflict_rows = [row for row in rows if row["metadata"]["leakage_probe_type"] == "conflicting_examples"]
    assert conflict_rows
    assert any(row["metadata"]["expected_solver_behavior"] == "abstain" for row in conflict_rows)
