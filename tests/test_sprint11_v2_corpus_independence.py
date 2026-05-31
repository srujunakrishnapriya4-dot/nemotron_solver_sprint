from kaggle_anti086.data.v2_corpus_independence import build_independence_report


def _row(row_id, source="src", prompt="p", answer="a", metadata=None):
    return {
        "id": row_id,
        "source_row_id": source,
        "prompt": prompt,
        "answer": answer,
        "normalized_answer": answer,
        "family": "format_only",
        "subfamily": "normalization",
        "rule_id": f"rule-{row_id}",
        "metadata": metadata or {"source_rule_id": f"sr-{row_id}", "source_prompt_hash": f"ph-{row_id}"},
    }


def test_solver_corrected_mirror_marked_not_independent():
    report = build_independence_report([_row("d", source="s")], [_row("c", source="s")])
    policy = report["policy_overlay"]["c"]
    assert policy["derived_from_direct"] is True
    assert policy["sampling_weight"] <= 0.5
    assert policy["counted_as_independent"] is False
    assert report["effective_sft_example_count"] == 1.5


def test_independent_corrected_row_counts_only_with_metadata():
    report = build_independence_report([_row("d", source="s")], [_row("c", source="other", metadata={"true_solver_correction": True})])
    assert report["independent_solver_corrected_rows"] == 1


def test_duplicate_direct_prompt_answer_pair_fails_unless_marked():
    rows = [_row("a", source="a", prompt="same"), _row("b", source="b", prompt="same")]
    assert build_independence_report(rows, [])["status"] == "FAIL"
    rows[1]["metadata"]["duplicate_for_format_variation"] = True
    assert build_independence_report(rows, [])["status"] == "PASS"
