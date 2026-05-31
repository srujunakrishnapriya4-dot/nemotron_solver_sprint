from kaggle_anti086.data.v2_corpus_leakage import build_leakage_report


def _direct(**overrides):
    row = {
        "id": "train_v2_direct_000001",
        "source_row_id": "src-1",
        "rule_id": "train_rule_a",
        "leakage_group": "train_lg_a",
        "family": "roman_numeral",
        "prompt": "templated user prompt",
        "answer": "IV",
        "normalized_answer": "IV",
        "messages": [{"role": "user", "content": "templated user prompt"}, {"role": "assistant", "content": "IV"}],
    }
    row.update(overrides)
    return row


def test_clean_corpus_passes_leakage():
    report = build_leakage_report([_direct()], {"rule_holdout": [{"rule_id": "holdout_rule"}], "anti_leak": [{"leakage_group": "anti_lg", "prompt": "different"}]})
    assert report["status"] == "PASS"


def test_rule_id_overlap_fails():
    report = build_leakage_report([_direct(rule_id="holdout_rule")], {"rule_holdout": [{"rule_id": "holdout_rule"}], "anti_leak": []})
    assert report["status"] == "FAIL"
    assert report["rule_id_overlap_count"] == 1


def test_leakage_group_overlap_fails():
    report = build_leakage_report([_direct(leakage_group="anti_lg")], {"rule_holdout": [], "anti_leak": [{"leakage_group": "anti_lg"}]})
    assert report["status"] == "FAIL"
    assert report["leakage_group_overlap_count"] == 1


def test_prompt_hash_overlap_fails():
    report = build_leakage_report([_direct()], {"rule_holdout": [], "anti_leak": [{"prompt": "templated user prompt"}]})
    assert report["status"] == "FAIL"
    assert report["prompt_hash_overlap_count"] == 1


def test_duplicate_source_row_id_fails():
    report = build_leakage_report([_direct(id="a"), _direct(id="b")], {"rule_holdout": [], "anti_leak": []})
    assert report["status"] == "FAIL"
    assert report["duplicate_source_row_id_count"] == 1
