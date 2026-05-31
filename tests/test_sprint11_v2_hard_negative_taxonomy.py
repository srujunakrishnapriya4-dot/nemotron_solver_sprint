from kaggle_anti086.data.v2_hard_negative_taxonomy import build_hard_negative_taxonomy, classify_hard_negative


def _row(**overrides):
    row = {
        "id": "h",
        "family": "numeric_formula",
        "wrong_prediction": "",
        "gold_answer": "10.00",
        "failure_type": "solver_abstained_on_answerable",
        "do_not_use_as_sft": True,
        "allowed_for_dpo": True,
    }
    row.update(overrides)
    return row


def test_taxonomy_assigns_known_failure_classes():
    assert classify_hard_negative(_row(wrong_prediction="10", gold_answer="10.00", failure_type="solver_wrong_answer")) == "wrong_precision"
    assert classify_hard_negative(_row(family="bit_manipulation", wrong_prediction="101", gold_answer="0101", failure_type="solver_wrong_answer")) == "wrong_binary_width"
    assert classify_hard_negative(_row(family="equation_operator")) == "unsupported_family"


def test_policy_violations_fail():
    report = build_hard_negative_taxonomy([_row(do_not_use_as_sft=False, allowed_for_dpo=False, training_allowed=True)])
    assert report["status"] == "FAIL"
    assert report["sft_violation_count"] == 1
    assert report["do_not_use_as_sft_missing_count"] == 1
    assert report["contrastive_or_repair_policy_missing_count"] == 1


def test_small_hard_negative_corpus_warns():
    report = build_hard_negative_taxonomy([_row(), _row(family="bit_manipulation", wrong_prediction="101", gold_answer="0101")])
    assert report["status"] == "WARN"
    assert report["needs_day9_expansion"] is True
