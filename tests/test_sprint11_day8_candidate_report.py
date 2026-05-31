from kaggle_anti086.data.v2_corpus_io import write_json_checked
from kaggle_anti086.training.day8_candidate_report import build_candidate_report


def test_candidate_report_needs_eval_when_missing(tmp_path):
    train = tmp_path / "train.json"
    eval_report = tmp_path / "eval.json"
    write_json_checked(train, {"status": "PASS", "trained": True, "adapter_dir": "x"}, field_name="train")
    write_json_checked(eval_report, {"status": "NEEDS_KAGGLE_MODEL_EVAL", "model_eval_completed": False}, field_name="eval")
    report = build_candidate_report(train, eval_report)
    assert report["candidate_quality"] == "NEEDS_KAGGLE_EVAL"
    assert report["packaging_allowed"] is False


def test_candidate_report_rejects_regression(tmp_path):
    train = tmp_path / "train.json"
    eval_report = tmp_path / "eval.json"
    write_json_checked(train, {"status": "PASS", "trained": True, "adapter_dir": "x"}, field_name="train")
    write_json_checked(eval_report, {"status": "PASS", "model_eval_completed": True, "evals": {"private_like_answerable_512": {"delta": -0.1}}, "regressions": ["private"]}, field_name="eval")
    report = build_candidate_report(train, eval_report)
    assert report["candidate_quality"] == "REJECT"


def test_candidate_report_promotes_safe_improvement(tmp_path):
    train = tmp_path / "train.json"
    eval_report = tmp_path / "eval.json"
    write_json_checked(train, {"status": "PASS", "trained": True, "adapter_dir": "x"}, field_name="train")
    write_json_checked(eval_report, {"status": "PASS", "model_eval_completed": True, "evals": {"private_like_answerable_512": {"delta": 0.03}, "rule_holdout_answerable_512": {"delta": 0.02}}, "regressions": []}, field_name="eval")
    report = build_candidate_report(train, eval_report)
    assert report["candidate_quality"] == "PROMOTE_TO_DAY9_FAILURE_MINING"
