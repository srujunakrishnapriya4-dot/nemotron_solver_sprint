from kaggle_anti086.training.post_train_eval import build_post_train_eval, compare_reports


def test_post_train_eval_needs_kaggle_when_inference_unavailable():
    report = build_post_train_eval("/kaggle/working/anti086_adapters/v2a_run")
    assert report["status"] == "NEEDS_KAGGLE_MODEL_EVAL"
    assert report["model_eval_completed"] is False


def test_compare_reports_detects_regression():
    result = compare_reports({"private": {"exact_match": 0.5}}, {"private": {"exact_match": 0.4}})
    assert result["evals"]["private"]["delta"] < 0
    assert result["regressions"] == ["private"]


def test_compare_reports_allows_improvement():
    result = compare_reports({"private": {"exact_match": 0.5}}, {"private": {"exact_match": 0.55}})
    assert result["regressions"] == []
