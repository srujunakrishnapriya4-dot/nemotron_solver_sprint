from kaggle_anti086.training.inference_eval_backend import compare_base_vs_adapter, run_model_eval
from kaggle_anti086.data.v2_corpus_io import write_jsonl_checked


def test_scoring_uses_normalizer(tmp_path):
    path = tmp_path / "eval.jsonl"
    write_jsonl_checked(path, [{"id": "r1", "family": "unit_conversion", "prompt": "p", "answer": "154.62"}], field_name="eval")
    report, preds = run_model_eval(None, None, path, generator=lambda prompt, family: "154.620 meters")
    assert report["exact_match"] == 1.0
    assert preds[0]["correct"] is True


def test_detects_verbose_and_empty(tmp_path):
    path = tmp_path / "eval.jsonl"
    write_jsonl_checked(path, [{"id": "r1", "family": "word_cipher", "prompt": "p", "answer": "cat"}], field_name="eval")
    report, _ = run_model_eval(None, None, path, generator=lambda prompt, family: "The answer is cat because")
    assert report["verbose_output_count"] == 1


def test_compare_delta_and_family_regression():
    base = {"exact_match": 0.5, "by_family": {"x": {"exact_match": 0.8}}}
    adapter = {"exact_match": 0.6, "by_family": {"x": {"exact_match": 0.7}}}
    delta = compare_base_vs_adapter(base, adapter)
    assert delta["delta"] > 0
    assert delta["regressions"] == ["x"]
