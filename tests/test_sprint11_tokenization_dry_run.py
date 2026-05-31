from kaggle_anti086.training.prepare_tokenization_dry_run import build_tokenization_dry_run
from kaggle_anti086.training.training_config_schema import load_training_config


def _config():
    return load_training_config("kaggle_anti086/training/configs/v2a_base_lora.yaml")


def test_balanced_sample_produced():
    report = build_tokenization_dry_run(_config())
    assert report["status"] == "PASS"
    assert report["sample_count"] >= 256
    assert "format_only" in report["family_counts"]
    assert "direct" in report["source_counts"]
    assert "solver_corrected" in report["source_counts"]


def test_empty_prompt_and_answer_fail(tmp_path):
    path = tmp_path / "bad.jsonl"
    row = '{"id":"x","family":"format_only","messages":[{"role":"user","content":""},{"role":"assistant","content":""}]}\n'
    path.write_text(row * 300, encoding="utf-8")
    cfg = _config()
    cfg["train_direct_path"] = str(path)
    cfg["train_solver_corrected_path"] = str(path)
    report = build_tokenization_dry_run(cfg)
    assert report["status"] == "FAIL"
    assert report["empty_prompt_count"] > 0
    assert report["empty_answer_count"] > 0


def test_unsupported_family_fails(tmp_path):
    path = tmp_path / "bad.jsonl"
    row = '{"id":"x","family":"equation_operator","messages":[{"role":"user","content":"p"},{"role":"assistant","content":"a"}]}\n'
    path.write_text(row * 300, encoding="utf-8")
    cfg = _config()
    cfg["train_direct_path"] = str(path)
    cfg["train_solver_corrected_path"] = str(path)
    assert build_tokenization_dry_run(cfg)["unsupported_sft_rows"] > 0
