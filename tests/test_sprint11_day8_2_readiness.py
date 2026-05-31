import json

from kaggle_anti086.training.day8_2_readiness import build_day8_2_readiness


def _write(path, status, extra=None):
    data = {"status": status}
    if extra:
        data.update(extra)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_local_missing_smoke_needs_kaggle(tmp_path):
    paths = {}
    for name in ["weighted_sampling", "truncation", "final_sft_leakage", "supervised_label_decode"]:
        path = tmp_path / f"{name}.json"
        _write(path, "PASS")
        paths[name] = str(path)
    report = build_day8_2_readiness(paths)
    assert report["decision"] == "NEEDS_KAGGLE_SMOKE_TRAIN"
    assert report["full_training_allowed"] is False


def test_truncation_failure_blocks_full_training(tmp_path):
    paths = {}
    for name, status in [("weighted_sampling", "PASS"), ("truncation", "FAIL"), ("supervised_label_decode", "PASS")]:
        path = tmp_path / f"{name}.json"
        _write(path, status)
        paths[name] = str(path)
    report = build_day8_2_readiness(paths)
    assert report["decision"] == "BLOCK_FULL_TRAINING_TRUNCATION_BUG"
    assert report["packaging_allowed"] is False
    assert report["submission_allowed"] is False
