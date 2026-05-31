from kaggle_anti086.data.v2_corpus_io import write_json_checked
from kaggle_anti086.training.day8_decision_gate import build_day8_decision


def _write(tmp_path, name, data):
    path = tmp_path / name
    write_json_checked(path, data, field_name=name)
    return path


def test_bad_label_mask_blocks_training(tmp_path):
    c = _write(tmp_path, "c.json", {"status": "FAIL"})
    t = _write(tmp_path, "t.json", {"status": "PASS", "trained": True})
    e = _write(tmp_path, "e.json", {"status": "PASS"})
    r = _write(tmp_path, "r.json", {"candidate_quality": "PROMOTE_TO_DAY9_FAILURE_MINING"})
    assert build_day8_decision(c, t, e, r)["decision"] == "BLOCK_TRAINING_BAD_LABEL_MASK"


def test_runtime_failure_blocks_day9(tmp_path):
    c = _write(tmp_path, "c.json", {"status": "PASS"})
    t = _write(tmp_path, "t.json", {"status": "FAIL", "trained": False})
    e = _write(tmp_path, "e.json", {"status": "PASS"})
    r = _write(tmp_path, "r.json", {"candidate_quality": "PROMOTE_TO_DAY9_FAILURE_MINING"})
    assert build_day8_decision(c, t, e, r)["decision"] == "BLOCK_TRAINING_RUNTIME_FAILURE"


def test_missing_eval_requires_kaggle_eval(tmp_path):
    c = _write(tmp_path, "c.json", {"status": "PASS"})
    t = _write(tmp_path, "t.json", {"status": "PASS", "trained": True})
    e = _write(tmp_path, "e.json", {"status": "NEEDS_KAGGLE_MODEL_EVAL"})
    r = _write(tmp_path, "r.json", {"candidate_quality": "NEEDS_KAGGLE_EVAL"})
    report = build_day8_decision(c, t, e, r)
    assert report["decision"] == "NEEDS_KAGGLE_MODEL_EVAL"
    assert report["packaging_allowed"] is False


def test_safe_improvement_allows_day9(tmp_path):
    c = _write(tmp_path, "c.json", {"status": "PASS"})
    t = _write(tmp_path, "t.json", {"status": "PASS", "trained": True})
    e = _write(tmp_path, "e.json", {"status": "PASS"})
    r = _write(tmp_path, "r.json", {"candidate_quality": "PROMOTE_TO_DAY9_FAILURE_MINING"})
    assert build_day8_decision(c, t, e, r)["decision"] == "ALLOW_DAY9_FAILURE_MINING"
