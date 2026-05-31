import json

from kaggle_anti086.training.day8_2c_smoke_readiness import build_smoke_readiness_report


def _write(path, status="PASS", extra=None):
    data = {"status": status}
    if extra:
        data.update(extra)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_orchestrator_needs_kaggle_propagates(tmp_path):
    orch = tmp_path / "orch.json"
    _write(orch, "WARN", {"decision": "NEEDS_KAGGLE_SMOKE_TRAIN"})
    memory = tmp_path / "mem.json"
    _write(memory, "WARN")
    report = build_smoke_readiness_report({"orchestrator": orch, "memory": memory, "smoke_train": tmp_path / "missing1", "adapter_artifact": tmp_path / "missing2", "smoke_eval": tmp_path / "missing3", "output_drift": tmp_path / "missing4"})
    assert report["decision"] == "NEEDS_KAGGLE_SMOKE_TRAIN"


def test_memory_fail_blocks(tmp_path):
    paths = {}
    for name in ["orchestrator", "memory", "smoke_train", "adapter_artifact", "smoke_eval", "output_drift"]:
        p = tmp_path / f"{name}.json"
        _write(p, "FAIL" if name == "memory" else "PASS", {"decision": "ALLOW_DAY8_3_FULL_V2A_TRAINING", "full_training_allowed": True} if name == "orchestrator" else None)
        paths[name] = p
    report = build_smoke_readiness_report(paths)
    assert report["decision"] == "BLOCK_FULL_TRAINING_MEMORY_BACKEND_BUG"


def test_all_pass_allows_full_training(tmp_path):
    paths = {}
    for name in ["orchestrator", "memory", "smoke_train", "adapter_artifact", "smoke_eval", "output_drift"]:
        p = tmp_path / f"{name}.json"
        _write(p, "PASS", {"decision": "ALLOW_DAY8_3_FULL_V2A_TRAINING", "full_training_allowed": True} if name == "orchestrator" else None)
        paths[name] = p
    report = build_smoke_readiness_report(paths)
    assert report["decision"] == "ALLOW_DAY8_3_FULL_V2A_TRAINING"
    assert report["full_training_allowed"] is True
