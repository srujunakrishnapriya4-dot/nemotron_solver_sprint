import types

from kaggle_anti086.training import model_lifecycle as ml


def test_cleanup_returns_report(monkeypatch):
    monkeypatch.setattr(ml, "capture_gpu_memory_snapshot", lambda stage: {"stage": stage, "status": "WARN"})
    report = ml.free_model(object(), reason="unit")
    assert report["status"] == "PASS"
    assert report["reason"] == "unit"


def test_cleanup_calls_empty_cache(monkeypatch):
    called = {"cache": False}
    monkeypatch.setattr(ml, "cuda_empty_cache", lambda reason: called.update(cache=True) or {"status": "PASS"})
    monkeypatch.setattr(ml, "capture_gpu_memory_snapshot", lambda stage: {"stage": stage, "status": "PASS"})
    ml.free_objects(object(), reason="cleanup")
    assert called["cache"] is True


def test_cleanup_failure_does_not_crash_by_default(monkeypatch):
    def boom(reason):
        raise RuntimeError("bad cache")

    monkeypatch.setattr(ml, "cuda_empty_cache", boom)
    report = ml.free_tokenizer(object(), reason="x")
    assert report["status"] == "FAIL"
    assert report["failures"]
