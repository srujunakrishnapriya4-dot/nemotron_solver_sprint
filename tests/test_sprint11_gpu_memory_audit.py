import json
import types

from kaggle_anti086.training import gpu_memory_audit as gma


def test_no_cuda_local_warn(monkeypatch):
    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    monkeypatch.setitem(__import__("sys").modules, "torch", types.SimpleNamespace(cuda=FakeCuda()))
    snap = gma.capture_gpu_memory_snapshot("local")
    assert snap["status"] == "WARN"
    assert snap["cuda_available"] is False


def test_mocked_cuda_snapshot(monkeypatch):
    class Props:
        name = "fake-gpu"
        total_memory = 16 * 1024**3

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def get_device_properties(index):
            return Props()

        @staticmethod
        def memory_allocated(index):
            return 1024**3

        @staticmethod
        def memory_reserved(index):
            return 2 * 1024**3

        @staticmethod
        def max_memory_allocated(index):
            return 3 * 1024**3

        @staticmethod
        def mem_get_info(index):
            return (8 * 1024**3, 16 * 1024**3)

    monkeypatch.setitem(__import__("sys").modules, "torch", types.SimpleNamespace(cuda=FakeCuda()))
    snap = gma.capture_gpu_memory_snapshot("cuda")
    assert snap["status"] == "PASS"
    assert snap["devices"][0]["free_memory_gb"] == 8.0


def test_empty_cache_action_recorded(monkeypatch):
    called = {"empty": False}

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def empty_cache():
            called["empty"] = True

    monkeypatch.setitem(__import__("sys").modules, "torch", types.SimpleNamespace(cuda=FakeCuda()))
    report = gma.cuda_empty_cache("test")
    assert report["empty_cache_called"] is True
    assert called["empty"] is True


def test_memory_report_writes_json(tmp_path):
    out = tmp_path / "gpu.json"
    report = gma.write_gpu_memory_report(out, [{"stage": "x", "status": "PASS"}])
    assert report["status"] == "PASS"
    assert json.loads(out.read_text(encoding="utf-8"))["snapshot_count"] == 1
