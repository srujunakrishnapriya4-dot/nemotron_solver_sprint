from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_anti086"))

import kaggle_backend_probe  # noqa: E402


def test_backend_probe_detects_missing_cuda_as_unsafe(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(kaggle_backend_probe, "has_module", lambda _name: False)
    report = kaggle_backend_probe.probe_backend(base_model_path=str(tmp_path / "missing_model"), input_base=tmp_path / "input", output_path=tmp_path / "probe.json")

    assert report["recommended_mode"] == "BACKEND_NOT_SAFE"
    assert report["cuda_available"] is False
    assert report["micro_stack_test_label"].startswith("INFRASTRUCTURE STACK TEST ONLY")


def test_backend_probe_finds_win_or_anti086_input_root(tmp_path: Path) -> None:
    root = tmp_path / "input" / "win"
    root.mkdir(parents=True)
    (root / "win_corpus_manifest.json").write_text("{}", encoding="utf-8")
    (root / "win_micro.jsonl").write_text("{}\n", encoding="utf-8")

    assert kaggle_backend_probe.detect_anti086_input_root(tmp_path / "input") == str(root)
