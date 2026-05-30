from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.adversarial_curriculum_optimizer import build_winmode_curricula  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_adversarial_curriculum_caps_synthetic_and_contrastive(tmp_path: Path) -> None:
    base = [{"id": f"b{i}", "family": "bit_manipulation", "corpus_type": "direct_raw", "text": "User:\np\nAssistant:\na", "answer": "a"} for i in range(120)]
    synth = [{"id": f"s{i}", "family": "equation_symbolic", "rule_id": "r", "prompt": "p", "answer": "1", "generation_hash": f"g{i}"} for i in range(80)]
    contrast = [{"id": f"c{i}", "family": "gravity_numeric", "corpus_type": "contrastive_error", "text": "t", "answer": "1"} for i in range(40)]
    _write_jsonl(tmp_path / "base.jsonl", base)
    _write_jsonl(tmp_path / "synth.jsonl", synth)
    _write_jsonl(tmp_path / "contrast.jsonl", contrast)
    (tmp_path / "private.json").write_text(json.dumps({"validation_hashes": []}), encoding="utf-8")

    report = build_winmode_curricula(tmp_path / "base.jsonl", tmp_path / "synth.jsonl", tmp_path / "contrast.jsonl", tmp_path / "private.json", tmp_path)

    assert report["synthetic_ratio_v2"] <= 0.25
    assert report["contrastive_ratio_v3"] <= 0.10
    assert report["file_counts"]["winmode_micro.jsonl"] <= 64

