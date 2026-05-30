from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.family_gap_curriculum import build_family_gap_curricula  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_curriculum_ratios_enforced(tmp_path: Path) -> None:
    base = [{"id": f"b{i}", "family": "bit_manipulation", "corpus_type": "direct_raw", "text": "User:\np\nAssistant:\na", "answer": "a"} for i in range(100)]
    synth = [{"id": f"s{i}", "family": "unit_conversion", "rule_id": "r", "prompt": "p", "answer": "1", "generation_hash": f"g{i}"} for i in range(60)]
    contrast = [{"id": f"c{i}", "family": "gravity_numeric", "corpus_type": "contrastive_error", "text": "t", "answer": "1"} for i in range(30)]
    write_jsonl(tmp_path / "base.jsonl", base)
    write_jsonl(tmp_path / "synth.jsonl", synth)
    write_jsonl(tmp_path / "contrast.jsonl", contrast)
    (tmp_path / "private_like_manifest.json").write_text(json.dumps({"validation_hashes": []}), encoding="utf-8")

    manifest = build_family_gap_curricula(tmp_path / "base.jsonl", tmp_path / "synth.jsonl", tmp_path / "contrast.jsonl", tmp_path / "private_like_manifest.json", tmp_path)

    assert manifest["synthetic_ratio_v2"] <= 0.35
    assert manifest["contrastive_ratio_v3"] <= 0.15
    assert manifest["file_counts"]["corpus_anti086_micro.jsonl"] <= 64

