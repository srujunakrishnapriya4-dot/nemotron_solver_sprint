from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.win_mode_program import build_win_mode_program  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_win_mode_program_builds_planning_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_jsonl(Path("artifacts/anti086/adversarial_synthetic_filtered.jsonl"), [
        {"id": f"s{i}", "family": "bit_manipulation", "rule_id": f"r{i%3}", "prompt": f"p {i}", "answer": "1", "generation_hash": f"g{i}", "ambiguity_score": 0.1, "novelty_score": 0.8, "difficulty_score": 0.8, "private_like_score": 0.8}
        for i in range(40)
    ])
    _write_jsonl(Path("artifacts/vex_progress/corpus_mixed_main.jsonl"), [
        {"id": f"b{i}", "family": "bit_manipulation", "corpus_type": "direct_raw", "text": "User:\np\nAssistant:\na", "answer": "a", "row_hash": f"b{i}"}
        for i in range(80)
    ])
    _write_jsonl(Path("artifacts/anti086/contrastive_error_corpus.jsonl"), [
        {"id": f"c{i}", "family": "equation_symbolic", "corpus_type": "contrastive_error", "text": "t", "answer": "1", "row_hash": f"c{i}"}
        for i in range(10)
    ])
    Path("artifacts/anti086/private_like_manifest.json").write_text(json.dumps({"validation_hashes": []}), encoding="utf-8")

    report = build_win_mode_program("artifacts/anti086")

    assert report["backend"]["verdict"]["amd_cloud_default"] == "data_generation_and_validation_only"
    assert (Path("artifacts/anti086/winmode_curriculum_report.json")).exists()
    assert (Path("artifacts/anti086/DAY8_SUBMISSION_STRATEGY.md")).exists()

