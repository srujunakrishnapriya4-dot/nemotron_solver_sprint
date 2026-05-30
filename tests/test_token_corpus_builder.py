from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.vex_progress_bridge.token_corpus_builder import build_vex_text_corpus  # noqa: E402


def write_verified(path: Path) -> None:
    rows = [
        {"id": "a", "prompt": "p", "answer": "1", "family": "bit", "verified_status": "verified_correct", "rule_id": "solved"},
        {"id": "b", "prompt": "p", "answer": "2", "family": "unit", "verified_status": "abstained", "rule_id": "none"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_token_corpus_builder_uses_verified_correct_only_for_traces(tmp_path: Path) -> None:
    verified = tmp_path / "verified_rules.jsonl"
    write_verified(verified)

    manifest = build_vex_text_corpus(verified, tmp_path)
    traces = (tmp_path / "corpus_short_rule_trace.jsonl").read_text(encoding="utf-8").splitlines()

    assert manifest["corpus_rows_by_type"]["direct_raw"] == 2
    assert manifest["corpus_rows_by_type"]["short_rule_trace"] == 1
    assert "synthetic_verified" in manifest["corpus_rows_by_type"]

