from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.contrastive_error_corpus import build_contrastive_error_corpus  # noqa: E402


def test_contrastive_corpus_capped() -> None:
    tmp = Path("tests/.tmp_contrastive")
    tmp.mkdir(exist_ok=True)
    src = tmp / "verified.jsonl"
    rows = [{"id": f"r{i}", "prompt": "p", "answer": "a", "family": "bit", "rule_id": "rule", "verified_status": "verified_correct"} for i in range(100)]
    src.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    try:
        manifest = build_contrastive_error_corpus(src, tmp, max_ratio_base_count=100)
        assert manifest["contrastive_count"] == 15
    finally:
        for path in tmp.iterdir():
            path.unlink()
        tmp.rmdir()

