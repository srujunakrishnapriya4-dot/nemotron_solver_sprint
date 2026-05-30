from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.corpus_quality_manifest import build_corpus_quality_manifest  # noqa: E402


def test_corpus_quality_manifest_reports_ratios() -> None:
    manifest = build_corpus_quality_manifest({"win_v2.jsonl": [{"family": "a", "corpus_type": "synthetic_verified"}], "win_v3.jsonl": [{"family": "a", "corpus_type": "contrastive_error"}]})
    assert manifest["synthetic_ratio_v2"] == 1.0
    assert manifest["contrastive_ratio_v3"] == 1.0
