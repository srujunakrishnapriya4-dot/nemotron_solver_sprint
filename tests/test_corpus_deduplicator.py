from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.corpus_deduplicator import deduplicate_corpus  # noqa: E402


def test_corpus_deduplicator_removes_duplicates() -> None:
    rows = [{"family": "a", "text": "x", "answer": "1"}, {"family": "a", "text": "x", "answer": "1"}]
    assert len(deduplicate_corpus(rows)) == 1
