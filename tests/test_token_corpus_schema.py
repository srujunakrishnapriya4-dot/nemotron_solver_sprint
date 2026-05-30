from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.vex_progress_bridge.token_corpus_schema import VexTokenCorpusError, VexTokenCorpusRow  # noqa: E402


def test_token_corpus_schema_accepts_text_row() -> None:
    row = VexTokenCorpusRow(row_id="1", source_id="s", family="bit", corpus_type="direct_raw", text="User:\nx\nAssistant:\ny", answer="y")

    assert row.row_hash


def test_token_corpus_schema_rejects_malformed_rows() -> None:
    with pytest.raises(VexTokenCorpusError):
        VexTokenCorpusRow(row_id="1", source_id="s", family="bit", corpus_type="bad", text="x", answer="y")

    with pytest.raises(VexTokenCorpusError):
        VexTokenCorpusRow(row_id="1", source_id="s", family="bit", corpus_type="direct_raw", text="x", answer="y", metadata={"test_id": "x"})


def test_token_corpus_schema_rejects_length_mismatch() -> None:
    with pytest.raises(VexTokenCorpusError):
        VexTokenCorpusRow(row_id="1", source_id="s", family="bit", corpus_type="direct_raw", text="x", answer="y", token_ids=[1], target_ids=[1, 2])

