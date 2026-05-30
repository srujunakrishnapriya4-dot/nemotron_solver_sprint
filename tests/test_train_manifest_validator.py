from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.training_contract.train_manifest_validator import validate_train_manifest  # noqa: E402


def test_train_manifest_validator_requires_contract_fields() -> None:
    manifest = {"row_count": 1, "total_tokens": 2, "supervised_tokens": 1, "prompt_leakage_count": 0, "zero_supervised_rows": 0, "max_seq_len": 10, "truncation_count": 0, "corpus_hash": "h"}
    assert validate_train_manifest(manifest) == manifest
