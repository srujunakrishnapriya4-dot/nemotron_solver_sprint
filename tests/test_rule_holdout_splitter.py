from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.rule_holdout_splitter import split_rule_holdout_v2  # noqa: E402


def test_rule_holdout_splitter_excludes_rule_ids() -> None:
    split = split_rule_holdout_v2([{"rule_id": "a"}, {"rule_id": "b"}], holdout_fraction=0.5)
    assert set(row["rule_id"] for row in split["train"]).isdisjoint(split["holdout_rule_ids"])
