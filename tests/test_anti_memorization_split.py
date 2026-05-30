from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.anti_memorization_split import exclude_validation_hashes, split_rule_holdout  # noqa: E402


def test_rule_holdout_split_excludes_rule_ids_from_training() -> None:
    rows = [{"id": str(i), "rule_id": f"r{i%4}", "generation_hash": f"h{i}"} for i in range(20)]
    split = split_rule_holdout(rows)

    assert not ({row["rule_id"] for row in split["train"]} & {row["rule_id"] for row in split["rule_holdout"]})


def test_validation_hashes_excluded_from_training() -> None:
    rows = [{"id": "a", "generation_hash": "x"}, {"id": "b", "generation_hash": "y"}]

    assert exclude_validation_hashes(rows, [{"generation_hash": "x"}]) == [{"id": "b", "generation_hash": "y"}]
