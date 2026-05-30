from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.novelty_filter_v2 import filter_novelty_v2  # noqa: E402


def test_novelty_filter_v2_rejects_near_duplicate() -> None:
    rows = [{"prompt": "same 1", "nearest_train_template_distance": 0.1}, {"prompt": "new 2", "nearest_train_template_distance": 0.8}]
    assert len(filter_novelty_v2(rows)) == 1
