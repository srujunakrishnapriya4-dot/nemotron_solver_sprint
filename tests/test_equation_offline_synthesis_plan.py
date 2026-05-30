from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.equation_offline_synthesis_plan import build_equation_offline_synthesis_plan  # noqa: E402


def test_equation_offline_synthesis_plan_is_cpu_only(tmp_path: Path) -> None:
    rows = [{"id": "a"}, {"id": "b"}]
    path = tmp_path / "candidates.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    config = build_equation_offline_synthesis_plan(path, tmp_path)
    assert config["candidate_count"] == 2
    assert "gpu_training" in config["forbidden"]
    assert (tmp_path / "EQUATION_OFFLINE_SYNTHESIS_PLAN.md").exists()
