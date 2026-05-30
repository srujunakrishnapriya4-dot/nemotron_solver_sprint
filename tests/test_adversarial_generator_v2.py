from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.adversarial_generator_v2 import generate_adversarial_v2  # noqa: E402


def test_adversarial_generator_v2_outputs_verified_fields(tmp_path: Path) -> None:
    manifest = generate_adversarial_v2(tmp_path, targets={"equation_symbolic": 5, "bit_manipulation": 5})
    rows = [json.loads(line) for line in (tmp_path / "adversarial_v2_filtered.jsonl").read_text(encoding="utf-8").splitlines()]
    assert manifest["filtered_count"] > 0
    assert all(row["verifier_status"] == "verified" for row in rows)
    assert all("nearest_train_template_distance" in row for row in rows)
    assert "unsafe equation train rows are not used" in manifest["equation_source_policy"]
