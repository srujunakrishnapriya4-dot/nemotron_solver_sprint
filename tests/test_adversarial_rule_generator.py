from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.adversarial_rule_generator import generate_adversarial_synthetic  # noqa: E402


def test_hard_synthetic_generation_produces_nonzero_feasible_families(tmp_path: Path) -> None:
    manifest = generate_adversarial_synthetic(tmp_path, targets={"bit_manipulation": 4, "cipher_symbol": 3, "unit_gravity": 4, "equation_operator": 3})
    rows = [json.loads(line) for line in (tmp_path / "adversarial_synthetic.jsonl").read_text(encoding="utf-8").splitlines()]

    assert manifest["total_generated"] == 14
    assert {row["family"] for row in rows} >= {"bit_manipulation", "cipher_text", "unit_conversion", "gravity_numeric", "equation_symbolic"}
    assert all(row["verification_trace"]["verified"] for row in rows)

