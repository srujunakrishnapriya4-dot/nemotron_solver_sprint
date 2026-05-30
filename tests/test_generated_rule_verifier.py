from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.generated_rule_verifier import verify_generated_rule  # noqa: E402


def test_generated_rule_verifier_rejects_ambiguous() -> None:
    row = {"prompt": "p", "answer": "1", "family": "bit", "rule_id": "r", "generation_hash": "h", "ambiguity_score": 0.9, "difficulty_score": 0.8}
    assert verify_generated_rule(row)["verifier_status"] == "reject"
