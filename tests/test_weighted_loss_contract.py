from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.training_contract.weighted_loss_contract import weighted_loss_sum  # noqa: E402


def test_weighted_loss_contract_averages_supervised_tokens() -> None:
    assert weighted_loss_sum([10.0, 2.0], [0.0, 1.0]) == 2.0
