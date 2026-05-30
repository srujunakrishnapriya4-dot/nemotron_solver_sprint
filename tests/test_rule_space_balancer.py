from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.rule_space_balancer import balance_rule_space  # noqa: E402


def test_rule_space_balancer_caps_family() -> None:
    rows = [{"family": "equation_symbolic", "private_like_score": 0.9, "generation_hash": str(i)} for i in range(5)]
    assert len(balance_rule_space(rows, {"equation_symbolic": 2})) == 2
