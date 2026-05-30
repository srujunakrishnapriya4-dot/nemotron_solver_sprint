from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.trace_style_policy import make_short_trace, validate_trace_style  # noqa: E402


def test_trace_style_policy_short_only() -> None:
    assert validate_trace_style(make_short_trace("rule", "42"))
    assert not validate_trace_style("word " * 100)
