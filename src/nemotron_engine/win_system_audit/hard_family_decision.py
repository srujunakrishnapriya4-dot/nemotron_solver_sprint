from __future__ import annotations

from pathlib import Path
from typing import Any

from nemotron_engine.anti086_barrier.bit_numeric_pivot_plan import build_bit_numeric_pivot_plan
from nemotron_engine.anti086_barrier.hard_family_roi_analyzer import analyze_hard_family_roi
from .equation_quarantine_policy import build_equation_quarantine


def make_hard_family_decision(output_dir: str | Path = "artifacts/win_system") -> dict[str, Any]:
    roi = analyze_hard_family_roi(output_dir, Path(output_dir) / "hard_family_roi_report.json")
    quarantine = build_equation_quarantine(output_dir, output_dir)
    pivot = build_bit_numeric_pivot_plan(Path(output_dir) / "solver_coverage_report.json", output_dir)
    decision = {
        "equation_action": roi["families"]["equation_symbolic"]["recommended_action"],
        "equation_status": quarantine["equation_status"],
        "pivot_recommendation": pivot["recommended_next_sprint"],
        "gpu_main_allowed": False,
        "reason": "equation coverage stalled; unsafe equation traces quarantined; bit/numeric have higher one-day ROI",
    }
    return decision
