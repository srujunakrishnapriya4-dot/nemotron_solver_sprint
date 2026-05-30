from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nemotron_engine.core.schemas import stable_hash


def build_bit_numeric_pivot_plan(
    coverage_path: str | Path = "artifacts/win_system/solver_coverage_report.json",
    output_dir: str | Path = "artifacts/win_system",
) -> dict[str, Any]:
    coverage = _read_json(Path(coverage_path))
    abstain = coverage.get("top_abstention_reasons", {})
    plan = {
        "bit": {
            "remaining_bit_candidate_budget_exhausted": int(abstain.get("bit_candidate_budget_exhausted", 0)),
            "recommended": "SPRINT-7.7_BIT_BREAKTHROUGH",
            "implementation": [
                "hashed truth-table pruning",
                "vectorized candidate search",
                "depth 3/4 only after pruning",
                "majority/choice/mask caching",
                "ambiguity clustering",
            ],
        },
        "numeric": {
            "remaining_inconsistent_numeric_precision": int(abstain.get("inconsistent_numeric_precision", 0)),
            "recommended": "SPRINT-7.7_NUMERIC_PRECISION",
            "implementation": [
                "interval rounding inference",
                "nearest/floor/ceil/fixed-decimal detection",
                "output-format matching",
                "justified exact/tolerant comparison only",
            ],
        },
        "equation": {
            "recommended": "pause_template_patching",
            "next": "SPRINT-7.7_EQUATION_OFFLINE_SYNTHESIS only if CPU/AMD search is run",
        },
    }
    if plan["bit"]["remaining_bit_candidate_budget_exhausted"] >= plan["numeric"]["remaining_inconsistent_numeric_precision"]:
        plan["recommended_next_sprint"] = "SPRINT-7.7_BIT_BREAKTHROUGH"
    else:
        plan["recommended_next_sprint"] = "SPRINT-7.7_NUMERIC_PRECISION"
    plan["plan_hash"] = stable_hash(plan)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "bit_numeric_pivot_config.json").write_text(json.dumps(plan, sort_keys=True, indent=2), encoding="utf-8")
    (out / "BIT_NUMERIC_PIVOT_PLAN.md").write_text(_plan_md(plan), encoding="utf-8")
    return plan


def _plan_md(plan: dict[str, Any]) -> str:
    return (
        "# Bit/Numeric Pivot Plan\n\n"
        f"Recommended next sprint: {plan['recommended_next_sprint']}\n\n"
        f"Bit budget exhausted rows: {plan['bit']['remaining_bit_candidate_budget_exhausted']}\n"
        f"Numeric precision blockers: {plan['numeric']['remaining_inconsistent_numeric_precision']}\n\n"
        "Equation should pause for offline synthesis instead of more random templates.\n"
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
