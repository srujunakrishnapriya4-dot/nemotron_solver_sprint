from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nemotron_engine.core.schemas import stable_hash


def build_compute_budget_plan(
    output_json: str | Path = "artifacts/anti086/compute_budget_plan.json",
    output_md: str | Path = "artifacts/anti086/COMPUTE_BUDGET_PLAN.md",
) -> dict[str, Any]:
    schedule = [
        {"day": 2, "tasks": ("finish gates", "micro training only", "smoke eval"), "forbidden": ("v1/v2/v3 broad training", "packaging custom child")},
        {"day": 3, "tasks": ("v1 only if micro passes", "eval public_like_60", "eval family_hard_60"), "forbidden": ("v2 if v1 no private-like signal",)},
        {"day": 4, "tasks": ("v2 adversarial synthetic if v1 passes", "eval rule_holdout_60", "eval family_hard_60"), "forbidden": ("scale synthetic-only gains",)},
        {"day": 5, "tasks": ("v3 contrastive if v2 passes", "check no public-like collapse"), "forbidden": ("GRPO", "DPO without SFT stability")},
        {"day": 6, "tasks": ("final selected recipe training", "package candidate only if gates pass"), "forbidden": ("new recipe exploration",)},
        {"day": 7, "tasks": ("submit 1-2 strongest candidates manually", "record public score", "choose final"), "forbidden": ("public-only overfit chase",)},
        {"day": 8, "tasks": ("freeze best candidate", "fallback to parent if evidence weak"), "forbidden": ("last-minute blind training",)},
    ]
    stop_loss = (
        "if micro fails, stop custom training",
        "if v1 has no private-like improvement, do not run v2",
        "if v2 only improves synthetic holdout, stop v2",
        "if public-like collapses, reject candidate",
        "if adapter size explodes, reject candidate",
    )
    payload = {
        "day_start": 2,
        "days_remaining": 8,
        "amd_credits_usd": 200,
        "kaggle_gpu_policy": "scarce; micro gates first",
        "amd_policy": "CPU-heavy synthesis/corpus/token validation; final training only if ROCm stack proven",
        "schedule": schedule,
        "stop_loss": stop_loss,
    }
    payload["plan_hash"] = stable_hash(payload)
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    Path(output_md).write_text(_markdown(payload), encoding="utf-8")
    return payload


def _markdown(payload: dict[str, Any]) -> str:
    lines = ["# Anti086 Compute Budget Plan", "", f"AMD credits: ${payload['amd_credits_usd']}", ""]
    for day in payload["schedule"]:
        lines.append(f"## Day {day['day']}")
        lines.append("Tasks: " + ", ".join(day["tasks"]))
        lines.append("Forbidden: " + ", ".join(day["forbidden"]))
        lines.append("")
    lines.append("## Stop Loss")
    lines.extend(f"- {item}" for item in payload["stop_loss"])
    return "\n".join(lines) + "\n"
