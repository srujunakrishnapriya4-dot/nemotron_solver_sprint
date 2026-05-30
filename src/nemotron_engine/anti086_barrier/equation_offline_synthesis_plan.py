from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nemotron_engine.core.schemas import stable_hash


def build_equation_offline_synthesis_plan(
    candidate_path: str | Path = "artifacts/win_system/equation_offline_synthesis_candidates.jsonl",
    output_dir: str | Path = "artifacts/win_system",
) -> dict[str, Any]:
    candidates = _read_jsonl(Path(candidate_path))
    config = {
        "candidate_count": len(candidates),
        "execution": "cpu_or_amd_cloud_only",
        "parallelism": "multiprocessing_by_row",
        "candidate_limits": {
            "programs_per_row": 250000,
            "wall_clock_hours": 6,
            "cache_program_signatures": True,
        },
        "families_first": ["digit_symbol_arithmetic", "binary_operator_arithmetic", "modulo_exact_division"],
        "stop_criteria": [
            "equation_verified_correct >= 120",
            "safe_new_verified_programs >= 75",
            "no meaningful increase after time_box",
        ],
        "forbidden": ["model_load", "gpu_training", "target_answer_fitting", "one_example_extrapolation"],
        "success_criteria": {"equation_verified_correct": 120, "safe_new_verified_programs": 75},
        "failure_criteria": "no meaningful verified increase after time-box",
    }
    config["config_hash"] = stable_hash(config)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "equation_offline_synthesis_config.json").write_text(json.dumps(config, sort_keys=True, indent=2), encoding="utf-8")
    (out / "EQUATION_OFFLINE_SYNTHESIS_PLAN.md").write_text(_plan_md(config), encoding="utf-8")
    return config


def _plan_md(config: dict[str, Any]) -> str:
    lines = [
        "# Equation Offline Synthesis Plan",
        "",
        f"Candidates: {config['candidate_count']}",
        "",
        "Run this only on CPU/AMD resources. It loads no model and performs no training.",
        "",
        "Priority families:",
    ]
    lines.extend(f"- {family}" for family in config["families_first"])
    lines.extend(["", "Stop criteria:"])
    lines.extend(f"- {item}" for item in config["stop_criteria"])
    lines.extend(["", "Forbidden:"])
    lines.extend(f"- {item}" for item in config["forbidden"])
    return "\n".join(lines) + "\n"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
