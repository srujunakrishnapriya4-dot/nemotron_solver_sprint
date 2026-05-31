from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import read_json, write_json_checked


ART = Path("artifacts/sprint11")
DEFAULTS = {
    "orchestrator": ART / "day8_2b_smoke_orchestrator_report.json",
    "memory": ART / "day8_2c_gpu_memory_report.json",
    "preflight": ART / "day8_2c_smoke_preflight_report.json",
    "smoke_train": ART / "day8_2_smoke_train_summary.json",
    "adapter_artifact": ART / "day8_2_adapter_artifact_report.json",
    "smoke_eval": ART / "day8_2b_smoke_eval_report.json",
    "output_drift": ART / "day8_2b_output_drift_report.json",
}


def build_smoke_readiness_report(paths: dict[str, str | Path] | None = None) -> dict[str, Any]:
    paths = {**DEFAULTS, **(paths or {})}
    reports: dict[str, Any] = {}
    missing: list[str] = []
    for name, path in paths.items():
        p = Path(path)
        if p.exists():
            reports[name] = read_json(p)
        elif name != "preflight":
            missing.append(name)
            reports[name] = {"status": "MISSING"}
    failures: list[str] = []
    warnings: list[str] = []
    orch = reports.get("orchestrator", {})
    memory = reports.get("memory", {})
    if missing:
        failures.extend(f"missing_{name}" for name in missing)
    if memory.get("status") == "FAIL":
        failures.append("memory_audit_failed")
    if orch.get("decision") == "NEEDS_KAGGLE_SMOKE_TRAIN":
        return _report("WARN", "NEEDS_KAGGLE_SMOKE_TRAIN", False, reports, warnings, failures)
    if orch.get("decision") != "ALLOW_DAY8_3_FULL_V2A_TRAINING":
        failures.append("orchestrator_not_allowing_full_training")
    for name, block in (
        ("smoke_train", "BLOCK_FULL_TRAINING_SMOKE_TRAIN"),
        ("adapter_artifact", "BLOCK_FULL_TRAINING_RUNTIME"),
        ("smoke_eval", "BLOCK_FULL_TRAINING_SMOKE_EVAL"),
        ("output_drift", "BLOCK_FULL_TRAINING_OUTPUT_DRIFT"),
    ):
        if reports.get(name, {}).get("status") != "PASS":
            failures.append(f"{name}_not_pass")
            if not any(item.startswith("missing_") for item in failures):
                return _report("FAIL", block, False, reports, warnings, failures)
    if memory.get("status") not in {"PASS", "WARN"}:
        return _report("FAIL", "BLOCK_FULL_TRAINING_MEMORY_BACKEND_BUG", False, reports, warnings, failures)
    if failures:
        decision = "BLOCK_FULL_TRAINING_MEMORY_BACKEND_BUG" if "memory_audit_failed" in failures else "BLOCK_FULL_TRAINING_RUNTIME"
        return _report("FAIL", decision, False, reports, warnings, failures)
    return _report("PASS", "ALLOW_DAY8_3_FULL_V2A_TRAINING", True, reports, warnings, failures)


def _report(status: str, decision: str, allowed: bool, reports: dict[str, Any], warnings: list[str], failures: list[str]) -> dict[str, Any]:
    return {
        "status": status,
        "decision": decision,
        "full_training_allowed": allowed,
        "reports": {key: value.get("status") for key, value in reports.items()},
        "warnings": warnings,
        "failures": failures,
        "packaging_allowed": False,
        "submission_allowed": False,
        "no_leaderboard_evidence": True,
        "no_0_95_evidence": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = build_smoke_readiness_report()
    write_json_checked(args.out, report, field_name="day8_2c_smoke_readiness_report")
    print(json.dumps({"status": report["status"], "decision": report["decision"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] in {"PASS", "WARN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
