from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import write_json_checked
from kaggle_anti086.training.training_config_schema import STATUS_FAIL, STATUS_NOT_RUNNABLE, STATUS_PARENT_MISSING, load_training_config, validate_training_config


def build_adapter_constraint_audit(config_dir: str | Path) -> dict:
    configs = []
    for path in sorted(Path(config_dir).glob("*.yaml")):
        result = validate_training_config(load_training_config(path), path=path)
        configs.append({"path": result.path, "stage": result.stage, "status": result.status, "runnable": result.runnable, "warnings": result.warnings, "failures": result.failures})
    failed = [cfg for cfg in configs if cfg["status"] == STATUS_FAIL]
    blocked = [cfg for cfg in configs if cfg["status"] in {STATUS_PARENT_MISSING, STATUS_NOT_RUNNABLE}]
    runnable = [cfg for cfg in configs if cfg["runnable"]]
    failures = []
    if failed:
        failures.append("failed_configs_present")
    if not runnable:
        failures.append("no_runnable_configs")
    return {
        "status": "PASS" if not failures else "FAIL",
        "configs_checked": len(configs),
        "runnable_configs": [cfg["stage"] for cfg in runnable],
        "blocked_configs": [cfg["stage"] for cfg in blocked],
        "failed_configs": [cfg["stage"] for cfg in failed],
        "rank_ok": not any("rank_above_max" in cfg["failures"] for cfg in configs),
        "target_modules_ok": not any(any("target_modules" in failure or "lm_head" in failure for failure in cfg["failures"]) for cfg in configs),
        "paths_safe": not any(any("unsafe_output_adapter_dir" in failure for failure in cfg["failures"]) for cfg in configs),
        "parent_paths_ok": not any(cfg["status"] == STATUS_FAIL and "parent" in " ".join(cfg["failures"]) for cfg in configs),
        "v3_blocked": any(cfg["stage"] == "v3_hard_family_repair" and cfg["status"] == STATUS_NOT_RUNNABLE for cfg in configs),
        "warnings": [warning for cfg in configs for warning in cfg["warnings"]],
        "failures": failures,
        "configs": configs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = build_adapter_constraint_audit(args.config_dir)
    write_json_checked(args.out, report, field_name="day7_adapter_constraint_report")
    print(json.dumps({"status": report["status"], "runnable_configs": report["runnable_configs"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
