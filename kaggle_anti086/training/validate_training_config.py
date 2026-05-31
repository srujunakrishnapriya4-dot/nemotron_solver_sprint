from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import write_json_checked
from kaggle_anti086.training.training_config_schema import STATUS_FAIL, load_training_config, validate_training_config


def validate_config_dir(config_dir: str | Path) -> dict:
    reports = []
    for path in sorted(Path(config_dir).glob("*.yaml")):
        result = validate_training_config(load_training_config(path), path=path)
        reports.append(
            {
                "path": result.path,
                "stage": result.stage,
                "status": result.status,
                "runnable": result.runnable,
                "warnings": result.warnings,
                "failures": result.failures,
            }
        )
    failed = [report for report in reports if report["status"] == STATUS_FAIL]
    runnable = [report for report in reports if report["runnable"]]
    return {
        "status": "PASS" if not failed and runnable else "FAIL",
        "config_count": len(reports),
        "runnable_count": len(runnable),
        "blocked_count": len([r for r in reports if not r["runnable"] and r["status"] != STATUS_FAIL]),
        "failed_count": len(failed),
        "configs": reports,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = validate_config_dir(args.config_dir)
    out_path = Path(args.out)
    stage_to_name = {
        "v2a_base_lora": "day7_v2a_config_report.json",
        "v2b_tinker_parent_lora": "day7_v2b_config_report.json",
        "v2c_best_parent_lora": "day7_v2c_config_report.json",
        "v3_hard_family_repair": "day7_v3_config_report.json",
    }
    for config in report["configs"]:
        name = stage_to_name.get(config["stage"])
        if name:
            write_json_checked(out_path.parent / name, config, field_name=name)
    write_json_checked(out_path, report, field_name="day7_config_validation_report")
    print(json.dumps({"status": report["status"], "runnable_count": report["runnable_count"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
