from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.kaggle_path_safety import safe_write_text


def compare_reports(solver_report: Path, base_report: Path | None = None, parent_report: Path | None = None) -> dict:
    solver = _load_required(solver_report)
    base = _load_optional(base_report)
    parent = _load_optional(parent_report)
    return {
        "schema_version": 1,
        "created_by": "SPRINT-11D",
        "solver": _summary(solver),
        "base_status": "NOT_RUN" if base is None else base.get("execution_status", "UNKNOWN"),
        "base": None if base is None else _summary(base),
        "parent_status": "NOT_RUN" if parent is None else parent.get("execution_status", "UNKNOWN"),
        "parent": None if parent is None else _summary(parent),
        "model_results_faked": False,
        "required_next_commands": [
            "Run Kaggle GPU parent/base eval before claiming model evidence.",
            "Do not package or submit from solver-only reports.",
        ],
    }


def _summary(report: dict) -> dict:
    keys = ["execution_status", "quality_status", "row_count", "behavior_accuracy", "answerable_exact_match", "unsafe_answer_rate", "correct_abstain_rate"]
    return {key: report.get(key) for key in keys if key in report}


def _load_required(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare solver/base/parent reports without faking missing model runs.")
    parser.add_argument("--solver-report", required=True)
    parser.add_argument("--base-report")
    parser.add_argument("--parent-report")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = compare_reports(
        Path(args.solver_report),
        None if args.base_report is None else Path(args.base_report),
        None if args.parent_report is None else Path(args.parent_report),
    )
    safe_write_text(args.out, json.dumps(report, indent=2, sort_keys=True) + "\n", field_name="solver_base_parent_comparison")
    print(json.dumps({"status": "WROTE", "out": args.out, "base_status": report["base_status"], "parent_status": report["parent_status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
