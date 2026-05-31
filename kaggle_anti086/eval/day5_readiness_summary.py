from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.kaggle_path_safety import safe_write_text


DEFAULT_PATHS = {
    "private_like": Path("artifacts/sprint11/day5_private_like_solver_report.json"),
    "rule_holdout": Path("artifacts/sprint11/day5_rule_holdout_solver_report.json"),
    "family_hard": Path("artifacts/sprint11/day5_family_hard_solver_report.json"),
    "anti_leak": Path("artifacts/sprint11/day5_anti_leak_solver_report.json"),
    "private_like_answerable": Path("artifacts/sprint11/day5_private_like_answerable_solver_report.json"),
    "rule_holdout_answerable": Path("artifacts/sprint11/day5_rule_holdout_answerable_solver_report.json"),
    "family_hard_answerable": Path("artifacts/sprint11/day5_family_hard_answerable_solver_report.json"),
    "manifest": Path("artifacts/sprint11/day5_eval_manifest.json"),
    "weak_subfamily": Path("artifacts/sprint11/day5_weak_subfamily_report.json"),
    "v2_eligibility": Path("artifacts/sprint11/day5_v2_eligibility_report.json"),
    "semantic_holdout": Path("artifacts/sprint11/day5_2_semantic_holdout_report.json"),
}


def build_readiness_summary(paths: dict[str, Path] | None = None) -> dict:
    paths = paths or DEFAULT_PATHS
    source_reports = {name: _load(path) for name, path in paths.items()}
    gates = {
        "private_like_answerable_exact_match": _metric_gate(source_reports, "private_like_answerable", "answerable_exact_match", 0.90),
        "rule_holdout_answerable_exact_match": _metric_gate(source_reports, "rule_holdout_answerable", "answerable_exact_match", 0.90),
        "family_hard_answerable_exact_match": _metric_gate(source_reports, "family_hard_answerable", "answerable_exact_match", 0.90),
        "unsafe_answer_rate_max": _unsafe_gate(source_reports, 0.0),
        "eligible_verified_answer_count": _metric_gate(source_reports, "v2_eligibility", "eligible_verified_answer_count", 1000),
        "eligible_abstain_safety_count": _metric_gate(source_reports, "v2_eligibility", "eligible_abstain_safety_count", 200),
        "format_only_ready": _format_only_gate(source_reports),
        "semantic_holdout_pass": _bool_gate(source_reports, "semantic_holdout", "semantic_holdout_pass", True),
    }
    remaining = [name for name, gate in gates.items() if gate["status"] != "PASS"]
    missing = [name for name, payload in source_reports.items() if payload is None]
    status = "PASS" if not remaining else "FAIL"
    return {
        "schema_version": 1,
        "status": status,
        "decision": "ALLOW_DAY6_CORPUS_BUILD" if status == "PASS" else "BLOCK_DAY6_CORPUS_BUILD",
        "training_allowed": False,
        "packaging_allowed": False,
        "submission_allowed": False,
        "reason": "all Day 5.2 readiness gates passed" if status == "PASS" else "one or more Day 5.2 readiness gates failed",
        "gates": gates,
        "remaining_blockers": remaining + [f"missing_{name}" for name in missing],
        "source_reports": {name: {"path": str(paths[name]), "present": payload is not None, "status": None if payload is None else payload.get("status", payload.get("quality_status"))} for name, payload in source_reports.items()},
        "no_training_evidence_exists": True,
        "no_leaderboard_evidence_exists": True,
        "no_0_95_evidence_exists": True,
    }


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_gate(reports: dict[str, dict | None], report_name: str, key: str, threshold: float) -> dict:
    report = reports.get(report_name)
    if report is None:
        return {"value": None, "threshold": threshold, "status": "MISSING"}
    value = report.get(key)
    if value is None:
        return {"value": None, "threshold": threshold, "status": "MISSING"}
    return {"value": value, "threshold": threshold, "status": "PASS" if float(value) >= threshold else "FAIL"}


def _bool_gate(reports: dict[str, dict | None], report_name: str, key: str, threshold: bool) -> dict:
    report = reports.get(report_name)
    if report is None or key not in report:
        return {"value": None, "threshold": threshold, "status": "MISSING"}
    value = bool(report.get(key))
    return {"value": value, "threshold": threshold, "status": "PASS" if value is threshold else "FAIL"}


def _unsafe_gate(reports: dict[str, dict | None], threshold: float) -> dict:
    names = ("private_like", "rule_holdout", "family_hard", "anti_leak", "private_like_answerable", "rule_holdout_answerable", "family_hard_answerable")
    values = []
    missing = False
    for name in names:
        report = reports.get(name)
        if report is None or "unsafe_answer_rate" not in report:
            missing = True
            continue
        values.append(float(report["unsafe_answer_rate"]))
    if missing and not values:
        return {"value": None, "threshold": threshold, "status": "MISSING"}
    value = max(values) if values else None
    status = "MISSING" if missing else ("PASS" if value <= threshold else "FAIL")
    if value is not None and value > threshold:
        status = "FAIL"
    return {"value": value, "threshold": threshold, "status": status}


def _format_only_gate(reports: dict[str, dict | None]) -> dict:
    weak = reports.get("weak_subfamily")
    if weak is None:
        return {"value": None, "threshold": True, "status": "MISSING"}
    rows = weak.get("subfamilies", [])
    format_rows = [row for row in rows if row.get("family") == "format_only"]
    ready = bool(format_rows) and any(row.get("day6_action") == "eligible_for_v2_direct_answer" for row in format_rows)
    return {"value": ready, "threshold": True, "status": "PASS" if ready else "FAIL"}


def write_readiness_summary(summary: dict, path: str | Path) -> None:
    safe_write_text(path, json.dumps(summary, indent=2, sort_keys=True) + "\n", field_name="day5_2_readiness_summary")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build canonical Day 5.2 readiness summary.")
    parser.add_argument("--out", default="artifacts/sprint11/day5_2_readiness_summary.json")
    args = parser.parse_args(argv)
    summary = build_readiness_summary()
    write_readiness_summary(summary, args.out)
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "out": args.out}, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
