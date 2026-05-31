from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import file_record, write_json_checked
from kaggle_anti086.training.training_config_schema import load_training_config


REQUIRED_CONFIG_PATH_FIELDS = ("train_direct_path", "train_solver_corrected_path")


def build_run_provenance(
    config_path: str | Path,
    *,
    collator_audit: str | Path | None = None,
    weighted_sampling_report: str | Path | None = None,
    truncation_audit: str | Path | None = None,
    label_decode_audit: str | Path | None = None,
    leakage_audit: str | Path | None = None,
    smoke_report: str | Path | None = None,
) -> dict[str, Any]:
    config = load_training_config(config_path)
    warnings: list[str] = []
    failures: list[str] = []
    git_info = _git_info(warnings)
    records = {"config": file_record(config_path)}
    for key in REQUIRED_CONFIG_PATH_FIELDS:
        path = config.get(key)
        record = file_record(path) if path else {"exists": False, "path": str(path)}
        records[key] = record
        if not record.get("exists"):
            failures.append(f"missing_required_file:{key}")
    optional = {
        "weighted_sampling_report_sha256": weighted_sampling_report,
        "collator_audit_sha256": collator_audit,
        "truncation_audit_sha256": truncation_audit,
        "label_decode_audit_sha256": label_decode_audit,
        "leakage_audit_sha256": leakage_audit,
        "smoke_report_sha256": smoke_report,
    }
    out: dict[str, Any] = {
        "status": "PASS" if not failures else "FAIL",
        "git_commit": git_info.get("git_commit"),
        "branch": git_info.get("branch"),
        "dirty_worktree": git_info.get("dirty_worktree"),
        "config_path": str(config_path),
        "config_sha256": records["config"].get("sha256"),
        "train_direct_path": str(config.get("train_direct_path", "")),
        "train_direct_sha256": records["train_direct_path"].get("sha256"),
        "train_solver_corrected_path": str(config.get("train_solver_corrected_path", "")),
        "train_solver_corrected_sha256": records["train_solver_corrected_path"].get("sha256"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "warnings": warnings,
        "failures": failures,
    }
    for field, path in optional.items():
        out[field] = file_record(path).get("sha256") if path and Path(path).exists() else None
    return out


def _git_info(warnings: list[str]) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL)
        return {"git_commit": commit, "branch": branch, "dirty_worktree": bool(status.strip())}
    except Exception as exc:
        warnings.append(f"git_metadata_unavailable:{type(exc).__name__}")
        return {"git_commit": None, "branch": None, "dirty_worktree": None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--collator-audit")
    parser.add_argument("--weighted-sampling-report")
    parser.add_argument("--truncation-audit")
    parser.add_argument("--label-decode-audit")
    parser.add_argument("--leakage-audit")
    parser.add_argument("--smoke-report")
    args = parser.parse_args(argv)
    report = build_run_provenance(
        args.config,
        collator_audit=args.collator_audit,
        weighted_sampling_report=args.weighted_sampling_report,
        truncation_audit=args.truncation_audit,
        label_decode_audit=args.label_decode_audit,
        leakage_audit=args.leakage_audit,
        smoke_report=args.smoke_report,
    )
    write_json_checked(args.out, report, field_name="run_provenance")
    print(json.dumps({"status": report["status"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
