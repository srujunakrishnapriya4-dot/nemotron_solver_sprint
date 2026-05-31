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


def build_package_submission_guard(paths: list[str | Path] | None = None, reports: list[str | Path] | None = None) -> dict[str, Any]:
    roots = [Path(".")]
    if Path("artifacts").exists():
        roots.append(Path("artifacts"))
    if Path("/kaggle/working").exists():
        roots.append(Path("/kaggle/working"))
    paths = [Path(p) for p in (paths or roots)]
    forbidden: list[str] = []
    for root in paths:
        if root.is_file():
            if root.name == "submission.zip":
                forbidden.append(str(root))
            continue
        if root.exists():
            forbidden.extend(str(p) for p in root.rglob("submission.zip"))
            forbidden.extend(str(p) for p in root.rglob("package_ready*"))
    package_allowed = False
    submission_allowed = False
    for report_path in reports or []:
        path = Path(report_path)
        if path.exists():
            data = read_json(path)
            package_allowed = package_allowed or bool(data.get("packaging_allowed", False) or data.get("package_allowed", False))
            submission_allowed = submission_allowed or bool(data.get("submission_allowed", False))
    failures = []
    if forbidden:
        failures.append("submission_or_package_artifact_found")
    if package_allowed:
        failures.append("packaging_allowed_true")
    if submission_allowed:
        failures.append("submission_allowed_true")
    return {
        "status": "PASS" if not failures else "FAIL",
        "submission_zip_found": any(Path(p).name == "submission.zip" for p in forbidden),
        "package_allowed": package_allowed,
        "submission_allowed": submission_allowed,
        "forbidden_files": sorted(forbidden),
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = build_package_submission_guard()
    write_json_checked(args.out, report, field_name="package_submission_guard")
    print(json.dumps({"status": report["status"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
