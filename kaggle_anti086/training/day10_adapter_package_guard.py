from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import write_json_checked


FORBIDDEN_FINAL_NAMES = {
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "rng_state.pth",
    "training_args.bin",
    "submission.zip",
}


def rehearse_adapter_package(adapter_dir: str | Path, *, repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root)
    adapter = Path(adapter_dir)
    failures: list[str] = []
    warnings: list[str] = []
    if not adapter.exists():
        failures.append("adapter_dir_missing")
    config_path = adapter / "adapter_config.json"
    model_path = adapter / "adapter_model.safetensors"
    if adapter.exists():
        if not config_path.exists():
            failures.append("adapter_config_missing_at_package_root")
        if not model_path.exists():
            failures.append("adapter_model_missing_at_package_root")
        nested = [p for p in adapter.rglob("adapter_model.safetensors") if p.parent != adapter]
        if nested:
            failures.append("nested_adapter_model_found")
        forbidden_inside = [str(p.relative_to(adapter)) for p in adapter.rglob("*") if p.name in FORBIDDEN_FINAL_NAMES or p.name.startswith("checkpoint-")]
        if forbidden_inside:
            failures.append("forbidden_training_artifacts_in_adapter_dir")
    forbidden_repo = _repo_forbidden_files(root)
    if forbidden_repo:
        failures.append("forbidden_package_artifact_inside_repo")
    return {
        "status": "PASS" if not failures else "FAIL",
        "dry_run_package": True,
        "adapter_dir": str(adapter),
        "expected_zip_members": ["adapter_config.json", "adapter_model.safetensors"],
        "adapter_config_at_root": config_path.exists(),
        "adapter_model_at_root": model_path.exists(),
        "nested_package_rejected": True,
        "forbidden_repo_artifacts": forbidden_repo,
        "package_created": False,
        "packaging_allowed": False,
        "submission_allowed": False,
        "submit_recommended": False,
        "warnings": warnings,
        "failures": failures,
    }


def _repo_forbidden_files(root: Path) -> list[str]:
    if not root.exists():
        return []
    forbidden = []
    for p in root.rglob("*"):
        if ".git" in p.parts:
            continue
        if p.name == "submission.zip":
            forbidden.append(str(p))
        if p.name == "adapter_model.safetensors" and "artifacts" in p.parts:
            forbidden.append(str(p))
    return sorted(forbidden)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run adapter-only package rehearsal guard.")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--out", default="artifacts/sprint11/day10_adapter_package_guard_report.json")
    parser.add_argument("--dry-run-package", action="store_true", required=True)
    args = parser.parse_args(argv)
    report = rehearse_adapter_package(args.adapter_dir)
    write_json_checked(args.out, report, field_name="day10_adapter_package_guard")
    print(json.dumps({"status": report["status"], "package_created": report["package_created"]}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
