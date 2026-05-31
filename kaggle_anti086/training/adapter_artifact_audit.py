from __future__ import annotations

import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import read_json, write_json_checked


FORBIDDEN = {"optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth", "submission.zip"}


def audit_adapter_artifacts(adapter_dir: str | Path, *, adapter_size_limit_mb: float = 1500) -> dict:
    target = Path(adapter_dir)
    config_path = target / "adapter_config.json"
    model_path = target / "adapter_model.safetensors"
    failures = []
    adapter_config_exists = config_path.exists()
    adapter_model_exists = model_path.exists()
    if not adapter_config_exists:
        failures.append("adapter_config_missing")
    if not adapter_model_exists:
        failures.append("adapter_model_missing")
    rank = None
    target_modules = []
    if adapter_config_exists:
        data = read_json(config_path)
        rank = data.get("r")
        target_modules = data.get("target_modules", [])
        if int(rank or 0) > 32:
            failures.append("rank_gt_32")
        if "lm_head" in target_modules or not set(target_modules).issubset({"q_proj", "v_proj", "o_proj"}):
            failures.append("invalid_target_modules")
    forbidden = [str(p.relative_to(target)) for p in target.rglob("*") if p.name in FORBIDDEN or p.name.startswith("checkpoint-")]
    if forbidden:
        failures.append("forbidden_files_present")
    size_mb = model_path.stat().st_size / 1024 / 1024 if model_path.exists() else 0.0
    if size_mb > adapter_size_limit_mb:
        failures.append("adapter_size_limit_exceeded")
    return {
        "status": "PASS" if not failures else "FAIL",
        "adapter_dir": str(target),
        "adapter_config_exists": adapter_config_exists,
        "adapter_model_exists": adapter_model_exists,
        "adapter_size_mb": size_mb,
        "rank": rank,
        "target_modules": target_modules,
        "forbidden_files": forbidden,
        "packageable": False,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    report = audit_adapter_artifacts(args.adapter_dir)
    write_json_checked(args.out, report, field_name="day8_2_adapter_artifact_report")
    print(json.dumps({"status": report["status"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
