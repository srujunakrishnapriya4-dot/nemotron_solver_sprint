from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import write_json_checked


def capture_gpu_memory_snapshot(stage: str, *, kaggle_mode: bool = False) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "stage": stage,
        "status": "WARN",
        "torch_available": False,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "warnings": [],
        "failures": [],
    }
    try:
        import torch  # type: ignore

        snapshot["torch_available"] = True
        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        if not snapshot["cuda_available"]:
            if kaggle_mode:
                snapshot["status"] = "FAIL"
                snapshot["failures"].append("cuda_unavailable_in_kaggle_mode")
            else:
                snapshot["warnings"].append("cuda_unavailable_local")
            return snapshot
        count = int(torch.cuda.device_count())
        snapshot["device_count"] = count
        devices = []
        for index in range(count):
            props = torch.cuda.get_device_properties(index)
            free_bytes = None
            total_bytes = int(getattr(props, "total_memory", 0) or 0)
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            except Exception:
                pass
            name = getattr(props, "name", None)
            if name is None:
                name = torch.cuda.get_device_name(index)
            devices.append(
                {
                    "index": index,
                    "name": str(name),
                    "allocated_gb": _gb(torch.cuda.memory_allocated(index)),
                    "reserved_gb": _gb(torch.cuda.memory_reserved(index)),
                    "max_allocated_gb": _gb(torch.cuda.max_memory_allocated(index)),
                    "total_memory_gb": _gb(total_bytes),
                    "free_memory_gb": _gb(free_bytes) if free_bytes is not None else None,
                }
            )
        snapshot["devices"] = devices
        snapshot["status"] = "PASS"
        return snapshot
    except Exception as exc:
        snapshot["failures" if kaggle_mode else "warnings"].append(f"gpu_memory_snapshot_failed:{type(exc).__name__}")
        snapshot["status"] = "FAIL" if kaggle_mode else "WARN"
        return snapshot


def cuda_empty_cache(reason: str) -> dict[str, Any]:
    report = {"reason": reason, "gc_collected": False, "empty_cache_called": False, "status": "PASS", "warnings": [], "failures": []}
    try:
        gc.collect()
        report["gc_collected"] = True
    except Exception as exc:
        report["warnings"].append(f"gc_collect_failed:{type(exc).__name__}")
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            report["empty_cache_called"] = True
    except Exception as exc:
        report["warnings"].append(f"empty_cache_failed:{type(exc).__name__}")
    return report


def write_gpu_memory_report(path: str | Path, snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [snap for snap in snapshots if snap.get("status") == "FAIL"]
    warnings = [snap for snap in snapshots if snap.get("status") == "WARN"]
    report = {
        "status": "FAIL" if failures else ("WARN" if warnings else "PASS"),
        "snapshot_count": len(snapshots),
        "snapshots": snapshots,
        "failures": [snap.get("stage") for snap in failures],
        "warnings": [snap.get("stage") for snap in warnings],
    }
    write_json_checked(path, report, field_name="day8_2c_gpu_memory_report")
    return report


def _gb(value: int | float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / (1024**3), 6)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage", default="manual")
    parser.add_argument("--kaggle-mode", action="store_true")
    args = parser.parse_args(argv)
    report = write_gpu_memory_report(args.out, [capture_gpu_memory_snapshot(args.stage, kaggle_mode=args.kaggle_mode)])
    print(json.dumps({"status": report["status"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] in {"PASS", "WARN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
