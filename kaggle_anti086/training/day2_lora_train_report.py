from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List


REQUIRED_REPORT_FIELDS = {
    "adapter_name",
    "dataset_variant",
    "target_modules",
    "rank",
    "learning_rate",
    "max_seq_len",
    "train_rows",
    "eval_rows",
    "train_loss_start",
    "train_loss_end",
    "loss_finite",
    "nan_count",
    "inf_count",
    "adapter_path",
    "safe_for_eval",
    "package_authorized",
    "submission_authorized",
    "leaderboard_claim",
}


ALLOWED_TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj",
    "out_proj",
    "up_proj",
    "down_proj",
}


ADAPTER_WEIGHT_FILES = {
    "adapter_model.safetensors",
    "adapter_model.bin",
}


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return obj


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def is_finite_number(value: Any) -> bool:
    if value is None:
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def adapter_dir_valid(adapter_path: str | Path) -> bool:
    p = Path(adapter_path)
    if not p.exists() or not p.is_dir():
        return False
    if not (p / "adapter_config.json").exists():
        return False
    return any((p / name).exists() for name in ADAPTER_WEIGHT_FILES)


def validate_single_train_report(report: Dict[str, Any], require_adapter_files: bool = True) -> List[str]:
    reasons: List[str] = []

    missing = sorted(REQUIRED_REPORT_FIELDS - set(report))
    if missing:
        reasons.append("missing_fields:" + ",".join(missing))
        return reasons

    adapter_name = report.get("adapter_name")
    if not isinstance(adapter_name, str) or not adapter_name:
        reasons.append("invalid_adapter_name")

    target_modules = report.get("target_modules")
    if not isinstance(target_modules, list) or not target_modules:
        reasons.append("invalid_target_modules")
    else:
        if len(set(target_modules)) != len(target_modules):
            reasons.append("duplicate_target_modules")

        bad = [m for m in target_modules if m not in ALLOWED_TARGET_MODULES]
        if bad:
            reasons.append("unknown_target_modules:" + ",".join(map(str, bad)))

        if any(str(m).lower() in {"all-linear", "all_linear", "*"} for m in target_modules):
            reasons.append("unsafe_all_linear_target_modules")

    rank = report.get("rank")
    if not isinstance(rank, int):
        reasons.append("rank_not_int")
    elif rank <= 0:
        reasons.append("rank_nonpositive")
    elif rank > 32:
        reasons.append("rank_exceeds_32")

    lr = report.get("learning_rate")
    if not isinstance(lr, (int, float)) or not math.isfinite(float(lr)):
        reasons.append("invalid_learning_rate")
    elif not (1e-6 <= float(lr) <= 8e-5):
        reasons.append("learning_rate_outside_safe_band")

    max_seq_len = report.get("max_seq_len")
    if max_seq_len not in {1024, 1536, 2048}:
        reasons.append("invalid_max_seq_len")

    for field in ("train_rows", "eval_rows"):
        value = report.get(field)
        if not isinstance(value, int) or value < 0:
            reasons.append(f"invalid_{field}")

    if not is_finite_number(report.get("train_loss_start")):
        reasons.append("invalid_train_loss_start")

    if not is_finite_number(report.get("train_loss_end")):
        reasons.append("invalid_train_loss_end")

    if report.get("loss_finite") is not True:
        reasons.append("loss_finite_not_true")

    if report.get("nan_count") != 0:
        reasons.append("nan_count_nonzero")

    if report.get("inf_count") != 0:
        reasons.append("inf_count_nonzero")

    if report.get("safe_for_eval") is not True:
        reasons.append("safe_for_eval_not_true")

    if report.get("package_authorized") is not False:
        reasons.append("package_authorized_must_be_false")

    if report.get("submission_authorized") is not False:
        reasons.append("submission_authorized_must_be_false")

    if report.get("leaderboard_claim") is not False:
        reasons.append("leaderboard_claim_must_be_false")

    adapter_path = report.get("adapter_path")
    if not isinstance(adapter_path, str) or not adapter_path:
        reasons.append("invalid_adapter_path")
    elif require_adapter_files and not adapter_dir_valid(adapter_path):
        reasons.append("adapter_dir_invalid")

    return reasons


def build_matrix_report(
    reports: Iterable[Dict[str, Any]],
    out_dir: str | Path,
    require_adapter_files: bool = True,
) -> Dict[str, Any]:
    report_list = list(reports)
    adapter_results: List[Dict[str, Any]] = []
    pass_count = 0

    for r in report_list:
        reasons = validate_single_train_report(r, require_adapter_files=require_adapter_files)
        status = "PASS" if not reasons else "FAIL"

        if status == "PASS":
            pass_count += 1

        adapter_results.append(
            {
                "adapter_name": r.get("adapter_name"),
                "status": status,
                "blocked_reasons": reasons,
                "adapter_path": r.get("adapter_path"),
                "dataset_variant": r.get("dataset_variant"),
                "rank": r.get("rank"),
                "target_modules": r.get("target_modules"),
                "train_loss_start": r.get("train_loss_start"),
                "train_loss_end": r.get("train_loss_end"),
                "safe_for_eval": r.get("safe_for_eval") is True,
            }
        )

    global_blocked_reasons: List[str] = []

    if pass_count < 3:
        global_blocked_reasons.append(f"fewer_than_3_adapters_passed:{pass_count}")

    unsafe_flags = []
    for r in report_list:
        if r.get("package_authorized") is not False:
            unsafe_flags.append(f"{r.get('adapter_name')}:package_authorized")
        if r.get("submission_authorized") is not False:
            unsafe_flags.append(f"{r.get('adapter_name')}:submission_authorized")
        if r.get("leaderboard_claim") is not False:
            unsafe_flags.append(f"{r.get('adapter_name')}:leaderboard_claim")

    if unsafe_flags:
        global_blocked_reasons.append("unsafe_flags:" + ",".join(unsafe_flags))

    status = "PASS" if not global_blocked_reasons else "FAIL"

    return {
        "schema_version": 1,
        "phase": "PHASE9_CORE_SFT_ADAPTER_TRAINING_MATRIX",
        "status": status,
        "trained_adapter_count": len(report_list),
        "passed_adapter_count": pass_count,
        "minimum_required_passed_adapters": 3,
        "adapter_results": adapter_results,
        "safe_for_phase10_adapter_eval": status == "PASS",
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
        "next_phase": "PHASE10_ADAPTER_ONLY_EVALUATION_LADDER" if status == "PASS" else None,
        "blocked_reasons": global_blocked_reasons,
        "out_dir": str(out_dir),
    }


def discover_single_reports(out_dir: Path) -> List[Path]:
    return sorted(out_dir.glob("day2_train_report_A*.json"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="artifacts/sprint11")
    ap.add_argument("--require-adapter-files", action="store_true", default=True)
    ap.add_argument("--no-require-adapter-files", dest="require_adapter_files", action="store_false")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    report_paths = discover_single_reports(out_dir)
    reports = [read_json(p) for p in report_paths]

    matrix = build_matrix_report(
        reports,
        out_dir=out_dir,
        require_adapter_files=args.require_adapter_files,
    )

    write_json(out_dir / "day2_lora_training_matrix_report.json", matrix)
    print(json.dumps(matrix, indent=2, sort_keys=True))

    return 0 if matrix["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
