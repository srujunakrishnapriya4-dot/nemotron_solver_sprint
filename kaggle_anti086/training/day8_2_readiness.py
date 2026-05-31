from __future__ import annotations

import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import read_json, write_json_checked


REPORTS = {
    "weighted_sampling": "artifacts/sprint11/day8_2_weighted_sampling_report.json",
    "truncation": "artifacts/sprint11/day8_2_truncation_audit_report.json",
    "lora_target": "artifacts/sprint11/day8_2_lora_target_report.json",
    "trainable_params": "artifacts/sprint11/day8_2_trainable_parameter_report.json",
    "final_sft_leakage": "artifacts/sprint11/day8_2_final_sft_leakage_report.json",
    "supervised_label_decode": "artifacts/sprint11/day8_2_supervised_label_decode_report.json",
    "adapter_artifact": "artifacts/sprint11/day8_2_adapter_artifact_report.json",
    "smoke_train": "artifacts/sprint11/day8_2_smoke_train_summary.json",
    "output_drift": "artifacts/sprint11/day8_2_output_drift_report.json",
    "day8_1_backend": "artifacts/sprint11/day8_1_backend_readiness_report.json",
    "day7_readiness": "artifacts/sprint11/day7_training_readiness_report.json",
}


def build_day8_2_readiness(paths: dict[str, str] | None = None) -> dict:
    paths = paths or REPORTS
    gates, failures, warnings = {}, [], []
    for name, path in paths.items():
        if not Path(path).exists():
            gates[name] = "MISSING"
            warnings.append(f"{name}_missing")
            continue
        data = read_json(path)
        status = str(data.get("status", "FAIL"))
        gates[name] = status
        if status == "FAIL" and not _allowed_pre_smoke_failure(name, data):
            failures.append(f"{name}_failed")
    smoke = gates.get("smoke_train")
    if smoke == "PASS" and not failures and gates.get("adapter_artifact") == "PASS" and gates.get("output_drift") in {"PASS", "WARN"}:
        decision = "ALLOW_DAY8_3_FULL_V2A_TRAINING"
        status = "PASS"
        full = True
    elif failures:
        decision = _block_decision(failures[0])
        status = "FAIL"
        full = False
    else:
        decision = "NEEDS_KAGGLE_SMOKE_TRAIN"
        status = "WARN"
        full = False
    return {
        "status": status,
        "decision": decision,
        "full_training_allowed": full,
        "smoke_training_required": not full,
        "packaging_allowed": False,
        "submission_allowed": False,
        "no_leaderboard_evidence": True,
        "no_0_95_evidence": True,
        "gates": gates,
        "warnings": warnings,
        "failures": failures,
    }


def _block_decision(failure: str) -> str:
    if "weighted" in failure:
        return "BLOCK_FULL_TRAINING_SAMPLING_BUG"
    if "truncation" in failure:
        return "BLOCK_FULL_TRAINING_TRUNCATION_BUG"
    if "label" in failure:
        return "BLOCK_FULL_TRAINING_LABEL_BUG"
    if "lora_target" in failure:
        return "BLOCK_FULL_TRAINING_LORA_TARGET_BUG"
    if "trainable" in failure:
        return "BLOCK_FULL_TRAINING_TRAINABLE_PARAM_BUG"
    if "leakage" in failure:
        return "BLOCK_FULL_TRAINING_SAMPLING_BUG"
    return "BLOCK_FULL_TRAINING_SMOKE_FAILURE"


def _allowed_pre_smoke_failure(name: str, data: dict) -> bool:
    failure_text = " ".join(str(x) for x in data.get("failures", []))
    if name in {"lora_target", "trainable_params"}:
        return "model_unavailable_local" in failure_text
    if name == "adapter_artifact":
        return "adapter_config_missing" in failure_text or "adapter_model_missing" in failure_text
    if name == "output_drift":
        return "no_predictions" in failure_text
    if name == "day8_1_backend":
        return data.get("training_backend_ready") is not True
    if name == "smoke_train":
        return data.get("status") == "NEEDS_KAGGLE_SMOKE_TRAIN"
    return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    report = build_day8_2_readiness()
    write_json_checked(args.out, report, field_name="day8_2_readiness_report")
    print(json.dumps({"status": report["status"], "decision": report["decision"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] in {"PASS", "WARN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
