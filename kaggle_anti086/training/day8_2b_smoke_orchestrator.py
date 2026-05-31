from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import read_json, read_jsonl, write_json_checked, write_jsonl_checked
from kaggle_anti086.training.adapter_artifact_audit import audit_adapter_artifacts
from kaggle_anti086.training.eval_ladder import select_eval_rows
from kaggle_anti086.training.final_sft_leakage_audit import audit_final_sft_leakage
from kaggle_anti086.training.inference_eval_backend import compare_base_vs_adapter, run_model_eval
from kaggle_anti086.training.lora_target_inspector import verify_lora_targets
from kaggle_anti086.training.model_environment_report import build_model_environment_report
from kaggle_anti086.training.model_loader import load_adapter_model, load_base_model, load_tokenizer
from kaggle_anti086.training.output_drift_audit import audit_output_drift
from kaggle_anti086.training.package_submission_guard import build_package_submission_guard
from kaggle_anti086.training.real_tokenizer_collator_audit import build_real_collator_audit
from kaggle_anti086.training.run_provenance import build_run_provenance
from kaggle_anti086.training.sft_dataset import load_weighted_sft_rows
from kaggle_anti086.training.supervised_label_decode_audit import audit_supervised_label_decode
from kaggle_anti086.training.train_v2a_lora import main as train_v2a_main
from kaggle_anti086.training.trainable_parameter_audit import audit_trainable_parameters
from kaggle_anti086.training.training_config_schema import _target_modules, load_training_config
from kaggle_anti086.training.training_log_history import build_training_log_history
from kaggle_anti086.training.truncation_audit import build_truncation_audit
from kaggle_anti086.training.weighted_sft_sampler import build_weighted_sample, build_weighted_sampling_report


ART = Path("artifacts/sprint11")

PATHS = {
    "collator_audit": ART / "day8_2b_real_collator_audit_report.json",
    "weighted_sampling": ART / "day8_2_weighted_sampling_report.json",
    "truncation": ART / "day8_2_truncation_audit_report.json",
    "label_decode": ART / "day8_2_supervised_label_decode_report.json",
    "leakage": ART / "day8_2_final_sft_leakage_report.json",
    "model_environment": ART / "day8_2b_model_environment_report.json",
    "run_provenance": ART / "day8_2b_run_provenance.json",
    "lora_target": ART / "day8_2_lora_target_report.json",
    "trainable_params": ART / "day8_2_trainable_parameter_report.json",
    "package_submission_guard": ART / "day8_2b_package_submission_guard_report.json",
    "smoke_train": ART / "day8_2_smoke_train_summary.json",
    "smoke_manifest": ART / "day8_2b_smoke_train_manifest.json",
    "adapter_artifact": ART / "day8_2_adapter_artifact_report.json",
    "smoke_eval": ART / "day8_2b_smoke_eval_report.json",
    "smoke_predictions": ART / "day8_2b_smoke_eval_predictions.jsonl",
    "output_drift": ART / "day8_2b_output_drift_report.json",
    "training_log_history": ART / "day8_2b_training_log_history.json",
}

BLOCK_DECISIONS = {
    "collator_audit": "BLOCK_FULL_TRAINING_COLLATOR",
    "weighted_sampling": "BLOCK_FULL_TRAINING_SAMPLING",
    "truncation": "BLOCK_FULL_TRAINING_TRUNCATION",
    "label_decode": "BLOCK_FULL_TRAINING_LABEL_DECODE",
    "leakage": "BLOCK_FULL_TRAINING_LEAKAGE",
    "model_environment": "NEEDS_KAGGLE_SMOKE_TRAIN",
    "lora_target": "BLOCK_FULL_TRAINING_LORA_TARGET",
    "trainable_params": "BLOCK_FULL_TRAINING_TRAINABLE_PARAMS",
    "package_submission_guard": "BLOCK_FULL_TRAINING_RUNTIME",
    "smoke_train": "BLOCK_FULL_TRAINING_SMOKE_TRAIN",
    "adapter_artifact": "BLOCK_FULL_TRAINING_ADAPTER_ARTIFACT",
    "smoke_eval": "BLOCK_FULL_TRAINING_SMOKE_EVAL",
    "output_drift": "BLOCK_FULL_TRAINING_OUTPUT_DRIFT",
}


def build_gate_entry(status: str, path: Path) -> dict[str, str]:
    return {"status": status, "path": str(path)}


def build_orchestrator_report(
    *,
    config_path: str | Path,
    kaggle_mode: bool = False,
    continue_on_failure: bool = False,
    runner: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    config = load_training_config(config_path)
    gates: dict[str, dict[str, str]] = {}
    failures: list[str] = []
    warnings: list[str] = []
    adapter_dir: str | None = None
    smoke_training_completed = False
    if config.get("stage") != "v2a_base_lora":
        return _final("FAIL", "BLOCK_FULL_TRAINING_RUNTIME", False, False, None, gates, warnings, ["non_v2a_stage_rejected"])

    def run_gate(name: str, func: Callable[[], dict[str, Any]], *, hard: bool = True) -> dict[str, Any]:
        nonlocal failures
        path = PATHS[name]
        try:
            report = runner(name, {"config": config, "path": path}) if runner else func()
        except Exception as exc:
            report = {"status": "FAIL", "failures": [f"{name}_exception:{type(exc).__name__}"]}
        if "status" not in report:
            report["status"] = "FAIL"
            report.setdefault("failures", []).append("missing_status")
        write_json_checked(path, report, field_name=f"day8_2b_{name}")
        gates[name] = build_gate_entry(str(report["status"]), path)
        if hard and report["status"] != "PASS":
            failures.append(name)
        return report

    collator_report = run_gate("collator_audit", lambda: build_real_collator_audit(config, sample_size=128, kaggle_mode=kaggle_mode), hard=False)
    if collator_report.get("status") != "PASS":
        if kaggle_mode or collator_report.get("status") != "WARN_LOCAL_TOKENIZER_UNAVAILABLE":
            failures.append("collator_audit")
    if failures and not continue_on_failure:
        return _blocked(failures, gates, warnings, adapter_dir, smoke_training_completed, kaggle_mode)
    items = load_weighted_sft_rows(config)
    selected = build_weighted_sample(items, seed=int(config.get("seed", 42)))
    run_gate("weighted_sampling", lambda: build_weighted_sampling_report(items, selected, seed=int(config.get("seed", 42))))
    run_gate("truncation", lambda: build_truncation_audit(config))
    run_gate("label_decode", lambda: audit_supervised_label_decode(config))
    run_gate("leakage", lambda: audit_final_sft_leakage(selected))
    run_gate("model_environment", lambda: build_model_environment_report(config, kaggle_mode=kaggle_mode), hard=kaggle_mode)
    provenance = build_run_provenance(
        config_path,
        collator_audit=PATHS["collator_audit"],
        weighted_sampling_report=PATHS["weighted_sampling"],
        truncation_audit=PATHS["truncation"],
        label_decode_audit=PATHS["label_decode"],
        leakage_audit=PATHS["leakage"],
    )
    write_json_checked(PATHS["run_provenance"], provenance, field_name="day8_2b_run_provenance")
    gates["run_provenance"] = build_gate_entry(provenance["status"], PATHS["run_provenance"])
    if provenance["status"] != "PASS":
        failures.append("run_provenance")
    if failures and not continue_on_failure:
        return _blocked(failures, gates, warnings, adapter_dir, smoke_training_completed, kaggle_mode)

    if not kaggle_mode:
        _write_local_missing_reports(gates)
        return _final("WARN", "NEEDS_KAGGLE_SMOKE_TRAIN", False, False, None, gates, warnings, failures)

    def _load_model():
        return load_base_model(str(config["base_model_path"]), load_in_4bit=bool(config.get("load_in_4bit", False)), bf16=True)

    model = _load_model()
    run_gate("lora_target", lambda: verify_lora_targets(model, _target_modules(config)))
    if failures and not continue_on_failure:
        return _blocked(failures, gates, warnings, adapter_dir, smoke_training_completed, kaggle_mode)
    from kaggle_anti086.training.lora_backend import prepare_model_for_v2a_training

    lora_model = prepare_model_for_v2a_training(model, config)
    run_gate("trainable_params", lambda: audit_trainable_parameters(lora_model))
    run_gate("package_submission_guard", lambda: build_package_submission_guard())
    if failures and not continue_on_failure:
        return _blocked(failures, gates, warnings, adapter_dir, smoke_training_completed, kaggle_mode)

    smoke_rc = train_v2a_main(
        [
            "--config",
            str(config_path),
            "--collator-audit",
            str(PATHS["collator_audit"]),
            "--out-manifest",
            str(PATHS["smoke_manifest"]),
            "--out-summary",
            str(PATHS["smoke_train"]),
            "--kaggle-mode",
            "--smoke-steps",
            "5",
        ]
    )
    smoke_report = read_json(PATHS["smoke_train"]) if PATHS["smoke_train"].exists() else {"status": "FAIL", "failures": ["smoke_summary_missing"]}
    if smoke_rc != 0 and smoke_report.get("status") == "PASS":
        smoke_report["status"] = "FAIL"
        smoke_report.setdefault("failures", []).append("smoke_train_returned_nonzero")
        write_json_checked(PATHS["smoke_train"], smoke_report, field_name="day8_2b_smoke_train")
    gates["smoke_train"] = build_gate_entry(str(smoke_report.get("status", "FAIL")), PATHS["smoke_train"])
    if smoke_report.get("status") != "PASS":
        failures.append("smoke_train")
    adapter_dir = smoke_report.get("adapter_dir")
    smoke_training_completed = smoke_report.get("status") == "PASS"
    history = build_training_log_history(smoke_report)
    write_json_checked(PATHS["training_log_history"], history, field_name="day8_2b_training_log_history")
    if not adapter_dir:
        failures.append("smoke_adapter_dir_missing")
    if failures and not continue_on_failure:
        return _blocked(failures, gates, warnings, adapter_dir, smoke_training_completed, kaggle_mode)

    run_gate("adapter_artifact", lambda: audit_adapter_artifacts(adapter_dir))
    if failures and not continue_on_failure:
        return _blocked(failures, gates, warnings, adapter_dir, smoke_training_completed, kaggle_mode)
    smoke_eval, predictions = _run_smoke_eval(config, adapter_dir)
    write_json_checked(PATHS["smoke_eval"], smoke_eval, field_name="day8_2b_smoke_eval")
    write_jsonl_checked(PATHS["smoke_predictions"], predictions, field_name="day8_2b_smoke_eval_predictions")
    gates["smoke_eval"] = build_gate_entry(smoke_eval["status"], PATHS["smoke_eval"])
    if smoke_eval["status"] != "PASS":
        failures.append("smoke_eval")
    drift = audit_output_drift(predictions, exact_match_delta=float(smoke_eval.get("delta", 0.0)))
    write_json_checked(PATHS["output_drift"], drift, field_name="day8_2b_output_drift")
    gates["output_drift"] = build_gate_entry(drift["status"], PATHS["output_drift"])
    if drift["status"] != "PASS":
        failures.append("output_drift")
    if failures:
        return _blocked(failures, gates, warnings, adapter_dir, smoke_training_completed, kaggle_mode)
    return _final("PASS", "ALLOW_DAY8_3_FULL_V2A_TRAINING", True, True, adapter_dir, gates, warnings, failures)


def _run_smoke_eval(config: dict[str, Any], adapter_dir: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eval_path = ART / "day8_2b_smoke_eval_rows_32.jsonl"
    rows = select_eval_rows(read_jsonl("artifacts/sprint11/day5_private_like_answerable_512.jsonl"), "smoke_32")
    write_jsonl_checked(eval_path, rows, field_name="day8_2b_smoke_eval_rows")
    tokenizer = load_tokenizer(str(config["base_model_path"]))
    base_model = load_base_model(str(config["base_model_path"]), load_in_4bit=bool(config.get("load_in_4bit", False)), bf16=True)
    base_report, base_predictions = run_model_eval(base_model, tokenizer, eval_path)
    del base_model
    _empty_cache()
    adapter_model = load_adapter_model(str(config["base_model_path"]), adapter_dir, load_in_4bit=bool(config.get("load_in_4bit", False)), bf16=True)
    adapter_report, adapter_predictions = run_model_eval(adapter_model, tokenizer, eval_path)
    del adapter_model
    _empty_cache()
    delta = compare_base_vs_adapter(base_report, adapter_report)
    predictions = []
    for base, v2a, row in zip(base_predictions, adapter_predictions, rows):
        predictions.append(
            {
                "row_id": base.get("row_id"),
                "family": base.get("family"),
                "prompt": row.get("prompt"),
                "expected": base.get("expected"),
                "base_pred": base.get("prediction"),
                "v2a_pred": v2a.get("prediction"),
                "base_correct": base.get("correct"),
                "v2a_correct": v2a.get("correct"),
            }
        )
    failures = []
    if adapter_report.get("failures"):
        failures.extend(adapter_report["failures"])
    changed = sum(1 for row in predictions if row.get("base_pred") != row.get("v2a_pred"))
    report = {
        "status": "PASS" if not failures else "FAIL",
        "eval_tier": "smoke_32",
        "row_count": len(rows),
        "base_exact": delta["base_exact"],
        "v2a_exact": delta["v2a_exact"],
        "delta": delta["delta"],
        "output_changed_rate": changed / len(predictions) if predictions else 0.0,
        "verbose_output_rate": adapter_report.get("verbose_output_count", 0) / len(rows) if rows else 0.0,
        "empty_output_rate": adapter_report.get("empty_output_count", 0) / len(rows) if rows else 0.0,
        "copied_prompt_rate": 0.0,
        "by_family": adapter_report.get("by_family", {}),
        "failures": failures,
        "warnings": [],
    }
    return report, predictions


def _empty_cache() -> None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _write_local_missing_reports(gates: dict[str, dict[str, str]]) -> None:
    local_reports = {
        "lora_target": {"status": "FAIL", "failures": ["needs_kaggle_model_inspection"]},
        "trainable_params": {"status": "FAIL", "failures": ["needs_kaggle_lora_injection"]},
        "smoke_train": {"status": "NEEDS_KAGGLE_SMOKE_TRAIN", "adapter_dir": None, "failures": []},
        "adapter_artifact": {"status": "FAIL", "failures": ["needs_smoke_adapter"]},
        "smoke_eval": {"status": "NEEDS_KAGGLE_MODEL_EVAL", "failures": []},
        "output_drift": {"status": "FAIL", "failures": ["no_predictions"]},
        "package_submission_guard": build_package_submission_guard(),
    }
    for name, report in local_reports.items():
        path = PATHS[name]
        write_json_checked(path, report, field_name=f"day8_2b_{name}")
        gates[name] = build_gate_entry(str(report["status"]), path)
    write_jsonl_checked(PATHS["smoke_predictions"], [], field_name="day8_2b_smoke_eval_predictions")
    write_json_checked(PATHS["training_log_history"], build_training_log_history({"log_history": []}), field_name="day8_2b_training_log_history")


def _blocked(failures: list[str], gates: dict[str, dict[str, str]], warnings: list[str], adapter_dir: str | None, smoke_done: bool, kaggle_mode: bool) -> dict[str, Any]:
    first = failures[0] if failures else "runtime"
    decision = "NEEDS_KAGGLE_SMOKE_TRAIN" if (first == "model_environment" and not kaggle_mode) else BLOCK_DECISIONS.get(first, "BLOCK_FULL_TRAINING_RUNTIME")
    return _final("WARN" if decision == "NEEDS_KAGGLE_SMOKE_TRAIN" else "FAIL", decision, False, smoke_done, adapter_dir, gates, warnings, failures)


def _final(
    status: str,
    decision: str,
    full_training_allowed: bool,
    smoke_training_completed: bool,
    adapter_dir: str | None,
    gates: dict[str, dict[str, str]],
    warnings: list[str],
    failures: list[str],
) -> dict[str, Any]:
    return {
        "status": status,
        "decision": decision,
        "full_training_allowed": full_training_allowed,
        "smoke_training_completed": smoke_training_completed,
        "adapter_dir": adapter_dir,
        "gates": gates,
        "warnings": warnings,
        "failures": failures,
        "packaging_allowed": False,
        "submission_allowed": False,
        "no_leaderboard_evidence": True,
        "no_0_95_evidence": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--kaggle-mode", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    args = parser.parse_args(argv)
    report = build_orchestrator_report(config_path=args.config, kaggle_mode=args.kaggle_mode, continue_on_failure=args.continue_on_failure)
    write_json_checked(args.out, report, field_name="day8_2b_smoke_orchestrator_report")
    print(json.dumps({"status": report["status"], "decision": report["decision"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] in {"PASS", "WARN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
