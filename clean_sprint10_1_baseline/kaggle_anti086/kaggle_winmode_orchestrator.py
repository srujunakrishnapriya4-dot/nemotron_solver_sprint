from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from kaggle_prepare_anti086_tokens import load_simple_yaml, resolve_anti086_input_root
from kaggle_path_safety import require_safe_config_output_paths, require_writable_output_dir, require_writable_output_path, safe_write_text


MICRO_LABEL = "INFRASTRUCTURE STACK TEST ONLY - NOT A 0.95 CANDIDATE - NOT MAIN TRAINING - NOT SUBMISSION READY"
V1_LABEL = "V1 SMALL TRAINING EXPERIMENT - NOT SUBMISSION READY - NO PACKAGE - NO KAGGLE SUBMIT"
V1B_LABEL = "V1B DIRECT-ANSWER PARENT-CONTINUATION EXPERIMENT - NO V2 - NO PACKAGE - NO SUBMISSION"

STAGES = {
    "backend_probe": None,
    "validate_artifacts": None,
    "prepare_micro": "validate_artifacts",
    "train_micro": "prepare_micro",
    "eval_micro": "train_micro",
    "micro_gate": "eval_micro",
    "prepare_v1": "validate_artifacts",
    "train_v1": "prepare_v1",
    "eval_v1": "train_v1",
    "prepare_v1b": "validate_artifacts",
    "train_v1b": "prepare_v1b",
    "eval_v1b": "train_v1b",
}

MICRO_CONFIG = "anti086_winmode_micro.yaml"
V1_CONFIG = "anti086_winmode_v1.yaml"
V1B_CONFIG = "anti086_winmode_v1b.yaml"


def gate_dir() -> Path:
    path = require_writable_output_dir("/kaggle/working/anti086_stage_logs", field_name="stage_log_dir")
    path.mkdir(parents=True, exist_ok=True)
    return path


def gate_path(stage: str) -> Path:
    return gate_dir() / f"{stage}.gate.json"


def read_gate(stage: str) -> dict | None:
    path = gate_path(stage)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_gate(stage: str, decision: str, reason: str, **extra) -> dict:
    payload = {"stage": stage, "decision": decision, "reason": reason, "micro_stack_test_label": MICRO_LABEL, **extra}
    safe_write_text(gate_path(stage), json.dumps(payload, sort_keys=True, indent=2), field_name="stage_log_dir")
    return payload


def require_previous(stage: str) -> None:
    previous = STAGES[stage]
    if previous is None:
        return
    gate = read_gate(previous)
    if not gate or gate.get("decision") != "PASS":
        raise SystemExit(f"refusing {stage}: missing PASS gate for {previous}")


def find_or_extract_input_root(input_base: str | Path = "/kaggle/input", work_root: str | Path = "/kaggle/working/anti086_input") -> Path:
    base = Path(input_base)
    work = Path(work_root)
    direct_markers = {"curriculum_manifest.json", "win_corpus_manifest.json", "solver_coverage_report.json"}
    if base.exists():
        for marker in direct_markers:
            for path in sorted(base.rglob(marker)):
                return path.parent
        for zip_name in ("anti086_kaggle_input.zip", "win_system_kaggle_input.zip"):
            zips = sorted(base.rglob(zip_name))
            if zips:
                work.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zips[0]) as archive:
                    archive.extractall(work)
                return work
    for local_zip in (Path("anti086_kaggle_input.zip"), Path("win_system_kaggle_input.zip")):
        if local_zip.exists():
            work.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(local_zip) as archive:
                archive.extractall(work)
            return work
    raise SystemExit("could not find anti086/win input root or Kaggle input zip")


def _find_micro_file(root: Path) -> Path | None:
    for name in ("winmode_micro.jsonl", "win_micro.jsonl", "corpus_anti086_micro.jsonl"):
        path = root / name
        if path.exists():
            return path
    return None


def _find_stage_file(root: Path, stage: str) -> Path | None:
    if stage == "micro":
        return _find_micro_file(root)
    for name in ("corpus_anti086_v1.jsonl", "winmode_v1.jsonl", "win_v1.jsonl"):
        path = root / name
        if path.exists():
            return path
    return None


def _unsafe_equation_traces_present(root: Path) -> bool:
    unsafe = root / "equation_unsafe_trace_excluded.jsonl"
    trace = root / "equation_safe_verified_trace.jsonl"
    if not unsafe.exists() or not trace.exists():
        return False
    unsafe_ids = {json.loads(line).get("id") for line in unsafe.read_text(encoding="utf-8").splitlines() if line.strip()}
    for line in trace.read_text(encoding="utf-8").splitlines():
        if line.strip() and json.loads(line).get("id") in unsafe_ids:
            return True
    return False


def validate_artifacts(stage: str = "micro") -> dict:
    root = resolve_anti086_input_root({})
    missing = []
    if not any((root / name).exists() for name in ("curriculum_manifest.json", "win_corpus_manifest.json")):
        missing.append("curriculum_manifest.json OR win_corpus_manifest.json")
    micro = _find_micro_file(root)
    v1 = _find_stage_file(root, "v1")
    if stage == "micro" and micro is None:
        missing.append("winmode_micro.jsonl OR win_micro.jsonl OR corpus_anti086_micro.jsonl")
    if stage in {"v1", "v1b"} and v1 is None:
        missing.append("corpus_anti086_v1.jsonl OR winmode_v1.jsonl OR win_v1.jsonl")
    if not (root / "equation_quarantine_manifest.json").exists():
        missing.append("equation_quarantine_manifest.json")
    if missing:
        raise SystemExit(f"missing required micro input files: {missing}")
    if _unsafe_equation_traces_present(root):
        raise SystemExit("unsafe equation traces appear in safe trace corpus")
    local = Path("artifacts/anti086")
    local.mkdir(parents=True, exist_ok=True)
    for path in root.glob("*"):
        if path.is_file():
            shutil.copy2(path, local / path.name)
    if micro is not None:
        shutil.copy2(micro, local / "winmode_micro.jsonl")
        shutil.copy2(micro, local / "win_micro.jsonl")
    if v1 is not None:
        shutil.copy2(v1, local / "corpus_anti086_v1.jsonl")
        shutil.copy2(v1, local / "winmode_v1.jsonl")
    return write_gate("validate_artifacts", "PASS", "micro input artifacts validated", anti086_input_root=str(root), micro_corpus=str(micro))


def derive_token_manifest(config: dict) -> dict:
    token_path = Path(str(config["token_output"]))
    if not token_path.exists():
        raise SystemExit(f"missing token output: {token_path}")
    row_count = 0
    zero = 0
    supervised = 0
    prompt_leakage = 0
    missing_weights = 0
    truncation = 0
    with token_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row_count += 1
            weights = row.get("loss_weights")
            if not weights:
                missing_weights += 1
                continue
            prompt_count = int(row.get("prompt_token_count", 0))
            sup = sum(1 for value in weights if float(value) > 0)
            supervised += sup
            zero += sup == 0
            if any(float(value) > 0 for value in weights[:prompt_count]):
                prompt_leakage += 1
            if len(row.get("input_ids", [])) >= int(config.get("max_seq_len", 10**9)):
                truncation += 1
    return {
        "row_count": row_count,
        "zero_supervised_rows": zero,
        "prompt_leakage": prompt_leakage,
        "supervised_token_count": supervised,
        "missing_loss_weights": missing_weights,
        "truncation_count": truncation,
    }


def prepare_micro() -> dict:
    return prepare_stage("micro", MICRO_CONFIG)


def prepare_stage(stage: str, config_path: str) -> dict:
    config = load_simple_yaml(config_path)
    require_safe_config_output_paths(config, stage=stage)
    validate_artifacts(stage)
    subprocess.check_call([sys.executable, "kaggle_prepare_anti086_tokens.py", "--config", config_path])
    manifest = derive_token_manifest(config)
    if manifest["row_count"] <= 0:
        raise SystemExit(f"token gate failed: {manifest}")
    if manifest["zero_supervised_rows"] or manifest["prompt_leakage"] or manifest["missing_loss_weights"]:
        raise SystemExit(f"token mask gate failed: {manifest}")
    if manifest["truncation_count"] > max(1, int(0.1 * manifest["row_count"])):
        raise SystemExit(f"truncation catastrophe: {manifest}")
    manifest_path = require_writable_output_path(str(config["token_manifest_path"]), field_name="token_manifest_path")
    safe_write_text(manifest_path, json.dumps({**manifest, "micro_stack_test_label": MICRO_LABEL}, sort_keys=True, indent=2), field_name="token_manifest_path")
    return write_gate(f"prepare_{stage}", "PASS", "assistant-only token corpus prepared", token_manifest=str(manifest_path), **manifest)


def train_micro() -> dict:
    return train_stage("micro", MICRO_CONFIG)


def train_stage(stage: str, config_path: str) -> dict:
    subprocess.check_call([sys.executable, "kaggle_train_stage.py", "--config", config_path])
    config = load_simple_yaml(config_path)
    require_safe_config_output_paths(config, stage=stage)
    manifest_path = require_writable_output_path(str(config.get("train_manifest_path", "/kaggle/working/anti086_train_manifest.json")), field_name="train_manifest_path")
    if not manifest_path.exists():
        raise SystemExit("missing train manifest after micro training")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("loss_finite", True):
        raise SystemExit(f"micro training loss not finite: {manifest}")
    if float(manifest.get("adapter_size_mb", 10**9)) > float(config["adapter_size_limit_mb"]):
        raise SystemExit(f"micro adapter exceeds size limit: {manifest}")
    if int(manifest.get("rank", 999)) > 32:
        raise SystemExit(f"micro adapter rank >32: {manifest}")
    return write_gate(f"train_{stage}", "PASS", f"{stage} training completed; not submission-ready", **manifest)


def eval_micro() -> dict:
    return eval_stage("micro", MICRO_CONFIG)


def eval_stage(stage: str, config_path: str) -> dict:
    subprocess.check_call([sys.executable, "kaggle_eval_stage.py", "--config", config_path, "--stage", f"eval_{stage}"])
    config = load_simple_yaml(config_path)
    require_safe_config_output_paths(config, stage=stage)
    summary_path = require_writable_output_dir(str(config["eval_output_dir"]), field_name="eval_output_dir") / "eval_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if stage == "v1" and summary.get("v1_gate_status") in {"INCONCLUSIVE_PARENT_MISSING", "FAIL_CHILD_REGRESSED", "INCONCLUSIVE_BOTH_ZERO"}:
        return write_gate("eval_v1", "FAIL", "v1 parent-vs-child gate did not pass", eval_summary=str(summary_path), **summary)
    if stage == "v1b" and summary.get("v1_gate_status") in {"INCONCLUSIVE_PARENT_MISSING", "FAIL_CHILD_REGRESSED", "INCONCLUSIVE_BOTH_ZERO", "INVALID_EVAL_PARENT_ZERO", "FAIL_CHILD_ZERO"}:
        return write_gate("eval_v1b", "FAIL", "v1b parent-vs-child gate did not pass", eval_summary=str(summary_path), **summary)
    return write_gate(f"eval_{stage}", "PASS", "real eval report validated", eval_summary=str(summary_path), **summary)


def micro_gate() -> dict:
    eval_gate = read_gate("eval_micro")
    train_gate = read_gate("train_micro")
    if not eval_gate or not train_gate:
        raise SystemExit("micro gate requires train_micro and eval_micro PASS gates")
    fail_reasons = []
    if float(eval_gate.get("empty_output_rate", 1.0)) > 0.05:
        fail_reasons.append("empty_output_rate")
    if float(eval_gate.get("prompt_copy_rate", 1.0)) > 0.01:
        fail_reasons.append("prompt_copy_rate")
    if float(eval_gate.get("answer_format_pass_rate", 0.0)) < 0.95:
        fail_reasons.append("answer_format_pass_rate")
    decision = "FAIL" if fail_reasons else "PASS"
    allowed = "PATCH_STACK" if fail_reasons else "STOP"
    return write_gate(
        "micro_gate",
        decision,
        "MICRO_STACK_TEST_PASS" if decision == "PASS" else "MICRO_STACK_TEST_FAIL",
        micro_stack_test_result="MICRO_STACK_TEST_PASS" if decision == "PASS" else "MICRO_STACK_TEST_FAIL",
        allowed_next_action=allowed,
        not_submission_ready=True,
        not_095_candidate=True,
        fail_reasons=fail_reasons,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    args = parser.parse_args()
    require_previous(args.stage)
    if args.stage == "backend_probe":
        subprocess.check_call([sys.executable, "kaggle_backend_probe.py"])
        write_gate(args.stage, "PASS", "backend probe completed; inspect recommended_mode")
    elif args.stage == "validate_artifacts":
        validate_artifacts()
    elif args.stage == "prepare_micro":
        prepare_micro()
    elif args.stage == "train_micro":
        train_micro()
    elif args.stage == "eval_micro":
        eval_micro()
    elif args.stage == "micro_gate":
        micro_gate()
    elif args.stage == "prepare_v1":
        prepare_stage("v1", V1_CONFIG)
    elif args.stage == "train_v1":
        train_stage("v1", V1_CONFIG)
    elif args.stage == "eval_v1":
        eval_stage("v1", V1_CONFIG)
    elif args.stage == "prepare_v1b":
        subprocess.check_call([sys.executable, "kaggle_build_parent_calibrated_eval.py", "--config", V1B_CONFIG])
        prepare_stage("v1b", V1B_CONFIG)
    elif args.stage == "train_v1b":
        train_stage("v1b", V1B_CONFIG)
    elif args.stage == "eval_v1b":
        eval_stage("v1b", V1B_CONFIG)
    print("Next allowed stages:", [name for name, prev in STAGES.items() if prev == args.stage])
    print(MICRO_LABEL if "micro" in args.stage else V1B_LABEL if "v1b" in args.stage else V1_LABEL)


if __name__ == "__main__":
    main()
