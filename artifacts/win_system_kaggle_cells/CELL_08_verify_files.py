from pathlib import Path
import hashlib
import json
import py_compile

required = [
    "anti086_winmode_micro.yaml", "anti086_winmode_v1.yaml", "anti086_winmode_v1b.yaml",
    "kaggle_path_safety.py", "kaggle_runtime_patches.py", "kaggle_prepare_anti086_tokens.py",
    "kaggle_train_anti086_adapter.py", "kaggle_train_stage.py", "kaggle_eval_anti086_vllm.py",
    "kaggle_eval_stage.py", "kaggle_build_parent_calibrated_eval.py", "kaggle_winmode_orchestrator.py",
    "kaggle_final_candidate_selector.py", "kaggle_backend_probe.py",
]
output_fields = {
    "token_output", "token_output_dir", "token_manifest_path", "output_adapter_dir",
    "train_manifest_path", "eval_output_dir", "calibrated_eval_path",
    "parent_calibration_report_path", "stage_log_dir", "candidate_decision_path",
    "submission_zip_path", "package_output_path",
}

def read_simple_yaml(path):
    out = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip().strip("'\"")
    return out

records = []
for name in required:
    path = Path(name)
    rec = {"path": name, "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0}
    rec["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    if not rec["exists"] or rec["size_bytes"] <= 0:
        raise SystemExit(f"required file missing/empty: {name}")
    records.append(rec)
for script in Path('.').glob('kaggle_*.py'):
    py_compile.compile(str(script), doraise=True)
    text = script.read_text(encoding='utf-8', errors='ignore')
    if 'from ' + 'nemotron_engine' in text or 'import ' + 'nemotron_engine' in text:
        raise SystemExit(f"forbidden nemotron_engine import: {script}")
for cfg_path in Path('.').glob('anti086_winmode_*.yaml'):
    cfg = read_simple_yaml(cfg_path)
    for field in output_fields:
        value = cfg.get(field, '')
        if str(value).startswith('/kaggle/input'):
            raise SystemExit(f"unsafe output path in {cfg_path}: {field}={value}")
print(json.dumps({"status": "PASS", "files": records}, sort_keys=True, indent=2))
