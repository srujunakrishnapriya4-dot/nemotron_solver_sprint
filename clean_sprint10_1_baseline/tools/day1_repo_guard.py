from __future__ import annotations

import argparse
import ast
import hashlib
import json
import py_compile
import shutil
import zipfile
from pathlib import Path
import sys


REQUIRED_NAMES = [
    "anti086_winmode_micro.yaml",
    "anti086_winmode_v1.yaml",
    "anti086_winmode_v1b.yaml",
    "kaggle_runtime_patches.py",
    "kaggle_path_safety.py",
    "kaggle_prepare_anti086_tokens.py",
    "kaggle_train_anti086_adapter.py",
    "kaggle_train_stage.py",
    "kaggle_eval_anti086_vllm.py",
    "kaggle_eval_stage.py",
    "kaggle_build_parent_calibrated_eval.py",
    "kaggle_winmode_orchestrator.py",
    "kaggle_final_candidate_selector.py",
    "kaggle_backend_probe.py",
]

PROTECTED_ROOTS = [Path("."), Path("kaggle_anti086"), Path("artifacts/win_system_kaggle_cells")]
ACTIVE_SCAN_ROOTS = [Path("kaggle_anti086"), Path("artifacts/win_system_kaggle_cells")]
EXPECTED_CELLS = [
    "CELL_01_write_configs.py",
    "CELL_02_write_runtime_patches.py",
    "CELL_03_write_prepare_tokens.py",
    "CELL_04_write_train.py",
    "CELL_05_write_eval.py",
    "CELL_06_write_parent_calibrated_eval.py",
    "CELL_07_write_orchestrator.py",
    "CELL_08_verify_files.py",
    "CELL_09_v1b_commands.py",
]
FORBIDDEN_IMPORT_PATTERNS = ("from nemotron_engine", "import nemotron_engine")
STALE_PATTERNS = (
    "SPRINT-8 permits micro stack test only",
    "SPRINT-9 permits micro or v1 only",
    "micro-only",
    "v1_v2_v3_evidence_required",
    "old fallback",
    "manual overlay validation passed",
    "kaggle_runtime_fallback_overlay",
)
HASH_REPORT_MARKERS = ("sha256", "getsize", "stat().st_size")
CHECKED_WRITE_HELPERS = ("write_file_checked", "_write_checked", "write_text_checked")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")
OUTPUT_CONFIG_FIELDS = {
    "token_output",
    "token_output_dir",
    "token_manifest_path",
    "output_adapter_dir",
    "train_manifest_path",
    "eval_output_dir",
    "calibrated_eval_path",
    "parent_calibration_report_path",
    "stage_log_dir",
    "candidate_decision_path",
    "submission_zip_path",
    "package_output_path",
}
WRITER_SCRIPT_NAMES = {
    "kaggle_prepare_anti086_tokens.py",
    "kaggle_train_anti086_adapter.py",
    "kaggle_train_stage.py",
    "kaggle_eval_anti086_vllm.py",
    "kaggle_eval_stage.py",
    "kaggle_build_parent_calibrated_eval.py",
    "kaggle_winmode_orchestrator.py",
    "kaggle_final_candidate_selector.py",
    "kaggle_package_anti086_adapter.py",
    "kaggle_backend_probe.py",
    "kaggle_parent_baseline_eval.py",
}


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_by_name(root: Path, name: str) -> Path | None:
    for base in PROTECTED_ROOTS:
        candidate = root / base / name
        if candidate.exists():
            return candidate
    matches = [p for p in root.glob(f"**/{name}") if "archive" not in p.parts and "__pycache__" not in p.parts]
    return matches[0] if matches else None


def active_files(root: Path) -> list[Path]:
    files = []
    for base in ACTIVE_SCAN_ROOTS:
        path = root / base
        if path.exists():
            files.extend(p for p in path.glob("kaggle_*.py") if p.is_file())
            files.extend(p for p in path.glob("CELL_*.py") if p.is_file())
    files.extend(p for p in root.glob("kaggle_*.py") if p.is_file())
    return sorted(set(files))


def line_failures(path: Path, root: Path, patterns: tuple[str, ...], check_name: str, message: str) -> list[dict]:
    failures = []
    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        if any(pattern in line for pattern in patterns):
            failures.append({"check": check_name, "path": rel(path, root), "line": idx, "message": message})
    return failures


def readonly_write_failures(path: Path, root: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    failures = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return failures
    write_methods = {"write_text", "write_bytes", "mkdir", "touch", "unlink", "rename", "replace"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            line = getattr(node, "lineno", 1)
            if isinstance(func, ast.Attribute) and func.attr in write_methods:
                blob = ast.get_source_segment(text, node) or ""
                if "/kaggle/input" in blob:
                    failures.append({"check": "readonly_input_writes", "path": rel(path, root), "line": line, "message": "write-like operation under /kaggle/input"})
            if isinstance(func, ast.Name) and func.id == "open":
                args = [ast.get_source_segment(text, arg) or "" for arg in node.args]
                if args and "/kaggle/input" in args[0] and len(args) > 1 and any(m in args[1] for m in ("w", "a", "+", "x")):
                    failures.append({"check": "readonly_input_writes", "path": rel(path, root), "line": line, "message": "open write under /kaggle/input"})
            if isinstance(func, ast.Attribute) and func.attr in {"copy", "copy2", "copytree", "move"}:
                blob = ast.get_source_segment(text, node) or ""
                if "/kaggle/input" in blob:
                    failures.append({"check": "readonly_input_writes", "path": rel(path, root), "line": line, "message": "copy/move may target /kaggle/input"})
    return failures


def indirect_readonly_input_write_risks(path: Path, root: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    failures = []
    lines = text.splitlines()
    input_aliases = set()
    for idx, line in enumerate(lines, start=1):
        compact = line.replace(" ", "")
        if "Path(\"/kaggle/input\")" in compact or "Path('/kaggle/input')" in compact:
            lhs = line.split("=", 1)[0].strip() if "=" in line else ""
            if lhs and lhs.isidentifier():
                input_aliases.add(lhs)
        if "_write_conservative_quarantine_overlay(path.parent)" in compact:
            window = "\n".join(lines[max(0, idx - 8) : min(len(lines), idx + 4)])
            routed = "copy_minimal_input_to_overlay" in window and ("overlay" in compact or "_write_conservative_quarantine_overlay(overlay" in window)
            if not routed and ("/kaggle/input" in text or any(f"{alias}.rglob" in text for alias in input_aliases)):
                failures.append(
                    {
                        "check": "indirect_readonly_input_write_risk",
                        "path": rel(path, root),
                        "line": idx,
                        "message": "write helper called on path.parent derived from /kaggle/input without writable overlay routing",
                    }
                )
        if "_write_conservative_quarantine_overlay(" in compact and "path.parent" in compact and "copy_minimal_input_to_overlay" not in text:
            if "/kaggle/input" in text or any(f"{alias}.rglob" in text for alias in input_aliases):
                failures.append(
                    {
                        "check": "indirect_readonly_input_write_risk",
                        "path": rel(path, root),
                        "line": idx,
                        "message": "quarantine overlay may write to unresolved input-derived root",
                    }
                )
    return failures


def load_simple_yaml(path: Path) -> dict:
    out = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip().strip("\"'")
    return out


def check_py_compile(root: Path, files: list[Path]) -> tuple[dict, list[dict]]:
    report = {}
    failures = []
    for path in files:
        if path.name.startswith("kaggle_"):
            try:
                py_compile.compile(str(path), doraise=True)
                report[rel(path, root)] = "PASS"
            except py_compile.PyCompileError as exc:
                report[rel(path, root)] = "FAIL"
                failures.append({"check": "py_compile", "path": rel(path, root), "line": 0, "message": str(exc)})
    return report, failures


def check_required(root: Path, strict: bool) -> tuple[dict, list[dict]]:
    files = {}
    failures = []
    for name in REQUIRED_NAMES:
        found = find_by_name(root, name)
        files[name] = rel(found, root) if found else None
        if strict and found is None:
            failures.append({"check": "required_files", "path": name, "line": 0, "message": "required baseline file missing"})
    return {"files": files}, failures


def check_parent_paths(root: Path) -> list[dict]:
    failures = []
    for name in ("anti086_winmode_v1.yaml", "anti086_winmode_v1b.yaml"):
        path = find_by_name(root, name)
        if not path:
            continue
        cfg = load_simple_yaml(path)
        if str(cfg.get("parent_adapter_path", "")).strip() == "auto_or_none":
            failures.append({"check": "parent_adapter_path", "path": rel(path, root), "line": 0, "message": "auto_or_none forbidden for serious stages"})
    return failures


def check_generated_hash_reporting(root: Path) -> tuple[dict, list[dict]]:
    cell_dir = root / "artifacts" / "win_system_kaggle_cells"
    report = {"cell_dir": rel(cell_dir, root), "files": {}}
    failures = []
    if not cell_dir.exists():
        return report, [{"check": "generated_hash_reporting", "path": rel(cell_dir, root), "line": 0, "message": "generated cell dir missing"}]
    for path in sorted(cell_dir.glob("CELL_*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "%%writefile" in text:
            failures.append({"check": "unchecked_writer_cell", "path": rel(path, root), "line": 0, "message": "raw %%writefile cells are forbidden; use write_file_checked"})
        if "_write(" not in text and "write_text(" not in text and "%%writefile" not in text:
            report["files"][rel(path, root)] = "not_a_writer"
            continue
        has_helper = any(f"def {name}" in text or f"{name}(" in text for name in CHECKED_WRITE_HELPERS)
        has_payload = "path" in text and "size_bytes" in text and "sha256" in text
        unchecked_lines = []
        try:
            tree = ast.parse(text)
            helper_ranges = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name in CHECKED_WRITE_HELPERS:
                    helper_ranges.append((node.lineno, getattr(node, "end_lineno", node.lineno)))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "write_text":
                    line = getattr(node, "lineno", 0)
                    if not any(start <= line <= end for start, end in helper_ranges):
                        unchecked_lines.append(line)
        except SyntaxError:
            unchecked_lines.append(0)
        ok = has_helper and has_payload and not unchecked_lines
        report["files"][rel(path, root)] = ok
        if not ok:
            failures.append({"check": "generated_hash_reporting", "path": rel(path, root), "line": unchecked_lines[0] if unchecked_lines else 0, "message": "writer cell must use write_file_checked/_write_checked/write_text_checked and report path, size_bytes, sha256"})
    return report, failures


def check_generated_cells_committed(root: Path) -> tuple[dict, list[dict]]:
    cell_dir = root / "artifacts" / "win_system_kaggle_cells"
    report = {"cell_dir": rel(cell_dir, root), "expected_cells": EXPECTED_CELLS, "present_cells": []}
    failures = []
    if not cell_dir.exists():
        return report, [{"check": "generated_cells_committed", "path": rel(cell_dir, root), "line": 0, "message": "generated cell directory missing"}]
    present = sorted(p.name for p in cell_dir.glob("CELL_*.py"))
    report["present_cells"] = present
    for name in EXPECTED_CELLS:
        path = cell_dir / name
        if not path.exists():
            failures.append({"check": "generated_cells_committed", "path": rel(path, root), "line": 0, "message": "expected generated cell missing"})
    extras = sorted(set(present) - set(EXPECTED_CELLS))
    for name in extras:
        failures.append({"check": "generated_cells_committed", "path": rel(cell_dir / name, root), "line": 0, "message": "unexpected generated cell present"})
    return report, failures


def check_generated_cells_zip_match(root: Path) -> tuple[dict, list[dict]]:
    cell_dir = root / "artifacts" / "win_system_kaggle_cells"
    zip_path = root / "artifacts" / "win_system_kaggle_cells.zip"
    report = {"zip_path": rel(zip_path, root), "zip_exists": zip_path.exists(), "matches": False}
    failures = []
    if not zip_path.exists():
        return report, [{"check": "generated_cells_zip_match", "path": rel(zip_path, root), "line": 0, "message": "generated cells zip missing"}]
    if not cell_dir.exists():
        return report, [{"check": "generated_cells_zip_match", "path": rel(cell_dir, root), "line": 0, "message": "generated cell directory missing"}]
    disk = {p.name: sha256(p) for p in sorted(cell_dir.glob("CELL_*.py"))}
    with zipfile.ZipFile(zip_path) as archive:
        zipped = {
            Path(info.filename).name: hashlib.sha256(archive.read(info.filename)).hexdigest()
            for info in archive.infolist()
            if not info.is_dir() and Path(info.filename).name.startswith("CELL_")
        }
    report.update({"directory": disk, "zip": zipped})
    if disk != zipped:
        failures.append({"check": "generated_cells_zip_match", "path": rel(zip_path, root), "line": 0, "message": "zip contents do not match generated cell directory by filename and SHA256"})
    else:
        report["matches"] = True
    return report, failures


def check_output_path_safety_static(root: Path) -> tuple[dict, list[dict]]:
    failures = []
    scanned = []
    for path in sorted((root / "kaggle_anti086").glob("*.yaml")) + sorted(root.glob("*.yaml")):
        if "archive" in path.parts:
            continue
        scanned.append(rel(path, root))
        cfg = load_simple_yaml(path)
        for field in OUTPUT_CONFIG_FIELDS:
            value = str(cfg.get(field, "")).strip()
            if value.startswith("/kaggle/input"):
                failures.append({"check": "output_path_safety_static", "path": rel(path, root), "line": 0, "message": f"{field} points under /kaggle/input: {value}"})
    return {"scanned": scanned}, failures


def check_config_output_path_validation_imported(root: Path, files: list[Path]) -> tuple[dict, list[dict]]:
    report = {}
    failures = []
    safety_tokens = ("require_safe_config_output_paths", "require_writable_output_path", "require_writable_output_dir", "safe_write_text", "safe_open_text_for_write")
    write_tokens = ("write_text(", ".open(\"w", ".open('w", "open(", "mkdir(", "copy2(", "copytree(", "ZipFile(")
    for path in files:
        if path.name not in WRITER_SCRIPT_NAMES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        writes = any(token in text for token in write_tokens)
        has_safety = "kaggle_path_safety" in text and any(token in text for token in safety_tokens)
        report[rel(path, root)] = {"writes": writes, "uses_path_safety": has_safety}
        if writes and path.name != "kaggle_path_safety.py" and not has_safety:
            failures.append({"check": "config_output_path_validation_imported", "path": rel(path, root), "line": 0, "message": "writer script must use kaggle_path_safety output validation helpers"})
    return report, failures


def check_submission_artifacts(root: Path, allow_submission_zip: bool) -> tuple[dict, list[dict]]:
    zips = [p for p in root.glob("**/submission.zip") if "archive" not in p.parts]
    failures = []
    if zips and not allow_submission_zip:
        failures = [{"check": "submission_artifacts", "path": rel(p, root), "line": 0, "message": "submission.zip present during Day 1 freeze"} for p in zips]
    return {"submission_zip": [rel(p, root) for p in zips]}, failures


def build_report(root: Path, strict: bool = True, allow_submission_zip: bool = False) -> dict:
    files = active_files(root)
    failures: list[dict] = []
    checks: dict = {}
    checks["required_files"], req_fail = check_required(root, strict)
    failures.extend(req_fail)
    checks["py_compile"], compile_fail = check_py_compile(root, files)
    failures.extend(compile_fail)
    import_fail = []
    stale_fail = []
    readonly_fail = []
    indirect_fail = []
    for path in files:
        if "archive" in path.parts:
            continue
        import_fail.extend(line_failures(path, root, FORBIDDEN_IMPORT_PATTERNS, "forbidden_imports", "nemotron_engine imports forbidden in active Kaggle files"))
        stale_fail.extend(line_failures(path, root, STALE_PATTERNS, "stale_sprint_contamination", "stale Sprint-8/Sprint-9 fragment in active file"))
        readonly_fail.extend(readonly_write_failures(path, root))
        indirect_fail.extend(indirect_readonly_input_write_risks(path, root))
    checks["forbidden_imports"] = {"failures": import_fail}
    checks["readonly_input_writes"] = {"failures": readonly_fail}
    checks["indirect_readonly_input_write_risk"] = {"failures": indirect_fail}
    checks["stale_sprint_contamination"] = {"failures": stale_fail}
    failures.extend(import_fail + readonly_fail + indirect_fail + stale_fail)
    parent_fail = check_parent_paths(root)
    checks["parent_adapter_path"] = {"failures": parent_fail}
    failures.extend(parent_fail)
    checks["generated_hash_reporting"], hash_report_fail = check_generated_hash_reporting(root)
    failures.extend(hash_report_fail)
    checks["generated_cells_committed"], cell_fail = check_generated_cells_committed(root)
    failures.extend(cell_fail)
    checks["generated_cells_zip_match"], zip_fail = check_generated_cells_zip_match(root)
    failures.extend(zip_fail)
    checks["output_path_safety_static"], output_fail = check_output_path_safety_static(root)
    failures.extend(output_fail)
    checks["config_output_path_validation_imported"], validation_fail = check_config_output_path_validation_imported(root, files)
    failures.extend(validation_fail)
    manifest = root / "artifacts" / "day1" / "day1_hash_manifest.json"
    checks["hash_manifest"] = {"path": rel(manifest, root), "exists": manifest.exists()}
    if strict and not manifest.exists():
        failures.append({"check": "hash_manifest", "path": rel(manifest, root), "line": 0, "message": "day1 hash manifest missing"})
    checks["submission_artifacts"], sub_fail = check_submission_artifacts(root, allow_submission_zip)
    failures.extend(sub_fail)
    checks["packaging_run"] = {"day1_packaging_not_allowed": True}
    checks["main_branch_awareness"] = {
        "branch_hint": _branch_hint(root),
        "day1_1_fix_present": bool((root / "kaggle_anti086" / "kaggle_prepare_anti086_tokens.py").exists() and "writable_overlay_root" in (root / "kaggle_anti086" / "kaggle_prepare_anti086_tokens.py").read_text(encoding="utf-8", errors="ignore")),
        "day1_2_output_safety_present": (root / "kaggle_anti086" / "kaggle_path_safety.py").exists(),
    }
    return {"status": "PASS" if not failures else "FAIL", "checks": checks, "failures": failures}


def _branch_hint(root: Path) -> str:
    head = root / ".git" / "HEAD"
    if not head.exists():
        return "unknown"
    text = head.read_text(encoding="utf-8", errors="ignore").strip()
    if text.startswith("ref: refs/heads/"):
        return text.replace("ref: refs/heads/", "", 1)
    return text[:12]


def role_for(path: Path) -> str:
    name = path.name
    if name.startswith("test_"):
        return "test"
    if name in {"day1_repo_guard.py", "hash_audit.py", "archive_old_notebook_fragments.py"}:
        return "guard"
    if name.endswith(".yaml"):
        return "config"
    if "runtime" in name:
        return "runtime"
    if "eval" in name or "calibrated" in name:
        return "eval"
    if "orchestrator" in name:
        return "orchestrator"
    if "selector" in name:
        return "selector"
    if name.startswith("CELL_"):
        return "cell"
    if name.endswith(".zip"):
        return "cell"
    if name.endswith("_manifest.json"):
        return "manifest"
    if name.endswith(".md"):
        return "runbook"
    return "runtime"


def snapshot(root: Path, dest: Path) -> dict:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    candidates = []
    for name in REQUIRED_NAMES:
        found = find_by_name(root, name)
        if found:
            candidates.append(found)
    runbook = root / "factory_reset_runbook.md"
    if runbook.exists():
        candidates.append(runbook)
    for extra in (
        root / "tools" / "day1_repo_guard.py",
        root / "tools" / "hash_audit.py",
        root / "tools" / "archive_old_notebook_fragments.py",
        root / "artifacts" / "day1" / "day1_generated_cells_manifest.json",
        root / "artifacts" / "win_system_kaggle_cells.zip",
    ):
        if extra.exists():
            candidates.append(extra)
    cell_dir = root / "artifacts" / "win_system_kaggle_cells"
    if cell_dir.exists():
        candidates.extend(sorted(cell_dir.glob("CELL_*.py")))
    for src in sorted(set(candidates)):
        if src.suffix.lower() in WEIGHT_SUFFIXES or src.name == "submission.zip":
            continue
        target = dest / rel(src, root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        copied.append({"path": rel(target, dest), "sha256": sha256(target), "size_bytes": target.stat().st_size, "role": role_for(src)})
    manifest = {"created_by": "SPRINT-11_DAY1_FREEZE", "purpose": "clean Sprint 10.1 baseline snapshot", "files": copied, "forbidden_actions": ["training", "packaging", "submission"]}
    (dest / "baseline_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="artifacts/day1/day1_guard_report.json")
    parser.add_argument("--snapshot")
    parser.add_argument("--strict", action="store_true", default=True)
    parser.add_argument("--allow-submission-zip", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if args.snapshot:
        manifest = snapshot(root, root / args.snapshot)
        print(json.dumps({"status": "SNAPSHOT_WRITTEN", "path": args.snapshot, "file_count": len(manifest["files"])}, sort_keys=True))
        return 0
    report = build_report(root, strict=args.strict, allow_submission_zip=args.allow_submission_zip)
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")
    summary = {"status": report["status"], "failure_count": len(report["failures"]), "out": rel(out, root)}
    print(json.dumps(summary, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
