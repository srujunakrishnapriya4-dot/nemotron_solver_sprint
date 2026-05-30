from __future__ import annotations

import argparse
import ast
import hashlib
import json
import py_compile
import shutil
from pathlib import Path
import sys


REQUIRED_NAMES = [
    "anti086_winmode_micro.yaml",
    "anti086_winmode_v1.yaml",
    "anti086_winmode_v1b.yaml",
    "kaggle_runtime_patches.py",
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
FORBIDDEN_IMPORT_PATTERNS = ("from nemotron_engine", "import nemotron_engine")
STALE_PATTERNS = (
    "SPRINT-8 permits micro stack test only",
    "SPRINT-9 permits micro or v1 only",
    "micro-only",
    "v1_v2_v3_evidence_required",
    "old fallback",
)
HASH_REPORT_MARKERS = ("sha256", "getsize", "stat().st_size")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")


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
        if "_write(" not in text and "write_text(" not in text and "%%writefile" not in text:
            report["files"][rel(path, root)] = "not_a_writer"
            continue
        ok = all(marker in text for marker in HASH_REPORT_MARKERS[:1]) and ("size" in text.lower() or "bytes" in text.lower())
        report["files"][rel(path, root)] = ok
        if not ok:
            failures.append({"check": "generated_hash_reporting", "path": rel(path, root), "line": 0, "message": "cell does not report file path, size bytes, and sha256"})
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
    for path in files:
        if "archive" in path.parts:
            continue
        import_fail.extend(line_failures(path, root, FORBIDDEN_IMPORT_PATTERNS, "forbidden_imports", "nemotron_engine imports forbidden in active Kaggle files"))
        stale_fail.extend(line_failures(path, root, STALE_PATTERNS, "stale_sprint_contamination", "stale Sprint-8/Sprint-9 fragment in active file"))
        readonly_fail.extend(readonly_write_failures(path, root))
    checks["forbidden_imports"] = {"failures": import_fail}
    checks["readonly_input_writes"] = {"failures": readonly_fail}
    checks["stale_sprint_contamination"] = {"failures": stale_fail}
    failures.extend(import_fail + readonly_fail + stale_fail)
    parent_fail = check_parent_paths(root)
    checks["parent_adapter_path"] = {"failures": parent_fail}
    failures.extend(parent_fail)
    checks["generated_hash_reporting"], hash_report_fail = check_generated_hash_reporting(root)
    failures.extend(hash_report_fail)
    manifest = root / "artifacts" / "day1" / "day1_hash_manifest.json"
    checks["hash_manifest"] = {"path": rel(manifest, root), "exists": manifest.exists()}
    if strict and not manifest.exists():
        failures.append({"check": "hash_manifest", "path": rel(manifest, root), "line": 0, "message": "day1 hash manifest missing"})
    checks["submission_artifacts"], sub_fail = check_submission_artifacts(root, allow_submission_zip)
    failures.extend(sub_fail)
    checks["packaging_run"] = {"day1_packaging_not_allowed": True}
    return {"status": "PASS" if not failures else "FAIL", "checks": checks, "failures": failures}


def role_for(path: Path) -> str:
    name = path.name
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
