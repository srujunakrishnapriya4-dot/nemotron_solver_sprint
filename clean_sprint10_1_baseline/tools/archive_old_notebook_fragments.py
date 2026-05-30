from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


STALE_PATTERNS = (
    "SPRINT-8 permits micro stack test only",
    "SPRINT-9 permits micro or v1 only",
    "micro-only",
    "v1_v2_v3_evidence_required",
    "old fallback",
)
PROTECTED = {
    "CELL_01_write_configs.py",
    "CELL_02_write_runtime_patches.py",
    "CELL_03_write_prepare_tokens.py",
    "CELL_04_write_train.py",
    "CELL_05_write_eval.py",
    "CELL_06_write_parent_calibrated_eval.py",
    "CELL_07_write_orchestrator.py",
    "CELL_08_verify_files.py",
    "CELL_09_v1b_commands.py",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def candidates(root: Path) -> list[tuple[Path, str]]:
    out = []
    search_roots = [root / "artifacts", root]
    for base in search_roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or "archive" in path.parts or "__pycache__" in path.parts:
                continue
            if path.name in PROTECTED and path.parent.name == "win_system_kaggle_cells":
                continue
            if not (path.name.startswith("CELL_") or path.suffix.lower() == ".ipynb"):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            hit = next((pattern for pattern in STALE_PATTERNS if pattern in text), None)
            if hit:
                out.append((path, hit))
    return sorted(set(out))


def archive(root: Path, apply: bool) -> dict:
    archive_root = root / "archive" / "old_notebook_fragments"
    records = []
    for src, reason in candidates(root):
        before = {"path": rel(src, root), "size_bytes": src.stat().st_size, "sha256": sha256(src), "reason": reason}
        dest = archive_root / rel(src, root)
        after = {"path": rel(dest, root), "exists": False}
        if apply:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            after = {"path": rel(dest, root), "exists": True, "size_bytes": dest.stat().st_size, "sha256": sha256(dest)}
        records.append({"original": before, "archived": after, "applied": apply})
    manifest = {"status": "APPLIED" if apply else "DRY_RUN", "records": records}
    archive_root.mkdir(parents=True, exist_ok=True)
    (archive_root / "archive_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = archive(Path(args.root).resolve(), apply=bool(args.apply))
    print(json.dumps({"status": manifest["status"], "count": len(manifest["records"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
