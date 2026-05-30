from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


SCHEMA_VERSION = 1


def normalize_path(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        rel = path
    return rel.as_posix()


def file_record(path: str | Path, root: Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    exists = p.exists()
    record = {"path": normalize_path(p, root), "exists": exists, "size_bytes": 0, "sha256": None}
    if exists and p.is_file():
        data = p.read_bytes()
        record.update({"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return record


def write_manifest(paths: list[str], manifest_path: Path, root: Path) -> dict:
    manifest = {"schema_version": SCHEMA_VERSION, "files": [file_record(path, root) for path in paths]}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    return manifest


def check_manifest(manifest_path: Path, root: Path) -> tuple[bool, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    checked = []
    for expected in manifest.get("files", []):
        current = file_record(expected["path"], root)
        checked.append(current)
        if not current["exists"]:
            failures.append({"path": expected["path"], "message": "missing"})
            continue
        if current.get("sha256") != expected.get("sha256") or current.get("size_bytes") != expected.get("size_bytes"):
            failures.append({"path": expected["path"], "message": "changed", "expected": expected, "current": current})
    return not failures, {"schema_version": SCHEMA_VERSION, "checked": checked, "failures": failures}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*")
    parser.add_argument("--root", default=".")
    parser.add_argument("--write")
    parser.add_argument("--check")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if bool(args.write) == bool(args.check):
        parser.error("provide exactly one of --write or --check")
    if args.write:
        manifest = write_manifest(args.files, Path(args.write), root)
        print(json.dumps({"status": "WROTE", "path": args.write, "file_count": len(manifest["files"])}, sort_keys=True))
        return 0
    ok, report = check_manifest(Path(args.check), root)
    print(json.dumps({"status": "PASS" if ok else "FAIL", **report}, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
