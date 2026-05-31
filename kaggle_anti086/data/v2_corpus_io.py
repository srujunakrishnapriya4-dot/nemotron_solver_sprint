from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from kaggle_anti086.kaggle_path_safety import require_writable_output_path, safe_write_text


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | Path, *, row_count: int | None = None) -> dict[str, Any]:
    target = Path(path)
    record: dict[str, Any] = {
        "path": str(target),
        "exists": target.exists(),
    }
    if target.exists():
        record.update(
            {
                "size_bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
    if row_count is not None:
        record["row_count"] = row_count
    return record


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def write_jsonl_checked(path: str | Path, rows: Iterable[dict[str, Any]], *, field_name: str) -> dict[str, Any]:
    target = _require_durable_output(path, field_name=field_name)
    payload_rows = list(rows)
    content = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in payload_rows)
    safe_write_text(target, content, field_name=field_name)
    record = file_record(target, row_count=len(payload_rows))
    print(json.dumps({"path": record["path"], "size_bytes": record["size_bytes"], "sha256": record["sha256"], "row_count": len(payload_rows)}, sort_keys=True))
    return record


def write_json_checked(path: str | Path, obj: dict[str, Any], *, field_name: str) -> dict[str, Any]:
    target = _require_durable_output(path, field_name=field_name)
    safe_write_text(target, json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", field_name=field_name)
    record = file_record(target)
    print(json.dumps({"path": record["path"], "size_bytes": record["size_bytes"], "sha256": record["sha256"]}, sort_keys=True))
    return record


def resolve_existing_path(path: str | Path, candidates: Iterable[str | Path] = ()) -> Path:
    requested = Path(path)
    if requested.exists():
        return requested
    for candidate in candidates:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path
    checked = ", ".join(str(Path(candidate)) for candidate in candidates)
    raise FileNotFoundError(f"required input artifact is missing: {requested}; checked alternatives: {checked}")


def _require_durable_output(path: str | Path, *, field_name: str) -> Path:
    target = require_writable_output_path(path, field_name=field_name)
    normalized = str(target).replace("\\", "/")
    if normalized == "/tmp" or normalized.startswith("/tmp/"):
        raise SystemExit(f"{field_name} must not write durable Day 6 artifacts under /tmp: {target}")
    return target
