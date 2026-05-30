from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from nemotron_engine.core.schemas import stable_hash


def build_equation_quarantine(
    input_dir: str | Path = "artifacts/win_system",
    output_dir: str | Path = "artifacts/win_system",
    *,
    direct_answer_cap: int = 500,
) -> dict[str, Any]:
    root = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    verified = [row for row in _read_jsonl(root / "solver_verified_correct.jsonl") if row.get("family") == "equation_symbolic"]
    wrong = [row for row in _read_jsonl(root / "solver_verified_wrong.jsonl") if row.get("family") == "equation_symbolic"]
    abstained = [row for row in _read_jsonl(root / "solver_abstained.jsonl") if row.get("family") == "equation_symbolic"]
    digit_examples = _read_jsonl(root / "digit_symbol_failure_examples.jsonl")
    offline_ids = {
        row["id"]
        for row in digit_examples
        if row.get("enough_constraints_for_unique_induction") and row.get("cluster") in {"digit_symbol_arithmetic", "binary_operator_arithmetic"}
    }
    safe_trace = [_safe_trace_row(row) for row in verified if row.get("prediction")]
    unsafe_source = wrong + abstained
    offline = [_offline_row(row) for row in unsafe_source if row.get("id") in offline_ids]
    unsafe = [_unsafe_row(row) for row in unsafe_source]
    direct = [_direct_row(row) for row in unsafe_source if row.get("answer")][:direct_answer_cap]
    _write_jsonl(out / "equation_safe_verified_trace.jsonl", safe_trace)
    _write_jsonl(out / "equation_direct_answer_only.jsonl", direct)
    _write_jsonl(out / "equation_unsafe_trace_excluded.jsonl", unsafe)
    _write_jsonl(out / "equation_offline_synthesis_candidates.jsonl", offline)
    status = "OFFLINE_SYNTHESIS_REQUIRED" if offline else "QUARANTINED"
    manifest = {
        "equation_safe_trace_count": len(safe_trace),
        "equation_direct_answer_only_count": len(direct),
        "equation_unsafe_excluded_count": len(unsafe),
        "equation_offline_synthesis_candidate_count": len(offline),
        "equation_status": status,
        "unsafe_reason_counts": dict(Counter(row.get("reason", "unknown") for row in unsafe_source).most_common()),
        "rules": {
            "unsafe_rows_cannot_enter_trace_corpus": True,
            "unsafe_rows_cannot_seed_synthetic_generation": True,
            "direct_answer_only_rows_capped": direct_answer_cap,
        },
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    (out / "equation_quarantine_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    return manifest


def _safe_trace_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "family": "equation_symbolic",
        "prediction": row.get("prediction"),
        "answer": row.get("answer"),
        "rule_id": row.get("reason"),
        "status": "safe_verified_trace",
        "trace_allowed": True,
    }


def _direct_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "family": "equation_symbolic",
        "answer": row.get("answer"),
        "status": "safe_direct_answer_only",
        "trace_allowed": False,
    }


def _unsafe_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "family": "equation_symbolic",
        "answer": row.get("answer"),
        "reason": row.get("reason"),
        "status": "unsafe_trace_excluded",
        "trace_allowed": False,
        "synthetic_seed_allowed": False,
    }


def _offline_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = _unsafe_row(row)
    payload["status"] = "offline_synthesis_candidate"
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
