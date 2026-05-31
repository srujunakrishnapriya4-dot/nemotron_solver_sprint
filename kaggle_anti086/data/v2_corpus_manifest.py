from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from kaggle_anti086.data.v2_corpus_io import file_record


def build_corpus_manifest(
    *,
    files: dict[str, str | Path],
    direct_rows: list[dict[str, Any]],
    solver_corrected_rows: list[dict[str, Any]],
    abstain_rows: list[dict[str, Any]],
    hard_negative_rows: list[dict[str, Any]],
    leakage_report: dict[str, Any],
    mixture_report: dict[str, Any],
    quality_gate: dict[str, Any],
    blocked_rows: dict[str, int] | None = None,
) -> dict[str, Any]:
    row_sets = {
        "verified_direct_answer": direct_rows,
        "solver_corrected": solver_corrected_rows,
        "abstain_safety": abstain_rows,
        "hard_negative": hard_negative_rows,
    }
    file_records = {
        name: file_record(path, row_count=len(row_sets[name]) if name in row_sets else None)
        for name, path in sorted(files.items())
    }
    family_distribution = Counter(str(row.get("family", "unknown")) for row in direct_rows)
    subfamily_distribution = Counter(f"{row.get('family', 'unknown')}::{row.get('subfamily', 'unknown')}" for row in direct_rows)
    source_eval_distribution = Counter(str(row.get("source_eval", "unknown")) for row in direct_rows)
    loss_scope_distribution = Counter(str(row.get("loss_scope", "unknown")) for row in direct_rows)
    decision = quality_gate.get("decision", "BLOCK_DAY7_TRAINING_CONFIG_PREP")
    return {
        "schema_version": 1,
        "created_by": "SPRINT-11E",
        "decision": decision,
        "status": "PASS" if decision == "ALLOW_DAY7_TRAINING_CONFIG_PREP" else "FAIL",
        "training_allowed": False,
        "packaging_allowed": False,
        "submission_allowed": False,
        "files": file_records,
        "family_distribution": dict(sorted(family_distribution.items())),
        "subfamily_distribution": dict(sorted(subfamily_distribution.items())),
        "rule_id_count": len({row.get("rule_id") for row in direct_rows}),
        "leakage_group_count": len({row.get("leakage_group") for row in direct_rows}),
        "source_eval_distribution": dict(sorted(source_eval_distribution.items())),
        "loss_scope_distribution": dict(sorted(loss_scope_distribution.items())),
        "blocked_rows": blocked_rows or {},
        "quality_gates": quality_gate.get("quality_gates", {}),
        "leakage_report_status": leakage_report.get("status"),
        "mixture_report_status": mixture_report.get("status"),
        "remaining_blockers": quality_gate.get("remaining_blockers", []),
        "no_training_performed": True,
        "no_adapter_packaged": True,
        "no_submission_created": True,
        "no_leaderboard_evidence": True,
        "no_0_95_evidence": True,
    }


def validate_corpus_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    for name, record in manifest.get("files", {}).items():
        if not record.get("exists"):
            failures.append(f"{name}_missing")
        if record.get("exists") and not record.get("sha256"):
            failures.append(f"{name}_missing_sha256")
        if record.get("exists") and "row_count" in record and record.get("row_count") is None:
            failures.append(f"{name}_missing_row_count")
    if manifest.get("training_allowed") is not False:
        failures.append("training_allowed_must_be_false")
    if manifest.get("no_training_performed") is not True:
        failures.append("no_training_performed_must_be_true")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures}
