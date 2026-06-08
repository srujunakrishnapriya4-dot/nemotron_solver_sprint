from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from kaggle_anti086.training.day2_21_failure_hardening_audit import FAILURE_MODE_KEYS, build_hardening_audit


TMP = Path("artifacts/test_tmp/day2_21_failure_hardening_audit")


def _clean() -> Path:
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)
    return TMP


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tiny_report(path: Path, score: float = 0.8, *, status: str = "PASS", source_breakdown: dict | None = None) -> Path:
    return _write_json(
        path,
        {
            "status": status,
            "exact_match": score,
            "row_count": 4,
            "by_family": {"symbol_mapping": score},
            "source_breakdown": source_breakdown or {},
        },
    )


def _manifest(root: Path) -> Path:
    datasets = {}
    for dataset in ("core_eval", "family_eval", "rule_holdout", "anti_leak"):
        datasets[dataset] = {}
        for mode in ("base_only", "solver_only", "adapter_only", "combined"):
            report = _tiny_report(root / f"{dataset}_{mode}.json", 0.8)
            datasets[dataset][mode] = {"path": str(report), "sha256": _sha(report), "size_bytes": report.stat().st_size}
    return _write_json(root / "manifest.json", {"schema_version": 1, "datasets": datasets})


def test_hardening_audit_has_all_21_failure_mode_keys():
    report = build_hardening_audit()

    assert tuple(report["failure_modes"].keys()) == FAILURE_MODE_KEYS
    assert len(report["failure_modes"]) == 21


def test_missing_required_repo_file_fails_not_silent_pass():
    root = _clean()
    (root / "kaggle_anti086").mkdir()
    report = build_hardening_audit(repo_root=root)

    assert report["failure_modes"]["01_repo_code_missing"]["status"] == "FAIL"
    assert "tests" in report["failure_modes"]["01_repo_code_missing"]["evidence"]["missing"]


def test_tracked_win_system_kaggle_cells_cleanup_is_guarded():
    report = build_hardening_audit()

    mode = report["failure_modes"]["02_bad_kaggle_dataset_packaging"]
    assert mode["status"] == "PASS"
    assert mode["evidence"]["tracked"] == []


def test_manifest_schema_mismatch_detected():
    root = _clean()
    bad_manifest = _write_json(root / "bad_manifest.json", {"schema_version": 99, "reports": {}})
    report = build_hardening_audit(manifest_path=bad_manifest)

    assert report["failure_modes"]["11_eval_manifest_schema_mismatch"]["status"] == "FAIL"


def test_eval_ladder_cli_writes_output_or_fails_nonzero():
    root = _clean()
    manifest = _manifest(root)
    out = root / "ladder.json"

    result = subprocess.run(
        [sys.executable, "kaggle_anti086/eval/eval_ladder.py", "--manifest", str(manifest), "--out", str(out)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8"))["created_by"] == "DAY2_EVAL_LADDER"


def test_combined_pass_with_zero_adapter_or_solver_source_is_invalid():
    root = _clean()
    reports = root / "reports"
    _tiny_report(reports / "core_eval_combined.json", source_breakdown={"solver": 4, "adapter": 0})

    report = build_hardening_audit(reports_dir=reports)

    assert report["failure_modes"]["10_full_wrapper_slow_incomplete_combined"]["status"] == "FAIL"
    assert report["failure_modes"]["15_combined_mode_not_measured"]["status"] == "FAIL"


def test_old_v1_unsupported_families_require_v2_answerable_confirmation():
    root = _clean()
    v2 = _write_json(
        root / "v2.json",
        {
            "schema_version": 2,
            "by_family": {
                "custom_numeral": {"answerable_wrong": 0},
                "equation_operator": {"answerable_wrong": 2},
            },
        },
    )

    report = build_hardening_audit(failure_report_v2_path=v2)

    assert report["failure_modes"]["17_old_v1_custom_numeral_unsupported"]["evidence"]["v2_status"] == "resolved_by_policy_aware_analysis"
    assert report["failure_modes"]["18_old_v1_equation_operator_unsupported"]["evidence"]["v2_status"] == "true_solver_repair_target"


def test_hardening_audit_never_authorizes_train_package_submission_or_leaderboard():
    report = build_hardening_audit()

    assert report["decision"]["v2a_150_authorized"] is False
    assert report["decision"]["package_authorized"] is False
    assert report["decision"]["submission_authorized"] is False
    assert report["leaderboard_claim"] is False
    assert report["no_0_93_evidence"] is True
    assert report["no_0_95_evidence"] is True
