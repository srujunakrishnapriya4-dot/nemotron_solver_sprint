from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.win_system_audit.repo_readiness_audit import run_repo_readiness_audit  # noqa: E402


def test_repo_readiness_audit_writes_verdict(tmp_path: Path) -> None:
    report = run_repo_readiness_audit(Path.cwd(), tmp_path)
    assert report["verdict"] in {"NOT_READY", "MICRO_READY", "MAIN_READY", "SUBMISSION_READY"}
    assert (tmp_path / "readiness_audit.json").exists()
    assert "eval_metric_correctness" in report["component_scores"]


def test_repo_readiness_audit_blocks_main_ready_when_coverage_zero(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    coverage = tmp_path / "artifacts" / "win_system"
    coverage.mkdir(parents=True)
    (coverage / "solver_coverage_report.json").write_text('{"solver_verified_correct_by_family": {}}', encoding="utf-8")

    report = run_repo_readiness_audit(tmp_path, tmp_path / "out")

    assert "zero_verified_train_solver_coverage" in report["critical_gaps"]
    assert report["verdict"] != "MAIN_READY"
