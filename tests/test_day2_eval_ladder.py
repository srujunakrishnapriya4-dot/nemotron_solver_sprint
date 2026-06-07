from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from kaggle_anti086.eval.eval_ladder import build_eval_ladder


TMP = Path("artifacts/test_tmp/day2_eval_ladder")


def _clean() -> Path:
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)
    return TMP


def _write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report(path: Path, score: float, *, leakage: float | None = None, fmt: float = 0.0, by_family: dict | None = None, loss: float | None = None) -> Path:
    data = {
        "status": "PASS",
        "exact_match": score,
        "format_error_rate": fmt,
        "invalid_answer_rate": fmt,
        "by_family": by_family or {"numeric_formula": {"exact_match": score}},
    }
    if leakage is not None:
        data["leakage_score"] = leakage
    if loss is not None:
        data["loss_end"] = loss
    return _write_json(path, data)


def _manifest(root: Path, *, base=0.70, adapter=0.75, solver=0.80, combined=0.83, rule_combined: float | None = None, include_hashes=True, high_loss=False) -> Path:
    datasets = {}
    for dataset in ("core_eval", "family_eval", "rule_holdout", "anti_leak"):
        datasets[dataset] = {}
        for mode, score in {
            "base_only": base,
            "solver_only": solver,
            "adapter_only": adapter,
            "combined": combined,
        }.items():
            actual = rule_combined if dataset == "rule_holdout" and mode == "combined" and rule_combined is not None else score
            leakage = 0.0 if dataset == "anti_leak" and mode == "combined" else None
            path = _report(root / f"{dataset}_{mode}.json", actual, leakage=leakage, loss=99.0 if high_loss else None)
            if include_hashes:
                datasets[dataset][mode] = {"path": str(path), "sha256": _sha(path), "size_bytes": path.stat().st_size}
            else:
                datasets[dataset][mode] = str(path)
    return _write_json(root / "manifest.json", {"schema_version": 1, "datasets": datasets})


def test_eval_ladder_missing_report_fails():
    root = _clean()
    manifest = _manifest(root)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["datasets"]["core_eval"]["base_only"]["path"] = str(root / "missing.json")
    manifest.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        build_eval_ladder(manifest)


def test_eval_ladder_hash_mismatch_fails():
    root = _clean()
    manifest = _manifest(root)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["datasets"]["core_eval"]["base_only"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="sha256_mismatch"):
        build_eval_ladder(manifest)


def test_eval_ladder_combined_must_beat_solver():
    root = _clean()
    report = build_eval_ladder(_manifest(root, base=0.70, adapter=0.72, solver=0.90, combined=0.89))

    assert report["decision"]["decision"] == "STOP_ADAPTER_SCALING"
    assert report["decision"]["train_v2a_150"] is False
    assert "combined_not_better_than_solver" in report["decision"]["reason_codes"]


def test_eval_ladder_train_allowed_when_all_gates_pass():
    root = _clean()
    report = build_eval_ladder(_manifest(root, base=0.70, adapter=0.75, solver=0.80, combined=0.83, rule_combined=0.80))

    assert report["decision"]["decision"] == "train_v2a_150"
    assert report["decision"]["train_v2a_150"] is True
    assert report["model_results_faked"] is False


def test_high_training_loss_does_not_affect_ladder():
    root_a = _clean()
    manifest_a = _manifest(root_a, high_loss=False)
    decision_a = build_eval_ladder(manifest_a)["decision"]
    root_b = _clean()
    manifest_b = _manifest(root_b, high_loss=True)
    decision_b = build_eval_ladder(manifest_b)["decision"]

    assert decision_a == decision_b
