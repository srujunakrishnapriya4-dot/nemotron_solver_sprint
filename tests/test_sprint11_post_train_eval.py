import hashlib
import json
from pathlib import Path
import shutil

from kaggle_anti086.training.post_train_eval import build_post_train_eval, compare_reports, main


def test_post_train_eval_needs_kaggle_when_inference_unavailable():
    report = build_post_train_eval("/kaggle/working/anti086_adapters/v2a_run")
    assert report["status"] == "NEEDS_KAGGLE_MODEL_EVAL"
    assert report["model_eval_completed"] is False


def test_compare_reports_detects_regression():
    result = compare_reports({"private": {"exact_match": 0.5}}, {"private": {"exact_match": 0.4}})
    assert result["evals"]["private"]["delta"] < 0
    assert result["regressions"] == ["private"]


def test_compare_reports_allows_improvement():
    result = compare_reports({"private": {"exact_match": 0.5}}, {"private": {"exact_match": 0.55}})
    assert result["regressions"] == []


def test_post_train_eval_writes_ladder():
    root = Path("artifacts/test_tmp/post_train_eval_ladder")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    manifest = _ladder_manifest(root)
    out_report = root / "post_report.json"
    out_predictions = root / "predictions.jsonl"
    out_ladder = root / "ladder.json"

    rc = main(
        [
            "--adapter-dir",
            "/kaggle/working/anti086_adapters/v2a_run",
            "--out-report",
            str(out_report),
            "--out-predictions",
            str(out_predictions),
            "--eval-ladder-manifest",
            str(manifest),
            "--out-eval-ladder",
            str(out_ladder),
        ]
    )

    assert rc == 0
    assert out_ladder.exists()
    data = json.loads(out_report.read_text(encoding="utf-8"))
    assert data["day2_eval_ladder_path"] == str(out_ladder)
    assert data["day2_eval_ladder_decision"] == "train_v2a_150"
    assert data["day2_eval_ladder_train_v2a_150"] is True
    shutil.rmtree(root, ignore_errors=True)


def _ladder_manifest(root: Path) -> Path:
    datasets = {}
    for dataset in ("core_eval", "family_eval", "rule_holdout", "anti_leak"):
        datasets[dataset] = {}
        for mode, score in {"base_only": 0.7, "solver_only": 0.8, "adapter_only": 0.75, "combined": 0.83}.items():
            path = root / f"{dataset}_{mode}.json"
            report = {"status": "PASS", "exact_match": score, "format_error_rate": 0.0, "by_family": {"numeric_formula": {"exact_match": score}}}
            if dataset == "anti_leak" and mode == "combined":
                report["leakage_score"] = 0.0
            path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
            datasets[dataset][mode] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "datasets": datasets}), encoding="utf-8")
    return manifest
