from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from kaggle_anti086.eval.day2_generate_eval_reports import build_day2_eval_reports
from kaggle_anti086.eval.eval_ladder import build_eval_ladder


TMP = Path("artifacts/test_tmp/day2_eval_report_generation")


def _clean() -> Path:
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)
    return TMP


def test_day2_generator_manifest_points_to_real_files():
    root = _clean()
    manifest = root / "manifest.json"
    ladder = root / "ladder.json"

    predictions = root / "predictions"
    result = build_day2_eval_reports(
        out_dir=root,
        predictions_dir=predictions,
        manifest_path=manifest,
        ladder_out=ladder,
        candidate_ranking_out=root / "ranking.json",
        decision_out=root / "decision.json",
        smoke_out=root / "smoke.json",
        validation_out=root / "validation.json",
    )

    assert result["status"] == "PASS"
    assert manifest.exists()
    assert ladder.exists()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    for dataset in ("core_eval", "family_eval", "rule_holdout", "anti_leak"):
        for mode in ("base_only", "solver_only", "adapter_only", "combined"):
            entry = data["datasets"][dataset][mode]
            path = Path(entry["path"])
            assert path.exists()
            assert entry["sha256"]
            assert entry["size_bytes"] == path.stat().st_size
            report = json.loads(path.read_text(encoding="utf-8"))
            assert "row_count" in report
            assert any(field in report for field in ("exact_match", "overall_accuracy", "answerable_exact_match", "behavior_accuracy", "overall_score"))
            if dataset == "family_eval":
                assert report["by_family"]
            pred_path = predictions / f"{dataset}_{mode}_predictions.jsonl"
            assert pred_path.exists()
    ladder_report = json.loads(ladder.read_text(encoding="utf-8"))
    assert ladder_report["model_results_faked"] is False
    assert ladder_report["decision"]["train_v2a_150"] is False
    shutil.rmtree(root, ignore_errors=True)


def test_ladder_rejects_manifest_missing_report_from_day2_generation():
    root = _clean()
    manifest = root / "manifest.json"
    build_day2_eval_reports(
        out_dir=root,
        predictions_dir=root / "predictions",
        manifest_path=manifest,
        ladder_out=root / "ladder.json",
        candidate_ranking_out=root / "ranking.json",
        decision_out=root / "decision.json",
        smoke_out=root / "smoke.json",
        validation_out=root / "validation.json",
    )
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["datasets"]["core_eval"]["base_only"]["path"] = str(root / "missing.json")
    manifest.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        build_eval_ladder(manifest)
    shutil.rmtree(root, ignore_errors=True)


def test_ladder_rejects_fake_report_without_score_metric():
    root = _clean()
    manifest = root / "manifest.json"
    build_day2_eval_reports(
        out_dir=root,
        predictions_dir=root / "predictions",
        manifest_path=manifest,
        ladder_out=root / "ladder.json",
        candidate_ranking_out=root / "ranking.json",
        decision_out=root / "decision.json",
        smoke_out=root / "smoke.json",
        validation_out=root / "validation.json",
    )
    data = json.loads(manifest.read_text(encoding="utf-8"))
    fake = root / "fake.json"
    fake.write_text(json.dumps({"status": "PASS", "row_count": 1, "loss_end": 0.01}), encoding="utf-8")
    data["datasets"]["core_eval"]["base_only"] = {"path": str(fake), "sha256": "skip", "size_bytes": fake.stat().st_size}
    manifest.write_text(json.dumps(data), encoding="utf-8")
    data["datasets"]["core_eval"]["base_only"].pop("sha256")
    manifest.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="missing_explicit_score"):
        build_eval_ladder(manifest)
    shutil.rmtree(root, ignore_errors=True)


def test_day2_generator_local_model_modes_are_blocked_with_reasons():
    root = _clean()
    result = build_day2_eval_reports(
        out_dir=root,
        predictions_dir=root / "predictions",
        manifest_path=root / "manifest.json",
        ladder_out=root / "ladder.json",
        candidate_ranking_out=root / "ranking.json",
        decision_out=root / "decision.json",
        smoke_out=root / "smoke.json",
        validation_out=root / "validation.json",
        max_rows=2,
    )

    assert result["blocked_modes"] == ["adapter_only", "base_only", "combined"]
    report = json.loads((root / "core_eval_adapter_only.json").read_text(encoding="utf-8"))
    assert report["status"] == "BLOCKED"
    assert report["model_results_faked"] is False
    assert report["packaging_allowed"] is False
    assert report["submission_allowed"] is False
    assert report["blocked_reason"]["reason"] in {"adapter_dir_missing", "adapter_config_missing", "adapter_model_safetensors_missing", "model_inference_requires_kaggle_mode"}
    assert (root / "predictions" / "core_eval_adapter_only_predictions.jsonl").exists()
    shutil.rmtree(root, ignore_errors=True)


def test_day2_generator_mocked_kaggle_adapter_and_combined_reports(monkeypatch):
    root = _clean()

    def fake_loader(*, base_model_path, adapter_dir, mode):
        def generator(row):
            return str(row.get("answer", ""))

        return generator, {
            "adapter_path": adapter_dir,
            "adapter_config_sha256": "abc",
            "adapter_model_sha256": "def" if adapter_dir else None,
            "adapter_target_modules": ["q_proj", "v_proj"] if adapter_dir else None,
            "adapter_rank": 32 if adapter_dir else None,
        }

    monkeypatch.setattr("kaggle_anti086.eval.day2_generate_eval_reports._load_model_generator", fake_loader)
    result = build_day2_eval_reports(
        out_dir=root,
        predictions_dir=root / "predictions",
        manifest_path=root / "manifest.json",
        ladder_out=root / "ladder.json",
        candidate_ranking_out=root / "ranking.json",
        decision_out=root / "decision.json",
        smoke_out=root / "smoke.json",
        validation_out=root / "validation.json",
        base_model_path="fake_model",
        adapter_dir="fake_adapter",
        kaggle_mode=True,
        max_rows=2,
    )

    assert result["blocked_modes"] == []
    adapter_report = json.loads((root / "core_eval_adapter_only.json").read_text(encoding="utf-8"))
    combined_report = json.loads((root / "core_eval_combined.json").read_text(encoding="utf-8"))
    assert adapter_report["status"] == "PASS"
    assert adapter_report["provenance"]["adapter_config_sha256"] == "abc"
    assert combined_report["source_breakdown"]
    assert (root / "predictions" / "core_eval_combined_predictions.jsonl").exists()
    shutil.rmtree(root, ignore_errors=True)
