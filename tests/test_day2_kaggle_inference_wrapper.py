from __future__ import annotations

import json
from pathlib import Path
import shutil
import zipfile

import pytest

from kaggle_anti086.eval.day2_generate_eval_reports import DATASETS, MODES, validate_day2_outputs
from kaggle_anti086.eval.run_day2_kaggle_inference import (
    create_export_bundle,
    decision_table,
    resolve_adapter_dir,
    restore_adapter_zip,
    run_day2_kaggle_inference,
)


TMP = Path("artifacts/test_tmp/day2_kaggle_wrapper")


def _clean() -> Path:
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)
    return TMP


def _adapter(path: Path, *, targets: list[str] | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 32,
                "target_modules": targets or ["q_proj", "v_proj"],
                "base_model_name_or_path": "base",
            }
        ),
        encoding="utf-8",
    )
    (path / "adapter_model.safetensors").write_bytes(b"not-real-weights")
    return path


def _blocked_report(path: Path, *, dataset: str, mode: str, status: str = "BLOCKED") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": status,
                "dataset": dataset,
                "mode": mode,
                "row_count": 0 if status == "BLOCKED" else 1,
                "exact_match": 0.0,
                "answerable_exact_match": 0.0,
                "behavior_accuracy": 0.0,
                "format_error_rate": 0.0,
                "by_family": {"numeric_formula": {"count": 1, "correct": 0, "exact_match": 0.0}},
                "by_subfamily": {},
                "source_breakdown": {},
                "provenance": {},
                "failures": ["blocked"] if status == "BLOCKED" else [],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )


def test_missing_base_model_blocks_before_generation(monkeypatch):
    root = _clean()

    def fail_generator(**_):
        raise AssertionError("generator should not run")

    monkeypatch.setattr("kaggle_anti086.eval.run_day2_kaggle_inference.build_day2_eval_reports", fail_generator)
    result = run_day2_kaggle_inference(
        base_model_path=root / "missing_model",
        adapter_dir=root / "missing_adapter",
        adapter_zip=None,
        auto_discover_adapter=False,
        out_dir=root / "reports",
        predictions_dir=root / "predictions",
        manifest_out=root / "manifest.json",
        ladder_out=root / "ladder.json",
        candidate_ranking_out=root / "ranking.json",
        decision_out=root / "decision.json",
        smoke_out=root / "smoke.json",
        validation_out=root / "validation.json",
        export_zip=None,
        kaggle_mode=True,
    )

    assert result["status"] == "BLOCKED"
    assert result["reason"] == "base_model_path_missing"
    assert json.loads((root / "smoke.json").read_text(encoding="utf-8"))["failures"] == ["base_model_path_missing"]


def test_missing_adapter_blocks_before_generation(monkeypatch):
    root = _clean()
    base = root / "base"
    base.mkdir()

    def fail_generator(**_):
        raise AssertionError("generator should not run")

    monkeypatch.setattr("kaggle_anti086.eval.run_day2_kaggle_inference.build_day2_eval_reports", fail_generator)
    result = run_day2_kaggle_inference(
        base_model_path=base,
        adapter_dir=root / "missing_adapter",
        adapter_zip=None,
        auto_discover_adapter=False,
        out_dir=root / "reports",
        predictions_dir=root / "predictions",
        manifest_out=root / "manifest.json",
        ladder_out=root / "ladder.json",
        candidate_ranking_out=root / "ranking.json",
        decision_out=root / "decision.json",
        smoke_out=root / "smoke.json",
        validation_out=root / "validation.json",
        export_zip=None,
        kaggle_mode=True,
    )

    assert result["status"] == "BLOCKED"
    assert result["reason"] == "adapter_dir_missing"


def test_adapter_zip_restore_finds_adapter():
    root = _clean()
    source = _adapter(root / "bundle" / "v2a_bf16_qv_50_20260606_141808_42a82878")
    archive = root / "day11a_pass_adapter_bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for file in source.iterdir():
            zf.write(file, Path("nested") / source.name / file.name)

    restore = restore_adapter_zip(archive, root / "restored")
    resolved = resolve_adapter_dir(adapter_dir=None, auto_discover=True, extra_roots=[root / "restored"])

    assert restore["status"] == "PASS"
    assert resolved["status"] == "PASS"
    assert "v2a_bf16_qv_50" in resolved["selected_adapter_dir"]


def test_multiple_adapters_requires_explicit_selection():
    root = _clean()
    _adapter(root / "a")
    _adapter(root / "b")

    resolved = resolve_adapter_dir(adapter_dir=None, auto_discover=True, extra_roots=[root])

    assert resolved["status"] == "BLOCKED"
    assert "multiple_adapter_candidates_require_explicit_adapter_dir" in resolved["failures"]


def test_smoke_failure_prevents_full_report_generation(monkeypatch):
    root = _clean()
    base = root / "base"
    base.mkdir()
    adapter = _adapter(root / "adapter")

    monkeypatch.setattr(
        "kaggle_anti086.eval.run_day2_kaggle_inference._run_inference_smoke",
        lambda **_: {"status": "BLOCKED", "failures": ["adapter_load_failed"]},
    )

    def fail_generator(**_):
        raise AssertionError("generator should not run")

    monkeypatch.setattr("kaggle_anti086.eval.run_day2_kaggle_inference.build_day2_eval_reports", fail_generator)
    result = run_day2_kaggle_inference(
        base_model_path=base,
        adapter_dir=adapter,
        adapter_zip=None,
        auto_discover_adapter=False,
        out_dir=root / "reports",
        predictions_dir=root / "predictions",
        manifest_out=root / "manifest.json",
        ladder_out=root / "ladder.json",
        candidate_ranking_out=root / "ranking.json",
        decision_out=root / "decision.json",
        smoke_out=root / "smoke.json",
        validation_out=root / "validation.json",
        export_zip=None,
        kaggle_mode=True,
    )

    assert result["status"] == "BLOCKED"
    assert result["reason"] == "smoke_preflight_failed"


def test_export_zip_excludes_safetensors():
    root = _clean()
    sprint = root / "sprint11"
    (sprint / "day2_reports").mkdir(parents=True)
    (sprint / "day2_predictions").mkdir()
    (sprint / "day2_reports" / "ok.json").write_text("{}", encoding="utf-8")
    (sprint / "day2_reports" / "adapter_model.safetensors").write_bytes(b"forbidden")
    (sprint / "day2_eval_manifest.json").write_text("{}", encoding="utf-8")
    export = root / "day2_real_inference_artifacts.zip"

    create_export_bundle(export, artifacts_root=sprint)

    with zipfile.ZipFile(export) as zf:
        names = zf.namelist()
    assert "day2_reports/ok.json" in names
    assert all(not name.endswith(".safetensors") for name in names)


def test_decision_table_prints_from_blocked_reports():
    root = _clean()
    reports = root / "reports"
    preds = root / "predictions"
    for dataset in DATASETS:
        for mode in MODES:
            _blocked_report(reports / f"{dataset}_{mode}.json", dataset=dataset, mode=mode)
            (preds / f"{dataset}_{mode}_predictions.jsonl").parent.mkdir(parents=True, exist_ok=True)
            (preds / f"{dataset}_{mode}_predictions.jsonl").write_text("", encoding="utf-8")
    ladder = root / "ladder.json"
    ladder.write_text(json.dumps({"status": "WARN", "decision": {"decision": "STOP_ADAPTER_SCALING", "train_v2a_150": False, "reason_codes": []}}), encoding="utf-8")
    ranking = root / "ranking.json"
    ranking.write_text(json.dumps({"decision": {}}), encoding="utf-8")

    table = decision_table(reports_dir=reports, ladder_path=ladder, ranking_path=ranking)

    assert "base_only:" in table
    assert "next_action:" in table


def test_validation_fails_if_non_blocked_report_lacks_predictions(tmp_path: Path):
    root = _clean()
    reports = root / "reports"
    preds = root / "predictions"
    for dataset in DATASETS:
        for mode in MODES:
            status = "PASS" if (dataset, mode) == ("core_eval", "base_only") else "BLOCKED"
            _blocked_report(reports / f"{dataset}_{mode}.json", dataset=dataset, mode=mode, status=status)
            if (dataset, mode) != ("core_eval", "base_only"):
                (preds / f"{dataset}_{mode}_predictions.jsonl").parent.mkdir(parents=True, exist_ok=True)
                (preds / f"{dataset}_{mode}_predictions.jsonl").write_text("", encoding="utf-8")
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "datasets": {}}), encoding="utf-8")
    ladder = root / "ladder.json"
    ladder.write_text("{}", encoding="utf-8")

    report = validate_day2_outputs(reports_dir=reports, predictions_dir=preds, manifest_path=manifest, ladder_path=ladder)

    assert report["status"] == "FAIL"
    assert any("missing_predictions:core_eval.base_only" in failure for failure in report["failures"])
