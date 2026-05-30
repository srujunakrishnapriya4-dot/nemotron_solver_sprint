from __future__ import annotations

import json
import sys
from pathlib import Path
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_anti086"))

import kaggle_backend_probe  # noqa: E402
import kaggle_eval_stage  # noqa: E402
import kaggle_final_candidate_selector  # noqa: E402
import kaggle_train_stage  # noqa: E402
import kaggle_winmode_orchestrator  # noqa: E402


def test_backend_probe_detects_missing_cuda_as_unsafe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kaggle_backend_probe, "has_module", lambda name: False)
    report = kaggle_backend_probe.probe_backend(base_model_path=str(tmp_path / "model"), input_base=tmp_path / "input", output_path=tmp_path / "probe.json")

    assert report["recommended_mode"] == "BACKEND_NOT_SAFE"
    assert report["cuda_available"] is False


def test_validate_artifacts_extracts_zip_or_finds_input_root(tmp_path: Path) -> None:
    input_dir = tmp_path / "input" / "dataset"
    input_dir.mkdir(parents=True)
    zip_path = input_dir / "anti086_kaggle_input.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("curriculum_manifest.json", "{}")
        archive.writestr("corpus_anti086_micro.jsonl", "{}\n")

    root = kaggle_winmode_orchestrator.find_or_extract_input_root(tmp_path / "input", tmp_path / "work")

    assert (root / "curriculum_manifest.json").exists()
    assert (root / "corpus_anti086_micro.jsonl").exists()


def test_prepare_stage_refuses_pass_without_token_manifest(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        kaggle_winmode_orchestrator.derive_token_manifest({"token_output": str(tmp_path / "missing.jsonl")})


def test_train_stage_allows_false_and_rejects_true_full_prompt_loss() -> None:
    kaggle_train_stage.validate_training_config({"full_prompt_loss": False, "rank": 32})

    with pytest.raises(SystemExit):
        kaggle_train_stage.validate_training_config({"full_prompt_loss": True, "rank": 32})


def test_eval_placeholder_impossible_and_real_report_required(tmp_path: Path) -> None:
    placeholder = tmp_path / "eval_summary.json"
    placeholder.write_text(json.dumps({"status": "EVAL_SCRIPT_PLACEHOLDER", "row_count": 1, "empty_output_count": 0, "prompt_copy_rate": 0, "exact_answer_format_pass_rate": 1}), encoding="utf-8")

    with pytest.raises(SystemExit):
        kaggle_eval_stage.validate_eval_summary(placeholder)
    with pytest.raises(SystemExit):
        kaggle_eval_stage.validate_eval_summary(tmp_path / "missing.json")


def test_orchestrator_refuses_v1_before_eval_micro_pass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(kaggle_winmode_orchestrator, "gate_dir", lambda: tmp_path)

    with pytest.raises(SystemExit):
        kaggle_winmode_orchestrator.require_previous("train_micro")


def test_final_selector_rejects_missing_eval_evidence() -> None:
    score, reasons = kaggle_final_candidate_selector.score_stage("v1")

    assert score == 0.0
    assert "missing_eval_evidence" in reasons


def test_package_requires_final_selector_pass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(kaggle_winmode_orchestrator, "gate_dir", lambda: tmp_path)

    with pytest.raises(SystemExit):
        kaggle_winmode_orchestrator.require_previous("micro_gate")
