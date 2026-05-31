import json
from pathlib import Path

from kaggle_anti086.training import day8_2b_smoke_orchestrator as orch


def _patch_paths(monkeypatch, tmp_path):
    paths = {name: tmp_path / f"{name}.json" for name in orch.PATHS}
    paths["smoke_predictions"] = tmp_path / "smoke_predictions.jsonl"
    monkeypatch.setattr(orch, "PATHS", paths)
    return paths


def _patch_common(monkeypatch, tmp_path):
    config = {
        "stage": "v2a_base_lora",
        "base_model_path": "/kaggle/input/model",
        "train_direct_path": str(tmp_path / "direct.jsonl"),
        "train_solver_corrected_path": str(tmp_path / "corrected.jsonl"),
        "seed": 42,
        "load_in_4bit": True,
        "rank": 32,
        "target_modules": ["q_proj", "v_proj", "o_proj"],
    }
    Path(config["train_direct_path"]).write_text("{}\n", encoding="utf-8")
    Path(config["train_solver_corrected_path"]).write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(orch, "load_training_config", lambda path: dict(config))
    monkeypatch.setattr(orch, "load_weighted_sft_rows", lambda config: [])
    monkeypatch.setattr(orch, "build_weighted_sample", lambda items, seed=42: [])
    monkeypatch.setattr(orch, "build_real_collator_audit", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(orch, "build_weighted_sampling_report", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(orch, "build_truncation_audit", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(orch, "audit_supervised_label_decode", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(orch, "audit_final_sft_leakage", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(orch, "build_model_environment_report", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(orch, "build_run_provenance", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(orch, "build_package_submission_guard", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(orch, "capture_gpu_memory_snapshot", lambda stage, kaggle_mode=False: {"stage": stage, "status": "PASS", "devices": [{"free_memory_gb": 16.0, "reserved_gb": 0.0, "allocated_gb": 0.0}]})
    monkeypatch.setattr(orch, "write_gpu_memory_report", lambda *a, **k: {"status": "PASS"})
    return config


def test_local_missing_kaggle_stack_needs_smoke(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    _patch_common(monkeypatch, tmp_path)
    report = orch.build_orchestrator_report(config_path="config.yaml", kaggle_mode=False)
    assert report["decision"] == "NEEDS_KAGGLE_SMOKE_TRAIN"
    assert report["full_training_allowed"] is False
    assert report["packaging_allowed"] is False
    assert report["submission_allowed"] is False


def test_gate_failure_maps_to_sampling_decision(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(orch, "build_weighted_sampling_report", lambda *a, **k: {"status": "FAIL", "failures": ["bad_weight"]})
    report = orch.build_orchestrator_report(config_path="config.yaml", kaggle_mode=False)
    assert report["decision"] == "BLOCK_FULL_TRAINING_SAMPLING"


def test_non_v2a_stage_rejected(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    config = _patch_common(monkeypatch, tmp_path)
    config["stage"] = "v2c_best_parent_lora"
    monkeypatch.setattr(orch, "load_training_config", lambda path: dict(config))
    report = orch.build_orchestrator_report(config_path="config.yaml", kaggle_mode=False)
    assert report["decision"] == "BLOCK_FULL_TRAINING_RUNTIME"
    assert "non_v2a_stage_rejected" in report["failures"]


def test_full_success_reads_adapter_dir(monkeypatch, tmp_path):
    paths = _patch_paths(monkeypatch, tmp_path)
    _patch_common(monkeypatch, tmp_path)
    adapter_dir = str(tmp_path / "adapter")
    monkeypatch.setattr(orch, "load_base_model", lambda *a, **k: object())
    monkeypatch.setattr(orch, "verify_lora_targets", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr("kaggle_anti086.training.lora_backend.prepare_model_for_v2a_training", lambda model, config: model)
    monkeypatch.setattr(orch, "audit_trainable_parameters", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(orch, "audit_adapter_artifacts", lambda path: {"status": "PASS", "adapter_dir": path})
    monkeypatch.setattr(orch, "audit_output_drift", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(orch, "_run_smoke_eval", lambda config, adapter: ({"status": "PASS", "delta": 0.0}, [{"base_pred": "a", "v2a_pred": "b"}]))

    def fake_train(argv):
        paths["smoke_train"].write_text(json.dumps({"status": "PASS", "adapter_dir": adapter_dir, "log_history": [{"loss": 1.0}], "steps_completed": 5}), encoding="utf-8")
        return 0

    monkeypatch.setattr(orch, "train_v2a_main", fake_train)
    report = orch.build_orchestrator_report(config_path="config.yaml", kaggle_mode=True)
    assert report["decision"] == "ALLOW_DAY8_3_FULL_V2A_TRAINING"
    assert report["full_training_allowed"] is True
    assert report["adapter_dir"] == adapter_dir
    assert any(snap["stage"] == "after_inspection_cleanup" for snap in report["memory_snapshots"])


def test_pre_smoke_low_memory_blocks(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(orch, "load_base_model", lambda *a, **k: object())
    monkeypatch.setattr(orch, "verify_lora_targets", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr("kaggle_anti086.training.lora_backend.prepare_model_for_v2a_training", lambda model, config: model)
    monkeypatch.setattr(orch, "audit_trainable_parameters", lambda *a, **k: {"status": "PASS"})

    def snapshot(stage, kaggle_mode=False):
        free = 0.25 if stage == "before_smoke_train" else 16.0
        return {"stage": stage, "status": "PASS", "devices": [{"free_memory_gb": free, "reserved_gb": 0.0, "allocated_gb": 0.0}]}

    monkeypatch.setattr(orch, "capture_gpu_memory_snapshot", snapshot)
    report = orch.build_orchestrator_report(config_path="config.yaml", kaggle_mode=True)
    assert report["decision"] == "BLOCK_FULL_TRAINING_MEMORY_BACKEND_BUG"


def test_missing_smoke_adapter_dir_blocks(monkeypatch, tmp_path):
    paths = _patch_paths(monkeypatch, tmp_path)
    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(orch, "load_base_model", lambda *a, **k: object())
    monkeypatch.setattr(orch, "verify_lora_targets", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr("kaggle_anti086.training.lora_backend.prepare_model_for_v2a_training", lambda model, config: model)
    monkeypatch.setattr(orch, "audit_trainable_parameters", lambda *a, **k: {"status": "PASS"})

    def fake_train(argv):
        paths["smoke_train"].write_text(json.dumps({"status": "PASS", "steps_completed": 5}), encoding="utf-8")
        return 0

    monkeypatch.setattr(orch, "train_v2a_main", fake_train)
    report = orch.build_orchestrator_report(config_path="config.yaml", kaggle_mode=True)
    assert report["decision"] == "BLOCK_FULL_TRAINING_RUNTIME"
    assert "smoke_adapter_dir_missing" in report["failures"]


def test_command_plan_has_no_placeholders(tmp_path):
    from kaggle_anti086.training.day8_2b_kaggle_commands import main

    out = tmp_path / "plan.txt"
    assert main(["--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "<" not in text and ">" not in text
    assert "package" not in "\n".join(line for line in text.splitlines() if not line.startswith("#")).lower()
    assert "submit" not in "\n".join(line for line in text.splitlines() if not line.startswith("#")).lower()
