from pathlib import Path

from kaggle_anti086.training import day8_2c_smoke_preflight as pf


def _patch_common(monkeypatch, tmp_path):
    paths = {
        "collator_audit": tmp_path / "collator.json",
        "weighted_sampling": tmp_path / "weighted.json",
        "truncation": tmp_path / "truncation.json",
        "label_decode": tmp_path / "label.json",
        "leakage": tmp_path / "leakage.json",
        "model_environment": tmp_path / "environment.json",
        "package_submission_guard": tmp_path / "package.json",
        "lora_target": tmp_path / "target.json",
        "trainable_params": tmp_path / "trainable.json",
    }
    monkeypatch.setattr(pf, "PATHS", paths)
    monkeypatch.setattr(pf, "PREFLIGHT_MEMORY", tmp_path / "memory.json")
    config = {
        "stage": "v2a_base_lora",
        "base_model_path": "/kaggle/input/model",
        "train_direct_path": str(tmp_path / "direct.jsonl"),
        "train_solver_corrected_path": str(tmp_path / "corrected.jsonl"),
        "seed": 42,
        "rank": 32,
        "target_modules": ["q_proj", "v_proj", "o_proj"],
        "load_in_4bit": True,
    }
    Path(config["train_direct_path"]).write_text("{}\n", encoding="utf-8")
    Path(config["train_solver_corrected_path"]).write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(pf, "load_training_config", lambda path: dict(config))
    monkeypatch.setattr(pf, "load_weighted_sft_rows", lambda config: [])
    monkeypatch.setattr(pf, "build_weighted_sample", lambda items, seed=42: [])
    monkeypatch.setattr(pf, "build_real_collator_audit", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(pf, "build_weighted_sampling_report", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(pf, "build_truncation_audit", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(pf, "audit_supervised_label_decode", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(pf, "audit_final_sft_leakage", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(pf, "build_package_submission_guard", lambda *a, **k: {"status": "PASS"})
    return config


def test_local_missing_stack_needs_kaggle(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(pf, "build_model_environment_report", lambda *a, **k: {"status": "WARN"})
    report = pf.build_smoke_preflight_report("config.yaml", kaggle_mode=False)
    assert report["decision"] == "NEEDS_KAGGLE_MODEL_STACK"


def test_mocked_pass_ready_for_smoke(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(pf, "build_model_environment_report", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(pf, "load_base_model", lambda *a, **k: object())
    monkeypatch.setattr(pf, "verify_lora_targets", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr("kaggle_anti086.training.lora_backend.prepare_model_for_v2a_training", lambda model, config: model)
    monkeypatch.setattr(pf, "audit_trainable_parameters", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(pf, "write_gpu_memory_report", lambda *a, **k: {"status": "PASS"})
    report = pf.build_smoke_preflight_report("config.yaml", kaggle_mode=True)
    assert report["decision"] == "PASS_PREFLIGHT_READY_FOR_SMOKE_TRAIN"


def test_target_fail_blocks(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(pf, "build_model_environment_report", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(pf, "load_base_model", lambda *a, **k: object())
    monkeypatch.setattr(pf, "verify_lora_targets", lambda *a, **k: {"status": "FAIL"})
    monkeypatch.setattr("kaggle_anti086.training.lora_backend.prepare_model_for_v2a_training", lambda model, config: model)
    monkeypatch.setattr(pf, "audit_trainable_parameters", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(pf, "write_gpu_memory_report", lambda *a, **k: {"status": "PASS"})
    report = pf.build_smoke_preflight_report("config.yaml", kaggle_mode=True)
    assert report["decision"] == "BLOCK_PREFLIGHT_LORA_TARGET"


def test_trainable_fail_blocks(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(pf, "build_model_environment_report", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(pf, "load_base_model", lambda *a, **k: object())
    monkeypatch.setattr(pf, "verify_lora_targets", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr("kaggle_anti086.training.lora_backend.prepare_model_for_v2a_training", lambda model, config: model)
    monkeypatch.setattr(pf, "audit_trainable_parameters", lambda *a, **k: {"status": "FAIL"})
    monkeypatch.setattr(pf, "write_gpu_memory_report", lambda *a, **k: {"status": "PASS"})
    report = pf.build_smoke_preflight_report("config.yaml", kaggle_mode=True)
    assert report["decision"] == "BLOCK_PREFLIGHT_TRAINABLE_PARAMS"
