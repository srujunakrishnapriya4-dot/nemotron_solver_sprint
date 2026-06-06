from __future__ import annotations

import json
from pathlib import Path

from kaggle_anti086.training import day10_train_solver_teacher_lora as launcher


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _teacher_row(answer: str = "7") -> dict:
    return {
        "id": "day10_teacher_numeric",
        "source_id": "day10_source_numeric",
        "family": "numeric_formula",
        "subfamily": "linear_offset",
        "prompt": "1 -> 3; 2 -> 5. Now solve: 3",
        "answer": answer,
        "expected_behavior": "answer",
        "solver_name": "solver_ensemble",
        "solver_source": "numeric_formula_solver",
        "solver_confidence": 0.95,
        "verified": True,
        "risk": "low",
        "metadata": {
            "source_rule_id": "day10_rule_numeric",
            "source_leakage_group": "day10_lg_numeric",
            "gold_used_for_target": False,
        },
    }


def _write_config(tmp_path: Path, teacher_path: Path) -> Path:
    config = tmp_path / "day10_small.yaml"
    config.write_text(
        "\n".join(
            [
                "stage: v4_solver_teacher_lora_small",
                "base_model_path: /kaggle/input/nemotron-model",
                "parent_adapter_path: none",
                f"train_teacher_path: {teacher_path.as_posix()}",
                "teacher_manifest_path: artifacts/sprint11/day10_solver_teacher_manifest.json",
                "teacher_audit_path: artifacts/sprint11/day10_solver_teacher_audit.json",
                "teacher_overlap_audit_path: artifacts/sprint11/day10_teacher_overlap_audit.json",
                "teacher_learnability_audit_path: artifacts/sprint11/day10_teacher_learnability_audit.json",
                "output_adapter_dir: /kaggle/working/anti086_adapters/v4_solver_teacher_small",
                "rank: 32",
                "lora_alpha: 32",
                "target_modules: q_proj,v_proj,o_proj",
                "learning_rate: 8e-8",
                "max_seq_len: 1024",
                "micro_batch_size: 1",
                "gradient_accumulation: 8",
                "num_steps: 500",
                "assistant_only_loss: true",
                "full_prompt_loss: false",
                "train_on_user: false",
                "teacher_direct_weight: 1.0",
                "abstain_weight: 0.0",
                "adapter_size_limit_mb: 1500",
                "not_submission_ready: true",
                "load_in_4bit: true",
                "bf16: auto",
            ]
        ),
        encoding="utf-8",
    )
    return config


def _write_gate_inputs(tmp_path: Path, teacher_path: Path) -> dict[str, Path]:
    audit = _write_json(tmp_path / "teacher_audit.json", {"status": "PASS", "row_count": 6144, "family_counts": {"format_only": 300}})
    overlap = _write_json(tmp_path / "overlap.json", {"status": "PASS"})
    learnability = _write_json(tmp_path / "learnability.json", {"status": "PASS", "subfamily_balance_warnings": {}})
    mix = _write_json(
        tmp_path / "mix.json",
        {
            "status": "PASS",
            "direct_answer_rows": 6144,
            "format_only_count": 300,
            "by_family": {"numeric_formula": 6144, "format_only": 300},
            "remaining_blockers": [],
            "smoke_training_eligible": True,
            "full_training_eligible": True,
        },
    )
    capacity = _write_json(tmp_path / "capacity.json", {"status": "PASS", "small_config_viable": True})
    abstain = tmp_path / "abstain.jsonl"
    abstain.write_text(json.dumps(_teacher_row("ABSTAIN")) + "\n", encoding="utf-8")
    config = _write_config(tmp_path, teacher_path)
    return {
        "config": config,
        "teacher_audit": audit,
        "overlap": overlap,
        "learnability": learnability,
        "mix": mix,
        "capacity": capacity,
        "abstain": abstain,
        "summary": tmp_path / "summary.json",
        "manifest": tmp_path / "manifest.json",
    }


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [max(2, ord(char) % 101) for char in text]


def _patch_light_backend(monkeypatch, tmp_path: Path, *, steps: int = 5, adapter_ok: bool = True):
    calls: dict[str, object] = {}
    monkeypatch.setattr(launcher, "capture_gpu_memory_snapshot", lambda stage, kaggle_mode=False: {"stage": stage, "status": "PASS"})
    monkeypatch.setattr(launcher, "cuda_empty_cache", lambda reason: {"status": "PASS", "reason": reason})
    monkeypatch.setattr(launcher, "load_tokenizer", lambda path: FakeTokenizer())
    monkeypatch.setattr(launcher, "load_base_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(launcher, "prepare_model_for_v2a_training", lambda model, config: {"model": model, "config": dict(config)})
    monkeypatch.setattr(launcher, "build_day10_smoke_output_dir", lambda config, config_path, smoke_steps: tmp_path / "adapter")

    def fake_train(model, tokenizer, dataset, config, output_dir):
        calls["num_steps"] = config["num_steps"]
        calls["dataset_len"] = len(dataset)
        calls["output_dir"] = str(output_dir)
        return {
            "status": "PASS",
            "steps_completed": steps,
            "loss_start": 1.5,
            "loss_end": 1.0,
            "loss_min": 1.0,
            "loss_max": 1.5,
            "loss_nan_detected": False,
            "grad_nan_detected": False,
            "adapter_dir": str(output_dir),
        }

    monkeypatch.setattr(launcher, "run_lora_training", fake_train)
    monkeypatch.setattr(
        launcher,
        "validate_saved_adapter",
        lambda output_dir: {
            "status": "PASS" if adapter_ok else "FAIL",
            "adapter_config_exists": adapter_ok,
            "adapter_model_exists": adapter_ok,
            "failures": [] if adapter_ok else ["adapter_model_safetensors_missing"],
        },
    )
    return calls


def test_dry_run_does_not_call_backend(monkeypatch, tmp_path):
    teacher = tmp_path / "teacher.jsonl"
    teacher.write_text(json.dumps(_teacher_row()) + "\n", encoding="utf-8")
    paths = _write_gate_inputs(tmp_path, teacher)
    monkeypatch.setattr(launcher, "run_day10_smoke_backend", lambda **kwargs: (_ for _ in ()).throw(AssertionError("backend_called")))

    rc = launcher.main(
        [
            "--config",
            str(paths["config"]),
            "--teacher-audit",
            str(paths["teacher_audit"]),
            "--overlap-audit",
            str(paths["overlap"]),
            "--learnability-audit",
            str(paths["learnability"]),
            "--mix-repair-report",
            str(paths["mix"]),
            "--abstain-policy-path",
            str(paths["abstain"]),
            "--capacity-audit",
            str(paths["capacity"]),
            "--out-summary",
            str(paths["summary"]),
            "--out-manifest",
            str(paths["manifest"]),
            "--dry-run",
        ]
    )

    assert rc == 0
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["trained"] is False
    assert summary["no_training_performed"] is True


def test_smoke_mode_calls_backend_when_gates_pass(monkeypatch, tmp_path):
    teacher = tmp_path / "teacher.jsonl"
    teacher.write_text(json.dumps(_teacher_row()) + "\n", encoding="utf-8")
    paths = _write_gate_inputs(tmp_path, teacher)

    def fake_backend(**kwargs):
        return {
            "backend_status": "PASS",
            "trained": True,
            "smoke_training_completed": True,
            "smoke_steps": kwargs["smoke_steps"],
            "steps_completed": kwargs["smoke_steps"],
            "intended_steps": kwargs["smoke_steps"],
            "loss_start": 1.0,
            "loss_end": 0.9,
            "loss_min": 0.9,
            "loss_max": 1.0,
            "loss_nan_detected": False,
            "grad_nan_detected": False,
            "adapter_dir": "/kaggle/working/anti086_adapters/fake",
            "adapter_config_exists": True,
            "adapter_model_exists": True,
            "dataset_report": {"status": "PASS"},
            "memory_snapshots": [],
            "training_log_history_path": "log.json",
            "run_provenance_path": "prov.json",
            "failures": [],
            "warnings": [],
            "full_training_allowed": False,
        }

    monkeypatch.setattr(launcher, "run_day10_smoke_backend", fake_backend)
    rc = launcher.main(
        [
            "--config",
            str(paths["config"]),
            "--teacher-audit",
            str(paths["teacher_audit"]),
            "--overlap-audit",
            str(paths["overlap"]),
            "--learnability-audit",
            str(paths["learnability"]),
            "--mix-repair-report",
            str(paths["mix"]),
            "--abstain-policy-path",
            str(paths["abstain"]),
            "--capacity-audit",
            str(paths["capacity"]),
            "--out-summary",
            str(paths["summary"]),
            "--out-manifest",
            str(paths["manifest"]),
            "--smoke-steps",
            "5",
            "--kaggle-mode",
        ]
    )

    assert rc == 0
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["trained"] is True
    assert summary["smoke_training_completed"] is True
    assert summary["full_training_allowed"] is False


def test_real_smoke_backend_mutates_num_steps_and_excludes_abstain_policy(monkeypatch, tmp_path):
    teacher = tmp_path / "teacher.jsonl"
    teacher.write_text(json.dumps(_teacher_row()) + "\n", encoding="utf-8")
    paths = _write_gate_inputs(tmp_path, teacher)
    summary = launcher.build_training_readiness(
        config_path=paths["config"],
        teacher_audit_path=paths["teacher_audit"],
        overlap_audit_path=paths["overlap"],
        learnability_audit_path=paths["learnability"],
        mix_repair_report_path=paths["mix"],
        capacity_audit_path=paths["capacity"],
        abstain_policy_path=paths["abstain"],
        dry_run=False,
        smoke_steps=5,
        kaggle_mode=True,
    )
    calls = _patch_light_backend(monkeypatch, tmp_path, steps=5, adapter_ok=True)

    backend = launcher.run_day10_smoke_backend(config_path=paths["config"], summary=summary, smoke_steps=5, out_summary_path=paths["summary"])

    assert backend["backend_status"] == "PASS"
    assert backend["trained"] is True
    assert calls["num_steps"] == 5
    assert calls["dataset_len"] == 1
    assert backend["dataset_report"]["direct_rows"] == 1


def test_smoke_backend_fails_incomplete_steps(monkeypatch, tmp_path):
    teacher = tmp_path / "teacher.jsonl"
    teacher.write_text(json.dumps(_teacher_row()) + "\n", encoding="utf-8")
    paths = _write_gate_inputs(tmp_path, teacher)
    summary = launcher.build_training_readiness(
        config_path=paths["config"],
        teacher_audit_path=paths["teacher_audit"],
        overlap_audit_path=paths["overlap"],
        learnability_audit_path=paths["learnability"],
        mix_repair_report_path=paths["mix"],
        capacity_audit_path=paths["capacity"],
        dry_run=False,
        smoke_steps=5,
        kaggle_mode=True,
    )
    _patch_light_backend(monkeypatch, tmp_path, steps=3, adapter_ok=True)

    backend = launcher.run_day10_smoke_backend(config_path=paths["config"], summary=summary, smoke_steps=5, out_summary_path=paths["summary"])

    assert backend["backend_status"] == "FAIL"
    assert "smoke_steps_incomplete" in backend["failures"]
    assert backend["trained"] is False


def test_smoke_backend_fails_missing_adapter_files(monkeypatch, tmp_path):
    teacher = tmp_path / "teacher.jsonl"
    teacher.write_text(json.dumps(_teacher_row()) + "\n", encoding="utf-8")
    paths = _write_gate_inputs(tmp_path, teacher)
    summary = launcher.build_training_readiness(
        config_path=paths["config"],
        teacher_audit_path=paths["teacher_audit"],
        overlap_audit_path=paths["overlap"],
        learnability_audit_path=paths["learnability"],
        mix_repair_report_path=paths["mix"],
        capacity_audit_path=paths["capacity"],
        dry_run=False,
        smoke_steps=5,
        kaggle_mode=True,
    )
    _patch_light_backend(monkeypatch, tmp_path, steps=5, adapter_ok=False)

    backend = launcher.run_day10_smoke_backend(config_path=paths["config"], summary=summary, smoke_steps=5, out_summary_path=paths["summary"])

    assert backend["backend_status"] == "FAIL"
    assert "adapter_model_safetensors_missing" in backend["failures"]
    assert backend["adapter_model_exists"] is False


def test_full_mode_without_smoke_report_remains_blocked(tmp_path):
    teacher = tmp_path / "teacher.jsonl"
    teacher.write_text(json.dumps(_teacher_row()) + "\n", encoding="utf-8")
    paths = _write_gate_inputs(tmp_path, teacher)

    rc = launcher.main(
        [
            "--config",
            str(paths["config"]),
            "--teacher-audit",
            str(paths["teacher_audit"]),
            "--overlap-audit",
            str(paths["overlap"]),
            "--learnability-audit",
            str(paths["learnability"]),
            "--mix-repair-report",
            str(paths["mix"]),
            "--capacity-audit",
            str(paths["capacity"]),
            "--out-summary",
            str(paths["summary"]),
            "--out-manifest",
            str(paths["manifest"]),
            "--kaggle-mode",
        ]
    )

    assert rc == 2
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["trained"] is False
    assert summary["full_training_allowed"] is False
    assert "full_training_requires_smoke_report" in summary["failures"]
