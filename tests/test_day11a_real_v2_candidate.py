from __future__ import annotations

import json
from pathlib import Path
import shutil

from kaggle_anti086.training import day11a_real_v2_candidate as day11a


TMP = Path("artifacts/test_tmp/day11a_real_v2_candidate")


def _clean() -> Path:
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)
    return TMP


def _json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _config(path: Path, *, output="/kaggle/working/anti086_adapters/v2a_bf16_qv_50", corpus="artifacts/sprint11/day10_solver_teacher_direct.jsonl", extra: list[str] | None = None) -> Path:
    lines = [
        "stage: v2a_bf16_qv_50",
        "base_model_path: /kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1",
        f"teacher_corpus_path: {corpus}",
        "abstain_policy_path: artifacts/sprint11/day10_solver_teacher_abstain_policy.jsonl",
        f"output_adapter_dir: {output}",
        "rank: 32",
        "lora_alpha: 32",
        "target_modules: q_proj,v_proj",
        "learning_rate: 8e-8",
        "max_seq_len: 1024",
        "micro_batch_size: 1",
        "gradient_accumulation: 8",
        "num_steps: 50",
        "assistant_only_loss: true",
        "full_prompt_loss: false",
        "train_on_user: false",
        "teacher_direct_weight: 1.0",
        "load_in_4bit: false",
        "bf16: true",
        "smoke_bf16_runtime_only: false",
        "real_candidate_training: true",
        "submission_ready: false",
        "public_submission_allowed: false",
    ]
    if extra:
        lines.extend(extra)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _gate_files(root: Path) -> dict[str, Path]:
    return {
        "teacher": _json(root / "teacher.json", {"status": "PASS", "row_count": 6144}),
        "overlap": _json(root / "overlap.json", {"status": "PASS"}),
        "learnability": _json(root / "learnability.json", {"status": "PASS"}),
        "mix": _json(root / "mix.json", {"status": "PASS", "direct_answer_rows": 6144, "format_only_count": 300}),
        "capacity": _json(root / "capacity.json", {"status": "PASS", "small_config_viable": True}),
    }


def _args(config: Path, gates: dict[str, Path], root: Path) -> list[str]:
    return [
        "--config",
        str(config),
        "--teacher-audit",
        str(gates["teacher"]),
        "--overlap-audit",
        str(gates["overlap"]),
        "--learnability-audit",
        str(gates["learnability"]),
        "--mix-repair-report",
        str(gates["mix"]),
        "--capacity-audit",
        str(gates["capacity"]),
        "--out-summary",
        str(root / "summary.json"),
        "--out-manifest",
        str(root / "manifest.json"),
    ]


def test_v2a_config_validates_bf16_qv_only():
    failures, _ = day11a.validate_day11a_config(day11a.load_training_config("kaggle_anti086/training/configs/v2a_bf16_qv_50.yaml"))
    assert failures == []


def test_rejects_unsafe_lora_shapes_and_paths():
    root = _clean()
    cases = [
        ["load_in_4bit: true"],
        ["target_modules: q_proj,v_proj,o_proj"],
        ["rank: 64"],
        ["output_adapter_dir: /kaggle/input/bad"],
        ["output_adapter_dir: artifacts/test_tmp/repo_adapter"],
    ]
    for idx, extra in enumerate(cases):
        config = day11a.load_training_config(_config(root / f"bad_{idx}.yaml", extra=extra))
        failures, _ = day11a.validate_day11a_config(config)
        assert failures


def test_dry_run_writes_summary_manifest_without_backend(monkeypatch):
    root = _clean()
    gates = _gate_files(root)
    config = _config(root / "config.yaml")
    monkeypatch.setattr(day11a, "load_tokenizer", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("model_loaded")))

    rc = day11a.main([*_args(config, gates, root), "--dry-run", "--no-submit"])

    assert rc == 0
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert summary["trained"] is False
    assert summary["packaging_allowed"] is False
    assert summary["submission_allowed"] is False
    assert manifest["submit_recommended"] is False


def test_direct_corpus_with_abstain_fails():
    root = _clean()
    corpus = root / "direct.jsonl"
    corpus.write_text(json.dumps({"id": "x", "prompt": "p", "answer": "ABSTAIN"}) + "\n", encoding="utf-8")
    config = _config(root / "config.yaml", corpus=corpus.as_posix())
    gates = _gate_files(root)

    summary = day11a.build_day11a_readiness(
        config_path=config,
        teacher_audit_path=gates["teacher"],
        overlap_audit_path=gates["overlap"],
        learnability_audit_path=gates["learnability"],
        mix_repair_report_path=gates["mix"],
        capacity_audit_path=gates["capacity"],
        dry_run=True,
    )

    assert summary["status"] == "FAIL"
    assert "direct_corpus_has_abstain" in summary["failures"]


def test_missing_audit_fails_explicitly():
    root = _clean()
    gates = _gate_files(root)
    config = _config(root / "config.yaml")
    summary = day11a.build_day11a_readiness(
        config_path=config,
        teacher_audit_path=root / "missing.json",
        overlap_audit_path=gates["overlap"],
        learnability_audit_path=gates["learnability"],
        mix_repair_report_path=gates["mix"],
        capacity_audit_path=gates["capacity"],
        dry_run=True,
    )
    assert any(item.startswith("missing_audit:teacher_audit") for item in summary["failures"])


def test_skip_train_reuses_adapter_only_when_both_files_exist():
    root = _clean()
    adapter = root / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_text("fake-test-only", encoding="utf-8")
    gates = _gate_files(root)
    config = _config(root / "config.yaml")
    summary = day11a.build_day11a_readiness(
        config_path=config,
        teacher_audit_path=gates["teacher"],
        overlap_audit_path=gates["overlap"],
        learnability_audit_path=gates["learnability"],
        mix_repair_report_path=gates["mix"],
        capacity_audit_path=gates["capacity"],
        adapter_dir=adapter,
        kaggle_mode=True,
        train=True,
        skip_train_if_adapter_exists=True,
    )
    assert summary["status"] == "PASS_REUSED_ADAPTER"
    assert summary["reused_adapter"] is True
    shutil.rmtree(adapter, ignore_errors=True)
