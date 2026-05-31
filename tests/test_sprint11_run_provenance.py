from pathlib import Path

import pytest

from kaggle_anti086.training.run_provenance import build_run_provenance


def _write_config(path: Path, direct: Path, corrected: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "stage: v2a_base_lora",
                "base_model_path: /kaggle/input/model",
                "parent_adapter_path: none",
                f"train_direct_path: {direct}",
                f"train_solver_corrected_path: {corrected}",
                "train_abstain_safety_path: unused.jsonl",
                "train_hard_negative_path: unused2.jsonl",
                "output_adapter_dir: /kaggle/working/anti086_adapters/v2a",
                "rank: 32",
                "lora_alpha: 32",
                "target_modules: q_proj,v_proj,o_proj",
                "learning_rate: 1e-7",
                "max_seq_len: 1024",
                "micro_batch_size: 1",
                "gradient_accumulation: 8",
                "num_steps: 500",
                "assistant_only_loss: true",
                "full_prompt_loss: false",
                "train_on_user: false",
                "direct_answer_weight: 1.0",
                "solver_corrected_weight: 0.5",
                "abstain_safety_weight: 0.0",
                "hard_negative_weight: 0.0",
                "adapter_size_limit_mb: 1500",
                "not_submission_ready: true",
            ]
        ),
        encoding="utf-8",
    )


def test_records_config_and_corpus_hashes(tmp_path):
    direct = tmp_path / "direct.jsonl"
    corrected = tmp_path / "corrected.jsonl"
    direct.write_text("{}\n", encoding="utf-8")
    corrected.write_text("{}\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    _write_config(config, direct, corrected)
    report = build_run_provenance(config)
    assert report["status"] == "PASS"
    assert report["config_sha256"]
    assert report["train_direct_sha256"]
    assert report["train_solver_corrected_sha256"]


def test_missing_required_corpus_file_fails(tmp_path):
    direct = tmp_path / "missing.jsonl"
    corrected = tmp_path / "corrected.jsonl"
    corrected.write_text("{}\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    _write_config(config, direct, corrected)
    report = build_run_provenance(config)
    assert report["status"] == "FAIL"
    assert any(item.startswith("missing_required_file") for item in report["failures"])
