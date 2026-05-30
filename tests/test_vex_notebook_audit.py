from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.vex_progress_bridge.notebook_audit import audit_vex_notebooks  # noqa: E402


def write_ipynb(path: Path, source: str) -> None:
    path.write_text(json.dumps({"cells": [{"cell_type": "code", "source": source.splitlines(True)}]}), encoding="utf-8")


def test_notebook_audit_detects_vex_training_and_eval_features(tmp_path: Path) -> None:
    train = tmp_path / "train.ipynb"
    eval_nb = tmp_path / "eval.ipynb"
    write_ipynb(
        train,
        "LORA_RANK=32\nMAX_SEQ_LEN=1024\nfrom cut_cross_entropy import linear_cross_entropy\n"
        "loss_weights = tokens['loss_weights']\nFastLanguageModel\nTARGET=['q_proj','k_proj','lm_head']\ntrain_tokens.jsonl\n",
    )
    write_ipynb(eval_nb, "from vllm import LLM\nLoRARequest\nmax_model_len=8192\nmax_tokens=16\ntemperature=0.0\ntop_p=1.0\n")

    audit = audit_vex_notebooks(train, eval_nb, tmp_path / "audit.json")

    assert audit["features"]["cut_cross_entropy"] is True
    assert audit["features"]["preprocessed_tokens_loaded"] is True
    assert audit["eval_settings"]["vllm_usage"] is True
    assert audit["eval_settings"]["lora_request_usage"] is True
    assert "lm_head" in audit["target_modules"]


def test_real_vex_notebook_constants_are_extracted(tmp_path: Path) -> None:
    train = Path(r"C:\Users\ADMIN\OneDrive\Desktop\vex506-reinst-kaggle-replica.ipynb")
    eval_nb = Path(r"C:\Users\ADMIN\OneDrive\Desktop\vex-eval-vex506-reinst.ipynb")
    audit = audit_vex_notebooks(train, eval_nb, tmp_path / "audit.json")

    assert audit["notebooks_found"] is True
    assert audit["extracted_training_config"]["lora_rank"] == 32
    assert audit["extracted_training_config"]["lora_alpha"] == 32
    assert audit["extracted_training_config"]["lora_dropout"] == 0.0
    assert audit["extracted_training_config"]["max_seq_len"] == 8192
    assert audit["extracted_training_config"]["num_steps"] == 665
    assert audit["extracted_training_config"]["batch_size"] == 32
    assert audit["extracted_training_config"]["micro_batch_size"] == 4
    assert audit["extracted_training_config"]["learning_rate"] == 2e-4
    assert audit["extracted_training_config"]["reset_weights"] is True
    assert audit["extracted_training_config"]["in_proj_only"] is False
    assert audit["extracted_training_config"]["moe_tie_weights"] is False
    assert audit["extracted_training_config"]["shuffle_dataset"] is True
    assert audit["extracted_training_config"]["target_modules"] == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "in_proj",
        "out_proj",
        "lm_head",
    )
    assert audit["extracted_eval_config"]["max_new_tokens"] == 7680
    assert audit["extracted_eval_config"]["max_model_len"] == 8192
    assert audit["extracted_eval_config"]["max_num_seqs"] == 64
    assert audit["extracted_eval_config"]["gpu_mem_util"] == 0.85
    assert audit["extracted_eval_config"]["held_variant_primary"] == "v2"
    assert audit["extracted_eval_config"]["held_variant_secondary"] == "v3"
    assert abs(sum(audit["extracted_eval_config"]["category_weights"].values()) - 1.0) < 1e-12


def test_notebook_missing_records_instruction(tmp_path: Path) -> None:
    audit = audit_vex_notebooks(tmp_path / "missing_train.ipynb", tmp_path / "missing_eval.ipynb", tmp_path / "audit.json")

    assert audit["notebooks_found"] is False
    assert "Copy vex506" in audit["missing_instruction"]
