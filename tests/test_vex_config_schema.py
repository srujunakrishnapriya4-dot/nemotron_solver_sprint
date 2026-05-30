from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.vex_progress_bridge.vex_config_schema import VexConfig, VexConfigError  # noqa: E402


def config(**kwargs) -> VexConfig:
    payload = dict(
        mode="public_safe_micro",
        variant_name="micro_sanity",
        base_model_path="/base",
        parent_adapter_path="/adapter",
        token_corpus_path="/tokens.jsonl",
        output_adapter_dir="/out",
        rank=32,
        lora_alpha=32,
        lora_dropout=0.03,
        learning_rate=2e-7,
        max_rows=64,
        max_seq_len=768,
        micro_batch_size=1,
        grad_accum=8,
        num_steps=20,
    )
    payload.update(kwargs)
    return VexConfig(**payload)


def test_vex_config_accepts_safe_micro_config() -> None:
    assert config().rank == 32


def test_vex_config_rejects_rank_over_32() -> None:
    with pytest.raises(VexConfigError):
        config(rank=64)


def test_vex_config_rejects_high_lr() -> None:
    with pytest.raises(VexConfigError):
        config(learning_rate=2e-5)


def test_vex_replica_config_refuses_missing_private_flag_or_audit() -> None:
    with pytest.raises(VexConfigError):
        config(mode="vex_replica_if_inputs_available", learning_rate=2e-4, require_vex_private_inputs=False, reset_weights=True)
    with pytest.raises(VexConfigError):
        config(
            mode="vex_replica_if_inputs_available",
            learning_rate=2e-4,
            require_vex_private_inputs=True,
            reset_weights=True,
            notebook_audit_found=False,
        )


def test_public_safe_micro_can_run_without_private_inputs() -> None:
    assert config(mode="public_safe_micro", public_safe_fallback=True).mode == "public_safe_micro"
