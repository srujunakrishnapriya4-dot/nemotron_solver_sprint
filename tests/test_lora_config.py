from __future__ import annotations

from dataclasses import replace
import pytest

from nemotron_engine.training.lora_config import (
    LoRAConfig,
    LoRAConfigError,
    load_lora_config,
    save_lora_config,
    validate_adapter_config_json,
    validate_lora_config,
)


def config(**updates: object) -> LoRAConfig:
    data = {
        "base_model_name": "nemotron-mini",
        "target_modules": ("q_proj", "v_proj"),
        "rank": 32,
        "alpha": 64,
        "dropout": 0.05,
    }
    data.update(updates)
    return LoRAConfig(**data)


def test_valid_rank_32_config_passes_and_hash_deterministic() -> None:
    first = config(metadata={"b": 2, "a": 1})
    second = config(metadata={"a": 1, "b": 2})

    assert validate_lora_config(first) == first
    assert first.config_hash == second.config_hash


def test_rejects_bad_rank_types_and_bounds() -> None:
    with pytest.raises(LoRAConfigError):
        config(rank=33)
    with pytest.raises(LoRAConfigError):
        config(rank=0)
    with pytest.raises(LoRAConfigError):
        config(rank=True)
    with pytest.raises(LoRAConfigError):
        config(rank="32")


def test_rejects_bad_dropout_alpha_and_targets() -> None:
    with pytest.raises(LoRAConfigError):
        config(dropout=1.0)
    with pytest.raises(LoRAConfigError):
        config(dropout=-0.1)
    with pytest.raises(LoRAConfigError):
        config(alpha=0)
    with pytest.raises(LoRAConfigError):
        config(target_modules=())


def test_adapter_config_json_rejects_rank_over_32_and_strings() -> None:
    for payload in ({"r": 33}, {"rank": 33}, {"lora_rank": 33}, {"peft_config": {"r": 33}}):
        with pytest.raises(LoRAConfigError):
            validate_adapter_config_json(payload)
    with pytest.raises(LoRAConfigError):
        validate_adapter_config_json({"rank": "32"})
    assert validate_adapter_config_json({"peft_config": {"r": 32}})


def test_config_hash_tampering_rejected() -> None:
    with pytest.raises(LoRAConfigError):
        replace(config(), config_hash="forged")


def test_save_load_round_trip_deterministic() -> None:
    class MemoryPath:
        def __init__(self) -> None:
            self.text = ""

        def write_text(self, text: str, encoding: str) -> None:
            assert encoding == "utf-8"
            self.text = text

        def read_text(self, encoding: str) -> str:
            assert encoding == "utf-8"
            return self.text

    path = MemoryPath()
    original = config()

    save_lora_config(original, path)
    loaded = load_lora_config(path)

    assert loaded == original
    assert loaded.config_hash == original.config_hash
