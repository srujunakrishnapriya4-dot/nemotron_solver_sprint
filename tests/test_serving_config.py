from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.runtime.serving_config import (  # noqa: E402
    ServingConfig,
    ServingConfigError,
    ServingRuntime,
    build_serving_config,
    load_serving_config,
    save_serving_config,
    validate_serving_config,
)


def test_default_serving_config_validates() -> None:
    config = ServingConfig()

    assert validate_serving_config(config) is config
    assert config.temperature == 0.0
    assert config.top_p == 1.0
    assert config.max_tokens == 512
    assert config.stop == ()
    assert config.num_samples == 1
    assert config.majority_vote is False
    assert config.batch_size == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temperature": 0.1}, "temperature"),
        ({"top_p": 0}, "top_p"),
        ({"top_p": 1.1}, "top_p"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"num_samples": 2}, "num_samples"),
        ({"majority_vote": True}, "majority_vote"),
        ({"batch_size": 0}, "batch_size"),
    ],
)
def test_strict_submission_mode_rejects_non_deterministic_or_invalid_settings(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ServingConfigError, match=message):
        ServingConfig(**kwargs)


def test_serving_config_exports_vllm_kwargs_and_command() -> None:
    config = ServingConfig(
        model_path="/models/nemotron",
        served_model_name="nemotron",
        tensor_parallel_size=8,
        max_model_len=32768,
        quantization="fp8",
        extra_args={"trust_remote_code": True},
    )

    assert config.model_name == "nemotron"
    assert config.to_vllm_kwargs()["quantization"] == "fp8"
    assert config.to_vllm_kwargs()["trust_remote_code"] is True
    command = config.to_vllm_command()
    assert command[:3] == ["vllm", "serve", "/models/nemotron"]
    assert "--served-model-name" in command
    assert "--trust-remote-code" in command


def test_hash_fields_and_save_load_round_trip_are_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    storage: dict[str, str] = {}
    config = build_serving_config(
        model_path="/m",
        runtime=ServingRuntime.OPENAI_COMPATIBLE,
        stop=("END",),
        prompt_template_hash="prompt-sha",
        tokenizer_hash="tokenizer-sha",
        model_hash="model-sha",
        adapter_hash="adapter-sha",
    )

    def write_text(self: Path, text: str, encoding: str = "utf-8") -> int:
        storage[str(self)] = text
        return len(text)

    def read_text(self: Path, encoding: str = "utf-8") -> str:
        return storage[str(self)]

    monkeypatch.setattr(Path, "write_text", write_text)
    monkeypatch.setattr(Path, "read_text", read_text)

    path = Path("serving_config.json")
    save_serving_config(config, path)
    first = storage[str(path)]
    loaded = load_serving_config(path)
    save_serving_config(loaded, path)
    second = storage[str(path)]

    assert loaded == config
    assert first == second
    assert loaded.to_dict()["runtime"] == "openai_compatible"
    assert loaded.to_dict()["stop"] == ["END"]
