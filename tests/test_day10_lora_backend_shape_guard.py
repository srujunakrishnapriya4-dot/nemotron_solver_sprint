from __future__ import annotations

import pytest

from kaggle_anti086.training.lora_backend import (
    _validate_lora_shape,
    inspect_target_modules,
    prepare_model_for_v2a_training,
    validate_target_modules_for_peft,
)


class Shape:
    def __init__(self, *values: int):
        self._values = values

    def __iter__(self):
        return iter(self._values)


class Weight:
    def __init__(self, *shape: int):
        self.shape = Shape(*shape)


class Linear4bit:
    def __init__(self, weight_shape=(1, 5505024), in_features=4096, out_features=14336):
        self.weight = Weight(*weight_shape)
        self.in_features = in_features
        self.out_features = out_features
        self.quant_state = object()


class Linear:
    def __init__(self, weight_shape=(14336, 4096), in_features=4096, out_features=14336):
        self.weight = Weight(*weight_shape)
        self.in_features = in_features
        self.out_features = out_features


class FakeModel:
    def __init__(self, modules):
        self.modules = modules

    def named_modules(self):
        yield "", self
        for name, module in self.modules.items():
            yield name, module


def test_compressed_linear4bit_shape_detected_before_peft():
    model = FakeModel({"layers.0.self_attn.q_proj": Linear4bit()})

    report = validate_target_modules_for_peft(model, ["q_proj"])

    assert report["status"] == "FAIL"
    assert "peft_linear4bit_shape_incompatible" in report["failures"]
    assert report["incompatible_modules"][0]["weight_shape"] == [1, 5505024]
    assert report["incompatible_modules"][0]["expected_weight_shape"] == [14336, 4096]


def test_compatible_linear_shape_passes_peft_guard():
    model = FakeModel({"layers.0.self_attn.q_proj": Linear(), "layers.0.self_attn.v_proj": Linear()})

    report = validate_target_modules_for_peft(model, ["q_proj", "v_proj"])

    assert report["status"] == "PASS"
    assert report["matched_target_counts"] == {"q_proj": 1, "v_proj": 1}
    assert report["incompatible_modules"] == []


def test_prepare_model_refuses_incompatible_target_before_get_peft(monkeypatch):
    model = FakeModel({"layers.0.self_attn.q_proj": Linear4bit(), "layers.0.self_attn.v_proj": Linear()})

    with pytest.raises(RuntimeError, match="peft_linear4bit_shape_incompatible"):
        prepare_model_for_v2a_training(model, {"rank": 32, "target_modules": ["q_proj", "v_proj"], "smoke_bf16_runtime_only": True, "load_in_4bit": False, "num_steps": 5})


def test_qv_only_requires_explicit_smoke_bf16_flag():
    with pytest.raises(ValueError, match="target_modules_must_be_q_proj_v_proj_o_proj"):
        _validate_lora_shape({"rank": 32, "load_in_4bit": False, "num_steps": 5}, ["q_proj", "v_proj"])

    _validate_lora_shape(
        {"rank": 32, "smoke_bf16_runtime_only": True, "load_in_4bit": False, "num_steps": 5},
        ["q_proj", "v_proj"],
    )


def test_smoke_bf16_flag_never_allows_long_run_or_4bit():
    with pytest.raises(ValueError, match="smoke_bf16_runtime_only_max_5_steps"):
        _validate_lora_shape({"rank": 32, "smoke_bf16_runtime_only": True, "load_in_4bit": False, "num_steps": 6}, ["q_proj", "v_proj"])
    with pytest.raises(ValueError, match="smoke_bf16_runtime_only_requires_load_in_4bit_false"):
        _validate_lora_shape({"rank": 32, "smoke_bf16_runtime_only": True, "load_in_4bit": True, "num_steps": 5}, ["q_proj", "v_proj"])


def test_inspect_target_modules_records_core_shape_fields():
    rows = inspect_target_modules(FakeModel({"layer.q_proj": Linear()}), ["q_proj"])

    assert rows[0]["class_name"] == "Linear"
    assert rows[0]["in_features"] == 4096
    assert rows[0]["out_features"] == 14336
    assert rows[0]["weight_shape"] == [14336, 4096]
