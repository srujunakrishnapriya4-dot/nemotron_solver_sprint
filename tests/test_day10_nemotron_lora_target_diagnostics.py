from __future__ import annotations

import json

from kaggle_anti086.training.day10_nemotron_lora_target_diagnostics import build_nemotron_lora_target_diagnostics, main
from kaggle_anti086.training import day10_nemotron_lora_target_diagnostics as diag


class Weight:
    def __init__(self, *shape):
        self.shape = shape


class Linear4bit:
    def __init__(self):
        self.weight = Weight(1, 5505024)
        self.in_features = 4096
        self.out_features = 14336
        self.quant_state = object()


class FakeModel:
    def named_modules(self):
        yield "", self
        yield "model.layers.0.self_attn.q_proj", Linear4bit()
        yield "model.layers.0.self_attn.v_proj", Linear4bit()


def _config(tmp_path, *, load_in_4bit=True):
    path = tmp_path / "config.yaml"
    path.write_text(
        "\n".join(
            [
                "stage: v4_solver_teacher_lora_small",
                "base_model_path: /kaggle/input/nemotron-model",
                "rank: 32",
                "lora_alpha: 32",
                "target_modules: q_proj,v_proj",
                f"load_in_4bit: {str(load_in_4bit).lower()}",
                "assistant_only_loss: true",
                "full_prompt_loss: false",
                "train_on_user: false",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_diagnostics_dry_inspection_warns_without_fake_pass(tmp_path):
    report = build_nemotron_lora_target_diagnostics(_config(tmp_path), load_model=False)

    assert report["status"] == "WARN"
    assert "model_not_loaded_dry_inspection" in report["warnings"]
    assert report["no_training_performed"] is True


def test_diagnostics_reports_linear4bit_incompatibility(monkeypatch, tmp_path):
    monkeypatch.setattr(diag, "load_tokenizer", lambda path: object())
    monkeypatch.setattr(diag, "load_base_model", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(diag, "check_training_dependencies", lambda *args, **kwargs: {"status": "PASS", "failures": []})

    report = build_nemotron_lora_target_diagnostics(_config(tmp_path), load_model=True, candidate_targets=["q_proj", "v_proj"])

    assert report["status"] == "FAIL"
    assert report["peft_linear4bit_shape_incompatible"] is True
    assert "peft_linear4bit_shape_incompatible" in report["failures"]
    assert report["target_modules"][0]["weight_shape"] == [1, 5505024]


def test_diagnostics_cli_writes_report(tmp_path):
    out = tmp_path / "diagnostics.json"
    rc = main(["--config", str(_config(tmp_path)), "--out", str(out)])

    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["status"] == "WARN"
    assert data["packaging_allowed"] is False
    assert data["submission_allowed"] is False
