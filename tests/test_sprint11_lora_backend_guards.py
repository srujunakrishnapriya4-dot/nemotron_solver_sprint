from pathlib import Path

import pytest

from kaggle_anti086.training.lora_backend import _validate_lora_shape, collate_sft_batch, validate_saved_adapter


def test_lora_rejects_rank_gt_32():
    with pytest.raises(ValueError):
        _validate_lora_shape({"rank": 64}, ["q_proj", "v_proj", "o_proj"])


def test_lora_rejects_lm_head():
    with pytest.raises(ValueError):
        _validate_lora_shape({"rank": 32}, ["q_proj", "v_proj", "o_proj", "lm_head"])


def test_lora_requires_exact_targets():
    with pytest.raises(ValueError):
        _validate_lora_shape({"rank": 32}, ["q_proj", "v_proj"])


def test_adapter_validation_requires_files(tmp_path):
    report = validate_saved_adapter(tmp_path)
    assert report["status"] == "FAIL"
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "adapter_model.safetensors").write_text("x")
    assert validate_saved_adapter(tmp_path)["status"] == "PASS"


def test_custom_collator_preserves_labels():
    batch = collate_sft_batch([{"input_ids": [1], "attention_mask": [1], "labels": [-100]}])
    assert "labels" in batch


def _tolist(value):
    return value.tolist() if hasattr(value, "tolist") else value


def test_custom_collator_uses_tokenizer_pad_id():
    class Tok:
        pad_token_id = 123
        eos_token_id = 456

    batch = collate_sft_batch(
        [
            {"input_ids": [1], "attention_mask": [1], "labels": [1]},
            {"input_ids": [2, 3], "attention_mask": [1, 1], "labels": [2, 3]},
        ],
        Tok(),
    )
    assert _tolist(batch["input_ids"])[0][-1] == 123
    assert _tolist(batch["labels"])[0][-1] == -100


def test_custom_collator_uses_eos_fallback_and_rejects_missing_ids():
    class EosTok:
        pad_token_id = None
        eos_token_id = 456

    batch = collate_sft_batch(
        [
            {"input_ids": [1], "attention_mask": [1], "labels": [1]},
            {"input_ids": [2, 3], "attention_mask": [1, 1], "labels": [2, 3]},
        ],
        EosTok(),
    )
    assert _tolist(batch["input_ids"])[0][-1] == 456
    class BadTok:
        pad_token_id = None
        eos_token_id = None
    with pytest.raises(ValueError):
        collate_sft_batch([{"input_ids": [1], "attention_mask": [1], "labels": [1]}], BadTok())
