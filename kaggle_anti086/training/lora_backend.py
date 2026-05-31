from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from kaggle_anti086.kaggle_path_safety import safe_mkdir_for_output
from kaggle_anti086.training.sft_dataset import Sprint11SFTDataset
from kaggle_anti086.training.training_safety_callbacks import Sprint11TrainingSafetyCallback
from kaggle_anti086.training.training_config_schema import _target_modules


def build_lora_config(config: dict[str, Any]):
    targets = _target_modules(config)
    _validate_lora_shape(config, targets)
    from peft import LoraConfig  # type: ignore

    return LoraConfig(
        r=int(config["rank"]),
        lora_alpha=int(config["lora_alpha"]),
        target_modules=targets,
        lora_dropout=float(config.get("lora_dropout", 0.0) or 0.0),
        bias="none",
        task_type="CAUSAL_LM",
    )


def prepare_model_for_v2a_training(model, config: dict[str, Any]):
    lora_config = build_lora_config(config)
    try:
        from peft import get_peft_model, prepare_model_for_kbit_training  # type: ignore

        if bool(config.get("load_in_4bit", False)):
            model = prepare_model_for_kbit_training(model)
        return get_peft_model(model, lora_config)
    except ImportError:
        raise RuntimeError("peft is required for LoRA training")


def run_lora_training(model, tokenizer, dataset: Sprint11SFTDataset, config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    _validate_lora_shape(config, _target_modules(config))
    safe_mkdir_for_output(output_dir, field_name="output_adapter_dir")
    try:
        from transformers import Trainer, TrainingArguments  # type: ignore
    except ImportError as exc:
        raise RuntimeError("transformers Trainer is required for real training") from exc
    args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=int(config["num_steps"]),
        per_device_train_batch_size=int(config["micro_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation"]),
        learning_rate=float(config["learning_rate"]),
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="no",
        bf16=bool(config.get("bf16", True)),
        fp16=False,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=lambda features: collate_sft_batch(features, tokenizer), tokenizer=tokenizer, callbacks=[Sprint11TrainingSafetyCallback()])
    result = trainer.train()
    losses = [float(row["loss"]) for row in trainer.state.log_history if "loss" in row]
    save_lora_adapter(model, output_dir)
    validation = validate_saved_adapter(output_dir)
    return {
        "status": "PASS" if validation["status"] == "PASS" and int(trainer.state.global_step) > 0 else "FAIL",
        "steps_completed": int(trainer.state.global_step),
        "loss_start": losses[0] if losses else None,
        "loss_end": losses[-1] if losses else None,
        "loss_min": min(losses) if losses else None,
        "loss_max": max(losses) if losses else None,
        "loss_nan_detected": any(math.isnan(loss) for loss in losses),
        "grad_nan_detected": False,
        "train_runtime": getattr(result, "metrics", {}).get("train_runtime") if result is not None else None,
        "samples_seen": len(dataset),
        "adapter_dir": str(output_dir),
        **validation,
    }


def collate_sft_batch(features: list[dict[str, Any]], tokenizer: Any | None = None) -> dict[str, Any]:
    if not features:
        return {"input_ids": [], "attention_mask": [], "labels": []}
    max_len = max(len(item["input_ids"]) for item in features)
    pad_id = _pad_id(tokenizer)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in features:
        pad = max_len - len(item["input_ids"])
        batch["input_ids"].append(item["input_ids"] + [pad_id] * pad)
        batch["attention_mask"].append(item["attention_mask"] + [0] * pad)
        batch["labels"].append(item["labels"] + [-100] * pad)
    try:
        import torch  # type: ignore

        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
    except Exception:
        return batch


def _pad_id(tokenizer: Any | None) -> int:
    if tokenizer is None:
        return 0
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is not None:
        return int(pad)
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        return int(eos)
    raise ValueError("tokenizer_missing_pad_and_eos_token_id")


def save_lora_adapter(model, output_dir: str | Path) -> None:
    safe_mkdir_for_output(output_dir, field_name="output_adapter_dir")
    model.save_pretrained(str(output_dir), safe_serialization=True)


def validate_saved_adapter(output_dir: str | Path) -> dict[str, Any]:
    target = Path(output_dir)
    adapter_config_exists = (target / "adapter_config.json").exists()
    adapter_model_exists = (target / "adapter_model.safetensors").exists()
    failures = []
    if not adapter_config_exists:
        failures.append("adapter_config_missing")
    if not adapter_model_exists:
        failures.append("adapter_model_safetensors_missing")
    return {
        "status": "PASS" if not failures else "FAIL",
        "adapter_config_exists": adapter_config_exists,
        "adapter_model_exists": adapter_model_exists,
        "failures": failures,
    }


def _validate_lora_shape(config: dict[str, Any], targets: list[str]) -> None:
    if int(config.get("rank", 0)) > 32:
        raise ValueError("rank_gt_32")
    if "lm_head" in targets:
        raise ValueError("lm_head_forbidden")
    if set(targets) != {"q_proj", "v_proj", "o_proj"}:
        raise ValueError("target_modules_must_be_q_proj_v_proj_o_proj")
