from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggle_prepare_vex_corpus import load_simple_yaml


DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "in_proj", "out_proj", "lm_head"]


def load_token_rows(path: str | Path, max_rows: int) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) >= max_rows:
                    break
    if not rows:
        raise SystemExit("no token rows")
    if all(sum(row["loss_weights"]) == 0 for row in rows):
        raise SystemExit("supervised tokens all zero")
    return rows


def collate(rows: list[dict], pad_id: int):
    import torch

    max_len = max(len(row["input_ids"]) for row in rows)
    input_ids, labels, weights, masks = [], [], [], []
    for row in rows:
        n = len(row["input_ids"])
        pad = max_len - n
        input_ids.append(row["input_ids"] + [pad_id] * pad)
        labels.append(row["target_ids"] + [-100] * pad)
        weights.append(row["loss_weights"] + [0.0] * pad)
        masks.append([1] * n + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "loss_weights": torch.tensor(weights, dtype=torch.float32),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }


def train(config: dict) -> None:
    mode = str(config.get("mode", "public_safe_micro"))
    if mode == "vex_replica_if_inputs_available":
        validate_vex_private_inputs(config)
    elif mode not in {"public_safe_micro", "public_safe_synthetic"}:
        raise SystemExit(f"unsupported VEX training mode: {mode}")
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_token_rows(config["token_corpus_path"], int(config["max_rows"]) if int(config.get("max_rows", 0)) > 0 else 10**12)
    tokenizer = AutoTokenizer.from_pretrained(config["base_model_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["base_model_path"], torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    model.gradient_checkpointing_enable()
    if config.get("parent_adapter_path"):
        model = PeftModel.from_pretrained(model, config["parent_adapter_path"], is_trainable=True)
    else:
        lora = LoraConfig(r=int(config["rank"]), lora_alpha=int(config["lora_alpha"]), lora_dropout=float(config["lora_dropout"]), bias="none", task_type="CAUSAL_LM", target_modules=target_modules_from_config(config))
        model = get_peft_model(model, lora)
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    if trainable <= 0:
        raise SystemExit("zero trainable LoRA params")
    print(f"trainable_params={trainable}")
    print(f"target_modules={target_modules_from_config(config)}")
    try:
        from cut_cross_entropy import linear_cross_entropy  # noqa: F401
        print("cut_cross_entropy_available=True")
    except Exception:
        print("cut_cross_entropy_available=False using torch.nn.functional.cross_entropy")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=float(config["learning_rate"]))
    model.train()
    step = 0
    while step < int(config["num_steps"]):
        row = rows[step % len(rows)]
        batch = collate([row], tokenizer.pad_token_id)
        batch = {key: value.to(model.device) for key, value in batch.items()}
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = out.logits[:, :-1, :]
        labels = batch["labels"][:, 1:]
        weights = batch["loss_weights"][:, 1:].to(logits.device)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none", ignore_index=-100).view_as(weights)
        denom = weights.sum()
        if denom <= 0:
            raise SystemExit("zero supervised tokens in batch")
        loss = (ce * weights).sum() / denom
        if not torch.isfinite(loss):
            raise SystemExit("loss is NaN/inf")
        loss.backward()
        if (step + 1) % int(config["grad_accum"]) == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            print(f"step={step+1} loss={float(loss.detach().cpu())} grad_norm={float(grad_norm)}")
        step += 1
    output = Path(config["output_adapter_dir"])
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    size_mb = (output / "adapter_model.safetensors").stat().st_size / (1024 * 1024)
    print(f"adapter_size_mb={size_mb}")
    if size_mb > int(config["adapter_size_limit_mb"]):
        raise SystemExit("adapter size exceeds limit")


def target_modules_from_config(config: dict) -> list[str]:
    raw = config.get("target_modules")
    if isinstance(raw, str) and raw.strip():
        return [item.strip() for item in raw.split(",") if item.strip()]
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return list(DEFAULT_TARGET_MODULES)


def validate_vex_private_inputs(config: dict) -> None:
    required = [
        "corpus_path",
        "train_order_path",
        "swap_cryptarithm_thk_path",
        "swap_v6_path",
        "swap_binary_fmt_path",
        "extra_hyprevise_synth_path",
        "heldout_file",
    ]
    missing = [str(config.get(key)) for key in required if not config.get(key) or not Path(str(config.get(key))).exists()]
    if missing:
        raise SystemExit(f"missing VEX private/token-swap inputs; full replica training blocked: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="vex_config_micro.yaml")
    args = parser.parse_args()
    train(load_simple_yaml(args.config))


if __name__ == "__main__":
    main()
