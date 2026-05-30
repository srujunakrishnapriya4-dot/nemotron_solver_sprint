from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from kaggle_prepare_anti086_tokens import load_simple_yaml, stable_hash, validate_token_mask_rows
from kaggle_runtime_patches import apply_runtime_patches, require_real_optional_path


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "in_proj", "out_proj", "lm_head"]
MICRO_LABEL = "INFRASTRUCTURE STACK TEST ONLY - NOT A 0.95 CANDIDATE - NOT MAIN TRAINING - NOT SUBMISSION READY"


def validate_train_config(config: dict) -> None:
    if config.get("stage") is not None and str(config.get("stage")) not in {"micro", "v1", "v1b"}:
        raise SystemExit("SPRINT-10 training path allows micro, v1, or v1b only")
    if bool(config.get("full_prompt_loss", False)):
        raise SystemExit("full_prompt_loss forbidden")
    if int(config["rank"]) > 32:
        raise SystemExit("rank > 32 forbidden")
    if str(config.get("stage")) in {"v1", "v1b"} and int(config["rank"]) > 16:
        raise SystemExit("first v1 rank must be <=16")
    if "target_modules" not in config:
        raise SystemExit("target_modules required")
    target_modules = [item.strip() for item in str(config.get("target_modules", "")).split(",") if item.strip()]
    if str(config.get("stage")) in {"v1", "v1b"} and "lm_head" in target_modules:
        raise SystemExit("lm_head forbidden for first v1/v1b")
    if str(config.get("parent_adapter_path", "")).strip() == "auto_or_none":
        raise SystemExit("parent_adapter_path must never be auto_or_none")
    require_real_optional_path(config.get("parent_adapter_path"), field_name="parent_adapter_path")


def load_token_rows(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_token_mask_rows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="anti086_config_micro.yaml")
    args = parser.parse_args()
    config = load_simple_yaml(args.config)
    runtime_patch = apply_runtime_patches()
    validate_train_config(config)
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_token_rows(config["token_output"])
    tokenizer = AutoTokenizer.from_pretrained(config["base_model_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["base_model_path"], torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    model.gradient_checkpointing_enable()
    parent_adapter = require_real_optional_path(config.get("parent_adapter_path"), field_name="parent_adapter_path")
    if parent_adapter:
        model = PeftModel.from_pretrained(model, parent_adapter, is_trainable=True)
    else:
        target_modules = str(config.get("target_modules", ",".join(TARGET_MODULES))).split(",")
        model = get_peft_model(model, LoraConfig(r=int(config["rank"]), lora_alpha=int(config.get("lora_alpha", config["rank"])), target_modules=target_modules, task_type="CAUSAL_LM", bias="none"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable_params={trainable}")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=float(config["learning_rate"]))
    for step in range(int(config["num_steps"])):
        row = rows[step % len(rows)]
        input_ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=model.device)
        labels = torch.tensor([row["target_ids"]], dtype=torch.long, device=model.device)
        weights = torch.tensor([row["loss_weights"]], dtype=torch.float32, device=model.device)
        out = model(input_ids=input_ids)
        logits = out.logits[:, :-1, :]
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), reduction="none", ignore_index=-100).view_as(weights[:, 1:])
        denom = weights[:, 1:].sum()
        if denom <= 0:
            raise SystemExit("zero supervised tokens in batch")
        loss = (ce * weights[:, 1:]).sum() / denom
        if not torch.isfinite(loss):
            raise SystemExit("loss not finite")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 10 == 0:
            print(f"step={step} loss_per_supervised_token={float(loss.detach().cpu())} grad_norm={float(grad_norm)}")
    output = Path(config["output_adapter_dir"])
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True, save_embedding_layers=False)
    size_mb = (output / "adapter_model.safetensors").stat().st_size / (1024 * 1024)
    if size_mb > int(config["adapter_size_limit_mb"]):
        raise SystemExit("adapter size exceeds limit")
    if size_mb > 4096:
        raise SystemExit("4GB unknown adapter forbidden")
    contract = validate_token_mask_rows(rows)
    manifest = {
        **contract,
        "micro_stack_test_label": MICRO_LABEL,
        "stage": config.get("stage", "micro"),
        "not_submission_ready": True,
        "not_095_candidate": True,
        "runtime_patch": runtime_patch,
        "max_seq_len": int(config["max_seq_len"]),
        "corpus_hash": stable_hash([row.get("metadata", {}).get("id") for row in rows]),
        "adapter_size_mb": size_mb,
        "rank": int(config["rank"]),
        "trainable_params": trainable,
    }
    manifest_path = Path(str(config.get("train_manifest_path", output / "train_manifest.json")))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
