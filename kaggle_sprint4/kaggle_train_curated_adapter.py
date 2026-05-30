from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import random
import shutil
import stat
import sys
import traceback


BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
CUTLASS_PATH = Path("/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/nvidia_cutlass_dsl/python_packages")
PTXAS_BLACKWELL_SOURCE = Path("/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/triton/backends/nvidia/bin/ptxas-blackwell")
PTXAS_BLACKWELL_TARGET = Path("/tmp/ptxas-blackwell")
KNOWN_VARIANTS = {
    "variant_a_family_tagged",
    "variant_b_distilled",
    "variant_c_hard_oversampled",
    "variant_d_direct_hard",
    "retention_raw_family_tagged",
    "distilled_low_lr",
    "hard_low_lr",
}
IGNORE_INDEX = -100


def load_config(path: str | Path) -> dict:
    try:
        import yaml  # type: ignore

        with Path(path).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except Exception:
        return load_simple_yaml(Path(path))


def load_simple_yaml(path: Path) -> dict:
    payload: dict = {}
    current_list: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("  - ") and current_list:
            if not isinstance(payload[current_list], list):
                payload[current_list] = []
            payload[current_list].append(parse_scalar(line[4:].strip()))
            continue
        if line.startswith("  ") and current_list and ":" in line:
            if not isinstance(payload[current_list], dict):
                payload[current_list] = {}
            key, value = line.strip().split(":", 1)
            payload[current_list][key.strip()] = parse_scalar(value.strip())
            continue
        if ":" in line and not line.startswith(" "):
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                payload[key] = []
                current_list = key
            else:
                payload[key] = parse_scalar(value)
                current_list = None
    return payload


def parse_scalar(value: str):
    value = value.strip().strip('"').strip("'")
    if value.lower() in {"null", "none", ""}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if any(ch in value.lower() for ch in (".", "e")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def validate_training_config(config: dict) -> dict:
    variant = str(config.get("variant_name", ""))
    if variant not in KNOWN_VARIANTS:
        raise ValueError(f"unknown variant_name: {variant}")
    rank = int(config.get("lora_rank", 0))
    if rank <= 0 or rank > 32:
        raise ValueError("lora_rank must be between 1 and 32")
    lr = float(config.get("learning_rate", 0.0))
    if lr > 0.000002 and not bool(config.get("allow_high_lr", False)):
        raise ValueError("learning_rate > 2e-6 requires allow_high_lr=true")
    answer_format = str(config.get("answer_format", "raw"))
    if answer_format not in {"raw", "boxed"}:
        raise ValueError("answer_format must be raw or boxed")
    assistant_only = bool(config.get("assistant_only_loss", True))
    if not assistant_only and not bool(config.get("allow_full_prompt_loss", False)):
        raise ValueError("full-prompt loss requires allow_full_prompt_loss=true")
    if not assistant_only:
        print("WARNING: FULL-PROMPT LOSS ENABLED. This can corrupt parent adapter behavior.")
    config["assistant_only_loss"] = assistant_only
    config["answer_format"] = answer_format
    config["dataset_mixture"] = parse_dataset_mixture(config.get("dataset_mixture"))
    return config


def parse_dataset_mixture(raw) -> list[tuple[str, float]]:
    if isinstance(raw, dict):
        items = [(str(key), float(value)) for key, value in raw.items()]
    elif isinstance(raw, (list, tuple)):
        items = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                items.append((str(item[0]), float(item[1])))
            else:
                items.append((str(item), 1.0))
    else:
        raise ValueError("dataset_mixture must be a list or mapping")
    if not items:
        raise ValueError("dataset_mixture must be non-empty")
    for filename, weight in items:
        if not filename.endswith(".jsonl"):
            raise ValueError(f"dataset_mixture entry must be a JSONL file: {filename}")
        if weight <= 0:
            raise ValueError(f"dataset_mixture weight must be positive: {filename}")
    return items


def fix_cutlass_path() -> None:
    if CUTLASS_PATH.exists():
        sys.path.insert(0, str(CUTLASS_PATH))
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "CUTLASS/CUTE import failed. Add the NVIDIA utility script input or verify "
            f"{CUTLASS_PATH} exists."
        ) from exc


def fix_triton_ptxas_blackwell() -> None:
    """Copy Kaggle's read-only ptxas-blackwell binary to executable /tmp."""

    print(f"ptxas_blackwell_source={PTXAS_BLACKWELL_SOURCE}")
    print(f"ptxas_blackwell_target={PTXAS_BLACKWELL_TARGET}")
    if not PTXAS_BLACKWELL_SOURCE.exists():
        print("ptxas_blackwell_fix=skipped source_missing")
        return
    try:
        shutil.copyfile(PTXAS_BLACKWELL_SOURCE, PTXAS_BLACKWELL_TARGET)
        current_mode = PTXAS_BLACKWELL_TARGET.stat().st_mode
        executable_bits = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        PTXAS_BLACKWELL_TARGET.chmod(current_mode | executable_bits)
        os.environ["TRITON_PTXAS_PATH"] = str(PTXAS_BLACKWELL_TARGET)
        os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = str(PTXAS_BLACKWELL_TARGET)
        print(f"ptxas_blackwell_fix=enabled target_exists={PTXAS_BLACKWELL_TARGET.exists()} mode={oct(PTXAS_BLACKWELL_TARGET.stat().st_mode)}")
        print(f"TRITON_PTXAS_PATH={os.environ['TRITON_PTXAS_PATH']}")
        print(f"TRITON_PTXAS_BLACKWELL_PATH={os.environ['TRITON_PTXAS_BLACKWELL_PATH']}")
    except Exception as exc:
        raise RuntimeError(f"failed to prepare executable ptxas-blackwell at {PTXAS_BLACKWELL_TARGET}: {exc}") from exc


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"non-object row in {path}")
                rows.append(payload)
    return rows


def build_training_rows(config: dict) -> list[dict]:
    root = Path(config["curated_dataset_path"])
    mixture = parse_dataset_mixture(config["dataset_mixture"])
    rows: list[dict] = []
    rng = random.Random(int(config["seed"]))
    for filename, weight in mixture:
        source_rows = load_jsonl(root / filename)
        if weight < 1.0:
            take = max(1, int(round(len(source_rows) * weight)))
            source_rows = sorted(source_rows, key=lambda row: str(row.get("example_hash", row.get("problem_id", ""))))
            rng.shuffle(source_rows)
            rows.extend(source_rows[:take])
        else:
            repeats = int(weight)
            fraction = weight - repeats
            for _ in range(repeats):
                rows.extend(source_rows)
            if fraction > 0:
                take = max(1, int(round(len(source_rows) * fraction)))
                source_rows = sorted(source_rows, key=lambda row: str(row.get("example_hash", row.get("problem_id", ""))))
                rng.shuffle(source_rows)
                rows.extend(source_rows[:take])
    rows = family_balanced_rows(rows, seed=int(config["seed"]))
    max_rows = int(config.get("max_train_rows") or 0)
    return rows[:max_rows] if max_rows > 0 else rows


def family_balanced_rows(rows: list[dict], *, seed: int) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(str(row.get("family", "unknown")), []).append(row)
    rng = random.Random(seed)
    balanced: list[dict] = []
    for family in sorted(buckets):
        bucket = list(buckets[family])
        rng.shuffle(bucket)
        balanced.extend(bucket)
    rng.shuffle(balanced)
    return balanced


def render_row(row: dict, tokenizer) -> str:
    return render_full_conversation(row, tokenizer, answer_format="boxed")


def assistant_answer_from_row(row: dict, *, answer_format: str) -> str:
    assistant = row["messages"][1]["content"].strip()
    if assistant.startswith("\\boxed{") and assistant.endswith("}"):
        answer = assistant[len("\\boxed{") : -1]
    else:
        answer = assistant
    if answer_format == "raw":
        return answer
    if answer_format == "boxed":
        return f"\\boxed{{{answer}}}"
    raise ValueError("answer_format must be raw or boxed")


def messages_for_answer_format(row: dict, *, answer_format: str) -> list[dict]:
    return [
        {"role": "user", "content": row["messages"][0]["content"]},
        {"role": "assistant", "content": assistant_answer_from_row(row, answer_format=answer_format)},
    ]


def render_prompt_prefix(row: dict, tokenizer) -> str:
    user_message = [{"role": "user", "content": row["messages"][0]["content"]}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(user_message, tokenize=False, add_generation_prompt=True)
    return f"USER: {user_message[0]['content']}\nASSISTANT: "


def render_full_conversation(row: dict, tokenizer, *, answer_format: str) -> str:
    messages = messages_for_answer_format(row, answer_format=answer_format)
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "\n".join(f"{message['role'].upper()}: {message['content']}" for message in messages)


def render_row_legacy(row: dict, tokenizer) -> str:
    messages = row["messages"]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "\n".join(f"{message['role'].upper()}: {message['content']}" for message in messages)


def tokenize_assistant_only(row: dict, tokenizer, *, max_seq_len: int, answer_format: str) -> dict:
    prefix_text = render_prompt_prefix(row, tokenizer)
    full_text = render_full_conversation(row, tokenizer, answer_format=answer_format)
    prefix_ids = tokenizer(prefix_text, truncation=True, max_length=max_seq_len, padding=False)["input_ids"]
    full_unpadded = tokenizer(full_text, truncation=True, max_length=max_seq_len, padding=False)["input_ids"]
    encoded = tokenizer(full_text, truncation=True, max_length=max_seq_len, padding="max_length")
    input_ids = list(encoded["input_ids"])
    labels = [IGNORE_INDEX for _ in input_ids]
    prompt_len = min(len(prefix_ids), len(full_unpadded), len(input_ids))
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    for index in range(prompt_len, min(len(full_unpadded), len(input_ids))):
        if pad_id is not None and input_ids[index] == pad_id:
            labels[index] = IGNORE_INDEX
        else:
            labels[index] = input_ids[index]
    encoded["labels"] = labels
    return encoded


def tokenize_full_prompt(row: dict, tokenizer, *, max_seq_len: int, answer_format: str) -> dict:
    full_text = render_full_conversation(row, tokenizer, answer_format=answer_format)
    encoded = tokenizer(full_text, truncation=True, max_length=max_seq_len, padding="max_length")
    pad_id = getattr(tokenizer, "pad_token_id", None)
    labels = [token if pad_id is None or token != pad_id else IGNORE_INDEX for token in encoded["input_ids"]]
    encoded["labels"] = labels
    return encoded


def supervision_diagnostics(tokenized_rows: list[dict], tokenizer) -> dict:
    supervised_counts = [sum(1 for token in row["labels"] if token != IGNORE_INDEX) for row in tokenized_rows]
    total_counts = [len(row["input_ids"]) for row in tokenized_rows]
    prompt_counts = [total - supervised for total, supervised in zip(total_counts, supervised_counts)]
    zero_count = sum(1 for count in supervised_counts if count == 0)
    ratios = [supervised / total for supervised, total in zip(supervised_counts, total_counts) if total]
    diagnostics = {
        "supervised_tokens_min": min(supervised_counts) if supervised_counts else 0,
        "supervised_tokens_max": max(supervised_counts) if supervised_counts else 0,
        "supervised_tokens_mean": sum(supervised_counts) / len(supervised_counts) if supervised_counts else 0.0,
        "prompt_tokens_mean": sum(prompt_counts) / len(prompt_counts) if prompt_counts else 0.0,
        "total_tokens_mean": sum(total_counts) / len(total_counts) if total_counts else 0.0,
        "zero_supervised_rows": zero_count,
        "supervised_ratio_mean": sum(ratios) / len(ratios) if ratios else 0.0,
    }
    print(json.dumps(diagnostics, sort_keys=True))
    for row in tokenized_rows[:2]:
        ids = [token for token, label in zip(row["input_ids"], row["labels"]) if label != IGNORE_INDEX]
        try:
            print(f"decoded_supervised_span={tokenizer.decode(ids)}")
        except Exception:
            print(f"supervised_token_ids={ids}")
    return diagnostics


def validate_supervision_diagnostics(diagnostics: dict, *, allow_low_supervision_ratio: bool = False) -> None:
    if diagnostics["zero_supervised_rows"]:
        raise ValueError("assistant-only masking produced zero supervised tokens")
    ratio = float(diagnostics["supervised_ratio_mean"])
    if ratio > 0.40:
        raise ValueError(f"supervised token ratio too high; prompt tokens may be leaking into labels: {ratio}")
    if ratio < 0.01 and not allow_low_supervision_ratio:
        raise ValueError(f"supervised token ratio too low: {ratio}")


def build_tokenized_dataset(rows: list[dict], tokenizer, config: dict) -> list[dict]:
    max_seq_len = int(config["max_seq_len"])
    answer_format = str(config.get("answer_format", "raw"))
    if config.get("assistant_only_loss", True):
        tokenized = [tokenize_assistant_only(row, tokenizer, max_seq_len=max_seq_len, answer_format=answer_format) for row in rows]
        diagnostics = supervision_diagnostics(tokenized, tokenizer)
        validate_supervision_diagnostics(diagnostics, allow_low_supervision_ratio=bool(config.get("allow_low_supervision_ratio", False)))
        return tokenized
    tokenized = [tokenize_full_prompt(row, tokenizer, max_seq_len=max_seq_len, answer_format=answer_format) for row in rows]
    diagnostics = supervision_diagnostics(tokenized, tokenizer)
    print("WARNING: full prompt tokens are supervised.")
    return tokenized


def train(config: dict) -> None:
    config = validate_training_config(config)
    fix_cutlass_path()
    fix_triton_ptxas_blackwell()
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    if int(config["lora_rank"]) > 32:
        raise ValueError("lora_rank must be <= 32")
    torch.cuda.empty_cache()
    gc.collect()
    rows = build_training_rows(config)
    model_path = os.environ.get("BASE_MODEL_PATH") or config.get("base_model_path") or BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = build_tokenized_dataset(rows, tokenizer, config)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    model.gradient_checkpointing_enable()
    parent = config.get("parent_adapter_path")
    if parent:
        model = PeftModel.from_pretrained(model, parent, is_trainable=True)
    else:
        lora = LoraConfig(
            r=int(config["lora_rank"]),
            lora_alpha=int(config["lora_alpha"]),
            lora_dropout=float(config["lora_dropout"]),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora)
    args = TrainingArguments(
        output_dir="/kaggle/working/sprint4_trainer_tmp",
        per_device_train_batch_size=int(config["batch_size"]),
        gradient_accumulation_steps=int(config["grad_accum"]),
        learning_rate=float(config["learning_rate"]),
        num_train_epochs=float(config["num_epochs"]),
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset)
    fix_triton_ptxas_blackwell()
    trainer.train()
    output = Path(config["output_adapter_dir"])
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    for required in ("adapter_config.json", "adapter_model.safetensors"):
        path = output / required
        if not path.is_file():
            raise RuntimeError(f"adapter output missing: {path}")
    print(f"adapter_saved={output}")


def print_failure_context() -> None:
    try:
        import torch

        print(f"torch_version={torch.__version__}")
        print(f"cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"gpu={torch.cuda.get_device_name(0)}")
            print(f"memory_reserved={torch.cuda.memory_reserved(0)}")
            print(f"memory_allocated={torch.cuda.memory_allocated(0)}")
    except Exception as exc:
        print(f"torch_context_unavailable={exc}")
    print(f"cutlass_path_exists={CUTLASS_PATH.exists()}")
    print(f"ptxas_blackwell_source_exists={PTXAS_BLACKWELL_SOURCE.exists()}")
    print(f"ptxas_blackwell_target_exists={PTXAS_BLACKWELL_TARGET.exists()}")
    print(f"TRITON_PTXAS_PATH={os.environ.get('TRITON_PTXAS_PATH')}")
    print(f"TRITON_PTXAS_BLACKWELL_PATH={os.environ.get('TRITON_PTXAS_BLACKWELL_PATH')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="sprint4_config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        train(config)
    except Exception:
        print_failure_context()
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
