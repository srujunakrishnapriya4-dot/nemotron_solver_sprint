from __future__ import annotations

from pathlib import Path
from typing import Any


def check_training_dependencies(base_model_path: str | None = None) -> dict[str, Any]:
    report = {
        "torch_available": False,
        "transformers_available": False,
        "peft_available": False,
        "bitsandbytes_available": False,
        "cuda_available": False,
        "device_name": None,
        "bf16_supported": False,
        "base_model_path_exists": Path(base_model_path or "").exists() if base_model_path else False,
        "tokenizer_loadable": False,
        "model_loadable": False,
        "failures": [],
    }
    try:
        import torch  # type: ignore

        report["torch_available"] = True
        report["cuda_available"] = bool(torch.cuda.is_available())
        if report["cuda_available"]:
            report["device_name"] = torch.cuda.get_device_name(0)
            report["bf16_supported"] = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    except Exception as exc:
        report["failures"].append(f"torch_unavailable:{type(exc).__name__}")
    try:
        import transformers  # noqa: F401

        report["transformers_available"] = True
    except Exception as exc:
        report["failures"].append(f"transformers_unavailable:{type(exc).__name__}")
    try:
        import peft  # noqa: F401

        report["peft_available"] = True
    except Exception as exc:
        report["failures"].append(f"peft_unavailable:{type(exc).__name__}")
    try:
        import bitsandbytes  # noqa: F401

        report["bitsandbytes_available"] = True
    except Exception:
        report["bitsandbytes_available"] = False
    if base_model_path and report["transformers_available"]:
        try:
            tokenizer = load_tokenizer(base_model_path)
            report["tokenizer_loadable"] = tokenizer is not None
        except Exception as exc:
            report["failures"].append(f"tokenizer_unloadable:{type(exc).__name__}")
    return report


def load_tokenizer(base_model_path: str):
    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=Path(base_model_path).exists(), trust_remote_code=True)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(base_model_path: str, *, load_in_4bit: bool = False, bf16: bool = True):
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM  # type: ignore

    kwargs: dict[str, Any] = {"local_files_only": Path(base_model_path).exists(), "trust_remote_code": True}
    if bf16 and torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        kwargs["torch_dtype"] = torch.bfloat16
    if load_in_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["device_map"] = "auto"
    return AutoModelForCausalLM.from_pretrained(base_model_path, **kwargs)


def load_adapter_model(base_model_path: str, adapter_dir: str, *, load_in_4bit: bool = False, bf16: bool = True):
    from peft import PeftModel  # type: ignore

    model = load_base_model(base_model_path, load_in_4bit=load_in_4bit, bf16=bf16)
    return PeftModel.from_pretrained(model, adapter_dir)


def detect_device_summary() -> dict[str, Any]:
    return check_training_dependencies()
