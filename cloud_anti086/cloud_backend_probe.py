from __future__ import annotations

import importlib.util
import json
import platform
from pathlib import Path


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    report = {
        "platform": platform.platform(),
        "torch_available": has_module("torch"),
        "transformers_available": has_module("transformers"),
        "peft_available": has_module("peft"),
        "vllm_available": has_module("vllm"),
        "cut_cross_entropy_available": has_module("cut_cross_entropy"),
        "final_training_allowed": False,
        "reason": "ROCm/Nemotron compatibility must be proven explicitly before training.",
    }
    Path("cloud_backend_probe.json").write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
