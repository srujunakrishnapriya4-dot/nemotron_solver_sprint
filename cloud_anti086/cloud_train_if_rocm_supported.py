from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    probe = Path("cloud_backend_probe.json")
    if not probe.exists():
        raise SystemExit("run cloud_backend_probe.py first")
    payload = json.loads(probe.read_text(encoding="utf-8"))
    required = ["torch_available", "transformers_available", "peft_available", "vllm_available", "cut_cross_entropy_available"]
    missing = [key for key in required if not payload.get(key)]
    if missing or not payload.get("final_training_allowed"):
        raise SystemExit(f"refusing AMD final training; missing or unproven: {missing}")
    raise SystemExit("training still requires explicit user approval and a proven ROCm model load")


if __name__ == "__main__":
    main()
