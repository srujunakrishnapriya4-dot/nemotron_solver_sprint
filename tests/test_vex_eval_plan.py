from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.vex_progress_bridge.vex_eval_plan import vllm_eval_settings  # noqa: E402


def test_vex_eval_plan_matches_notebook_smoke_settings() -> None:
    settings = vllm_eval_settings("smoke_16")

    assert settings["temperature"] == 0.0
    assert settings["top_p"] == 1.0
    assert settings["max_model_len"] == 8192
    assert settings["max_num_seqs"] == 64
    assert settings["uses_lora_request"] is True
    assert settings["enable_prefix_caching"] is True
    assert settings["enable_chunked_prefill"] is True


def test_vex_eval_plan_serious_eval_uses_7680_tokens_and_weights() -> None:
    settings = vllm_eval_settings("vex_held_v2")

    assert settings["max_tokens"] == 7680
    assert settings["gpu_memory_utilization"] == 0.85
    assert abs(sum(settings["category_weights"].values()) - 1.0) < 1e-12
