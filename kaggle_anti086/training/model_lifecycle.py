from __future__ import annotations

import contextlib
import gc
from typing import Any, Iterator

from kaggle_anti086.training.gpu_memory_audit import capture_gpu_memory_snapshot, cuda_empty_cache


def free_model(model: Any, *, reason: str, strict: bool = False) -> dict[str, Any]:
    return free_objects(model, reason=reason, strict=strict)


def free_tokenizer(tokenizer: Any, *, reason: str, strict: bool = False) -> dict[str, Any]:
    return free_objects(tokenizer, reason=reason, strict=strict)


def free_objects(*objects: Any, reason: str, strict: bool = False) -> dict[str, Any]:
    report = {"status": "PASS", "reason": reason, "object_count": len(objects), "warnings": [], "failures": []}
    try:
        for obj in objects:
            del obj
        gc.collect()
        cleanup = cuda_empty_cache(reason)
        report["cleanup"] = cleanup
        report["memory_snapshot"] = capture_gpu_memory_snapshot(f"after_cleanup:{reason}")
    except Exception as exc:
        report["status"] = "FAIL"
        report["failures"].append(f"cleanup_failed:{type(exc).__name__}")
        if strict:
            raise
    return report


@contextlib.contextmanager
def memory_safe_stage(stage_name: str) -> Iterator[dict[str, Any]]:
    state = {"stage": stage_name, "before": capture_gpu_memory_snapshot(f"before_{stage_name}"), "after": None}
    try:
        yield state
    finally:
        cuda_empty_cache(stage_name)
        state["after"] = capture_gpu_memory_snapshot(f"after_{stage_name}")
