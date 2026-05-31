from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


TIERS = {"smoke_32": 32, "fast_128": 128, "full_512": 512}


def select_eval_rows(rows: list[dict], tier: str) -> list[dict]:
    return rows[: TIERS[tier]]


def should_run_full_eval(fast_report: dict) -> bool:
    return fast_report.get("status") == "PASS" and not fast_report.get("failures")


def write_predictions_incremental(path: str | Path, rows: Iterable[dict], *, resume: bool = False) -> int:
    target = Path(path)
    seen = set()
    if resume and target.exists():
        with target.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    seen.add(json.loads(line).get("row_id"))
    count = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a" if resume else "w", encoding="utf-8") as handle:
        for row in rows:
            if resume and row.get("row_id") in seen:
                continue
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


class ModelLifecycleRecorder:
    def __init__(self):
        self.events: list[str] = []

    def free_base_before_adapter(self) -> None:
        self.events.append("base_freed")

    def adapter_loaded(self) -> None:
        self.events.append("adapter_loaded")
