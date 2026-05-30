from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ComponentScore:
    name: str
    score: int
    status: str
    reasons: tuple[str, ...]


def score_component(name: str, *, evidence: dict[str, Any]) -> ComponentScore:
    score = 0
    reasons: list[str] = []
    for key, weight in evidence.get("checks", {}).items():
        if evidence.get(key):
            score += int(weight)
        else:
            reasons.append(f"missing_{key}")
    if evidence.get("fake_metric_detected"):
        score = min(score, 20)
        reasons.append("fake_metric_detected")
    if evidence.get("full_prompt_loss_path"):
        score = min(score, 10)
        reasons.append("full_prompt_loss_path")
    status = "ready" if score >= 80 else "partial" if score >= 45 else "not_ready"
    return ComponentScore(name=name, score=score, status=status, reasons=tuple(sorted(set(reasons))))


def file_contains(path: str | Path, needle: str) -> bool:
    p = Path(path)
    return p.exists() and needle in p.read_text(encoding="utf-8", errors="ignore")


def file_exists(path: str | Path) -> bool:
    return Path(path).exists()
