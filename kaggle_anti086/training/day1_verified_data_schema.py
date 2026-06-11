from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class VerifiedSFTRecord:
    record_id: str
    family: str
    rule_id: str
    difficulty: str
    prompt_style: str
    prompt: str
    answer: str
    trace: str
    target_text: str
    source: str
    verification_status: str
    ambiguity_count: int
    trainable: bool
    split: str
    seed: int
    teacher_version: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class HardNegativePair:
    pair_id: str
    family: str
    rule_id: str
    difficulty: str
    prompt_style: str
    prompt: str
    chosen: str
    rejected: str
    correct_answer: str
    rejected_answer: str | None
    reason_rejected: str
    verifier_status: str
    split: str
    seed: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RejectedRow:
    rejected_id: str
    family: str | None
    prompt: str | None
    reason: str
    source: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class FactoryManifest:
    schema_version: int
    created_by: str
    seed: int
    generated_at: str
    input_reports: dict[str, str]
    output_files: dict[str, str]
    counts: dict[str, int]
    family_counts: dict[str, dict[str, int]]
    prompt_style_counts: dict[str, int]
    difficulty_counts: dict[str, int]
    rejection_counts: dict[str, int]
    gates: dict[str, bool]
    safe_to_train_lora: bool
    safe_to_package: bool
    safe_to_submit: bool
    leaderboard_claim: bool
    no_0_93_evidence: bool
    no_0_95_evidence: bool


def to_json_dict(record: Any) -> dict[str, Any]:
    data = asdict(record) if is_dataclass(record) else dict(record)
    if isinstance(record, VerifiedSFTRecord) or {"prompt", "target_text"}.issubset(data):
        data.setdefault(
            "messages",
            [
                {"role": "user", "content": data["prompt"]},
                {"role": "assistant", "content": data["target_text"]},
            ],
        )
    return data


def stable_record_id(family: str, prompt: str, answer: str, rule_id: str, seed: int) -> str:
    return _stable_id("sft", family, prompt, answer, rule_id, str(seed))


def stable_pair_id(family: str, prompt: str, chosen: str, rejected: str, seed: int) -> str:
    return _stable_id("pair", family, prompt, chosen, rejected, str(seed))


def stable_rejected_id(family: str | None, prompt: str | None, reason: str, seed: int) -> str:
    return _stable_id("reject", family or "", prompt or "", reason, str(seed))


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(to_json_dict(row), sort_keys=True, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _stable_id(*parts: str) -> str:
    joined = "\x1f".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:24]
