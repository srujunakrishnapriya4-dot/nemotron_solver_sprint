from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any

from nemotron_engine.competition_sprint.competition_prompt_adapter import parse_competition_prompt
from nemotron_engine.competition_sprint.competition_runner import _solve_problem
from nemotron_engine.core.schemas import stable_hash


class VerifiedRuleExporterError(ValueError):
    pass


STATUSES = ("verified_correct", "verified_wrong", "abstained", "parse_failed", "unknown", "unsafe")


def export_verified_rules(
    train_csv_path: str | Path = "data/nemotron_competition/train.csv",
    output_dir: str | Path = "artifacts/vex_progress",
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    with Path(train_csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            problem_id = str(row.get("id") or row.get("problem_id") or f"row-{index}")
            prompt = str(row.get("prompt") or row.get("raw_prompt") or "")
            answer = str(row.get("answer") or "").strip()
            base: dict[str, Any] = {
                "id": problem_id,
                "prompt": prompt,
                "answer": answer,
                "family": "unknown",
                "solver_name": None,
                "rule_id": None,
                "rule_metadata": {},
                "verified_status": "unknown",
                "target_prediction": None,
                "train_answer_match": False,
                "rejection_reason": None,
            }
            try:
                problem = parse_competition_prompt(problem_id, prompt, answer)
                base["family"] = problem.family
                family_counts[problem.family] += 1
                prediction, reason, diagnostics = _solve_problem(problem)
                base["solver_name"] = f"competition_{problem.family}_solver"
                base["rule_id"] = reason
                base["rule_metadata"] = diagnostics
                base["target_prediction"] = prediction
                if prediction is None:
                    base["verified_status"] = "abstained"
                    base["rejection_reason"] = reason
                elif str(prediction).strip() == answer:
                    base["verified_status"] = "verified_correct"
                    base["train_answer_match"] = True
                else:
                    base["verified_status"] = "verified_wrong"
                    base["rejection_reason"] = "prediction_mismatch"
            except Exception as exc:
                base["verified_status"] = "parse_failed"
                base["rejection_reason"] = str(exc)
            base["stable_hash"] = stable_hash(base)
            counts[base["verified_status"]] += 1
            rows.append(base)
    _write_jsonl(out / "verified_rules.jsonl", rows)
    manifest = {
        "source_train_csv": str(train_csv_path),
        "total_rows": len(rows),
        "status_counts": {status: counts.get(status, 0) for status in STATUSES},
        "family_counts": dict(sorted(family_counts.items())),
        "verified_correct_only_for_trace": True,
        "manifest_hash": stable_hash({"rows": [row["stable_hash"] for row in rows], "counts": dict(counts)}),
    }
    (out / "verified_rule_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    return manifest


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
