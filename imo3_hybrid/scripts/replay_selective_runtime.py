from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VENV_SITE_PACKAGES = ROOT / ".venv" / "lib" / "python3.12" / "site-packages"
if VENV_SITE_PACKAGES.exists() and str(VENV_SITE_PACKAGES) not in sys.path:
    sys.path.append(str(VENV_SITE_PACKAGES))

from src.common.constants import ANSWER_MAX, ANSWER_MIN
from src.common.schemas import AttemptEntropySource, AttemptRecord, BudgetPlan, Difficulty, RouteDecision
from src.online.inference_engine import InferenceEngine
from src.online.kaggle_runner import build_inference_engine_config, load_kaggle_runner_config


class ReplayProblemRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problem_id: str
    problem: str = ""
    expected_answer: str | None = None
    route: dict[str, Any] = Field(default_factory=dict)
    attempts: list[AttemptRecord] = Field(default_factory=list)
    attempt_batch_stopped_reason: str = "attempt_budget_exhausted"
    escalated_prediction: dict[str, Any] | None = None
    full_prediction: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class ModeOutcome:
    mode: str
    predicted_answer: str | None
    correct: bool | None
    early_stop_triggered: bool
    should_escalate: bool
    escalation_used: bool
    attempt_weighting_mode: str
    confidence: float
    warning_count: int
    escalation_reasons: tuple[str, ...]
    metadata: dict[str, Any]


def _canonical_answer(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text.isdigit():
        return None
    numeric = int(text)
    if numeric < ANSWER_MIN or numeric > ANSWER_MAX:
        return None
    return str(numeric)


def _prediction_answer(payload: Mapping[str, Any] | None) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    for key in ("final_answer", "submission_answer", "predicted_answer", "answer"):
        answer = _canonical_answer(payload.get(key))
        if answer is not None:
            return answer
    return None


def _prediction_confidence(payload: Mapping[str, Any] | None) -> float:
    if not isinstance(payload, Mapping):
        return 0.0
    for key in ("confidence", "final_confidence", "score"):
        if key in payload and payload[key] is not None:
            try:
                return max(0.0, min(1.0, float(payload[key])))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _problem_text_from_payload(payload: Mapping[str, Any]) -> str:
    for key in ("problem", "question", "problem_text"):
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _gold_answer_from_payload(payload: Mapping[str, Any]) -> str | None:
    for key in ("expected_answer", "gold_answer", "target", "gold"):
        value = payload.get(key)
        answer = _canonical_answer(value)
        if answer is not None:
            return answer
    return None


def _predicted_answer_from_payload(payload: Mapping[str, Any]) -> str | None:
    answer = _prediction_answer(payload.get("full_prediction")) or _prediction_answer(payload.get("escalated_prediction"))
    if answer is not None:
        return answer
    for key in ("predicted_answer", "final_answer", "answer", "prediction"):
        value = payload.get(key)
        resolved = _canonical_answer(value)
        if resolved is not None:
            return resolved
    return None


def _synthesized_attempt(problem_id: str, predicted_answer: str | None) -> AttemptRecord:
    answer = _canonical_answer(predicted_answer)
    raw_text = f"\\boxed{{{answer}}}" if answer is not None else str(predicted_answer or "")
    return AttemptRecord(
        attempt_id=f"{problem_id}::attempt::000",
        seed=0,
        raw_text=raw_text,
        extracted_answer=answer,
        valid_answer=answer is not None,
        mean_token_entropy=None,
        entropy_source=AttemptEntropySource.UNAVAILABLE,
        runtime_sec=0.0,
        stopped_reason="synthesized_minimal_record",
        metadata={"synthesized_from_prediction_row": True},
    )


def _normalize_replay_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    problem_id = str(payload.get("problem_id") or payload.get("id") or "").strip()
    if not problem_id:
        raise ValueError("Replay row is missing problem_id")
    problem = _problem_text_from_payload(payload)
    expected_answer = _gold_answer_from_payload(payload)
    predicted_answer = _predicted_answer_from_payload(payload)
    attempts = payload.get("attempts")
    if not isinstance(attempts, list):
        attempts = []
    if not attempts:
        attempts = [_synthesized_attempt(problem_id, predicted_answer).model_dump(mode="json")]

    full_prediction = payload.get("full_prediction")
    if not isinstance(full_prediction, Mapping):
        full_prediction = (
            {"final_answer": predicted_answer, "confidence": 0.0, "status": "unavailable"}
            if predicted_answer is not None
            else {"status": "unavailable"}
        )
    escalated_prediction = payload.get("escalated_prediction")
    if escalated_prediction is not None and not isinstance(escalated_prediction, Mapping):
        escalated_prediction = None

    metadata = dict(payload.get("metadata") or {})
    if "input_compatibility_mode" not in metadata:
        metadata["input_compatibility_mode"] = "minimal_prediction_row" if not payload.get("attempts") else "attempt_level"

    route = payload.get("route")
    if not isinstance(route, Mapping):
        route = {}

    return {
        "problem_id": problem_id,
        "problem": problem,
        "expected_answer": expected_answer,
        "route": dict(route),
        "attempts": attempts,
        "attempt_batch_stopped_reason": str(payload.get("attempt_batch_stopped_reason", "attempt_budget_exhausted")),
        "escalated_prediction": dict(escalated_prediction) if isinstance(escalated_prediction, Mapping) else None,
        "full_prediction": dict(full_prediction),
        "metadata": metadata,
    }


def _default_route(problem_id: str, route_payload: Mapping[str, Any]) -> RouteDecision:
    budget_payload = route_payload.get("budget_plan", {}) if isinstance(route_payload, Mapping) else {}
    if isinstance(budget_payload, Mapping):
        try:
            budget = BudgetPlan.model_validate(budget_payload)
        except Exception:
            budget = BudgetPlan()
    else:
        budget = BudgetPlan()
    difficulty_value = str(route_payload.get("difficulty", Difficulty.MEDIUM.value)) if isinstance(route_payload, Mapping) else Difficulty.MEDIUM.value
    try:
        difficulty = Difficulty(difficulty_value)
    except Exception:
        difficulty = Difficulty.MEDIUM
    difficulty_score = 0.5
    route_uncertainty = 0.0
    if isinstance(route_payload, Mapping):
        try:
            difficulty_score = float(route_payload.get("difficulty_score", 0.5))
        except (TypeError, ValueError):
            difficulty_score = 0.5
        try:
            route_uncertainty = float(route_payload.get("route_uncertainty", 0.0))
        except (TypeError, ValueError):
            route_uncertainty = 0.0
    return RouteDecision(
        problem_id=problem_id,
        difficulty=difficulty,
        difficulty_score=max(0.0, min(1.0, difficulty_score)),
        route_uncertainty=max(0.0, min(1.0, route_uncertainty)),
        problem_type_probs=dict(route_payload.get("problem_type_probs", {}) or {"mixed": 1.0}) if isinstance(route_payload, Mapping) else {"mixed": 1.0},
        archetype_probs=dict(route_payload.get("archetype_probs", {}) or {}) if isinstance(route_payload, Mapping) else {},
        operator_prior=dict(route_payload.get("operator_prior", {}) or {}) if isinstance(route_payload, Mapping) else {},
        budget_plan=budget,
        retrieval_depth=int(route_payload.get("retrieval_depth", budget.retrieval_depth)) if isinstance(route_payload, Mapping) else budget.retrieval_depth,
        branch_budget=int(route_payload.get("branch_budget", budget.branch_budget)) if isinstance(route_payload, Mapping) else budget.branch_budget,
        repair_threshold=float(route_payload.get("repair_threshold", 0.5)) if isinstance(route_payload, Mapping) else 0.5,
        verifier_mode=str(route_payload.get("verifier_mode", "default")) if isinstance(route_payload, Mapping) else "default",
        route_rationale=list(route_payload.get("route_rationale", []) or []) if isinstance(route_payload, Mapping) else [],
        compute_signals=dict(route_payload.get("compute_signals", {}) or {}) if isinstance(route_payload, Mapping) else {},
    )


def _load_records(path: str | Path) -> list[ReplayProblemRecord]:
    target = Path(path)
    records: list[ReplayProblemRecord] = []
    with target.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, Mapping):
                raise ValueError(f"Replay row {line_no} must be a JSON object")
            records.append(ReplayProblemRecord.model_validate(_normalize_replay_payload(payload)))
    return records


def _evaluate_minimal_mode(
    engine: InferenceEngine,
    record: ReplayProblemRecord,
    *,
    allow_escalation: bool,
) -> ModeOutcome:
    route = _default_route(record.problem_id, record.route)
    selection = engine._select_from_attempt_records(
        problem_id=record.problem_id,
        attempts=tuple(record.attempts),
        stopped_reason=record.attempt_batch_stopped_reason,
    )
    early_stop_triggered = record.attempt_batch_stopped_reason == "early_stop_consensus"
    should_escalate = engine._should_escalate_after_attempts(
        route=route,
        selection=selection,
        attempts=tuple(record.attempts),
        early_stop_triggered=early_stop_triggered,
    )
    predicted_answer = _canonical_answer(selection.submission_answer)
    if allow_escalation and should_escalate:
        escalated_answer = _prediction_answer(record.escalated_prediction) or _prediction_answer(record.full_prediction)
        if escalated_answer is not None:
            predicted_answer = escalated_answer
    gold = _canonical_answer(record.expected_answer)
    correct = None if gold is None or predicted_answer is None else predicted_answer == gold
    warning_count = len([item for item in selection.warnings if getattr(item, "value", str(item)) != "none"])
    return ModeOutcome(
        mode="selective-escalation" if allow_escalation else "minimal-only",
        predicted_answer=predicted_answer,
        correct=correct,
        early_stop_triggered=early_stop_triggered,
        should_escalate=should_escalate,
        escalation_used=bool(allow_escalation and should_escalate and predicted_answer == (_prediction_answer(record.escalated_prediction) or _prediction_answer(record.full_prediction))),
        attempt_weighting_mode=str(selection.metadata.get("attempt_weighting_mode", "equal_weight")),
        confidence=float(selection.prediction.confidence),
        warning_count=warning_count,
        escalation_reasons=tuple(engine._attempt_escalation_reasons(
            route=route,
            selection=selection,
            attempts=tuple(record.attempts),
            early_stop_triggered=early_stop_triggered,
        )),
        metadata={
            "selection_warnings": [str(item.value if hasattr(item, "value") else item) for item in selection.warnings],
            "selection_method": selection.prediction.method_used,
            "selection_metadata": dict(selection.metadata or {}),
            "gate_thresholds": engine._selective_gate_thresholds(),
            "route_metrics": engine._route_runtime_metrics(route=route),
        },
    )


def _evaluate_full_mode(record: ReplayProblemRecord) -> ModeOutcome:
    predicted_answer = _prediction_answer(record.full_prediction) or _prediction_answer(record.escalated_prediction)
    gold = _canonical_answer(record.expected_answer)
    correct = None if gold is None or predicted_answer is None else predicted_answer == gold
    payload = record.full_prediction or record.escalated_prediction or {}
    return ModeOutcome(
        mode="full-architecture",
        predicted_answer=predicted_answer,
        correct=correct,
        early_stop_triggered=False,
        should_escalate=False,
        escalation_used=False,
        attempt_weighting_mode="n/a",
        confidence=_prediction_confidence(payload),
        warning_count=int(payload.get("warning_count", 0)) if isinstance(payload, Mapping) else 0,
        escalation_reasons=(),
        metadata={"source": "full_prediction" if record.full_prediction else "escalated_prediction"},
    )


def _summarize_mode(rows: Sequence[ModeOutcome]) -> dict[str, Any]:
    labeled = [row for row in rows if row.correct is not None]
    correct = sum(1 for row in labeled if row.correct)
    return {
        "problem_count": len(rows),
        "labeled_problem_count": len(labeled),
        "correct_count": correct,
        "accuracy": 0.0 if not labeled else round(correct / len(labeled), 6),
        "escalation_used_count": sum(1 for row in rows if row.escalation_used),
        "suggested_escalation_count": sum(1 for row in rows if row.should_escalate),
        "early_stop_count": sum(1 for row in rows if row.early_stop_triggered),
        "weighting_modes": _count_by_key(row.attempt_weighting_mode for row in rows),
        "escalation_reason_counts": _count_reasons(rows),
    }


def _count_by_key(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _count_reasons(rows: Sequence[ModeOutcome]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for reason in row.escalation_reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _pairwise_delta(lhs: Sequence[ModeOutcome], rhs: Sequence[ModeOutcome]) -> dict[str, Any]:
    helped = 0
    hurt = 0
    neutral = 0
    for a, b in zip(lhs, rhs):
        if a.correct is None or b.correct is None:
            continue
        if not a.correct and b.correct:
            helped += 1
        elif a.correct and not b.correct:
            hurt += 1
        else:
            neutral += 1
    return {"helped": helped, "hurt": hurt, "neutral": neutral}


def run_replay(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    online_config_path: str | Path = "configs/online.yaml",
) -> dict[str, Any]:
    records = _load_records(input_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    base_runner_config = load_kaggle_runner_config(online_config_path)
    base_engine_config = build_inference_engine_config(base_runner_config)

    minimal_engine = InferenceEngine(
        config=base_engine_config.model_copy(
            update={
                "force_runtime_path": "minimal",
                "minimal_mode_allow_post_attempt_escalation": False,
            }
        )
    )
    selective_engine = InferenceEngine(
        config=base_engine_config.model_copy(
            update={
                "force_runtime_path": "minimal",
                "minimal_mode_allow_post_attempt_escalation": True,
            }
        )
    )

    per_problem_rows: list[dict[str, Any]] = []
    minimal_rows: list[ModeOutcome] = []
    selective_rows: list[ModeOutcome] = []
    full_rows: list[ModeOutcome] = []

    for record in records:
        minimal = _evaluate_minimal_mode(minimal_engine, record, allow_escalation=False)
        selective = _evaluate_minimal_mode(selective_engine, record, allow_escalation=True)
        full = _evaluate_full_mode(record)

        minimal_rows.append(minimal)
        selective_rows.append(selective)
        full_rows.append(full)

        per_problem_rows.append(
            {
                "problem_id": record.problem_id,
                "expected_answer": _canonical_answer(record.expected_answer),
                "minimal_only": asdict(minimal),
                "selective_escalation": asdict(selective),
                "full_architecture": asdict(full),
            }
        )
        for key in ("minimal_only", "selective_escalation", "full_architecture"):
            per_problem_rows[-1][key]["metadata"] = {
                **per_problem_rows[-1][key]["metadata"],
                "problem_id": record.problem_id,
            }

    replay_results = {
        "input_path": str(Path(input_path)),
        "problem_count": len(records),
        "online_config_path": str(Path(online_config_path)),
        "gate_thresholds": selective_engine._selective_gate_thresholds(),
        "modes": {
            "minimal_only": _summarize_mode(minimal_rows),
            "selective_escalation": _summarize_mode(selective_rows),
            "full_architecture": _summarize_mode(full_rows),
        },
        "pairwise_deltas": {
            "selective_vs_minimal": _pairwise_delta(minimal_rows, selective_rows),
            "full_vs_selective": _pairwise_delta(selective_rows, full_rows),
            "full_vs_minimal": _pairwise_delta(minimal_rows, full_rows),
        },
    }

    escalation_helped_vs_hurt = {
        "online_config_path": str(Path(online_config_path)),
        "gate_thresholds": selective_engine._selective_gate_thresholds(),
        "escalation_triggered_count": sum(1 for row in selective_rows if row.should_escalate),
        "escalation_used_count": sum(1 for row in selective_rows if row.escalation_used),
        "reason_counts": _count_reasons(selective_rows),
        "helped": sum(
            1
            for minimal, selective in zip(minimal_rows, selective_rows)
            if minimal.correct is False and selective.correct is True
        ),
        "hurt": sum(
            1
            for minimal, selective in zip(minimal_rows, selective_rows)
            if minimal.correct is True and selective.correct is False
        ),
        "neutral": sum(
            1
            for minimal, selective in zip(minimal_rows, selective_rows)
            if minimal.correct is not None and selective.correct is not None and minimal.correct == selective.correct
        ),
    }

    (output_root / "replay_results.json").write_text(
        json.dumps(replay_results, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    with (output_root / "per_problem_modes.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_problem_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output_root / "escalation_helped_vs_hurt.json").write_text(
        json.dumps(escalation_helped_vs_hurt, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return {
        "replay_results_path": str(output_root / "replay_results.json"),
        "per_problem_modes_path": str(output_root / "per_problem_modes.jsonl"),
        "escalation_helped_vs_hurt_path": str(output_root / "escalation_helped_vs_hurt.json"),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay selective-runtime mode comparisons from archived attempt traces.")
    parser.add_argument("--input-jsonl", required=True, help="JSONL file with replay problem records.")
    parser.add_argument("--output-dir", required=True, help="Output directory for replay artifacts.")
    parser.add_argument("--online-config", default="configs/online.yaml", help="Online runtime config used to load selective thresholds.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_replay(input_path=args.input_jsonl, output_dir=args.output_dir, online_config_path=args.online_config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
