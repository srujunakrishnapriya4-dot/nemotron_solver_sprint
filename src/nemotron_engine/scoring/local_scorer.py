"""Strict local exact-match scoring for boxed completions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .answer_extractor import AnswerExtractionError, extract_boxed_answer
from .format_policy import AnswerFormatError, CanonicalAnswer, FormatPolicy, normalize_answer


@dataclass(frozen=True)
class ScoreResult:
    predicted: CanonicalAnswer | None
    expected: CanonicalAnswer | None
    correct: bool
    score: float
    error: str | None = None

    @property
    def predicted_answer(self) -> str | None:
        return self.predicted.normalized if self.predicted is not None else None

    @property
    def gold_answer(self) -> str | None:
        return self.expected.normalized if self.expected is not None else None


def score_completion(
    completion: str,
    expected_answer: Any,
    *,
    policy: FormatPolicy | None = None,
    strict: bool = True,
) -> ScoreResult:
    """Score only the boxed answer present in ``completion`` against ``expected_answer``."""

    resolved = policy or FormatPolicy()
    try:
        extracted = extract_boxed_answer(completion)
        predicted = normalize_answer(extracted.raw, policy=resolved)
        expected = normalize_answer(expected_answer, policy=resolved)
    except (AnswerExtractionError, AnswerFormatError) as exc:
        if strict:
            return ScoreResult(
                predicted=None,
                expected=None,
                correct=False,
                score=0.0,
                error=str(exc),
            )
        return ScoreResult(predicted=None, expected=None, correct=False, score=0.0, error=str(exc))

    correct = predicted.normalized == expected.normalized
    return ScoreResult(
        predicted=predicted,
        expected=expected,
        correct=correct,
        score=1.0 if correct else 0.0,
        error=None if correct else "Normalized prediction does not match expected answer.",
    )


def score_answer(prediction: Any, gold: Any, *, policy: FormatPolicy | None = None) -> ScoreResult:
    """Backward-compatible wrapper that treats ``prediction`` as completion text."""

    if not isinstance(prediction, str):
        return ScoreResult(
            predicted=None,
            expected=None,
            correct=False,
            score=0.0,
            error="Prediction must be completion text, not a hidden/internal answer field.",
        )
    return score_completion(prediction, gold, policy=policy)


def score_batch(records: Iterable[Mapping[str, Any]], *, policy: FormatPolicy | None = None) -> list[ScoreResult]:
    """Score mappings with ``completion``/``expected_answer`` or legacy ``prediction``/``gold`` keys."""

    results: list[ScoreResult] = []
    for record in records:
        if "completion" in record and "expected_answer" in record:
            results.append(score_completion(record["completion"], record["expected_answer"], policy=policy))
        elif "prediction" in record and "gold" in record:
            results.append(score_answer(record["prediction"], record["gold"], policy=policy))
        else:
            results.append(
                ScoreResult(
                    predicted=None,
                    expected=None,
                    correct=False,
                    score=0.0,
                    error="Record must contain completion/expected_answer or prediction/gold.",
                )
            )
    return results


def accuracy(results: Iterable[ScoreResult]) -> float:
    materialized = list(results)
    if not materialized:
        return 0.0
    return sum(item.score for item in materialized) / len(materialized)


__all__ = [
    "ScoreResult",
    "accuracy",
    "score_answer",
    "score_batch",
    "score_completion",
]
