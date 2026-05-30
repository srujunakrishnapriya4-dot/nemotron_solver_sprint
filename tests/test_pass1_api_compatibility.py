from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_direct_pass1_api_imports() -> None:
    from nemotron_engine.runtime.serving_config import (
        ServingConfig,
        load_serving_config,
        save_serving_config,
        validate_serving_config,
    )
    from nemotron_engine.scoring.answer_extractor import (
        ExtractedAnswer,
        extract_boxed_answer,
        find_boxed_spans,
        has_truncation_risk,
    )
    from nemotron_engine.scoring.format_policy import (
        AnswerType,
        CanonicalAnswer,
        FormatPolicy,
        normalize_answer,
        validate_answer,
    )
    from nemotron_engine.scoring.local_scorer import ScoreResult, score_completion
    from nemotron_engine.submission.submission_validator import (
        SubmissionValidationResult,
        validate_adapter_dir,
        validate_submission_zip,
    )

    assert ExtractedAnswer is not None
    assert extract_boxed_answer is not None
    assert find_boxed_spans is not None
    assert has_truncation_risk is not None
    assert AnswerType is not None
    assert FormatPolicy is not None
    assert CanonicalAnswer is not None
    assert normalize_answer is not None
    assert validate_answer is not None
    assert ScoreResult is not None
    assert score_completion is not None
    assert SubmissionValidationResult is not None
    assert validate_adapter_dir is not None
    assert validate_submission_zip is not None
    assert ServingConfig is not None
    assert validate_serving_config is not None
    assert load_serving_config is not None
    assert save_serving_config is not None


def test_package_level_pass1_api_imports() -> None:
    from nemotron_engine.runtime import (
        ServingConfig,
        load_serving_config,
        save_serving_config,
        validate_serving_config,
    )
    from nemotron_engine.scoring import (
        CanonicalAnswer,
        ExtractedAnswer,
        FormatPolicy,
        ScoreResult,
        extract_boxed_answer,
        normalize_answer,
        score_completion,
    )
    from nemotron_engine.submission import (
        SubmissionValidationResult,
        validate_adapter_dir,
        validate_submission_zip,
    )

    assert ExtractedAnswer is not None
    assert extract_boxed_answer is not None
    assert FormatPolicy is not None
    assert CanonicalAnswer is not None
    assert normalize_answer is not None
    assert ScoreResult is not None
    assert score_completion is not None
    assert SubmissionValidationResult is not None
    assert validate_adapter_dir is not None
    assert validate_submission_zip is not None
    assert ServingConfig is not None
    assert validate_serving_config is not None
    assert load_serving_config is not None
    assert save_serving_config is not None

