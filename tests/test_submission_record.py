from __future__ import annotations

from dataclasses import replace

import pytest

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.runbook import SubmissionRecord, SubmissionRecordError


def record_id(**overrides: object) -> str:
    payload = {
        "manual_submission_performed": False,
        "human_approval_hash": "h" * 64,
        "reproducibility_manifest_hash": "r" * 64,
        "submission_reference": None,
        "artifact_hash": None,
    }
    payload.update(overrides)
    return stable_hash(payload)


def record(**kwargs: object) -> SubmissionRecord:
    payload = {
        "record_id": record_id(),
        "manual_submission_performed": False,
        "submitted_by": None,
        "external_platform": None,
        "artifact_hash": None,
        "submission_reference": None,
        "submitted_at_text": None,
        "public_score_text": None,
        "private_score_text": None,
        "notes": None,
        "human_approval_hash": "h" * 64,
        "reproducibility_manifest_hash": "r" * 64,
        "metadata": {},
    }
    payload.update(kwargs)
    return SubmissionRecord(**payload)


def test_no_submission_record_passes() -> None:
    assert record().record_hash


@pytest.mark.parametrize(
    "field,value",
    [
        ("submitted_by", "operator"),
        ("external_platform", "kaggle"),
        ("submission_reference", "ref"),
        ("submitted_at_text", "today"),
        ("public_score_text", "0.1"),
        ("private_score_text", "0.1"),
    ],
)
def test_no_submission_with_submission_details_rejected(field: str, value: str) -> None:
    with pytest.raises(SubmissionRecordError):
        record(**{field: value})


def test_manual_submission_requires_submitted_by_reference_and_artifact_hash() -> None:
    with pytest.raises(SubmissionRecordError):
        record(record_id=record_id(manual_submission_performed=True), manual_submission_performed=True)
    assert record(
        record_id=record_id(manual_submission_performed=True, submission_reference="ref", artifact_hash="a" * 64),
        manual_submission_performed=True,
        submitted_by="operator",
        external_platform="kaggle",
        artifact_hash="a" * 64,
        submission_reference="ref",
    )


def test_scores_require_human_recorded_true() -> None:
    with pytest.raises(SubmissionRecordError):
        record(
            record_id=record_id(manual_submission_performed=True, submission_reference="ref", artifact_hash="a" * 64),
            manual_submission_performed=True,
            submitted_by="operator",
            artifact_hash="a" * 64,
            submission_reference="ref",
            public_score_text="0.1",
        )
    assert record(
        record_id=record_id(manual_submission_performed=True, submission_reference="ref", artifact_hash="a" * 64),
        manual_submission_performed=True,
        submitted_by="operator",
        artifact_hash="a" * 64,
        submission_reference="ref",
        public_score_text="0.1",
        metadata={"human_recorded": True},
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("notes", "score guaranteed"),
        ("public_score_text", "95+ guaranteed"),
        ("private_score_text", "private score guaranteed"),
        ("external_platform", "automatic submission platform"),
        ("submission_reference", "auto submitted by automation"),
        ("submitted_at_text", "Kaggle success at noon"),
    ],
)
def test_free_text_fake_claims_rejected(field: str, value: str) -> None:
    payload = {
        "record_id": record_id(manual_submission_performed=True, submission_reference="ref", artifact_hash="a" * 64),
        "manual_submission_performed": True,
        "submitted_by": "operator",
        "artifact_hash": "a" * 64,
        "submission_reference": "ref",
        "metadata": {"human_recorded": True},
    }
    payload[field] = value
    if field == "submission_reference":
        payload["record_id"] = record_id(manual_submission_performed=True, submission_reference=value, artifact_hash="a" * 64)
    with pytest.raises(SubmissionRecordError):
        record(**payload)


def test_ordinary_human_recorded_score_text_allowed() -> None:
    assert record(
        record_id=record_id(manual_submission_performed=True, submission_reference="ref", artifact_hash="a" * 64),
        manual_submission_performed=True,
        submitted_by="operator",
        artifact_hash="a" * 64,
        submission_reference="ref",
        public_score_text="0.123",
        private_score_text="pending",
        metadata={"human_recorded": True},
    )


@pytest.mark.parametrize("metadata", [{"automatic_submission": True}, {"score": "95+ guaranteed"}])
def test_automatic_submission_and_score_guarantee_metadata_rejected(metadata: dict[str, object]) -> None:
    with pytest.raises(SubmissionRecordError):
        record(metadata=metadata)


def test_forged_record_hash_rejected() -> None:
    with pytest.raises(SubmissionRecordError):
        replace(record(), record_hash="forged")
