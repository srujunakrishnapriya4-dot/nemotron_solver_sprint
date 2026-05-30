from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.core.registry import (  # noqa: E402
    load_registry_jsonl,
    quarantine_problem,
    split_allows_sft,
    validate_registry,
    write_registry_jsonl,
)
from nemotron_engine.core.schemas import ImmutableProblemRecord, ProblemSource, RegistryError, SplitName, stable_json_dumps  # noqa: E402


_TMP = Path(__file__).resolve().parent / ".tmp_pass2"


def _record(problem_id: str = "p1", **updates: object) -> ImmutableProblemRecord:
    data = {
        "problem_id": problem_id,
        "source": ProblemSource.MANUAL_FIXTURE,
        "source_hash": "sha",
        "family_id": "fam",
        "primitive_family_id": "prim",
        "format_family_id": "fmt",
        "prompt_wrapper_id": "wrap",
        "split": SplitName.TRAIN,
        "allowed_for_sft": True,
        "allowed_for_dpo": True,
        "allowed_for_eval": True,
    }
    data.update(updates)
    return ImmutableProblemRecord(**data)


def _unsafe_record(**updates: object) -> ImmutableProblemRecord:
    record = _record()
    forged = object.__new__(ImmutableProblemRecord)
    for name in record.__dataclass_fields__:
        object.__setattr__(forged, name, getattr(record, name))
    for key, value in updates.items():
        object.__setattr__(forged, key, value)
    return forged


def test_write_load_registry_jsonl() -> None:
    root = _TMP / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    try:
        path = root / "registry.jsonl"
        records = [_record("p1"), _record("p2")]
        write_registry_jsonl(records, path)
        assert load_registry_jsonl(path) == records
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)


def test_duplicate_problem_ids_rejected() -> None:
    with pytest.raises(RegistryError):
        validate_registry([_record("p1"), _record("p1")])


def test_contamination_disables_training_and_quarantine_is_immutable() -> None:
    record = _record(contamination_flags=("leak",))
    quarantined = quarantine_problem(record, "leak")

    assert split_allows_sft(record) is False
    assert quarantined.problem_id == record.problem_id
    assert record.contamination_flags == ("leak",)


def test_validate_registry_rejects_contaminated_training_flags() -> None:
    with pytest.raises(RegistryError):
        validate_registry([_record(contamination_flags=("leak",))])


def test_forbidden_holdout_never_allows_training() -> None:
    record = _record(split=SplitName.FORBIDDEN_HOLDOUT, allowed_for_sft=False, allowed_for_dpo=False)

    assert split_allows_sft(record) is False


def test_validate_registry_rejects_forged_forbidden_and_stress_training_flags() -> None:
    with pytest.raises(RegistryError):
        validate_registry([_unsafe_record(split=SplitName.FORBIDDEN_HOLDOUT, allowed_for_sft=True)])
    with pytest.raises(RegistryError):
        validate_registry([_unsafe_record(split=SplitName.STRESS_ONLY, allowed_for_dpo=True)])


def test_load_registry_jsonl_rejects_unknown_fields_and_non_object_rows() -> None:
    root = _TMP / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    try:
        unknown = root / "unknown.jsonl"
        unknown.write_text(stable_json_dumps({**_record().__dict__, "extra": "bad"}) + "\n", encoding="utf-8")
        with pytest.raises(RegistryError):
            load_registry_jsonl(unknown)

        non_object = root / "non_object.jsonl"
        non_object.write_text("[]\n", encoding="utf-8")
        with pytest.raises(RegistryError):
            load_registry_jsonl(non_object)
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
