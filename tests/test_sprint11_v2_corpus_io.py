import json
from pathlib import Path

import pytest

from kaggle_anti086.data.v2_corpus_io import read_jsonl, write_json_checked, write_jsonl_checked


def test_jsonl_round_trip_and_hash_report(tmp_path):
    path = tmp_path / "rows.jsonl"
    record = write_jsonl_checked(path, [{"id": "a"}, {"id": "b"}], field_name="unit_test_jsonl")
    assert record["row_count"] == 2
    assert record["size_bytes"] > 0
    assert len(record["sha256"]) == 64
    assert read_jsonl(path) == [{"id": "a"}, {"id": "b"}]


def test_json_write_reports_hash(tmp_path):
    path = tmp_path / "report.json"
    record = write_json_checked(path, {"status": "PASS"}, field_name="unit_test_json")
    assert record["size_bytes"] == path.stat().st_size
    assert json.loads(path.read_text())["status"] == "PASS"


def test_kaggle_input_output_rejected():
    with pytest.raises(SystemExit):
        write_jsonl_checked(Path("/kaggle/input/bad.jsonl"), [], field_name="bad_output")


def test_tmp_durable_output_rejected():
    with pytest.raises(SystemExit):
        write_json_checked(Path("/tmp/day6.json"), {"bad": True}, field_name="bad_tmp_output")
