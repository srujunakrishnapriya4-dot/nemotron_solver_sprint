import json

from kaggle_anti086.training.eval_ladder import ModelLifecycleRecorder, select_eval_rows, should_run_full_eval, write_predictions_incremental


def test_eval_tiers_and_full_gate():
    rows = [{"id": i} for i in range(600)]
    assert len(select_eval_rows(rows, "smoke_32")) == 32
    assert len(select_eval_rows(rows, "fast_128")) == 128
    assert len(select_eval_rows(rows, "full_512")) == 512
    assert should_run_full_eval({"status": "PASS", "failures": []}) is True
    assert should_run_full_eval({"status": "FAIL"}) is False


def test_prediction_resume_and_lifecycle(tmp_path):
    path = tmp_path / "preds.jsonl"
    write_predictions_incremental(path, [{"row_id": "a"}, {"row_id": "b"}])
    added = write_predictions_incremental(path, [{"row_id": "a"}, {"row_id": "c"}], resume=True)
    assert added == 1
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["row_id"] for r in rows] == ["a", "b", "c"]
    rec = ModelLifecycleRecorder()
    rec.free_base_before_adapter()
    rec.adapter_loaded()
    assert rec.events == ["base_freed", "adapter_loaded"]
