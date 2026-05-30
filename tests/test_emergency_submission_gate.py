from __future__ import annotations

import json
import sys
from pathlib import Path
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.emergency_score_recovery.adapter_inventory import inspect_adapter_dir  # noqa: E402
from nemotron_engine.emergency_score_recovery.adapter_package_plan import package_existing_adapter  # noqa: E402
from nemotron_engine.emergency_score_recovery.adapter_scorebook import default_scorebook_entries  # noqa: E402
from nemotron_engine.emergency_score_recovery.submission_decision_gate import decide_submission  # noqa: E402


def make_adapter(root: Path, *, rank: int = 32, public_parent: bool = False) -> Path:
    if public_parent:
        root = root / "huikang" / "nemotron-adapter" / "transformers" / "default" / "20"
    root.mkdir(parents=True)
    (root / "adapter_config.json").write_text(json.dumps({"r": rank}), encoding="utf-8")
    (root / "adapter_model.safetensors").write_bytes(b"x" * 2048)
    return root


def test_decision_gate_outputs_submit_baseline_for_known_parent(tmp_path: Path) -> None:
    adapter = inspect_adapter_dir(make_adapter(tmp_path, public_parent=True))
    decision = decide_submission(adapter, scorebook_entries=default_scorebook_entries())

    assert decision.decision == "SUBMIT_BASELINE"


def test_decision_gate_rejects_failed_child_labels_input_ids(tmp_path: Path) -> None:
    adapter = inspect_adapter_dir(make_adapter(tmp_path / "custom"))
    decision = decide_submission(
        adapter,
        training_metadata={"labels_equal_input_ids": True, "train_loss": 0.1},
        probe_evidence={"sane_outputs": True},
        operator_selected=True,
    )

    assert decision.decision == "DO_NOT_SUBMIT"
    assert "labels_input_ids_failed_branch" in decision.reasons


def test_decision_gate_rejects_high_loss_and_missing_probe(tmp_path: Path) -> None:
    adapter = inspect_adapter_dir(make_adapter(tmp_path / "custom"))
    decision = decide_submission(adapter, training_metadata={"train_loss": 30.0}, operator_selected=True)

    assert decision.decision == "DO_NOT_SUBMIT"
    assert "train_loss_gt_20" in decision.reasons
    assert "missing_probe_evidence" in decision.reasons


def test_decision_gate_allows_selected_sane_candidate(tmp_path: Path) -> None:
    adapter = inspect_adapter_dir(make_adapter(tmp_path / "custom"))
    decision = decide_submission(adapter, training_metadata={"train_loss": 1.2}, probe_evidence={"sane_outputs": True}, operator_selected=True)

    assert decision.decision == "SUBMIT_CANDIDATE"


def test_package_script_includes_only_adapter_files(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path / "adapter")
    zip_path = tmp_path / "submission.zip"

    result = package_existing_adapter(adapter, zip_path)

    assert result["contents"] == ["adapter_config.json", "adapter_model.safetensors"]
    with zipfile.ZipFile(zip_path) as archive:
        assert sorted(archive.namelist()) == ["adapter_config.json", "adapter_model.safetensors"]


def test_package_script_rejects_csv_predictions(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path / "adapter")
    (adapter / "predictions.csv").write_text("id,answer\n", encoding="utf-8")

    try:
        package_existing_adapter(adapter, tmp_path / "submission.zip")
    except Exception as exc:
        assert "forbidden file" in str(exc)
    else:
        raise AssertionError("expected forbidden CSV rejection")

