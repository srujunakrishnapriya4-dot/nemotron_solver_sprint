from kaggle_anti086.training.loss_mask_audit import build_loss_mask_audit
from kaggle_anti086.training.training_config_schema import load_training_config


def _config(**overrides):
    cfg = load_training_config("kaggle_anti086/training/configs/v2a_base_lora.yaml")
    cfg.update(overrides)
    return cfg


def test_assistant_answer_tokens_supervised_only():
    report = build_loss_mask_audit(_config())
    assert report["status"] == "PASS"
    assert report["answer_supervision_count"] > 0
    assert report["user_token_supervision_count"] == 0
    assert report["system_token_supervision_count"] == 0


def test_full_prompt_and_train_on_user_fail():
    assert "full_prompt_loss_detected" in build_loss_mask_audit(_config(full_prompt_loss=True))["failures"]
    assert "train_on_user_detected" in build_loss_mask_audit(_config(train_on_user=True))["failures"]


def test_zero_supervised_sft_row_fails(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id":"x","family":"format_only","messages":[{"role":"user","content":"p"},{"role":"assistant","content":""}]}\n', encoding="utf-8")
    cfg = _config(train_direct_path=str(path), train_solver_corrected_path=str(path))
    report = build_loss_mask_audit(cfg)
    assert report["zero_supervised_sft_rows"] > 0


def test_hard_negative_and_abstain_not_supervised():
    report = build_loss_mask_audit(_config())
    assert report["hard_negative_supervised_count"] == 0
    assert report["abstain_safety_supervised_count"] == 0
