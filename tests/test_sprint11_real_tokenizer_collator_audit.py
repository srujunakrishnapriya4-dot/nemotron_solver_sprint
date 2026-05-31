from kaggle_anti086.training.real_tokenizer_collator_audit import FallbackAuditTokenizer, build_labels_for_row, build_real_collator_audit
from kaggle_anti086.training.training_config_schema import load_training_config


def test_real_collator_passes_assistant_only_labels():
    config = load_training_config("kaggle_anti086/training/configs/v2a_base_lora.yaml")
    report = build_real_collator_audit(config, sample_size=16, tokenizer=FallbackAuditTokenizer())
    assert report["status"] == "PASS"
    assert report["user_tokens_supervised"] == 0
    assert report["system_tokens_supervised"] == 0
    assert report["assistant_answer_tokens_supervised"] > 0


def test_real_collator_reports_tokenizer_unavailable_honestly():
    config = load_training_config("kaggle_anti086/training/configs/v2a_base_lora.yaml")
    report = build_real_collator_audit(config, sample_size=8)
    assert report["status"] in {"WARN_LOCAL_TOKENIZER_UNAVAILABLE", "PASS"}
    if report["status"] != "PASS":
        assert report["tokenizer_loaded"] is False


def test_build_labels_masks_user_and_system_tokens():
    row = {"messages": [{"role": "user", "content": "Input: 1"}, {"role": "assistant", "content": "I"}]}
    features = build_labels_for_row(row, FallbackAuditTokenizer(), supervise=True, max_seq_len=1024)
    labels = features["labels"]
    assert all(label == -100 for label in labels[: features["answer_span"][0]])
    assert any(label != -100 for label in labels[features["answer_span"][0] :])


def test_audit_detects_bad_label_builder(monkeypatch):
    config = load_training_config("kaggle_anti086/training/configs/v2a_base_lora.yaml")

    def bad_builder(row, tokenizer, *, supervise, max_seq_len):
        return {
            "input_ids": [1, 2, 3],
            "labels": [1, 2, 3],
            "system_span": [0, 1],
            "user_span": [1, 2],
            "assistant_prefix_span": [2, 2],
            "answer_span": [2, 3],
            "answer_token_count": 1,
            "truncated": False,
        }

    monkeypatch.setattr("kaggle_anti086.training.real_tokenizer_collator_audit.build_labels_for_row", bad_builder)
    report = build_real_collator_audit(config, sample_size=4, tokenizer=FallbackAuditTokenizer())
    assert report["status"] == "FAIL"
    assert "user_tokens_supervised_nonzero" in report["failures"]
