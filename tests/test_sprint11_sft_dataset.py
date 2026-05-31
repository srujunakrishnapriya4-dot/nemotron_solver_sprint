from kaggle_anti086.training.real_tokenizer_collator_audit import FallbackAuditTokenizer, build_labels_for_row
from kaggle_anti086.training.sft_dataset import build_assistant_only_features, build_sft_dataset_report, load_weighted_sft_rows
from kaggle_anti086.training.training_config_schema import load_training_config


def test_direct_rows_supervise_answer_only():
    row = {"messages": [{"role": "user", "content": "x -> ?"}, {"role": "assistant", "content": "y"}], "family": "word_cipher", "verification_status": "verified", "rule_id": "r", "leakage_group": "g"}
    features = build_assistant_only_features(row, FallbackAuditTokenizer(), 128)
    assert all(label == -100 for label in features["labels"][: features["answer_span"][0]])
    assert any(label != -100 for label in features["labels"][features["answer_span"][0] :])


def test_weighted_sft_rows_exclude_abstain_and_hard_negative():
    config = load_training_config("kaggle_anti086/training/configs/v2a_base_lora.yaml")
    rows = load_weighted_sft_rows(config)
    sources = {item.source for item in rows}
    assert sources == {"direct_answer", "solver_corrected"}
    assert any(item.sampling_weight == 0.5 for item in rows if item.source == "solver_corrected")


def test_dataset_report_blocks_unsupported_and_zero_answer():
    rows = [
        type("Item", (), {"row": {"id": "x", "family": "equation_operator", "verification_status": "verified", "answer": "1", "rule_id": "r", "leakage_group": "g"}, "source": "direct_answer", "sampling_weight": 1.0})(),
        type("Item", (), {"row": {"id": "y", "family": "word_cipher", "verification_status": "verified", "answer": "", "rule_id": "r2", "leakage_group": "g2"}, "source": "direct_answer", "sampling_weight": 1.0})(),
    ]
    report = build_sft_dataset_report({"max_seq_len": 128}, rows)  # type: ignore[arg-type]
    assert report["status"] == "FAIL"
    assert report["unsupported_rows"] == 1
    assert report["zero_answer_rows"] == 1


def test_audit_and_dataset_share_label_builder():
    row = {"messages": [{"role": "user", "content": "Input: 1"}, {"role": "assistant", "content": "I"}]}
    tokenizer = FallbackAuditTokenizer()
    assert build_labels_for_row(row, tokenizer, supervise=True, max_seq_len=128)["labels"] == build_assistant_only_features(row, tokenizer, 128)["labels"]
