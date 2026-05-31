import pytest

from kaggle_anti086.training.real_tokenizer_collator_audit import FallbackAuditTokenizer
from kaggle_anti086.training.sft_dataset import SFTDatasetRejectedRowError, SFTItem, Sprint11SFTDataset


def _item(row):
    base = {
        "id": "r1",
        "family": "word_cipher",
        "subfamily": "x",
        "rule_id": "rule",
        "leakage_group": "leak",
        "verification_status": "verified",
        "messages": [{"role": "user", "content": "a -> b. c -> ?"}, {"role": "assistant", "content": "d"}],
        "answer": "d",
    }
    base.update(row)
    return SFTItem(row=base, source="direct_answer", sampling_weight=1.0)


def test_valid_row_passes():
    dataset = Sprint11SFTDataset([_item({})], FallbackAuditTokenizer(), 128)
    assert dataset[0]["supervised_token_count"] > 0


def test_unsupported_row_raises():
    with pytest.raises(SFTDatasetRejectedRowError, match="unsupported_family"):
        Sprint11SFTDataset([_item({"family": "equation_operator"})], FallbackAuditTokenizer(), 128)


def test_zero_supervised_row_raises():
    with pytest.raises(SFTDatasetRejectedRowError, match="missing_answer_span|zero_supervised"):
        Sprint11SFTDataset([_item({"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": ""}], "answer": ""})], FallbackAuditTokenizer(), 128)


def test_truncated_answer_row_raises():
    long_prompt = " ".join(f"tok{i}" for i in range(200))
    with pytest.raises(SFTDatasetRejectedRowError, match="answer_truncated"):
        Sprint11SFTDataset([_item({"messages": [{"role": "user", "content": long_prompt}, {"role": "assistant", "content": "final"}], "answer": "final"})], FallbackAuditTokenizer(), 16)
