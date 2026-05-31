from kaggle_anti086.training.real_tokenizer_collator_audit import FallbackAuditTokenizer
from kaggle_anti086.training.sft_dataset import SFTItem
from kaggle_anti086.training.supervised_label_decode_audit import audit_supervised_label_decode


def test_exact_decoded_answer_passes(monkeypatch):
    row = {"id": "r1", "prompt": "Input: x", "answer": "cat book", "family": "word_cipher", "rule_id": "r", "leakage_group": "g", "verification_status": "verified"}
    monkeypatch.setattr("kaggle_anti086.training.supervised_label_decode_audit.load_weighted_sft_rows", lambda config: [SFTItem(row, "direct_answer", 1.0)])
    report = audit_supervised_label_decode({"max_seq_len": 100}, FallbackAuditTokenizer(), limit=1)
    assert report["status"] == "PASS"


def test_mismatch_fails(monkeypatch):
    row = {"id": "r1", "prompt": "Input: x", "answer": "gold", "family": "word_cipher", "rule_id": "r", "leakage_group": "g", "verification_status": "verified"}
    monkeypatch.setattr("kaggle_anti086.training.supervised_label_decode_audit.load_weighted_sft_rows", lambda config: [SFTItem(row, "direct_answer", 1.0)])
    class BadTokenizer(FallbackAuditTokenizer):
        def decode(self, ids, skip_special_tokens=True):
            return "ASSISTANT: wrong because noisy"
    report = audit_supervised_label_decode({"max_seq_len": 100}, BadTokenizer(), limit=1)
    assert report["status"] == "FAIL"
    assert "decoded_supervision_mismatch" in report["failures"]
