from kaggle_anti086.training.real_tokenizer_collator_audit import FallbackAuditTokenizer
from kaggle_anti086.training.sft_dataset import build_assistant_only_features


def _row(prompt="short", answer="ANSWER"):
    return {"id": "r1", "prompt": prompt, "answer": answer, "family": "word_cipher", "rule_id": "r", "leakage_group": "g", "verification_status": "verified"}


def test_answer_truncation_is_flagged():
    tok = FallbackAuditTokenizer()
    features = build_assistant_only_features(_row(" ".join(["p"] * 50), "gold"), tok, max_seq_len=12)
    assert features["answer_truncated"] is True
    assert features["rejected_reason"] == "answer_truncated"


def test_zero_answer_is_missing_span():
    features = build_assistant_only_features(_row("prompt", ""), FallbackAuditTokenizer(), max_seq_len=100)
    assert features["missing_answer_span"] is True
    assert features["rejected_reason"] == "missing_answer_span"
