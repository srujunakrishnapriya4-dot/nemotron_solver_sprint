from kaggle_anti086.data.v2_corpus_quality_gate import PER_FAMILY_MINIMUMS, build_quality_gate
from kaggle_anti086.data.v2_prompt_templates import build_direct_answer_messages


def _direct_rows(counts=None):
    counts = counts or PER_FAMILY_MINIMUMS
    rows = []
    idx = 0
    for family, count in counts.items():
        for _ in range(count):
            idx += 1
            answer = "IV" if family == "roman_numeral" else "x"
            row = {
                "id": f"train_v2_direct_{idx:06d}",
                "source_row_id": f"src-{idx}",
                "family": family,
                "subfamily": "s",
                "rule_id": f"r-{idx}",
                "leakage_group": f"lg-{idx}",
                "prompt": f"prompt {idx}",
                "answer": answer,
                "messages": build_direct_answer_messages({"prompt": f"prompt {idx}"}, answer),
                "loss_scope": "assistant_only",
            }
            rows.append(row)
    while len(rows) < 1000:
        idx += 1
        rows.append(
            {
                "id": f"train_v2_direct_{idx:06d}",
                "source_row_id": f"src-{idx}",
                "family": "bit_manipulation",
                "subfamily": "s",
                "rule_id": f"r-{idx}",
                "leakage_group": f"lg-{idx}",
                "prompt": f"prompt {idx}",
                "answer": "0101",
                "messages": build_direct_answer_messages({"prompt": f"prompt {idx}"}, "0101"),
                "loss_scope": "assistant_only",
            }
        )
    return rows


def _pass_reports():
    return {"status": "PASS", "rule_id_overlap_count": 0, "leakage_group_overlap_count": 0, "prompt_hash_overlap_count": 0}, {"status": "PASS"}


def test_passes_clean_corpus():
    leakage, mixture = _pass_reports()
    report = build_quality_gate(_direct_rows(), [{}] * 200, [{}] * 20, leakage, mixture)
    assert report["status"] == "PASS"
    assert report["decision"] == "ALLOW_DAY7_TRAINING_CONFIG_PREP"
    assert report["training_allowed"] is False


def test_fails_if_direct_rows_below_1000():
    leakage, mixture = _pass_reports()
    report = build_quality_gate(_direct_rows({"format_only": 20})[:999], [{}] * 200, [{}] * 20, leakage, mixture)
    assert report["quality_gates"]["verified_direct_answer_rows"]["status"] == "FAIL"


def test_fails_if_format_only_below_20():
    counts = dict(PER_FAMILY_MINIMUMS)
    counts["format_only"] = 19
    leakage, mixture = _pass_reports()
    report = build_quality_gate(_direct_rows(counts), [{}] * 200, [{}] * 20, leakage, mixture)
    assert report["quality_gates"]["format_only_rows"]["status"] == "FAIL"


def test_fails_if_unsupported_direct_row_exists():
    rows = _direct_rows()
    rows[0]["family"] = "equation_operator"
    leakage, mixture = _pass_reports()
    report = build_quality_gate(rows, [{}] * 200, [{}] * 20, leakage, mixture)
    assert report["quality_gates"]["unsupported_family_direct_rows"]["status"] == "FAIL"


def test_fails_if_full_prompt_loss_or_verbose_assistant():
    rows = _direct_rows()
    rows[0]["loss_scope"] = "full_prompt"
    rows[1]["messages"][1]["content"] = "Answer: x"
    leakage, mixture = _pass_reports()
    report = build_quality_gate(rows, [{}] * 200, [{}] * 20, leakage, mixture)
    assert report["quality_gates"]["full_prompt_loss_rows"]["status"] == "FAIL"
    assert report["quality_gates"]["verbose_assistant_rows"]["status"] == "FAIL"


def test_fails_if_leakage_report_fails():
    leakage, mixture = {"status": "FAIL", "rule_id_overlap_count": 1, "leakage_group_overlap_count": 0, "prompt_hash_overlap_count": 0}, {"status": "PASS"}
    report = build_quality_gate(_direct_rows(), [{}] * 200, [{}] * 20, leakage, mixture)
    assert report["quality_gates"]["leakage_report_status"]["status"] == "FAIL"
