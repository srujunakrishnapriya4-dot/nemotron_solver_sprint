from kaggle_anti086.training.sft_dataset import SFTItem
from kaggle_anti086.training.weighted_sft_sampler import build_weighted_sample, build_weighted_sampling_report


def _item(i, source="direct_answer", family="bit_manipulation"):
    return SFTItem(
        row={
            "id": f"r{i}",
            "prompt": f"Prompt {i}",
            "answer": "x",
            "family": family,
            "rule_id": f"rule_{i}",
            "leakage_group": f"lg_{i}",
            "verification_status": "verified",
        },
        source=source,
        sampling_weight=1.0 if source == "direct_answer" else 0.5,
    )


def test_solver_corrected_half_selected_and_seed_deterministic():
    items = [_item(i, "direct_answer", "bit_manipulation") for i in range(10)]
    items += [_item(100 + i, "solver_corrected", "word_cipher") for i in range(10)]
    a = build_weighted_sample(items, seed=7)
    b = build_weighted_sample(items, seed=7)
    report = build_weighted_sampling_report(items, a, seed=7)
    assert [x.row["id"] for x in a] == [x.row["id"] for x in b]
    assert report["direct_rows_selected"] == 10
    assert report["solver_corrected_rows_selected"] == 5
    assert report["status"] == "PASS"


def test_abstain_hard_negative_source_fails_if_selected():
    selected = [_item(1), _item(2, "abstain_safety")]
    report = build_weighted_sampling_report(selected, selected)
    assert report["status"] == "FAIL"
    assert "non_sft_source_selected" in report["failures"]
