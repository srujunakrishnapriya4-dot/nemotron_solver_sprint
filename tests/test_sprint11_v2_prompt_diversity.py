from kaggle_anti086.data.v2_prompt_diversity import build_prompt_diversity_report


def _row(i, family="bit_manipulation", prompt=None, generator="g", rule=None, noise="clean"):
    return {
        "id": str(i),
        "family": family,
        "subfamily": "s",
        "prompt": prompt or f"Example {i}: 0001 -> 0010. Input: {i}",
        "metadata": {"generator_id": generator, "rule_signature": rule or f"rule-{i}", "surface_noise_profile": noise},
    }


def test_single_prompt_template_over_20_percent_fails():
    rows = [_row(i, prompt="1 -> 2. Input: 3", generator=f"g{i}") for i in range(10)]
    report = build_prompt_diversity_report(rows)
    assert report["status"] == "FAIL"
    assert "prompt_template_signature_share_above_20_percent" in report["violations"]


def test_single_generator_over_35_percent_fails():
    rows = [_row(i, generator="one-generator", prompt=f"prompt unique {i} value {i}") for i in range(20)]
    report = build_prompt_diversity_report(rows)
    assert "generator_share_above_35_percent" in report["violations"]


def test_rule_signature_dominance_warns_for_major_family():
    rows = [_row(i, prompt=f"shape {i}", generator=f"g{i}", rule="same-rule", noise="clean" if i % 2 else "quoted") for i in range(60)]
    report = build_prompt_diversity_report(rows)
    assert any("rule_signature_share" in warning for warning in report["warnings"])


def test_multiple_templates_and_noise_profiles_pass():
    families = ["bit_manipulation", "word_cipher", "format_only"]
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india"]
    rows = []
    for i in range(45):
        rows.append(_row(i, family=families[i % 3], prompt=f"{words[i % len(words)]} template shape {i % 5} query token", generator=f"g{i}", noise="clean" if i % 2 else "quoted"))
    report = build_prompt_diversity_report(rows)
    assert report["max_prompt_template_share"] < 0.20
    assert report["surface_noise_profiles_per_family"]["bit_manipulation"] >= 2
