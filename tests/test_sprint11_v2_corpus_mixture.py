from kaggle_anti086.data.v2_corpus_mixture import build_mixture_report


def test_family_distribution_computed():
    rows = [{"family": "bit_manipulation", "source_row_id": "a"}, {"family": "word_cipher", "source_row_id": "b"}]
    report = build_mixture_report(rows)
    assert report["family_counts"] == {"bit_manipulation": 1, "word_cipher": 1}
    assert report["oversampling_used"] is False


def test_underrepresented_families_reported():
    report = build_mixture_report([{"family": "bit_manipulation", "source_row_id": "a"}])
    assert "symbol_mapping" in report["underrepresented_families"]


def test_source_duplication_fails():
    report = build_mixture_report([{"family": "bit_manipulation", "source_row_id": "a"}, {"family": "bit_manipulation", "source_row_id": "a"}])
    assert report["status"] == "FAIL"
    assert report["max_duplicate_source_row_count"] == 2
