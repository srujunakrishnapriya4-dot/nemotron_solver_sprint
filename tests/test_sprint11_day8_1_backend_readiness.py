from kaggle_anti086.data.v2_corpus_io import write_json_checked
from kaggle_anti086.training.day8_1_backend_readiness import build_backend_readiness


def test_backend_readiness_fails_without_collator_pass(tmp_path):
    audit = tmp_path / "audit.json"
    write_json_checked(audit, {"status": "FAIL"}, field_name="audit")
    report = build_backend_readiness(collator_path=audit)
    assert report["status"] == "FAIL"
    assert "collator_audit_not_pass" in report["failures"]
