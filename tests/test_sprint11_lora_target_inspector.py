from kaggle_anti086.training.lora_target_inspector import verify_lora_targets


class Linear:
    pass


class FakeModel:
    def __init__(self, names):
        self.names = names

    def named_modules(self):
        return [(name, Linear()) for name in self.names]


def test_fake_model_with_qvo_targets_passes():
    report = verify_lora_targets(FakeModel(["a.q_proj", "a.v_proj", "a.o_proj"]), ["q_proj", "v_proj", "o_proj"])
    assert report["status"] == "PASS"


def test_missing_target_and_lm_head_fail():
    report = verify_lora_targets(FakeModel(["a.q_proj", "a.v_proj"]), ["q_proj", "v_proj", "o_proj", "lm_head"])
    assert report["status"] == "FAIL"
    assert "lm_head_target_forbidden" in report["failures"]
    assert "target_missing:o_proj" in report["failures"]
