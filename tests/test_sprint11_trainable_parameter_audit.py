from kaggle_anti086.training.trainable_parameter_audit import audit_trainable_parameters


class Param:
    def __init__(self, n, requires_grad):
        self.n = n
        self.requires_grad = requires_grad

    def numel(self):
        return self.n


class Model:
    def __init__(self, params):
        self.params = params

    def named_parameters(self):
        return self.params


def test_lora_only_trainable_passes():
    model = Model([("base.weight", Param(1000, False)), ("layer.lora_A.weight", Param(10, True))])
    report = audit_trainable_parameters(model)
    assert report["status"] == "PASS"
    assert report["base_params_frozen"] is True


def test_base_or_lm_head_trainable_fails():
    model = Model([("base.weight", Param(100, True)), ("lm_head.weight", Param(10, True))])
    report = audit_trainable_parameters(model)
    assert report["status"] == "FAIL"
    assert "base_params_not_frozen" in report["failures"]
    assert "lm_head_trainable" in report["failures"]
