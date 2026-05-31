from kaggle_anti086.training import model_loader


def test_dependency_report_has_required_fields():
    report = model_loader.check_training_dependencies("/missing/model")
    for key in ("torch_available", "transformers_available", "peft_available", "cuda_available", "tokenizer_loadable"):
        assert key in report


def test_pad_token_is_set_if_missing(monkeypatch):
    class Tok:
        pad_token = None
        eos_token = "<eos>"

    class Auto:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return Tok()

    monkeypatch.setitem(__import__("sys").modules, "transformers", type("M", (), {"AutoTokenizer": Auto}))
    tok = model_loader.load_tokenizer("/missing")
    assert tok.pad_token == "<eos>"
