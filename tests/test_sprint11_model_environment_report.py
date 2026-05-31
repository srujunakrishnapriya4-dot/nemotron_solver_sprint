from kaggle_anti086.training import model_environment_report as mer


def _config():
    return {"base_model_path": "/kaggle/input/model", "load_in_4bit": True, "device_map": "auto", "gradient_checkpointing": True}


def test_local_missing_gpu_warns(monkeypatch):
    monkeypatch.setattr(
        mer,
        "check_training_dependencies",
        lambda *args, **kwargs: {
            "failures": ["cuda_unavailable", "tokenizer_unavailable"],
            "cuda_available": False,
            "tokenizer_loadable": False,
            "base_model_path_exists": False,
        },
    )
    report = mer.build_model_environment_report(_config(), kaggle_mode=False)
    assert report["status"] == "WARN"
    assert "local_model_or_gpu_unavailable" in report["warnings"]


def test_kaggle_missing_tokenizer_fails(monkeypatch):
    monkeypatch.setattr(
        mer,
        "check_training_dependencies",
        lambda *args, **kwargs: {
            "failures": ["tokenizer_unavailable"],
            "cuda_available": True,
            "tokenizer_loadable": False,
            "base_model_path_exists": True,
        },
    )
    report = mer.build_model_environment_report(_config(), kaggle_mode=True)
    assert report["status"] == "FAIL"
    assert "tokenizer_unavailable" in report["failures"]


def test_kaggle_four_bit_missing_bitsandbytes_fails(monkeypatch):
    monkeypatch.setattr(
        mer,
        "check_training_dependencies",
        lambda *args, **kwargs: {
            "failures": ["bitsandbytes_unavailable_for_4bit"],
            "cuda_available": True,
            "tokenizer_loadable": True,
            "base_model_path_exists": True,
        },
    )
    report = mer.build_model_environment_report(_config(), kaggle_mode=True)
    assert report["status"] == "FAIL"
    assert "bitsandbytes_unavailable_for_4bit" in report["failures"]
