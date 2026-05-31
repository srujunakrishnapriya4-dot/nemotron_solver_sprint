from kaggle_anti086.training.sampling_policy_consumption import build_sampling_policy_consumption
from kaggle_anti086.training.training_config_schema import load_training_config


def test_sampling_policy_consumed_safely():
    report = build_sampling_policy_consumption(load_training_config("kaggle_anti086/training/configs/v2a_base_lora.yaml"))
    assert report["status"] == "PASS"
    assert report["direct_answer_rows"] == 1350
    assert report["solver_corrected_rows"] == 1350
    assert report["solver_corrected_effective_weight"] == 675
    assert report["abstain_safety_sft_rows"] == 0
    assert report["hard_negative_sft_rows"] == 0
    assert report["unsupported_sft_rows"] == 0
