from kaggle_anti086.data.v2_sampling_policy import build_sampling_policy


def _direct(**overrides):
    row = {"id": "d", "family": "format_only", "verification_status": "verified", "loss_scope": "assistant_only"}
    row.update(overrides)
    return row


def test_verified_direct_row_allowed_for_sft():
    report = build_sampling_policy([_direct()], [], [], [], {})
    assert report["row_policy"]["d"]["allowed_for_sft"] is True
    assert report["row_policy"]["d"]["train_on_user"] is False


def test_solver_corrected_mirror_weighted_half():
    report = build_sampling_policy([], [_direct(id="c")], [], [], {"policy_overlay": {"c": {"derived_from_direct": True}}})
    assert report["row_policy"]["c"]["sampling_weight"] == 0.5


def test_abstain_and_hard_negative_not_allowed_for_sft():
    report = build_sampling_policy([], [], [{"id": "a"}], [{"id": "h"}], {})
    assert report["row_policy"]["a"]["allowed_for_sft"] is False
    assert report["row_policy"]["h"]["allowed_for_sft"] is False


def test_train_on_user_full_prompt_unsupported_unverified_fail():
    bad = _direct(family="equation_operator", verification_status="unverified")
    report = build_sampling_policy([bad], [], [], [], {})
    assert report["unsupported_sft_family_violations"] == 1
    assert report["unverified_sft_violations"] == 1
