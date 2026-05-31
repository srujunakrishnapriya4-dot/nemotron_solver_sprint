from kaggle_anti086.training.training_safety_callbacks import Sprint11TrainingSafetyCallback


class Control:
    should_training_stop = False


def test_nan_inf_and_huge_loss_stop():
    for value in [float("nan"), float("inf"), 21.0]:
        cb = Sprint11TrainingSafetyCallback(max_loss_abort=20.0)
        control = cb.on_log(control=Control(), logs={"loss": value})
        assert control.should_training_stop is True
        assert cb.report()["failure_reason"]


def test_normal_loss_records_metrics():
    cb = Sprint11TrainingSafetyCallback()
    control = cb.on_log(control=Control(), logs={"loss": 2.5})
    assert control.should_training_stop is False
    assert cb.report()["loss_start"] == 2.5
