from __future__ import annotations

from types import SimpleNamespace

from kaggle_anti086.training.training_safety_callbacks import Sprint11TrainingSafetyCallback, TrainerCallback


def test_training_safety_callback_is_transformers_callback():
    callback = Sprint11TrainingSafetyCallback()

    assert isinstance(callback, TrainerCallback)
    assert hasattr(callback, "on_init_end")
    assert hasattr(callback, "on_step_end")
    assert hasattr(callback, "on_log")
    report = callback.report()
    assert set(report) >= {"loss_start", "loss_end", "loss_min", "loss_max", "failure_reason"}


def test_training_safety_callback_stops_on_invalid_loss():
    callback = Sprint11TrainingSafetyCallback(max_loss_abort=20.0)
    control = SimpleNamespace(should_training_stop=False)

    returned = callback.on_log(control=control, logs={"loss": "nan"})

    assert returned is control
    assert control.should_training_stop is True
    assert callback.report()["failure_reason"] == "nan_loss"
