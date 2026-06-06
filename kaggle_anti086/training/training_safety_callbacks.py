from __future__ import annotations

import math
from typing import Any

try:
    from transformers import TrainerCallback  # type: ignore
except ImportError:
    class TrainerCallback:  # type: ignore
        pass


class Sprint11TrainingSafetyCallback(TrainerCallback):
    def __init__(self, *, max_loss_abort: float = 20.0):
        super().__init__()
        self.max_loss_abort = max_loss_abort
        self.loss_start = None
        self.loss_end = None
        self.loss_min = None
        self.loss_max = None
        self.failure_reason = None

    def on_init_end(self, args=None, state=None, control=None, **kwargs):
        return control

    def on_train_begin(self, args=None, state=None, control=None, **kwargs):
        return control

    def on_step_end(self, args=None, state=None, control=None, **kwargs):
        return control

    def on_log(self, args=None, state=None, control=None, logs=None, **kwargs):
        logs = logs or {}
        if "loss" not in logs:
            return control
        try:
            loss = float(logs["loss"])
        except (TypeError, ValueError):
            self.failure_reason = "invalid_loss"
            if control is not None:
                control.should_training_stop = True
            return control
        if self.loss_start is None:
            self.loss_start = loss
        self.loss_end = loss
        self.loss_min = loss if self.loss_min is None else min(self.loss_min, loss)
        self.loss_max = loss if self.loss_max is None else max(self.loss_max, loss)
        if math.isnan(loss):
            self.failure_reason = "nan_loss"
        elif math.isinf(loss):
            self.failure_reason = "inf_loss"
        elif loss > self.max_loss_abort:
            self.failure_reason = "loss_above_abort_threshold"
        if self.failure_reason and control is not None:
            control.should_training_stop = True
        return control

    def report(self) -> dict[str, Any]:
        return {
            "loss_start": self.loss_start,
            "loss_end": self.loss_end,
            "loss_min": self.loss_min,
            "loss_max": self.loss_max,
            "failure_reason": self.failure_reason,
        }
