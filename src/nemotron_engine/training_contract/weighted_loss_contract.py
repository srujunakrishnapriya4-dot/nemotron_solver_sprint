from __future__ import annotations


def weighted_loss_sum(losses: list[float], weights: list[float]) -> float:
    if len(losses) != len(weights):
        raise ValueError("loss/weight length mismatch")
    denom = sum(weights)
    if denom <= 0:
        raise ValueError("zero supervised weight")
    return sum(loss * weight for loss, weight in zip(losses, weights)) / denom
