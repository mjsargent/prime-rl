from __future__ import annotations


def accuracy_above_chance(accuracy: float, num_classes: int) -> float:
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    return accuracy - 1.0 / num_classes
