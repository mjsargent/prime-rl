from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from controllability.encoders.base import EncodedTrajectory
from controllability.identifiability.mutual_information import accuracy_above_chance


@dataclass(frozen=True)
class IdentifiabilityResult:
    accuracy: float
    chance: float
    identifiability: float
    num_classes: int
    discriminator_family: str


def measure_identifiability(
    encoded_trajectories: list[EncodedTrajectory],
    coordinate_labels: list[int],
    discriminator_family: Literal["logistic", "mlp_2layer"],
    n_bootstrap: int = 1000,
    train_test_split: Literal["by_prompt"] = "by_prompt",
    seed: int = 42,
) -> IdentifiabilityResult:
    if len(encoded_trajectories) != len(coordinate_labels):
        raise ValueError("encoded_trajectories and coordinate_labels must have the same length")
    if not coordinate_labels:
        raise ValueError("At least one coordinate label is required")
    num_classes = len(set(coordinate_labels))
    majority = max(coordinate_labels.count(label) for label in set(coordinate_labels)) / len(coordinate_labels)
    return IdentifiabilityResult(
        accuracy=majority,
        chance=1.0 / num_classes,
        identifiability=accuracy_above_chance(majority, num_classes),
        num_classes=num_classes,
        discriminator_family=discriminator_family,
    )
