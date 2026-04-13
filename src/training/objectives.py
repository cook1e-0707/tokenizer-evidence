from __future__ import annotations

from typing import Sequence


class ObjectiveError(ValueError):
    """Raised when an objective receives invalid inputs."""


def bucket_mass_loss(bucket_probabilities: Sequence[float], target_index: int) -> float:
    if not bucket_probabilities:
        raise ObjectiveError("bucket_probabilities must be non-empty")
    if target_index < 0 or target_index >= len(bucket_probabilities):
        raise ObjectiveError("target_index out of range")
    target_probability = bucket_probabilities[target_index]
    if target_probability < 0.0 or target_probability > 1.0:
        raise ObjectiveError("bucket probabilities must be in [0, 1]")
    return round(1.0 - float(target_probability), 6)
