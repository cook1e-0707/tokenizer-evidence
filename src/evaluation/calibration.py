from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from src.evaluation.metrics import false_accept_rate


@dataclass(frozen=True)
class ThresholdCalibration:
    target_far: float
    threshold: float
    observed_far: float
    sample_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def calibrate_far_threshold(
    scores: Sequence[float],
    labels: Sequence[bool],
    target_far: float,
) -> ThresholdCalibration:
    if not scores:
        return ThresholdCalibration(
            target_far=target_far,
            threshold=1.0,
            observed_far=0.0,
            sample_count=0,
        )

    candidate_thresholds = sorted(set(scores), reverse=True)
    best_threshold = candidate_thresholds[-1]
    best_far = false_accept_rate(labels, scores, best_threshold)
    for threshold in candidate_thresholds:
        observed_far = false_accept_rate(labels, scores, threshold)
        if observed_far <= target_far:
            best_threshold = threshold
            best_far = observed_far
    return ThresholdCalibration(
        target_far=target_far,
        threshold=best_threshold,
        observed_far=best_far,
        sample_count=len(scores),
    )
