from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from src.evaluation.metrics import false_accept_rate


@dataclass(frozen=True)
class FarSummary:
    sample_count: int
    threshold: float
    false_accept_rate: float
    negative_count: int
    false_accept_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def evaluate_far(scores: Sequence[float], labels: Sequence[bool], threshold: float) -> FarSummary:
    negative_count = sum(1 for label in labels if not label)
    false_accept_count = sum(
        1 for score, label in zip(scores, labels, strict=False) if not label and score >= threshold
    )
    return FarSummary(
        sample_count=len(scores),
        threshold=threshold,
        false_accept_rate=false_accept_rate(labels, scores, threshold),
        negative_count=negative_count,
        false_accept_count=false_accept_count,
    )
