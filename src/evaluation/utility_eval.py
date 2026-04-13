from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from src.evaluation.metrics import acceptance_rate


@dataclass(frozen=True)
class UtilitySummary:
    sample_count: int
    success_count: int
    acceptance_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def evaluate_utility(acceptances: Sequence[bool]) -> UtilitySummary:
    success_count = sum(1 for item in acceptances if item)
    return UtilitySummary(
        sample_count=len(acceptances),
        success_count=success_count,
        acceptance_rate=acceptance_rate(acceptances),
    )
