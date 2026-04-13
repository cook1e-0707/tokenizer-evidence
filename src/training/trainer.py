from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrainingPlan:
    dataset_name: str
    objective: str
    batch_size: int
    epochs: int
    learning_rate: float


@dataclass(frozen=True)
class TrainingOutcome:
    status: str
    steps: int
    examples_seen: int
    final_loss: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def execute_training(plan: TrainingPlan, dataset_size: int) -> TrainingOutcome:
    effective_size = max(dataset_size, 1)
    steps_per_epoch = max(1, math.ceil(effective_size / max(plan.batch_size, 1)))
    steps = steps_per_epoch * max(plan.epochs, 1)
    final_loss = round(1.0 / (effective_size + max(plan.batch_size, 1)), 6)
    return TrainingOutcome(
        status="completed",
        steps=steps,
        examples_seen=effective_size * max(plan.epochs, 1),
        final_loss=final_loss,
    )
