from __future__ import annotations

from typing import Sequence

from src.training.dataset import TrainingExample


def collate_training_examples(examples: Sequence[TrainingExample]) -> dict[str, object]:
    return {
        "batch_size": len(examples),
        "prompts": [example.prompt for example in examples],
        "target_symbols": [list(example.target_symbols) for example in examples],
        "metadata": [example.metadata for example in examples],
    }
