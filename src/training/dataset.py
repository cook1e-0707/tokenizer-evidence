from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class TrainingExample:
    prompt: str
    target_symbols: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


class JsonlDataset:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.examples = load_training_examples(path)

    def __iter__(self) -> Iterator[TrainingExample]:
        return iter(self.examples)

    def __len__(self) -> int:
        return len(self.examples)


def load_training_examples(path: Path) -> list[TrainingExample]:
    if path.suffix == ".jsonl":
        examples: list[TrainingExample] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            examples.append(
                TrainingExample(
                    prompt=str(payload.get("prompt", "")),
                    target_symbols=tuple(payload.get("target_symbols", [])),
                    metadata=dict(payload.get("metadata", {})),
                )
            )
        return examples

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        prompt = str(payload.get("prompt", "synthetic prompt"))
        expected_sequence = tuple(payload.get("expected_sequence", []))
        return [
            TrainingExample(
                prompt=prompt,
                target_symbols=expected_sequence,
                metadata={"source_path": str(path)},
            )
        ]
    raise ValueError(f"Unsupported dataset format for {path}")
