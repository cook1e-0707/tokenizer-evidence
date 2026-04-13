from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def save_checkpoint_metadata(run_dir: Path, name: str, payload: Mapping[str, Any]) -> Path:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{name}.json"
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_checkpoint_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
