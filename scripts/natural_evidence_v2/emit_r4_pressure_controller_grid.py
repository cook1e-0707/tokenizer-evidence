from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.validate_r4_positive_evidence_contract import load_yaml  # noqa: E402


def _float_list(grid: Mapping[str, Any], key: str) -> list[float]:
    values = grid.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"controller_grid.{key} must be a non-empty list")
    floats = [float(item) for item in values]
    if floats != sorted(floats):
        raise ValueError(f"controller_grid.{key} must be sorted ascending")
    if len(set(floats)) != len(floats):
        raise ValueError(f"controller_grid.{key} must contain unique values")
    return floats


def grid_values(config: Mapping[str, Any], grid_index: int) -> dict[str, Any]:
    grid = config.get("controller_grid")
    if not isinstance(grid, Mapping):
        raise ValueError("controller_grid must be a mapping")
    bonus_values = _float_list(grid, "bonus_nats")
    penalty_values = _float_list(grid, "penalty_nats")
    max_target_mass_values = _float_list(grid, "max_target_mass")
    max_kl_values = _float_list(grid, "max_kl_budget")
    combinations = list(product(bonus_values, penalty_values, max_target_mass_values, max_kl_values))
    if grid_index < 0 or grid_index >= len(combinations):
        raise ValueError(f"grid_index {grid_index} outside controller grid size {len(combinations)}")
    bonus, penalty, max_target_mass, max_kl = combinations[grid_index]
    return {
        "controller_bonus_nats": bonus,
        "controller_grid_size": len(combinations),
        "controller_max_kl_budget": max_kl,
        "controller_max_target_mass": max_target_mass,
        "controller_mode": str(grid.get("controller_mode", "additive")),
        "controller_penalty_nats": penalty,
        "grid_index": grid_index,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit one R4 pressure-controller grid assignment from a route config.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--grid-index", type=int, required=True)
    parser.add_argument("--format", choices=["json", "shell"], default="json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    values = grid_values(load_yaml(config_path), args.grid_index)
    if args.format == "json":
        print(json.dumps(values, indent=2, sort_keys=True))
        return 0
    for key in sorted(values):
        print(f"{key.upper()}={values[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
