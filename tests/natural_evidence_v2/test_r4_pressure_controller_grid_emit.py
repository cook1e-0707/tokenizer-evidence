from __future__ import annotations

from pathlib import Path

import pytest

from scripts.natural_evidence_v2.emit_r4_pressure_controller_grid import grid_values
from scripts.natural_evidence_v2.validate_r4_positive_evidence_contract import load_yaml

SAFETY_BOUND_CONFIG = Path("configs/natural_evidence_v2/r4_controller_only_safety_bound_pressure_route.yaml")


def test_emit_grid_values_from_route_config() -> None:
    config = load_yaml(SAFETY_BOUND_CONFIG)

    first = grid_values(config, 0)
    last = grid_values(config, 23)

    assert first["controller_grid_size"] == 24
    assert first["controller_bonus_nats"] == 1.5
    assert first["controller_penalty_nats"] == 0.25
    assert first["controller_max_target_mass"] == 0.35
    assert first["controller_max_kl_budget"] == 0.10
    assert last["controller_grid_size"] == 24
    assert last["controller_bonus_nats"] == 2.0
    assert last["controller_penalty_nats"] == 0.50
    assert last["controller_max_target_mass"] == 0.50
    assert last["controller_max_kl_budget"] == 0.20


def test_emit_grid_values_rejects_out_of_range_index() -> None:
    config = load_yaml(SAFETY_BOUND_CONFIG)

    with pytest.raises(ValueError, match="outside controller grid size 24"):
        grid_values(config, 24)
