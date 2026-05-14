from __future__ import annotations

import pytest

from scripts.natural_evidence_v2.train_wp5_micro_slot_lora import target_mass_ceiling_loss, target_mass_floor_loss


def test_target_mass_floor_zero_above_floor() -> None:
    assert target_mass_floor_loss(0.30, 0.20) == 0.0


def test_target_mass_floor_positive_below_floor() -> None:
    assert target_mass_floor_loss(0.05, 0.20) == pytest.approx(0.15)


def test_target_mass_ceiling_disabled_when_zero() -> None:
    assert target_mass_ceiling_loss(0.90, 0.0) == 0.0


def test_target_mass_ceiling_zero_below_ceiling() -> None:
    assert target_mass_ceiling_loss(0.30, 0.50) == 0.0


def test_target_mass_ceiling_positive_above_ceiling() -> None:
    assert target_mass_ceiling_loss(0.65, 0.50) == pytest.approx(0.15)
