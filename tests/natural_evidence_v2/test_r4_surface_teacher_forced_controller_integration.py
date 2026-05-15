from __future__ import annotations

from argparse import Namespace

import pytest

from scripts.natural_evidence_v2.score_r4_surface_teacher_forced_mass import (
    ControllerConfig,
    controller_applies_to_condition,
    controller_config_from_args,
    score_next_token_surface_masses,
)


def namespace(**overrides):
    values = {
        "controller_mode": "disabled",
        "controller_bonus_nats": 0.0,
        "controller_penalty_nats": 0.0,
        "controller_max_target_mass": None,
        "controller_max_kl_budget": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_controller_config_disabled_by_default() -> None:
    config = controller_config_from_args(namespace())

    assert config == ControllerConfig()
    assert config.enabled is False
    assert config.to_json()["mode"] == "disabled"


def test_controller_config_rejects_enabled_without_positive_bonus() -> None:
    with pytest.raises(ValueError, match="bonus"):
        controller_config_from_args(namespace(controller_mode="additive", controller_bonus_nats=0.0))


def test_controller_config_rejects_unsafe_caps() -> None:
    with pytest.raises(ValueError, match="max-target-mass"):
        controller_config_from_args(
            namespace(controller_mode="additive", controller_bonus_nats=1.0, controller_max_target_mass=0.75)
        )
    with pytest.raises(ValueError, match="max-kl-budget"):
        controller_config_from_args(
            namespace(controller_mode="additive", controller_bonus_nats=1.0, controller_max_kl_budget=0.50)
        )


def test_controller_only_applies_to_protected_conditions() -> None:
    config = ControllerConfig(mode="additive", bonus_nats=1.0)

    assert controller_applies_to_condition("protected", config) is True
    assert controller_applies_to_condition("protected_gain_2", config) is True
    assert controller_applies_to_condition("base", config) is False
    assert controller_applies_to_condition("task_only", config) is False


def test_disabled_controller_does_not_apply_to_protected() -> None:
    assert controller_applies_to_condition("protected", ControllerConfig()) is False


def test_torch_mass_helper_increases_target_mass_with_controller() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor([0.0, 0.0, 0.0, 0.0])

    base = score_next_token_surface_masses(logits=logits, target_ids=[1], other_ids=[2])
    controlled = score_next_token_surface_masses(
        logits=logits,
        target_ids=[1],
        other_ids=[2],
        controller_config=ControllerConfig(mode="additive", bonus_nats=1.0, penalty_nats=0.25),
    )

    assert controlled["controller_applied"] is True
    assert controlled["target_mass"] > base["target_mass"]
    assert controlled["other_mass"] < base["other_mass"]


def test_torch_mass_helper_rejects_target_other_overlap() -> None:
    torch = pytest.importorskip("torch")

    with pytest.raises(ValueError, match="overlap"):
        score_next_token_surface_masses(logits=torch.tensor([0.0, 0.0, 0.0]), target_ids=[1], other_ids=[1])
