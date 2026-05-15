from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from scripts.natural_evidence_v2.score_r4_surface_teacher_forced_mass import (
    ControllerConfig,
    condition_plan,
    controller_applies_to_condition,
    controller_config_from_args,
    controller_token_ids_for_policy,
    deterministic_wrong_key_bit,
    score_next_token_surface_masses,
)


def namespace(**overrides):
    values = {
        "controller_mode": "disabled",
        "controller_bonus_nats": 0.0,
        "controller_penalty_nats": 0.0,
        "controller_max_target_mass": None,
        "controller_max_kl_budget": None,
        "controller_condition_set": "standard",
        "protected_adapter": None,
        "task_only_adapter": None,
        "protected_adapter_gains": "1.0",
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
    assert controller_applies_to_condition("controlled_protected", config) is True
    assert controller_applies_to_condition("wrong_key_controlled", config) is True
    assert controller_applies_to_condition("wrong_payload_controlled", config) is True
    assert controller_applies_to_condition("base", config) is False
    assert controller_applies_to_condition("task_only", config) is False


def test_disabled_controller_does_not_apply_to_protected() -> None:
    assert controller_applies_to_condition("protected", ControllerConfig()) is False


def test_pressure_control_condition_plan_has_reviewed_arms() -> None:
    plan = condition_plan(
        namespace(
            controller_condition_set="pressure_controls",
            controller_mode="additive",
            controller_bonus_nats=0.5,
            protected_adapter=Path("protected_adapter"),
            task_only_adapter=Path("task_only_adapter"),
        )
    )

    assert [condition for condition, _, _ in plan] == [
        "base",
        "task_only",
        "controlled_protected",
        "wrong_key_controlled",
        "wrong_payload_controlled",
    ]


def test_pressure_control_condition_plan_requires_enabled_controller() -> None:
    with pytest.raises(ValueError, match="controller-mode"):
        condition_plan(
            namespace(
                controller_condition_set="pressure_controls",
                protected_adapter=Path("protected_adapter"),
                task_only_adapter=Path("task_only_adapter"),
            )
        )


def test_wrong_payload_controller_policy_uses_complement_tokens() -> None:
    result = controller_token_ids_for_policy(
        policy="complement",
        row={"target_bit": 1},
        committed_target_ids=[10],
        committed_other_ids=[20, 21],
    )

    assert result["controller_target_ids"] == [20, 21]
    assert result["controller_other_ids"] == [10]
    assert result["controller_policy_detail"]["policy"] == "complement"


def test_wrong_key_controller_policy_is_deterministic_and_precommitted() -> None:
    row = {"prompt_id": "p1", "prompt_index": 7, "coordinate_id": 3, "target_bit": 1}
    first = deterministic_wrong_key_bit(row)
    second = deterministic_wrong_key_bit(dict(row))
    result = controller_token_ids_for_policy(
        policy="coordinate_hash_v1",
        row=row,
        committed_target_ids=[10],
        committed_other_ids=[20],
    )

    assert first == second
    assert result["controller_policy_detail"]["wrong_key_bit"] == first
    assert result["controller_policy_detail"]["policy"] == "coordinate_hash_v1"
    assert result["controller_target_ids"] in ([10], [20])


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
