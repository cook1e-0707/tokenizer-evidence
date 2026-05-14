from __future__ import annotations

import pytest

from scripts.natural_evidence_v2.r4_prefix_native_soft_logit_controller import apply_controller, softmax


def test_disabled_controller_returns_identical_logits() -> None:
    logits = [0.0, 0.1, 0.2, 0.3]
    result = apply_controller(logits, [1], [2], bonus=2.0, mode="disabled")

    assert result["controlled_logits"] == logits
    assert result["kl_to_base"] == 0.0


def test_target_mass_increases_monotonically_with_bonus() -> None:
    logits = [0.0, 0.0, 0.0, 0.0]
    small = apply_controller(logits, [1], [2], bonus=0.25)
    large = apply_controller(logits, [1], [2], bonus=1.0)

    assert large["target_mass"] > small["target_mass"]
    assert large["other_mass"] < small["other_mass"]


def test_overlap_fails_closed() -> None:
    with pytest.raises(ValueError, match="overlap"):
        apply_controller([0.0, 0.0, 0.0], [1], [1], bonus=1.0)


def test_target_mass_cap_prevents_collapse() -> None:
    result = apply_controller(
        [0.0, 0.0, 0.0, 0.0],
        [1],
        [2],
        bonus=10.0,
        max_target_mass=0.40,
    )

    assert result["target_mass"] <= 0.400000001
    assert result["scale"] < 1.0


def test_kl_budget_cap_prevents_over_forcing() -> None:
    result = apply_controller(
        [0.0, 0.0, 0.0, 0.0],
        [1],
        [2],
        bonus=10.0,
        max_kl_budget=0.01,
    )

    assert result["kl_to_base"] <= 0.010000001
    assert result["scale"] < 1.0


def test_wrong_key_target_sets_differ() -> None:
    logits = [0.0, 0.0, 0.0, 0.0]
    right = apply_controller(logits, [1], [2], bonus=1.0)
    wrong = apply_controller(logits, [3], [2], bonus=1.0)

    assert right["controlled_logits"] != wrong["controlled_logits"]
