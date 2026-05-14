from __future__ import annotations

from copy import deepcopy

from scripts.natural_evidence_v2.validate_r4_adapter_gain_sweep_plan import DEFAULT_CONFIG, load_yaml, validate_plan
from scripts.natural_evidence_v2.score_r4_surface_teacher_forced_mass import (
    gain_condition_name,
    parse_gain_values,
    scale_peft_lora_adapters,
)


def test_default_adapter_gain_sweep_plan_passes() -> None:
    summary = validate_plan(load_yaml(DEFAULT_CONFIG))

    assert summary["status"] == "PASS_R4_ADAPTER_GAIN_SWEEP_PLAN_VALIDATION"
    assert summary["row_count"] == 8192
    assert summary["generation_started"] is False
    assert summary["training_started"] is False


def test_adapter_gain_plan_rejects_base_gain_condition() -> None:
    plan = load_yaml(DEFAULT_CONFIG)
    plan["conditions"] = list(plan["conditions"]) + ["base_gain_2_0"]
    summary = validate_plan(plan)

    assert summary["status"].startswith("FAIL")
    assert any("base/task_only gain" in error for error in summary["errors"])


def test_adapter_gain_plan_rejects_generation_unlock() -> None:
    plan = load_yaml(DEFAULT_CONFIG)
    plan["generation_allowed"] = True
    summary = validate_plan(plan)

    assert summary["status"].startswith("FAIL")
    assert "generation_allowed must be false" in summary["errors"]


def test_parse_gain_values_requires_unique_nonnegative_values() -> None:
    assert parse_gain_values("0.5,1.0,2") == [0.5, 1.0, 2.0]

    import pytest

    with pytest.raises(ValueError, match="non-negative"):
        parse_gain_values("-1.0")
    with pytest.raises(ValueError, match="unique"):
        parse_gain_values("1.0,1.0")


def test_gain_condition_name_preserves_historical_single_gain() -> None:
    assert gain_condition_name(1.0, historical_single_gain=True) == "protected"
    assert gain_condition_name(1.0, historical_single_gain=False) == "protected_gain_1"
    assert gain_condition_name(0.5, historical_single_gain=False) == "protected_gain_0_5"


def test_scale_peft_lora_adapters_multiplies_scaling_tables() -> None:
    class Leaf:
        def __init__(self) -> None:
            self.scaling = {"default": 2.0}

    class Dummy:
        def __init__(self) -> None:
            self.leaf = Leaf()

        def modules(self):
            return [self, self.leaf]

    dummy = Dummy()
    summary = scale_peft_lora_adapters(dummy, 1.5)

    assert dummy.leaf.scaling["default"] == 3.0
    assert summary["lora_scaling_keys_touched"] == 1
