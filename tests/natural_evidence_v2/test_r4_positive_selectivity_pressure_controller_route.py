from __future__ import annotations

from pathlib import Path

from scripts.natural_evidence_v2.validate_r4_positive_evidence_contract import load_yaml
from scripts.natural_evidence_v2.validate_r4_positive_selectivity_pressure_controller_route import (
    DEFAULT_CONFIG,
    validate_route,
)

CONTROLLER_ONLY_CONFIG = Path("configs/natural_evidence_v2/r4_positive_selectivity_controller_only_route.yaml")


def test_pressure_controller_route_plan_passes() -> None:
    summary = validate_route(load_yaml(DEFAULT_CONFIG))

    assert summary["status"] == "PASS_R4_POSITIVE_SELECTIVITY_PRESSURE_CONTROLLER_ROUTE_PLAN_NO_COMPUTE"
    assert summary["score_row_count"] == 8192
    assert summary["current_compute_unlocked"] is False
    assert summary["model_scoring_started"] is False
    assert summary["generation_started"] is False


def test_controller_only_route_plan_passes() -> None:
    summary = validate_route(load_yaml(CONTROLLER_ONLY_CONFIG))

    assert summary["status"] == "PASS_R4_POSITIVE_SELECTIVITY_PRESSURE_CONTROLLER_ROUTE_PLAN_NO_COMPUTE"
    assert summary["score_row_count"] == 8192
    assert summary["current_compute_unlocked"] is False
    assert summary["model_scoring_started"] is False
    assert summary["generation_started"] is False


def test_pressure_controller_route_requires_scorer_integration_before_slurm() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    config["scorer_integration_required_before_slurm"] = False

    summary = validate_route(config)

    assert summary["status"].startswith("FAIL")
    assert "scorer_integration_required_before_slurm must be true" in summary["errors"]


def test_pressure_controller_route_rejects_generation_unlock() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    config["current_permissions"]["generation_allowed"] = True

    summary = validate_route(config)

    assert summary["status"].startswith("FAIL")
    assert "current_permissions.generation_allowed must be false" in summary["errors"]


def test_pressure_controller_route_caps_target_mass_grid() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    config["controller_grid"]["max_target_mass"] = [0.25, 0.60]

    summary = validate_route(config)

    assert summary["status"].startswith("FAIL")
    assert "controller_grid.max_target_mass must not exceed 0.50" in summary["errors"]


def test_pressure_controller_route_rejects_posthoc_wrong_key_mapping() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    config["controller_control_mapping"]["wrong_key_controlled_policy"] = "posthoc_transcript_search"

    summary = validate_route(config)

    assert summary["status"].startswith("FAIL")
    assert (
        "controller_control_mapping.wrong_key_controlled_policy must be coordinate_hash_v1"
        in summary["errors"]
    )


def test_controller_only_route_rejects_protected_adapter_controller_arms() -> None:
    config = load_yaml(CONTROLLER_ONLY_CONFIG)
    config["controller_control_mapping"]["protected_adapter_loaded_for_controller_arms"] = True

    summary = validate_route(config)

    assert summary["status"].startswith("FAIL")
    assert (
        "controller_control_mapping.protected_adapter_loaded_for_controller_arms must be false"
        in summary["errors"]
    )
