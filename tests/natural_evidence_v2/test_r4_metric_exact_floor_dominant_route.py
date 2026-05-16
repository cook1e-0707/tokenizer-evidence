from __future__ import annotations

from scripts.natural_evidence_v2.validate_r4_metric_exact_floor_dominant_route import DEFAULT_CONFIG, validate_route
from scripts.natural_evidence_v2.validate_r4_positive_evidence_contract import load_yaml


def test_floor_dominant_route_passes_static_validation() -> None:
    summary = validate_route(load_yaml(DEFAULT_CONFIG))

    assert summary["status"] == "PASS_R4_METRIC_EXACT_FLOOR_DOMINANT_ROUTE_STATIC_VALIDATION_NO_COMPUTE"
    assert summary["source_job_id"] == "864332"
    assert summary["selected_repair"] == "floor_dominant_metric_exact_micro_overfit"
    assert summary["allowlist_entry"] == "v2_r4_candidate_v3_floor_dominant_micro_overfit_h200"
    assert summary["current_compute_unlocked"] is False
    assert summary["slurm_job_submitted"] is False
    assert summary["training_started"] is False
    assert summary["generation_started"] is False


def test_floor_dominant_route_requires_task_ce_disabled() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    config["route_parameters"]["task_ce_weight"] = 1.0

    summary = validate_route(config)

    assert summary["status"].startswith("FAIL")
    assert "route_parameters.task_ce_weight must be 0.0" in summary["errors"]


def test_floor_dominant_route_requires_strong_floor_lambda() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    config["route_parameters"]["target_mass_floor_lambda"] = 5.0

    summary = validate_route(config)

    assert summary["status"].startswith("FAIL")
    assert "route_parameters.target_mass_floor_lambda must be >= 50" in summary["errors"]


def test_floor_dominant_route_rejects_generation_unlock() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    config["not_unlocked_by_this_route_package"]["generation"] = False

    summary = validate_route(config)

    assert summary["status"].startswith("FAIL")
    assert "not_unlocked_by_this_route_package.generation must be true" in summary["errors"]
