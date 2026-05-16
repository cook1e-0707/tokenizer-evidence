from __future__ import annotations

from scripts.natural_evidence_v2.validate_r4_after_864117_pivot_package import DEFAULT_CONFIG, validate_package
from scripts.natural_evidence_v2.validate_r4_positive_evidence_contract import load_yaml


def test_after_864117_pivot_package_passes_static_validation() -> None:
    summary = validate_package(load_yaml(DEFAULT_CONFIG))

    assert summary["status"] == "PASS_R4_AFTER_864117_PIVOT_PACKAGE_STATIC_VALIDATION_NO_COMPUTE"
    assert summary["selected_next_route"] == "metric_exact_objective_repair"
    assert summary["source_failed_jobs"] == ["859672", "863274", "864117"]
    assert summary["current_compute_unlocked"] is False
    assert summary["slurm_job_submitted"] is False
    assert summary["generation_started"] is False
    assert summary["training_started"] is False


def test_after_864117_pivot_requires_864117_review() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    config["source_reviews"] = config["source_reviews"][:-1]

    summary = validate_package(config)

    assert summary["status"].startswith("FAIL")
    assert "source_reviews job set mismatch: ['859672', '863274']" in summary["errors"]


def test_after_864117_pivot_rejects_more_scalar_controller_grid() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    config["controller_route_decision"]["no_additional_scalar_controller_grid_without_new_design"] = False

    summary = validate_package(config)

    assert summary["status"].startswith("FAIL")
    assert (
        "controller_route_decision.no_additional_scalar_controller_grid_without_new_design must be true"
        in summary["errors"]
    )


def test_after_864117_pivot_rejects_generation_unlock() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    config["current_permissions"]["generation_allowed"] = True

    summary = validate_package(config)

    assert summary["status"].startswith("FAIL")
    assert "current_permissions.generation_allowed must be false" in summary["errors"]


def test_after_864117_pivot_requires_metric_exact_route() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    config["controller_route_decision"]["selected_next_route"] = "row_adaptive_controller"

    summary = validate_package(config)

    assert summary["status"].startswith("FAIL")
    assert "controller_route_decision.selected_next_route must be metric_exact_objective_repair" in summary["errors"]
