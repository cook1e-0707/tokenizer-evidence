from scripts.natural_evidence_v2.validate_r4_after_868260_quality_repair_confirmation_route import (
    DEFAULT_CONFIG,
    validate_route,
)
from scripts.natural_evidence_v2.validate_r4_positive_evidence_contract import load_yaml


def test_after_868260_quality_repair_confirmation_route_passes_plan_only() -> None:
    summary = validate_route(load_yaml(DEFAULT_CONFIG))
    assert summary["status"] == "PASS_R4_AFTER_868260_QUALITY_REPAIR_CONFIRMATION_ROUTE_PLAN_ONLY_NO_SUBMIT"
    assert summary["errors"] == []
    assert summary["slurm_allowed"] is False
    assert summary["generation_started"] is False
    assert summary["reclassifies_868260"] is False


def test_after_868260_quality_repair_confirmation_route_rejects_duplicate_gate_relaxation() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    config["quality_gates"]["global_duplicate_response_hash_count_max"] = 1
    summary = validate_route(config)
    assert summary["status"].startswith("FAIL_")
    assert any("global_duplicate_response_hash_count_max must be 0" in error for error in summary["errors"])


def test_after_868260_quality_repair_confirmation_route_rejects_full_mode_unlock() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    config["compute_policy"]["slurm_allowed_now"] = True
    summary = validate_route(config)
    assert summary["status"].startswith("FAIL_")
    assert any("slurm_allowed_now must be false" in error for error in summary["errors"])
