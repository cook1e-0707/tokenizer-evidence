from __future__ import annotations

from scripts.natural_evidence_v2.validate_r4_positive_evidence_contract import load_yaml
from scripts.natural_evidence_v2.validate_r4_positive_selectivity_dev_diagnostic_route import (
    DEFAULT_CONFIG,
    validate_route,
)


def test_selectivity_dev_diagnostic_route_scope_passes() -> None:
    summary = validate_route(load_yaml(DEFAULT_CONFIG))

    assert summary["status"] == "PASS_R4_POSITIVE_SELECTIVITY_DEV_DIAGNOSTIC_ROUTE_SCOPE_REVIEW_NO_SUBMIT"
    assert summary["current_compute_unlocked"] is False
    assert summary["slurm_job_submitted"] is False
    assert summary["allowlist_enabled"] is False
    assert summary["generation_started"] is False


def test_selectivity_route_scope_rejects_generation_unlock() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    config["current_permissions"]["generation_allowed"] = True

    summary = validate_route(config)

    assert summary["status"].startswith("FAIL")
    assert "current_permissions.generation_allowed must be false in route-scope review" in summary["errors"]


def test_selectivity_route_scope_requires_prompt_hash_match() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    config["expected_prompt_bank_hash"] = "bad-hash"

    summary = validate_route(config)

    assert summary["status"].startswith("FAIL")
    assert "expected_prompt_bank_hash does not match prompt policy package" in summary["errors"]


def test_selectivity_route_scope_requires_h200_pomplun_policy() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    config["future_route_scope"]["gpu"] = "a100"

    summary = validate_route(config)

    assert summary["status"].startswith("FAIL")
    assert "future route GPU must be H200" in summary["errors"]
