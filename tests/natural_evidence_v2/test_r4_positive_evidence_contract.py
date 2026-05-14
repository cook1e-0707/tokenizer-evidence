from __future__ import annotations

from scripts.natural_evidence_v2.validate_r4_positive_evidence_contract import DEFAULT_CONFIG, load_yaml, validate_contract


def test_default_positive_evidence_contract_static_validation_passes() -> None:
    summary = validate_contract(load_yaml(DEFAULT_CONFIG))

    assert summary["status"] == "PASS_R4_POSITIVE_EVIDENCE_CONTRACT_STATIC_VALIDATION_NO_COMPUTE"
    assert summary["contract_id"] == "r4_keyed_correlation_evidence_v1"
    assert summary["current_compute_unlocked"] is False
    assert summary["generation_started"] is False
    assert summary["training_started"] is False


def test_contract_rejects_generation_unlock_before_route_review() -> None:
    contract = load_yaml(DEFAULT_CONFIG)
    contract["current_permissions"]["generation_allowed"] = True

    summary = validate_contract(contract)

    assert summary["status"].startswith("FAIL")
    assert "current_permissions.generation_allowed must be false before compute route review" in summary["errors"]


def test_contract_requires_key_payload_specificity_before_accept() -> None:
    contract = load_yaml(DEFAULT_CONFIG)
    contract["key_payload_specificity_policy"]["required_before_accept"][
        "protected_minus_best_wrong_specificity_margin_min"
    ] = 1.0

    summary = validate_contract(contract)

    assert summary["status"].startswith("FAIL")
    assert "specificity margin before accept must be >= 3.0" in summary["errors"]


def test_contract_rejects_structure_dependent_decoder() -> None:
    contract = load_yaml(DEFAULT_CONFIG)
    contract["structural_leakage_policy"]["bullet_count_used_by_decoder"] = True

    summary = validate_contract(contract)

    assert summary["status"].startswith("FAIL")
    assert "structural_leakage_policy.bullet_count_used_by_decoder must be false" in summary["errors"]


def test_contract_requires_wrong_key_and_wrong_payload_controls() -> None:
    contract = load_yaml(DEFAULT_CONFIG)
    contract["pre_registered_dev_gate"]["conditions"].remove("wrong_payload")

    summary = validate_contract(contract)

    assert summary["status"].startswith("FAIL")
    assert "pre_registered_dev_gate.conditions mismatch" in summary["errors"]
