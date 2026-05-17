from scripts.natural_evidence_v2.validate_r4_after_868348_global_unique_dev_diagnostic_route import (
    DEFAULT_CONFIG,
    validate_route,
)
from scripts.natural_evidence_v2.validate_r4_positive_evidence_contract import load_yaml


def test_after_868348_global_unique_dev_route_passes_plan_only() -> None:
    summary = validate_route(load_yaml(DEFAULT_CONFIG))
    assert summary["status"] == "PASS_R4_AFTER_868348_GLOBAL_UNIQUE_DEV_DIAGNOSTIC_ROUTE_PLAN_ONLY_NO_SUBMIT"
    assert summary["errors"] == []
    assert summary["slurm_allowed"] is False
    assert summary["generation_started"] is False
    assert summary["skip_allowlist_state_check"] is False


def test_after_868348_global_unique_dev_route_can_skip_runtime_allowlist_state_check() -> None:
    summary = validate_route(load_yaml(DEFAULT_CONFIG), skip_allowlist_state_check=True)
    assert summary["status"] == "PASS_R4_AFTER_868348_GLOBAL_UNIQUE_DEV_DIAGNOSTIC_ROUTE_PLAN_ONLY_NO_SUBMIT"
    assert summary["errors"] == []
    assert summary["skip_allowlist_state_check"] is True


def test_after_868348_global_unique_dev_route_rejects_duplicate_gate_relaxation() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    config["quality_gates"]["global_duplicate_response_hash_count_max"] = 1
    summary = validate_route(config)
    assert summary["status"].startswith("FAIL_")
    assert any("global_duplicate_response_hash_count_max must be 0" in error for error in summary["errors"])


def test_after_868348_global_unique_dev_route_rejects_missing_tokenizer_pass() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    config["source_artifacts"]["qwen_tokenizer_review"] = (
        "results/natural_evidence_v2/status/"
        "r4_after_868348_global_unique_qwen_tokenizer_boundary_preflight_869298/"
        "r4_prefix_native_tokenizer_boundary_preflight_summary.json"
    )
    summary = validate_route(config)
    assert summary["status"].startswith("FAIL_")
    assert any("Qwen tokenizer review status mismatch" in error for error in summary["errors"])


def test_after_868348_global_unique_dev_route_rejects_locked_scale_claim() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    config["generation_scope"]["locked_scale_claim"] = True
    summary = validate_route(config)
    assert summary["status"].startswith("FAIL_")
    assert any("generation_scope.locked_scale_claim must be false" in error for error in summary["errors"])
