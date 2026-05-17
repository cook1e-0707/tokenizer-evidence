from scripts.natural_evidence_v2.plan_r4_after_868299_first_token_event_dev_diagnostic import (
    DEFAULT_BASE_ALLOCATION,
    build_cyclic_allocation,
)
from scripts.natural_evidence_v2.r4_cover_natural_common import read_jsonl
from scripts.natural_evidence_v2.validate_r4_after_868299_first_token_event_dev_diagnostic_route import (
    DEFAULT_CONFIG,
    validate_route,
)
from scripts.natural_evidence_v2.validate_r4_positive_evidence_contract import load_yaml


def test_after_868299_first_token_event_dev_route_passes_plan_only() -> None:
    summary = validate_route(load_yaml(DEFAULT_CONFIG))
    assert summary["status"] == "PASS_R4_AFTER_868299_FIRST_TOKEN_EVENT_DEV_DIAGNOSTIC_ROUTE_PLAN_ONLY_NO_SUBMIT"
    assert summary["errors"] == []
    assert summary["slurm_allowed"] is False
    assert summary["generation_started"] is False
    assert summary["skip_allowlist_state_check"] is False


def test_after_868299_first_token_event_dev_route_can_skip_runtime_allowlist_state_check() -> None:
    summary = validate_route(load_yaml(DEFAULT_CONFIG), skip_allowlist_state_check=True)
    assert summary["status"] == "PASS_R4_AFTER_868299_FIRST_TOKEN_EVENT_DEV_DIAGNOSTIC_ROUTE_PLAN_ONLY_NO_SUBMIT"
    assert summary["errors"] == []
    assert summary["skip_allowlist_state_check"] is True


def test_after_868299_first_token_event_dev_route_rejects_duplicate_gate_relaxation() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    config["quality_gates"]["global_duplicate_response_hash_count_max"] = 1
    summary = validate_route(config)
    assert summary["status"].startswith("FAIL_")
    assert any("global_duplicate_response_hash_count_max must be 0" in error for error in summary["errors"])


def test_after_868299_first_token_event_dev_route_rejects_locked_scale_claim() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    config["generation_scope"]["locked_scale_claim"] = True
    summary = validate_route(config)
    assert summary["status"].startswith("FAIL_")
    assert any("generation_scope.locked_scale_claim must be false" in error for error in summary["errors"])


def test_after_868299_cyclic_allocation_preserves_within_shard_uniqueness() -> None:
    rows = read_jsonl(DEFAULT_BASE_ALLOCATION)
    allocation_rows, manifest = build_cyclic_allocation(rows, target_shards=32, base_shards=4)
    assert manifest["status"] == "PASS_R4_AFTER_868299_FIRST_TOKEN_EVENT_DEV_DIAGNOSTIC_ALLOCATION_PLAN_NO_SUBMIT"
    assert manifest["total_rows"] == 32768
    assert manifest["within_shard_duplicate_prompt_prefix_pair_count_max"] == 0
    assert len(allocation_rows) == 32768
    assert {row["assigned_shard_index"] for row in allocation_rows} == set(range(32))
