from __future__ import annotations

from scripts.natural_evidence_v2.validate_r4_after_868212_repaired_first_token_event_generation_route import (
    DEFAULT_CONFIG,
    validate_route,
)
from scripts.natural_evidence_v2.validate_r4_positive_evidence_contract import load_yaml


def test_repaired_first_token_event_generation_route_passes() -> None:
    summary = validate_route(load_yaml(DEFAULT_CONFIG))
    assert summary["status"] == "PASS_R4_AFTER_868212_REPAIRED_FIRST_TOKEN_EVENT_GENERATION_ROUTE_VALIDATION_NO_SUBMIT"
    assert summary["errors"] == []
    assert summary["generation_started"] is False


def test_repaired_first_token_event_generation_route_rejects_12_coordinate_scope() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    config["generation_scope"]["selected_coordinate_count"] = 12
    summary = validate_route(config)
    assert summary["status"].startswith("FAIL_")
    assert any("selected_coordinate_count must be 16" in item for item in summary["errors"])
