from __future__ import annotations

from pathlib import Path

import yaml

from scripts.natural_evidence_v2.build_r4_positive_selectivity_prompt_policy import (
    build_fixture_rows,
    build_prompt_rows,
    elicited_prompt_text,
    forbidden_hits,
)
from scripts.natural_evidence_v2.extract_r4_positive_support_window_events import load_event_window_bank


CONFIG_PATH = Path("configs/natural_evidence_v2/r4_positive_selectivity_prompt_policy.yaml")
EVENT_BANK = Path("results/natural_evidence_v2/precommit/r4_positive_selectivity_repair_package_20260514_2158/event_window_bank.json")

SOURCE_ROW = {
    "angle": "common mistakes",
    "audience": "a new team",
    "constraint": "keeping the tone calm",
    "domain": "volunteer coordination",
    "family": "practical_advice_short",
    "prompt_id": "source_prompt",
    "prompt_text_sha256": "abc",
    "split": "dev",
}


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_elicited_prompt_has_semantic_cues_without_public_protocol_terms() -> None:
    config = load_config()
    family = config["elicitation_policy"]["families"][0]
    text = elicited_prompt_text(SOURCE_ROW, family)

    assert "constraints" in text
    assert "tradeoffs" in text
    assert forbidden_hits(text, config["forbidden_public_terms"]) == []


def test_build_prompt_rows_preserves_no_compute_flags_and_balances_families() -> None:
    config = load_config()
    source_rows = [{**SOURCE_ROW, "prompt_id": f"source_{idx}", "family": "planning_guidance"} for idx in range(12)]
    rows, violations, family_counts = build_prompt_rows(source_rows, config=config)

    assert violations == []
    assert len(rows) == 12
    assert len({row["prompt_id"] for row in rows}) == 12
    assert max(family_counts.values()) / len(rows) <= config["elicitation_policy"]["max_policy_family_fraction"]
    assert all(row["generation_allowed"] is False for row in rows)
    assert all(row["model_scoring_allowed"] is False for row in rows)
    assert all(row["paper_claim_allowed"] is False for row in rows)


def test_fixture_rows_cover_selectivity_event_families() -> None:
    config = load_config()
    event_bank = load_event_window_bank(EVENT_BANK)
    rows, summary = build_fixture_rows(config, event_bank)

    assert len(rows) == len(config["elicitation_policy"]["families"])
    assert summary["total_fixture_events"] >= config["static_validation"]["min_total_fixture_events"]
    assert all(
        count >= config["static_validation"]["min_fixture_events_per_family"]
        for count in summary["family_event_counts"].values()
    )
