from __future__ import annotations

from scripts.natural_evidence_v2.build_r4_transfer_gap_prompt_bank import (
    build_prompt_bank,
    forbidden_hits,
    repaired_prompt_text,
)
from scripts.natural_evidence_v2.validate_r4_transfer_gap_repair_plan import DEFAULT_CONFIG, load_yaml


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


def test_repaired_prompt_text_has_next_action_without_public_structure() -> None:
    text = repaired_prompt_text(SOURCE_ROW)

    assert "concrete next actions" in text
    assert "headings" in text
    assert forbidden_hits(text) == []


def test_build_prompt_bank_preserves_row_count_and_no_compute_flags() -> None:
    rows, violations = build_prompt_bank([SOURCE_ROW, {**SOURCE_ROW, "family": "planning_guidance"}], plan=load_yaml(DEFAULT_CONFIG))

    assert violations == []
    assert len(rows) == 2
    assert len({row["prompt_id"] for row in rows}) == 2
    assert all(row["generation_allowed"] is False for row in rows)
    assert all(row["training_allowed"] is False for row in rows)
    assert all(row["paper_claim_allowed"] is False for row in rows)


def test_forbidden_hits_catches_step_and_protocol_literals() -> None:
    assert "Step " in forbidden_hits("Use Step 1 as a label.")
    assert "bucket" in forbidden_hits("Mention the bucket.")
