from __future__ import annotations

from scripts.natural_evidence_v2.build_r4_positive_support_repair_package import build_event_window_bank
from scripts.natural_evidence_v2.extract_r4_positive_support_window_events import (
    extract_support_window_events,
    normalize_text,
    tokenize_for_events,
)


CONFIG = {
    "event_window_policy": {
        "source_rule_id": "test",
        "event_type": "support_window_event",
        "normalization_rule": "lowercase_strip_punctuation_collapse_space_simple_stem",
        "max_token_distance": 8,
        "forbidden_public_technical_literals": [
            "fingerprint",
            "watermark",
            "payload",
            "secret key",
            "decoder",
            "hidden signal",
            "evidence block",
            "bucket",
            "coordinate",
        ],
    }
}


BANK = build_event_window_bank(CONFIG)


def _ids(text: str) -> set[str]:
    return {event["surface_id"] for event in extract_support_window_events(text, BANK, scrub_mode="all")}


def test_support_window_extractor_strips_public_structure() -> None:
    normalized = normalize_text("- Next action: Confirm the main constraint.", scrub_structure=True)

    assert "next action" not in normalized
    assert "confirm" in normalized


def test_support_window_matches_verb_and_cue_within_window() -> None:
    surface_ids = _ids("Confirm the current constraint before you make the plan.")

    assert "r4sr_clarify_01_01" in surface_ids


def test_support_window_rejects_verb_without_cue() -> None:
    surface_ids = _ids("Confirm everything before continuing.")

    assert "r4sr_clarify_01_01" not in surface_ids


def test_support_window_rejects_cue_too_far_from_verb() -> None:
    surface_ids = _ids("Confirm the owner, timeline, venue, supplies, staffing, and budget before the constraint changes.")

    assert "r4sr_clarify_01_01" not in surface_ids


def test_tokenizer_for_events_stems_simple_inflections() -> None:
    assert "confirm" in tokenize_for_events("confirmed constraints")
    assert "constraint" in tokenize_for_events("confirmed constraints")
