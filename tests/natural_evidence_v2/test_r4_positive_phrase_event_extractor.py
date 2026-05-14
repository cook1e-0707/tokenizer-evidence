from __future__ import annotations

from scripts.natural_evidence_v2.extract_r4_positive_phrase_events import (
    extract_phrase_events,
    load_surface_bank,
    normalize_text,
)


SURFACE_BANK = load_surface_bank()


def _surface_ids(text: str, scrub_mode: str = "all") -> set[str]:
    return {event["surface_id"] for event in extract_phrase_events(text, SURFACE_BANK, scrub_mode=scrub_mode)}


def test_normalize_text_strips_structural_labels() -> None:
    normalized = normalize_text("- Next action: Make a short plan.", scrub_structure=True)

    assert "next action" not in normalized
    assert normalized == "make a short plan"


def test_extractor_matches_phrase_after_format_scrub() -> None:
    surface_ids = _surface_ids("1. Ask a focused question before choosing anything.")

    assert "r4pe_clarify_00" in surface_ids


def test_extractor_does_not_depend_on_bullet_or_numbering() -> None:
    plain = _surface_ids("Ask a focused question before choosing anything.")
    numbered = _surface_ids("1) Ask a focused question before choosing anything.")
    bulleted = _surface_ids("- Ask a focused question before choosing anything.")

    assert plain == numbered == bulleted


def test_extractor_uses_word_boundaries() -> None:
    surface_ids = _surface_ids("The team will reask a focused question later.")

    assert "r4pe_clarify_00" not in surface_ids


def test_extractor_can_match_multiple_families() -> None:
    surface_ids = _surface_ids(
        "Use a calm tone, compare with a known case, and choose the lower risk option."
    )

    assert "r4pe_communicate_01" in surface_ids
    assert "r4pe_troubleshooting_04" in surface_ids
    assert "r4pe_safety_11" in surface_ids
