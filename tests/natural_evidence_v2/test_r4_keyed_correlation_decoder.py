from __future__ import annotations

import pytest

from scripts.natural_evidence_v2.r4_keyed_correlation_decoder import (
    SurfaceEvent,
    build_toy_aligned_events,
    decide_keyed_correlation,
    map_surface_to_coordinate_and_polarity,
    score_events,
)


SURFACES = [f"surface_{index:03d}" for index in range(512)]


def test_mapping_is_deterministic_and_payload_specific() -> None:
    first = map_surface_to_coordinate_and_polarity(
        audit_key="key-a",
        payload_id="payload-a",
        surface_id="surface-1",
    )
    second = map_surface_to_coordinate_and_polarity(
        audit_key="key-a",
        payload_id="payload-a",
        surface_id="surface-1",
    )
    other_payload = map_surface_to_coordinate_and_polarity(
        audit_key="key-a",
        payload_id="payload-b",
        surface_id="surface-1",
    )

    assert first == second
    assert first != other_payload


def test_keyed_correlation_accepts_aligned_events_and_rejects_controls() -> None:
    events = build_toy_aligned_events(
        audit_key="correct-key",
        payload_id="payload-a",
        surface_ids=SURFACES,
    )[:28]

    decision = decide_keyed_correlation(
        events,
        audit_key="correct-key",
        payload_id="payload-a",
        wrong_audit_key="wrong-key",
        wrong_payload_id="payload-b",
        min_observed_events=24,
        min_distinct_coordinates=20,
        min_keyed_correlation_score=6.0,
        min_specificity_margin=3.0,
        max_wrong_score=8.0,
    )

    assert decision.accept is True
    assert decision.keyed_correlation_score >= 24
    assert decision.specificity_margin >= 3


def test_support_is_not_acceptance_for_structural_events_only() -> None:
    events = [SurfaceEvent(surface_id=f"struct_{i}", event_type="bullet_marker") for i in range(64)]

    decision = decide_keyed_correlation(
        events,
        audit_key="correct-key",
        payload_id="payload-a",
        wrong_audit_key="wrong-key",
        wrong_payload_id="payload-b",
        min_observed_events=1,
        min_distinct_coordinates=1,
    )

    assert decision.observed_events == 0
    assert decision.accept is False


def test_reusable_unaligned_support_does_not_pass_specificity_gate() -> None:
    events = [SurfaceEvent(surface_id=f"surface_{index:03d}") for index in range(40)]
    score, observed, distinct = score_events(events, audit_key="correct-key", payload_id="payload-a")
    decision = decide_keyed_correlation(
        events,
        audit_key="correct-key",
        payload_id="payload-a",
        wrong_audit_key="wrong-key",
        wrong_payload_id="payload-b",
        min_observed_events=24,
        min_distinct_coordinates=10,
        min_keyed_correlation_score=max(score + 1.0, 6.0),
        min_specificity_margin=3.0,
        max_wrong_score=100.0,
    )

    assert observed == 40
    assert distinct >= 10
    assert decision.accept is False


def test_mapping_requires_nonempty_identifiers() -> None:
    with pytest.raises(ValueError, match="required"):
        map_surface_to_coordinate_and_polarity(audit_key="", payload_id="payload", surface_id="surface")
