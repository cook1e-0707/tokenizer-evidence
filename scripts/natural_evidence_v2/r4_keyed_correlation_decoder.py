from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class SurfaceEvent:
    surface_id: str
    weight: float = 1.0
    event_type: str = "normalized_phrase_event"


@dataclass(frozen=True)
class CorrelationDecision:
    accept: bool
    keyed_correlation_score: float
    wrong_key_correlation_score: float
    wrong_payload_correlation_score: float
    specificity_margin: float
    observed_events: int
    distinct_coordinates: int


IGNORED_EVENT_TYPES = {
    "bullet_marker",
    "line_number",
    "heading_text",
    "colon_after_label",
    "next_action_label",
}


def _digest(*parts: str) -> bytes:
    key = parts[0].encode("utf-8")
    message = "\x1f".join(parts[1:]).encode("utf-8")
    return hmac.new(key, message, hashlib.sha256).digest()


def map_surface_to_coordinate_and_polarity(
    *,
    audit_key: str,
    payload_id: str,
    surface_id: str,
    coordinate_count: int = 32,
) -> tuple[int, int]:
    if coordinate_count <= 0:
        raise ValueError("coordinate_count must be positive")
    if not audit_key or not payload_id or not surface_id:
        raise ValueError("audit_key, payload_id, and surface_id are required")
    payload = _digest(audit_key, payload_id, surface_id)
    coordinate = int.from_bytes(payload[:8], "big") % coordinate_count
    polarity = 1 if payload[8] & 1 else -1
    return coordinate, polarity


def score_events(
    events: Iterable[SurfaceEvent | Mapping[str, object]],
    *,
    audit_key: str,
    payload_id: str,
    coordinate_count: int = 32,
) -> tuple[float, int, int]:
    score = 0.0
    observed = 0
    coordinates: set[int] = set()
    for event in events:
        if isinstance(event, SurfaceEvent):
            surface_id = event.surface_id
            weight = event.weight
            event_type = event.event_type
        else:
            surface_id = str(event.get("surface_id", ""))
            weight = float(event.get("weight", 1.0))
            event_type = str(event.get("event_type", "normalized_phrase_event"))
        if event_type in IGNORED_EVENT_TYPES:
            continue
        coordinate, polarity = map_surface_to_coordinate_and_polarity(
            audit_key=audit_key,
            payload_id=payload_id,
            surface_id=surface_id,
            coordinate_count=coordinate_count,
        )
        score += float(weight) * polarity
        observed += 1
        coordinates.add(coordinate)
    return score, observed, len(coordinates)


def decide_keyed_correlation(
    events: Iterable[SurfaceEvent | Mapping[str, object]],
    *,
    audit_key: str,
    payload_id: str,
    wrong_audit_key: str,
    wrong_payload_id: str,
    coordinate_count: int = 32,
    min_observed_events: int = 24,
    min_distinct_coordinates: int = 20,
    min_keyed_correlation_score: float = 6.0,
    min_specificity_margin: float = 3.0,
    max_wrong_score: float = 1.5,
) -> CorrelationDecision:
    materialized = list(events)
    keyed_score, observed, distinct = score_events(
        materialized,
        audit_key=audit_key,
        payload_id=payload_id,
        coordinate_count=coordinate_count,
    )
    wrong_key_score, _, _ = score_events(
        materialized,
        audit_key=wrong_audit_key,
        payload_id=payload_id,
        coordinate_count=coordinate_count,
    )
    wrong_payload_score, _, _ = score_events(
        materialized,
        audit_key=audit_key,
        payload_id=wrong_payload_id,
        coordinate_count=coordinate_count,
    )
    best_wrong = max(wrong_key_score, wrong_payload_score)
    specificity_margin = keyed_score - best_wrong
    accept = (
        observed >= min_observed_events
        and distinct >= min_distinct_coordinates
        and keyed_score >= min_keyed_correlation_score
        and wrong_key_score <= max_wrong_score
        and wrong_payload_score <= max_wrong_score
        and specificity_margin >= min_specificity_margin
    )
    return CorrelationDecision(
        accept=accept,
        keyed_correlation_score=keyed_score,
        wrong_key_correlation_score=wrong_key_score,
        wrong_payload_correlation_score=wrong_payload_score,
        specificity_margin=specificity_margin,
        observed_events=observed,
        distinct_coordinates=distinct,
    )


def build_toy_aligned_events(
    *,
    audit_key: str,
    payload_id: str,
    surface_ids: Iterable[str],
    coordinate_count: int = 32,
    target_polarity: int = 1,
) -> list[SurfaceEvent]:
    if target_polarity not in {-1, 1}:
        raise ValueError("target_polarity must be -1 or 1")
    events: list[SurfaceEvent] = []
    seen_coordinates: set[int] = set()
    for surface_id in surface_ids:
        coordinate, polarity = map_surface_to_coordinate_and_polarity(
            audit_key=audit_key,
            payload_id=payload_id,
            surface_id=surface_id,
            coordinate_count=coordinate_count,
        )
        if polarity == target_polarity and coordinate not in seen_coordinates:
            events.append(SurfaceEvent(surface_id=surface_id))
            seen_coordinates.add(coordinate)
    return events
