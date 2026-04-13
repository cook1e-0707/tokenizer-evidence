from __future__ import annotations

from typing import Iterable, Sequence


def acceptance_rate(values: Sequence[bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def exact_match_ratio(expected: Sequence[str], recovered: Sequence[str]) -> float:
    if not expected:
        return 0.0
    matches = sum(1 for left, right in zip(expected, recovered) if left == right)
    return matches / len(expected)


def false_accept_rate(labels: Iterable[bool], scores: Iterable[float], threshold: float) -> float:
    negatives = 0
    false_accepts = 0
    for label, score in zip(labels, scores):
        if label:
            continue
        negatives += 1
        if score >= threshold:
            false_accepts += 1
    if negatives == 0:
        return 0.0
    return false_accepts / negatives
