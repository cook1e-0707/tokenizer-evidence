from __future__ import annotations

from typing import Any, Mapping

from scripts.natural_evidence_v1.build_bucket_bank import _entry_from_record


def construct_buckets_from_topk_record(
    *,
    record: Mapping[str, Any],
    tokenizer_name: str,
    protocol_id: str,
    bank_id: str,
    audit_key_id: str,
    bucket_count: int,
    candidate_top_k: int,
    min_probability: float,
    min_members_per_bucket: int,
    min_bucket_mass: float,
    strict_min_bucket_mass: bool,
    forbidden_patterns: list[str],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Construct deterministic opportunity buckets from a scored observed prefix.

    This is the verifier-facing policy boundary: an observed transcript prefix is
    first scored by the configured reference model to produce the same top-k
    candidate-record schema used by Phase A, then this function filters and
    partitions candidates under the audit key. Precomputed opportunity-bank
    files are caches/calibration samples, not the only source of truth.
    """

    return _entry_from_record(
        record=record,
        tokenizer_name=tokenizer_name,
        protocol_id=protocol_id,
        bank_id=bank_id,
        audit_key_id=audit_key_id,
        bucket_count=bucket_count,
        candidate_top_k=candidate_top_k,
        min_probability=min_probability,
        min_members_per_bucket=min_members_per_bucket,
        min_bucket_mass=min_bucket_mass,
        strict_min_bucket_mass=strict_min_bucket_mass,
        forbidden_patterns=forbidden_patterns,
    )
