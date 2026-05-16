from __future__ import annotations

from scripts.natural_evidence_v2.validate_r4_after_868212_reliability_duplicate_repair_plan import (
    duplicate_taxonomy,
    singleton_bit_failures,
)


def test_singleton_bit_failures_rejects_coordinate_26_as_sole_coordinate() -> None:
    decoder_spec = {
        "pair_to_bit_mapping": [
            {"bit_index": 0, "coordinates": [6, 22], "source_coordinates": [6, 22]},
            {"bit_index": 1, "coordinates": [26], "source_coordinates": [10, 26], "erased_source_coordinates": [10]},
        ]
    }

    failures = singleton_bit_failures(decoder_spec, min_active_coordinates=2)

    assert {failure["failure_reason"] for failure in failures} == {
        "active_coordinate_count_below_minimum",
        "coordinate_26_is_sole_active_coordinate",
    }


def test_duplicate_taxonomy_separates_cross_arm_and_cross_shard_duplicates() -> None:
    rows = [
        {
            "response_text_sha256": "same-cross-arm",
            "generation_condition": "protected",
            "shard_id": "shard_00",
        },
        {
            "response_text_sha256": "same-cross-arm",
            "generation_condition": "raw",
            "shard_id": "shard_00",
        },
        {
            "response_text_sha256": "same-cross-shard",
            "generation_condition": "task_only",
            "shard_id": "shard_00",
        },
        {
            "response_text_sha256": "same-cross-shard",
            "generation_condition": "task_only",
            "shard_id": "shard_01",
        },
    ]

    taxonomy = duplicate_taxonomy(rows)

    assert taxonomy["duplicate_hash_groups"] == 2
    assert taxonomy["global_duplicate_response_hash_count"] == 2
    assert taxonomy["cross_arm_duplicate_groups"] == 1
    assert taxonomy["cross_shard_duplicate_groups"] == 1
    assert taxonomy["within_arm_duplicate_groups"] == 1
    assert taxonomy["within_shard_duplicate_groups"] == 1
