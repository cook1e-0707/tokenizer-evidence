from __future__ import annotations

from scripts.natural_evidence_v2.build_r4_after_868212_repaired_first_token_event_precommit import build_codebook
from scripts.natural_evidence_v2.validate_r4_after_868212_repaired_first_token_event_plan import (
    allocation_failures,
    singleton_failures,
)


def test_repaired_codebook_restores_two_coordinates_per_bit() -> None:
    source_codebook = {
        "payload_bits": 4,
        "checksum_bits": 4,
        "pair_to_bit_mapping": [
            {"bit_index": index, "coordinates": [index, index + 16]} for index in range(8)
        ],
    }
    rows_summary = {"expected_codeword_bits": [1, 0, 1, 0, 0, 1, 0, 1]}

    codebook = build_codebook(source_codebook, rows_summary)

    assert codebook["min_active_coordinates_per_bit"] == 2
    assert len(codebook["selected_coordinates"]) == 16
    assert singleton_failures(codebook) == []


def test_allocation_failures_accepts_unique_prompt_prefix_per_shard() -> None:
    coordinates = list(range(16))
    manifest = {
        "shards": 2,
        "rows_per_coordinate_per_shard": 1,
        "coordinates": coordinates,
    }
    rows = [
        {
            "assigned_shard_index": shard,
            "coordinate_id": coordinate,
            "prompt_index": coordinate,
            "prefix_family_id": f"p{shard}",
        }
        for shard in range(2)
        for coordinate in coordinates
    ]

    assert allocation_failures(rows, manifest) == []


def test_allocation_failures_rejects_duplicate_prompt_prefix_within_shard() -> None:
    manifest = {
        "shards": 1,
        "rows_per_coordinate_per_shard": 1,
        "coordinates": [1, 17],
    }
    rows = [
        {"assigned_shard_index": 0, "coordinate_id": 1, "prompt_index": 0, "prefix_family_id": "a"},
        {"assigned_shard_index": 0, "coordinate_id": 17, "prompt_index": 0, "prefix_family_id": "a"},
    ]

    failures = allocation_failures(rows, manifest)

    assert any("duplicate prompt/prefix" in failure for failure in failures)
