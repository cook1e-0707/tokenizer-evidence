from __future__ import annotations

from scripts.natural_evidence_v2.plan_r4_after_868151_first_token_event_quality_repair import (
    build_duplicate_safe_allocation,
    is_coordination_domain_prompt,
    literal_policy,
)


def _row(prompt_index: int, prefix: str, coordinate: int) -> dict[str, object]:
    return {
        "coordinate_id": coordinate,
        "prefix_family_id": prefix,
        "prompt_id": f"p{prompt_index}",
        "prompt_index": prompt_index,
        "prompt_text": "Write a short useful answer.",
        "row_key": f"p{prompt_index}|c{coordinate}|{prefix}",
        "target_surface": f"surface {coordinate}",
    }


def test_duplicate_safe_allocation_splits_prompt_prefix_pairs() -> None:
    rows = [
        _row(0, "next", 1),
        _row(0, "next", 2),
        _row(1, "next", 1),
        _row(1, "next", 2),
        _row(2, "option", 1),
        _row(2, "option", 2),
        _row(3, "option", 1),
        _row(3, "option", 2),
    ]

    assigned, manifest = build_duplicate_safe_allocation(
        rows,
        shards=2,
        rows_per_coordinate_per_shard=2,
    )

    assert manifest["status"] == "PASS_DUPLICATE_SAFE_ROW_ALLOCATION_ARTIFACT_ONLY"
    assert manifest["total_rows"] == 8
    for shard in manifest["shard_summaries"]:
        assert shard["row_count"] == 4
        assert shard["duplicate_prompt_prefix_pair_count"] == 0
        assert shard["coordinate_counts"] == {"1": 2, "2": 2}

    by_shard: dict[int, set[str]] = {0: set(), 1: set()}
    for row in assigned:
        shard = int(row["assigned_shard_index"])
        pair = str(row["duplicate_pair_key"])
        assert pair not in by_shard[shard]
        by_shard[shard].add(pair)


def test_literal_policy_keeps_coordinate_contextual_not_hard_forbidden() -> None:
    policy = literal_policy()

    assert "coordinate" not in policy["hard_forbid_literals"]
    assert "coordinate" in policy["contextual_literals"]
    assert policy["final_generation_gate"]["technical_public_literal_count_max"] == 0


def test_coordination_domain_prompt_detection() -> None:
    assert is_coordination_domain_prompt("Advice for volunteer coordination.")
    assert is_coordination_domain_prompt("Help coordinators plan the work.")
    assert not is_coordination_domain_prompt("Advice for household planning.")
