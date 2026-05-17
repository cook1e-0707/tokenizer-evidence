from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_cover_natural_common import (  # noqa: E402
    read_json,
    read_jsonl,
    sha256_file,
    write_csv_new,
    write_json_new,
    write_jsonl_new,
    write_text_new,
)


DEFAULT_BASE_ALLOCATION = (
    ROOT / "results/natural_evidence_v2/status/r4_after_868212_full16_quality_repair_plan_20260516/row_allocation_rows.jsonl"
)
DEFAULT_REVIEW = (
    ROOT
    / "results/natural_evidence_v2/status/r4_after_868260_quality_repair_confirmation_868299_quality_review/"
    "quality_repair_confirmation_review_summary.json"
)
DEFAULT_OUTPUT = ROOT / "results/natural_evidence_v2/status/r4_after_868299_first_token_event_dev_diagnostic_plan_20260517"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan the 32-block first-token event dev diagnostic after the 868299 "
            "quality-repair confirmation pass. This is artifact-only: no model "
            "calls, no generation, no Slurm submission."
        )
    )
    parser.add_argument("--base-allocation", type=Path, default=DEFAULT_BASE_ALLOCATION)
    parser.add_argument("--quality-review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-shards", type=int, default=32)
    parser.add_argument("--base-shards", type=int, default=4)
    return parser.parse_args()


def row_pair_key(row: Mapping[str, Any]) -> str:
    return f"{int(row['prompt_index'])}|{row.get('prefix_family_id', '')}"


def validate_quality_review(review: Mapping[str, Any]) -> None:
    if review.get("status") != "PASS_R4_AFTER_868260_QUALITY_REPAIR_CONFIRMATION_FIRST_TOKEN_EVENT_GATE":
        raise ValueError(f"quality review did not pass: {review.get('status')}")
    if int(review.get("protected_strict_accepts", -1)) != 4:
        raise ValueError("quality review must preserve protected strict accepts 4/4")
    if int(review.get("protected_accepts_ignoring_quality", -1)) != 4:
        raise ValueError("quality review must preserve protected ignoring-quality accepts 4/4")
    controls = review.get("control_accepts", {})
    if not isinstance(controls, Mapping) or any(int(controls.get(arm, -1)) != 0 for arm in ("raw", "task_only", "wrong_key", "wrong_payload")):
        raise ValueError("quality review controls must be 0/4 each")
    if int(review.get("global_duplicate_response_hash_count", -1)) != 0:
        raise ValueError("quality review must have zero global duplicate response hashes")
    if int(review.get("trace_binding_invalid_rows", -1)) != 0:
        raise ValueError("quality review must have zero trace-binding invalid rows")


def validate_base_rows(rows: Sequence[Mapping[str, Any]], *, base_shards: int) -> dict[str, Any]:
    by_shard: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_shard[int(row.get("assigned_shard_index", -1))].append(row)
    expected_shards = set(range(base_shards))
    if set(by_shard) != expected_shards:
        raise ValueError(f"base allocation shards mismatch: {sorted(by_shard)} != {sorted(expected_shards)}")

    shard_summaries: list[dict[str, Any]] = []
    rows_per_shard = 0
    selected_coordinates: set[int] = set()
    for shard in sorted(by_shard):
        shard_rows = by_shard[shard]
        rows_per_shard = rows_per_shard or len(shard_rows)
        if len(shard_rows) != rows_per_shard:
            raise ValueError(f"base shard {shard} row count mismatch")
        row_keys = Counter(str(row.get("row_key", "")) for row in shard_rows)
        duplicate_row_keys = sum(count - 1 for count in row_keys.values() if count > 1)
        pairs = Counter(row_pair_key(row) for row in shard_rows)
        duplicate_pairs = sum(count - 1 for count in pairs.values() if count > 1)
        coordinates = sorted({int(row["coordinate_id"]) for row in shard_rows})
        selected_coordinates.update(coordinates)
        if duplicate_row_keys or duplicate_pairs:
            raise ValueError(f"base shard {shard} duplicate row_key={duplicate_row_keys} pair={duplicate_pairs}")
        shard_summaries.append(
            {
                "base_shard_index": shard,
                "row_count": len(shard_rows),
                "selected_coordinate_count": len(coordinates),
                "unique_prompt_prefix_pairs": len(pairs),
                "duplicate_prompt_prefix_pair_count": duplicate_pairs,
            }
        )
    return {
        "base_shards": base_shards,
        "rows_per_shard": rows_per_shard,
        "selected_coordinates": sorted(selected_coordinates),
        "selected_coordinate_count": len(selected_coordinates),
        "base_shard_summaries": shard_summaries,
    }


def build_cyclic_allocation(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_shards: int,
    base_shards: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if target_shards <= 0 or base_shards <= 0:
        raise ValueError("target_shards and base_shards must be positive")
    if target_shards % base_shards != 0:
        raise ValueError("target_shards must be a multiple of base_shards for the cyclic reuse policy")
    validation = validate_base_rows(rows, base_shards=base_shards)
    by_shard: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_shard[int(row["assigned_shard_index"])].append(row)

    output_rows: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []
    for shard in range(target_shards):
        base_shard = shard % base_shards
        cycle_index = shard // base_shards
        shard_rows: list[dict[str, Any]] = []
        for row in by_shard[base_shard]:
            copied = dict(row)
            copied["source_row_key"] = str(row.get("row_key", ""))
            copied["source_assigned_shard_index"] = int(row["assigned_shard_index"])
            copied["assigned_shard_index"] = shard
            copied["replicate_group_id"] = f"first_token_event_dev_shard_{shard:02d}"
            copied["dev_diagnostic_cycle_index"] = cycle_index
            copied["dev_diagnostic_base_shard_index"] = base_shard
            copied["dev_diagnostic_allocation_policy"] = "cyclic_reuse_reviewed_4_block_full16_allocation"
            copied["row_key"] = f"{row.get('row_key', '')}|devdiag{shard:02d}"
            shard_rows.append(copied)
            output_rows.append(copied)
        pairs = Counter(row_pair_key(row) for row in shard_rows)
        shard_summaries.append(
            {
                "shard_index": shard,
                "replicate_group_id": f"first_token_event_dev_shard_{shard:02d}",
                "base_shard_index": base_shard,
                "cycle_index": cycle_index,
                "row_count": len(shard_rows),
                "unique_prompt_prefix_pairs": len(pairs),
                "duplicate_prompt_prefix_pair_count": sum(count - 1 for count in pairs.values() if count > 1),
            }
        )

    manifest = {
        "schema_name": "natural_evidence_v2_r4_after_868299_first_token_event_dev_diagnostic_allocation_manifest_v1",
        "status": "PASS_R4_AFTER_868299_FIRST_TOKEN_EVENT_DEV_DIAGNOSTIC_ALLOCATION_PLAN_NO_SUBMIT",
        "target_shards": target_shards,
        "base_shards": base_shards,
        "reuse_cycle_count": target_shards // base_shards,
        "allocation_policy": "cyclic_reuse_reviewed_4_block_full16_allocation",
        "reuse_scope": "dev_diagnostic_only_not_locked_scale",
        "rows_per_shard": int(validation["rows_per_shard"]),
        "total_rows": len(output_rows),
        "selected_coordinates": validation["selected_coordinates"],
        "selected_coordinate_count": int(validation["selected_coordinate_count"]),
        "base_shard_summaries": validation["base_shard_summaries"],
        "shard_summaries": shard_summaries,
        "within_shard_duplicate_prompt_prefix_pair_count_max": max(
            int(item["duplicate_prompt_prefix_pair_count"]) for item in shard_summaries
        ),
        "global_prompt_prefix_reuse_is_expected": True,
        "global_response_hash_duplicate_gate_remains_zero": True,
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "slurm_submitted": False,
        "paper_claim_allowed": False,
    }
    return output_rows, manifest


def write_report(output_dir: Path, *, manifest: Mapping[str, Any], review: Mapping[str, Any]) -> None:
    text = f"""# R4 After-868299 First-Token Event Dev Diagnostic Plan

Date: 2026-05-17

## Status

`{manifest["status"]}`

This is a 32-block Qwen dev diagnostic route plan for the provider-side keyed
first-token event channel. It is not a locked-scale result and does not unlock
paper-facing claims, Llama, FAR, sanitizer, training, or payload diversity.

## Source Confirmation

The route is only allowed because job `868299` passed the strict 4-block
quality-repair confirmation:

- protected strict accepts: `{review["protected_strict_accepts"]}/4`
- protected accepts ignoring quality: `{review["protected_accepts_ignoring_quality"]}/4`
- raw/task-only/wrong-key/wrong-payload accepts: `{review["control_accepts"]}`
- global duplicate response hashes: `{review["global_duplicate_response_hash_count"]}`
- trace-binding invalid rows: `{review["trace_binding_invalid_rows"]}`

## Allocation Policy

The reviewed full16 row bank contains four fully unique 1024-row shards. A
blind 32-block dev diagnostic therefore cannot claim 32 independent prompt
allocations from the current bank. This route precommits cyclic reuse of the
reviewed four-shard allocation across `{manifest["reuse_cycle_count"]}` cycles.
Each shard still has zero within-shard prompt/prefix duplicates, and each shard
uses a distinct public shard id and public sampling seed.

This reuse is acceptable only for a dev diagnostic. It must not be described as
locked-scale independent evidence. The global exact response-hash duplicate
gate remains zero.

## Next Allowed Action

Run local/remote route validation and wrapper plan-only smoke. If those pass,
the existing user authorization permits one reviewed H200 Slurm submission with
the allowlist enabled for exactly one entry and disabled immediately after
`sbatch`.
"""
    write_text_new(output_dir / "dev_diagnostic_plan.md", text)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    base_allocation = args.base_allocation if args.base_allocation.is_absolute() else ROOT / args.base_allocation
    quality_review_path = args.quality_review if args.quality_review.is_absolute() else ROOT / args.quality_review
    review = read_json(quality_review_path)
    validate_quality_review(review)
    rows = read_jsonl(base_allocation)
    allocation_rows, manifest = build_cyclic_allocation(
        rows,
        target_shards=int(args.target_shards),
        base_shards=int(args.base_shards),
    )
    summary = {
        "schema_name": "natural_evidence_v2_r4_after_868299_first_token_event_dev_diagnostic_plan_v1",
        "status": "PASS_R4_AFTER_868299_FIRST_TOKEN_EVENT_DEV_DIAGNOSTIC_PLAN_NO_SUBMIT",
        "source_quality_review": str(quality_review_path),
        "source_quality_review_sha256": sha256_file(quality_review_path),
        "source_quality_review_status": str(review.get("status", "")),
        "base_allocation": str(base_allocation),
        "base_allocation_sha256": sha256_file(base_allocation),
        "allocation_manifest": str(output_dir / "row_allocation_manifest.json"),
        "allocation_rows": str(output_dir / "row_allocation_rows.jsonl"),
        "target_shards": int(args.target_shards),
        "base_shards": int(args.base_shards),
        "rows_per_shard": int(manifest["rows_per_shard"]),
        "total_rows": int(manifest["total_rows"]),
        "selected_coordinate_count": int(manifest["selected_coordinate_count"]),
        "allocation_policy": str(manifest["allocation_policy"]),
        "dev_diagnostic_only_not_locked_scale": True,
        "global_response_hash_duplicate_gate_remains_zero": True,
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "slurm_submitted": False,
        "paper_claim_allowed": False,
    }
    write_json_new(output_dir / "row_allocation_manifest.json", manifest)
    write_jsonl_new(output_dir / "row_allocation_rows.jsonl", allocation_rows)
    write_csv_new(
        output_dir / "allocation_shards.csv",
        manifest["shard_summaries"],
        [
            "shard_index",
            "replicate_group_id",
            "base_shard_index",
            "cycle_index",
            "row_count",
            "unique_prompt_prefix_pairs",
            "duplicate_prompt_prefix_pair_count",
        ],
    )
    write_json_new(output_dir / "dev_diagnostic_plan_summary.json", summary)
    write_report(output_dir, manifest=manifest, review=review)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
