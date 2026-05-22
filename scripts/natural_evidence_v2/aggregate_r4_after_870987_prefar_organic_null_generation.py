from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.aggregate_r4_after_869348_locked_scale_generation import (  # noqa: E402
    discover_shards,
    int_value,
    read_csv,
    read_json,
    read_jsonl,
    summarize_first_token,
    summarize_full_phrase,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate R4 pre-FAR organic-null raw-only generation shards.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-shards", type=int, default=256)
    parser.add_argument("--expected-generated-rows-per-shard", type=int, default=1024)
    parser.add_argument("--organic-accepts-max", type=int, default=0)
    parser.add_argument("--global-duplicate-extra-max", type=int, default=0)
    parser.add_argument("--within-block-duplicate-max", type=int, default=0)
    parser.add_argument("--forbidden-public-surface-max", type=int, default=0)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def summarize_generated_jsonl(shard_dirs: Mapping[int, Path]) -> dict[str, Any]:
    response_hash_counts: Counter[str] = Counter()
    generation_id_counts: Counter[str] = Counter()
    rows_by_condition: Counter[str] = Counter()
    row_count = 0
    for shard_dir in shard_dirs.values():
        for row in read_jsonl(shard_dir / "r4_generated_outputs.jsonl"):
            row_count += 1
            response_hash = str(row.get("output_text_sha256") or row.get("response_text_sha256") or "")
            generation_id = str(row.get("generation_id", ""))
            condition = str(row.get("generation_condition", ""))
            if response_hash:
                response_hash_counts[response_hash] += 1
            if generation_id:
                generation_id_counts[generation_id] += 1
            if condition:
                rows_by_condition[condition] += 1
    duplicate_response_groups = {key: count for key, count in response_hash_counts.items() if count > 1}
    duplicate_generation_id_groups = {key: count for key, count in generation_id_counts.items() if count > 1}
    return {
        "generated_rows": row_count,
        "rows_by_condition": dict(sorted(rows_by_condition.items())),
        "unique_response_hashes": len(response_hash_counts),
        "global_duplicate_response_hash_extra_rows": sum(count - 1 for count in duplicate_response_groups.values()),
        "global_duplicate_response_hash_group_count": len(duplicate_response_groups),
        "max_response_hash_group_size": max(duplicate_response_groups.values(), default=1),
        "duplicate_generation_id_extra_rows": sum(count - 1 for count in duplicate_generation_id_groups.values()),
        "duplicate_generation_id_group_count": len(duplicate_generation_id_groups),
        "max_generation_id_group_size": max(duplicate_generation_id_groups.values(), default=1),
    }


def main() -> int:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    roots = [resolve(path) for path in args.shard_roots]
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"shard root missing: {root}")
    complete, partial, missing, incomplete = discover_shards(
        roots,
        expected_shards=int(args.expected_shards),
        require_generated_jsonl=True,
    )
    first_token_rows: list[dict[str, Any]] = []
    full_phrase_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    shard_matrix: list[dict[str, Any]] = []
    for shard_index in range(int(args.expected_shards)):
        shard_dir = complete.get(shard_index)
        status = "complete" if shard_dir is not None else ("partial" if partial.get(shard_index) else "missing")
        shard_matrix.append(
            {
                "shard_index": shard_index,
                "shard_id": f"shard_{shard_index:02d}",
                "status": status,
                "selected_shard_dir": str(shard_dir or ""),
            }
        )
        if shard_dir is None:
            continue
        for row in read_csv(shard_dir / "first_token_event_decode/first_token_event_per_block.csv"):
            first_token_rows.append(dict(row) | {"shard_index": shard_index, "source_shard_dir": str(shard_dir)})
        for mode_dir in ("decode_all", "decode_none"):
            for row in read_csv(shard_dir / f"{mode_dir}/per_block_decode.csv"):
                full_phrase_rows.append(dict(row) | {"shard_index": shard_index, "source_shard_dir": str(shard_dir)})
        trace = read_json(shard_dir / "trace_binding_validation.json")
        trace_rows.append(
            {
                "shard_index": shard_index,
                "shard_id": f"shard_{shard_index:02d}",
                "status": trace.get("status", ""),
                "checked_rows": int_value(trace.get("checked_rows")),
                "invalid_rows": int_value(trace.get("invalid_rows")),
                "source_shard_dir": str(shard_dir),
            }
        )
    first_token_summary = summarize_first_token(first_token_rows)
    full_phrase_summary = summarize_full_phrase(full_phrase_rows)
    generation_summary = summarize_generated_jsonl(complete)
    expected_generated_rows = int(args.expected_shards) * int(args.expected_generated_rows_per_shard)
    raw_summary = first_token_summary.get("raw", {})
    trace_checked_rows = sum(int(row["checked_rows"]) for row in trace_rows)
    trace_invalid_rows = sum(int(row["invalid_rows"]) for row in trace_rows)
    trace_status_pass = bool(trace_rows) and all(row["status"] == "PASS_R4_FIRST_TOKEN_EVENT_TRACE_BINDING" for row in trace_rows)
    all_complete = len(complete) == int(args.expected_shards)
    gate_pass = (
        all_complete
        and int(raw_summary.get("blocks", -1)) == int(args.expected_shards)
        and int(raw_summary.get("accepts", -1)) <= int(args.organic_accepts_max)
        and int(raw_summary.get("accepts_ignoring_quality", -1)) <= int(args.organic_accepts_max)
        and int(raw_summary.get("duplicate_response_hash_count", -1)) <= int(args.within_block_duplicate_max)
        and int(raw_summary.get("forbidden_public_surface_count", -1)) <= int(args.forbidden_public_surface_max)
        and trace_status_pass
        and trace_invalid_rows == 0
        and trace_checked_rows == expected_generated_rows
        and generation_summary["generated_rows"] == expected_generated_rows
        and generation_summary["rows_by_condition"] == {"raw": expected_generated_rows}
        and generation_summary["global_duplicate_response_hash_extra_rows"] <= int(args.global_duplicate_extra_max)
    )
    status = (
        "PASS_R4_AFTER_870987_PREFAR_ORGANIC_NULL_GENERATION_GATE"
        if gate_pass
        else (
            "INCOMPLETE_R4_AFTER_870987_PREFAR_ORGANIC_NULL_GENERATION_NO_GATE"
            if not all_complete
            else "FAIL_R4_AFTER_870987_PREFAR_ORGANIC_NULL_GENERATION_GATE"
        )
    )
    summary = {
        "schema_name": "natural_evidence_v2_r4_after_870987_prefar_organic_null_generation_aggregate_v1",
        "status": status,
        "prefar_organic_null_gate_pass": bool(gate_pass),
        "all_shards_complete": all_complete,
        "complete_shard_count": len(complete),
        "expected_shards": int(args.expected_shards),
        "missing_shards": missing,
        "incomplete_shards": incomplete,
        "first_token_event_summary_by_arm": first_token_summary,
        "organic_null_raw_summary": {
            "blocks": int(raw_summary.get("blocks", 0)),
            "accepts": int(raw_summary.get("accepts", 0)),
            "accepts_ignoring_quality": int(raw_summary.get("accepts_ignoring_quality", 0)),
        },
        "full_phrase_decoder_report_only_summary": full_phrase_summary,
        "trace_binding": {
            "checked_rows": trace_checked_rows,
            "expected_checked_rows": expected_generated_rows,
            "invalid_rows": trace_invalid_rows,
            "all_shard_trace_status_pass": trace_status_pass,
        },
        "generation_duplicate_summary": generation_summary,
        "claim_control": {
            "paper_claim_allowed": False,
            "training_allowed": False,
            "llama_allowed": False,
            "same_family_null_allowed": False,
            "sanitizer_allowed": False,
            "far_aggregation_allowed": False,
            "payload_diversity_tested": False,
            "text_only_phrase_decoder_success_claim": False,
            "full_far_claim_allowed": False,
        },
    }
    write_csv(
        output_dir / "prefar_organic_null_shard_matrix.csv",
        shard_matrix,
        ["shard_index", "shard_id", "status", "selected_shard_dir"],
    )
    write_csv(
        output_dir / "prefar_organic_null_first_token_blocks.csv",
        first_token_rows,
        [
            "shard_index",
            "source_shard_dir",
            "block_id",
            "arm",
            "source_condition",
            "accept",
            "accept_ignoring_quality",
            "complete_pairs",
            "required_pairs",
            "decoded_bits",
            "expected_bits",
            "bits_match_condition",
            "checksum_valid",
            "forbidden_public_surface_count",
            "duplicate_response_hash_count",
        ],
    )
    write_csv(
        output_dir / "prefar_organic_null_full_phrase_blocks.csv",
        full_phrase_rows,
        ["shard_index", "source_shard_dir", "block_id", "arm", "accept", "complete_pairs", "required_pairs", "format_scrub_mode"],
    )
    write_csv(
        output_dir / "prefar_organic_null_trace_binding_by_shard.csv",
        trace_rows,
        ["shard_index", "shard_id", "status", "checked_rows", "invalid_rows", "source_shard_dir"],
    )
    write_json(output_dir / "prefar_organic_null_summary.json", summary)
    print(json.dumps({"status": status, "output_dir": str(output_dir)}, sort_keys=True))
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
