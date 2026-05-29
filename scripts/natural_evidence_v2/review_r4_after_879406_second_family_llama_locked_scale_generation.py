#!/usr/bin/env python3
"""Review R4 after-879406 second-family Llama 96-block locked-scale generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.aggregate_r4_after_869348_locked_scale_generation import (  # noqa: E402
    ARMS,
    CONTROL_ARMS,
    discover_shards,
    int_value,
    read_csv,
    read_json,
    resolve,
    summarize_first_token,
    summarize_full_phrase,
    summarize_generated_jsonl,
    write_csv,
    write_json,
)
from scripts.natural_evidence_v2.r4_cover_natural_common import write_text_new  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job-id", default="879555")
    parser.add_argument("--expected-shards", type=int, default=96)
    parser.add_argument("--expected-generated-rows-per-shard", type=int, default=2048)
    parser.add_argument("--protected-strict-min", type=int, default=80)
    parser.add_argument("--protected-ignoring-quality-min", type=int, default=85)
    parser.add_argument("--control-accepts-max", type=int, default=0)
    parser.add_argument("--global-duplicate-extra-max", type=int, default=0)
    parser.add_argument("--within-block-duplicate-max", type=int, default=0)
    parser.add_argument("--technical-forbidden-max", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = resolve(args.run_root)
    output_dir = resolve(args.output_dir)
    shard_root = run_root / "shards"
    complete, partial, missing, incomplete = discover_shards(
        [shard_root],
        expected_shards=args.expected_shards,
        require_generated_jsonl=True,
    )

    first_token_rows: list[dict[str, Any]] = []
    full_phrase_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    shard_matrix: list[dict[str, Any]] = []

    for shard_index in range(args.expected_shards):
        shard_dir = complete.get(shard_index)
        partial_dirs = partial.get(shard_index, [])
        status = "complete" if shard_dir is not None else ("partial" if partial_dirs else "missing")
        shard_matrix.append(
            {
                "shard_index": shard_index,
                "shard_id": f"shard_{shard_index:02d}",
                "status": status,
                "selected_shard_dir": str(shard_dir or ""),
                "partial_dirs": ";".join(str(path) for path in partial_dirs),
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
    expected_generated_rows = args.expected_shards * args.expected_generated_rows_per_shard
    trace_checked_rows = sum(int(row["checked_rows"]) for row in trace_rows)
    trace_invalid_rows = sum(int(row["invalid_rows"]) for row in trace_rows)
    trace_status_pass = bool(trace_rows) and all(
        row["status"] == "PASS_R4_FIRST_TOKEN_EVENT_TRACE_BINDING" for row in trace_rows
    )

    protected = first_token_summary["protected"]
    control_strict_accepts = {arm: first_token_summary[arm]["accepts"] for arm in CONTROL_ARMS}
    control_ignoring_quality_accepts = {
        arm: first_token_summary[arm]["accepts_ignoring_quality"] for arm in CONTROL_ARMS
    }
    technical_forbidden = sum(arm["forbidden_public_surface_count"] for arm in first_token_summary.values())
    within_block_duplicates = sum(arm["duplicate_response_hash_count"] for arm in first_token_summary.values())
    global_duplicate_extra = generation_summary["global_duplicate_response_hash_extra_rows"]

    all_complete = len(complete) == args.expected_shards
    generated_rows_ok = generation_summary["generated_rows"] == expected_generated_rows
    gate_pass = (
        all_complete
        and protected["blocks"] == args.expected_shards
        and protected["accepts"] >= args.protected_strict_min
        and protected["accepts_ignoring_quality"] >= args.protected_ignoring_quality_min
        and all(value <= args.control_accepts_max for value in control_strict_accepts.values())
        and all(value <= args.control_accepts_max for value in control_ignoring_quality_accepts.values())
        and within_block_duplicates <= args.within_block_duplicate_max
        and technical_forbidden <= args.technical_forbidden_max
        and trace_status_pass
        and trace_invalid_rows == 0
        and trace_checked_rows == expected_generated_rows
        and generated_rows_ok
        and global_duplicate_extra <= args.global_duplicate_extra_max
    )
    status = (
        f"PASS_R4_AFTER_879406_SECOND_FAMILY_LLAMA_LOCKED_SCALE_GENERATION_{args.job_id}_REVIEWED"
        if gate_pass
        else (
            f"INCOMPLETE_R4_AFTER_879406_SECOND_FAMILY_LLAMA_LOCKED_SCALE_GENERATION_{args.job_id}_NO_GATE"
            if not all_complete
            else f"FAIL_R4_AFTER_879406_SECOND_FAMILY_LLAMA_LOCKED_SCALE_GENERATION_{args.job_id}_REVIEWED_NO_ADOPT"
        )
    )

    summary: Mapping[str, Any] = {
        "schema_name": "natural_evidence_v2_r4_after_879406_second_family_llama_locked_scale_generation_review_v1",
        "status": status,
        "job_id": str(args.job_id),
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "scale_gate_pass": bool(gate_pass),
        "all_shards_complete": all_complete,
        "complete_shards": sorted(complete),
        "complete_shard_count": len(complete),
        "expected_shards": args.expected_shards,
        "missing_shards": missing,
        "incomplete_shards": incomplete,
        "run_root": str(run_root),
        "first_token_event_summary_by_arm": first_token_summary,
        "control_strict_accepts": control_strict_accepts,
        "control_ignoring_quality_accepts": control_ignoring_quality_accepts,
        "full_phrase_decoder_report_only_summary": full_phrase_summary,
        "trace_binding": {
            "checked_rows": trace_checked_rows,
            "expected_checked_rows": expected_generated_rows,
            "invalid_rows": trace_invalid_rows,
            "all_shard_trace_status_pass": trace_status_pass,
        },
        "generation_duplicate_summary": generation_summary,
        "technical_forbidden_public_surface_count": technical_forbidden,
        "within_block_duplicate_response_hash_count": within_block_duplicates,
        "gate_targets": {
            "protected_strict_accepts_min": args.protected_strict_min,
            "protected_accepts_ignoring_quality_min": args.protected_ignoring_quality_min,
            "control_accepts_max_per_condition": args.control_accepts_max,
            "within_block_duplicate_response_hash_count_max": args.within_block_duplicate_max,
            "global_duplicate_response_hash_extra_rows_max": args.global_duplicate_extra_max,
            "technical_forbidden_public_surface_count_max": args.technical_forbidden_max,
            "trace_binding_validity_required": 1.0,
            "full_phrase_decoder_policy": "report_only_not_text_only_success_claim",
        },
        "claim_control": {
            "internal_second_family_locked_scale_statement_allowed": bool(gate_pass),
            "paper_claim_allowed": False,
            "full_far_claim_allowed": False,
            "sanitizer_robustness_claim_allowed": False,
            "payload_diversity_tested": False,
            "text_only_phrase_decoder_success_claim_allowed": False,
        },
        "next_allowed_action": (
            "Record expert/project route decision for post-Llama-locked-scale work; paper-facing claims, "
            "FAR, sanitizer, and payload diversity remain gated by explicit downstream route review."
            if gate_pass
            else "Analyze failed locked-scale gates before rerun or downstream route unlock."
        ),
    }

    write_csv(
        output_dir / "llama_locked_scale_shard_matrix.csv",
        shard_matrix,
        ["shard_index", "shard_id", "status", "selected_shard_dir", "partial_dirs"],
    )
    write_csv(
        output_dir / "llama_locked_scale_first_token_blocks.csv",
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
        output_dir / "llama_locked_scale_full_phrase_blocks.csv",
        full_phrase_rows,
        [
            "shard_index",
            "source_shard_dir",
            "block_id",
            "arm",
            "accept",
            "complete_pairs",
            "required_pairs",
            "selected_coordinates_observed",
            "selected_coordinates_total",
            "min_pair_support",
            "matched_surface_count",
            "selected_surface_count",
            "checksum_valid",
            "bits_match_condition",
            "forbidden_public_surface_count",
            "format_scrub_mode",
        ],
    )
    write_csv(
        output_dir / "llama_locked_scale_trace_binding_by_shard.csv",
        trace_rows,
        ["shard_index", "shard_id", "status", "checked_rows", "invalid_rows", "source_shard_dir"],
    )
    write_json(output_dir / "llama_locked_scale_generation_duplicate_summary.json", generation_summary)
    write_json(output_dir / "review_summary.json", summary)

    review_lines = [
        f"# R4 After-879406 Llama Locked-Scale Generation {args.job_id} Review",
        "",
        f"Status: `{status}`",
        "",
        "This is an artifact-only review over completed Slurm outputs. It does not make a paper-facing claim.",
        "",
        "## Gate Summary",
        "",
        f"- Complete shards: `{len(complete)}/{args.expected_shards}`",
        f"- Generated rows: `{generation_summary['generated_rows']}/{expected_generated_rows}`",
        f"- Protected strict accepts: `{protected['accepts']}/{args.expected_shards}`",
        f"- Protected accepts ignoring quality: `{protected['accepts_ignoring_quality']}/{args.expected_shards}`",
        f"- Raw accepts: `{first_token_summary['raw']['accepts']}/{args.expected_shards}`",
        f"- Task-only accepts: `{first_token_summary['task_only']['accepts']}/{args.expected_shards}`",
        f"- Wrong-key accepts: `{first_token_summary['wrong_key']['accepts']}/{args.expected_shards}`",
        f"- Wrong-payload accepts: `{first_token_summary['wrong_payload']['accepts']}/{args.expected_shards}`",
        f"- Global duplicate extra rows: `{global_duplicate_extra}`",
        f"- Within-block duplicate count: `{within_block_duplicates}`",
        f"- Technical forbidden public surface count: `{technical_forbidden}`",
        f"- Trace binding invalid rows: `{trace_invalid_rows}/{trace_checked_rows}`",
        "",
        "## Claim Control",
        "",
        "- Passing this gate permits only an internal second-family locked-scale statement for the first-token event route.",
        "- Full-phrase text-only decoding, FAR, sanitizer robustness, payload diversity, and paper-facing claims remain gated.",
    ]
    write_text_new(output_dir / "review.md", "\n".join(review_lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
