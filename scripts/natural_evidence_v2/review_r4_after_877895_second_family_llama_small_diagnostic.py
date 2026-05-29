#!/usr/bin/env python3
"""Review the R4 after-877895 second-family Llama small diagnostic outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_cover_natural_common import write_json_new, write_text_new  # noqa: E402


ARMS = ("protected", "raw", "task_only", "wrong_key", "wrong_payload")
GATED_ARMS = ("protected", "raw", "wrong_key", "wrong_payload")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--slurm-sacct", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=4)
    parser.add_argument("--expected-generated-rows-per-shard", type=int, default=2048)
    parser.add_argument("--protected-accepts-min", type=int, default=4)
    parser.add_argument("--control-accepts-max", type=int, default=0)
    parser.add_argument("--global-duplicate-extra-max", type=int, default=0)
    parser.add_argument("--technical-forbidden-max", type=int, default=0)
    parser.add_argument("--ambiguous-forbidden-max", type=int, default=0)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            rows.append(payload)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def intish(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(value)


def discover_shards(run_root: Path) -> dict[int, Path]:
    shards: dict[int, Path] = {}
    for path in sorted((run_root / "shards").glob("shard_*")):
        if not path.is_dir():
            continue
        try:
            index = int(path.name.split("_", maxsplit=1)[1])
        except (IndexError, ValueError):
            continue
        shards[index] = path
    return shards


def summarize_first_token(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, Counter[str]] = {arm: Counter() for arm in ARMS}
    for row in rows:
        arm = str(row.get("arm", ""))
        if arm not in summary:
            summary[arm] = Counter()
        summary[arm]["blocks"] += 1
        summary[arm]["accepts"] += int(boolish(row.get("accept")))
        summary[arm]["accepts_ignoring_quality"] += int(boolish(row.get("accept_ignoring_quality")))
        summary[arm]["forbidden_public_surface_count"] += intish(row.get("forbidden_public_surface_count"))
        summary[arm]["duplicate_response_hash_count"] += intish(row.get("duplicate_response_hash_count"))
    return {arm: dict(counter) for arm, counter in sorted(summary.items())}


def summarize_generation(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    response_hashes = Counter(str(row.get("output_text_sha256") or row.get("response_text_sha256") or "") for row in rows)
    generation_ids = Counter(str(row.get("generation_id", "")) for row in rows)
    by_condition = Counter(str(row.get("generation_condition", row.get("arm", ""))) for row in rows)
    duplicates = {key: count for key, count in response_hashes.items() if key and count > 1}
    duplicate_generation_ids = {key: count for key, count in generation_ids.items() if key and count > 1}
    return {
        "generated_rows": len(rows),
        "rows_by_condition": dict(sorted(by_condition.items())),
        "unique_response_hashes": len([key for key in response_hashes if key]),
        "global_duplicate_response_hash_extra_rows": sum(count - 1 for count in duplicates.values()),
        "global_duplicate_response_hash_group_count": len(duplicates),
        "max_response_hash_group_size": max(duplicates.values(), default=1),
        "duplicate_generation_id_extra_rows": sum(count - 1 for count in duplicate_generation_ids.values()),
        "duplicate_generation_id_group_count": len(duplicate_generation_ids),
        "max_generation_id_group_size": max(duplicate_generation_ids.values(), default=1),
        "wrong_key_replay_accept_rows": sum(int(bool(row.get("wrong_key_replay_accept"))) for row in rows),
        "wrong_payload_replay_accept_rows": sum(int(bool(row.get("wrong_payload_replay_accept"))) for row in rows),
    }


def main() -> int:
    args = parse_args()
    run_root = resolve(args.run_root)
    output_dir = resolve(args.output_dir)
    shards = discover_shards(run_root)

    per_block_rows: list[dict[str, Any]] = []
    full_phrase_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    generated_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []
    errors: list[str] = []

    if len(shards) != int(args.expected_shards):
        errors.append(f"expected {args.expected_shards} shards, found {sorted(shards)}")

    for shard_index, shard_dir in sorted(shards.items()):
        generated_path = shard_dir / "r4_generated_outputs.jsonl"
        first_token_path = shard_dir / "first_token_event_decode/first_token_event_per_block.csv"
        first_event_path = shard_dir / "first_token_event_decode/first_token_event_rows.jsonl"
        trace_path = shard_dir / "trace_binding_validation.json"
        if not generated_path.exists():
            errors.append(f"missing generated outputs: {generated_path}")
            continue
        shard_generated = read_jsonl(generated_path)
        generated_rows.extend(shard_generated)
        if len(shard_generated) != int(args.expected_generated_rows_per_shard):
            errors.append(f"shard_{shard_index:02d} generated rows mismatch: {len(shard_generated)}")
        shard_first_token = read_csv(first_token_path) if first_token_path.exists() else []
        if not shard_first_token:
            errors.append(f"missing first-token per-block rows for shard_{shard_index:02d}")
        for row in shard_first_token:
            row["shard_index"] = shard_index
            row["source_shard_dir"] = str(shard_dir)
            per_block_rows.append(dict(row))
        if first_event_path.exists():
            for row in read_jsonl(first_event_path):
                row["shard_index"] = shard_index
                event_rows.append(row)
        for mode in ("decode_all", "decode_none"):
            path = shard_dir / mode / "per_block_decode.csv"
            if path.exists():
                for row in read_csv(path):
                    full_phrase_rows.append(dict(row) | {"shard_index": shard_index, "format_mode": mode})
        if trace_path.exists():
            trace = read_json(trace_path)
            trace_rows.append(
                {
                    "shard_index": shard_index,
                    "status": trace.get("status", ""),
                    "checked_rows": intish(trace.get("checked_rows")),
                    "invalid_rows": intish(trace.get("invalid_rows")),
                }
            )
        else:
            errors.append(f"missing trace binding validation for shard_{shard_index:02d}")
        shard_summaries.append(
            {
                "shard_index": shard_index,
                "generated_rows": len(shard_generated),
                "first_token_summary_by_arm": summarize_first_token(shard_first_token),
                "trace_invalid_rows": trace_rows[-1]["invalid_rows"] if trace_rows else "",
            }
        )

    first_token_summary = summarize_first_token(per_block_rows)
    generation_summary = summarize_generation(generated_rows)
    trace_checked = sum(row["checked_rows"] for row in trace_rows)
    trace_invalid = sum(row["invalid_rows"] for row in trace_rows)
    event_statuses = Counter(str(row.get("event_status", "")) for row in event_rows)
    event_sources = Counter(str(row.get("event_source", "")) for row in event_rows)

    protected = first_token_summary.get("protected", {})
    raw = first_token_summary.get("raw", {})
    wrong_key = first_token_summary.get("wrong_key", {})
    wrong_payload = first_token_summary.get("wrong_payload", {})
    technical_forbidden = sum(intish(first_token_summary.get(arm, {}).get("forbidden_public_surface_count")) for arm in GATED_ARMS)
    duplicate_within_block = sum(intish(first_token_summary.get(arm, {}).get("duplicate_response_hash_count")) for arm in GATED_ARMS)
    gate_pass = (
        not errors
        and intish(protected.get("accepts")) >= int(args.protected_accepts_min)
        and intish(protected.get("accepts_ignoring_quality")) >= int(args.protected_accepts_min)
        and intish(raw.get("accepts")) <= int(args.control_accepts_max)
        and intish(wrong_key.get("accepts")) <= int(args.control_accepts_max)
        and intish(wrong_payload.get("accepts")) <= int(args.control_accepts_max)
        and int(generation_summary["global_duplicate_response_hash_extra_rows"]) <= int(args.global_duplicate_extra_max)
        and duplicate_within_block == 0
        and technical_forbidden <= int(args.technical_forbidden_max)
        and trace_invalid == 0
    )
    status = (
        "PASS_R4_AFTER_877895_SECOND_FAMILY_LLAMA_SMALL_DIAGNOSTIC_879102_REVIEWED"
        if gate_pass
        else "FAIL_R4_AFTER_877895_SECOND_FAMILY_LLAMA_SMALL_DIAGNOSTIC_879102_REVIEWED_NO_ADOPT"
    )
    summary = {
        "schema_name": "r4_after_877895_second_family_llama_small_diagnostic_review_v1",
        "status": status,
        "errors": errors,
        "job_id": "879102",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "complete_shards": len(shards),
        "expected_shards": int(args.expected_shards),
        "generated_rows": len(generated_rows),
        "expected_generated_rows": int(args.expected_shards) * int(args.expected_generated_rows_per_shard),
        "generation_summary": generation_summary,
        "first_token_event_summary_by_arm": first_token_summary,
        "event_sources": dict(sorted(event_sources.items())),
        "event_statuses": dict(sorted(event_statuses.items())),
        "trace_binding": {
            "checked_rows": trace_checked,
            "invalid_rows": trace_invalid,
        },
        "technical_forbidden_public_surface_count": technical_forbidden,
        "duplicate_within_block_count": duplicate_within_block,
        "full_phrase_decoder_policy": "report_only_not_success_claim",
        "same_family_raw_null_rejection_already_supported_by_877895": True,
        "second_family_small_diagnostic_gate_pass": bool(gate_pass),
        "second_family_small_diagnostic_internal_claim_allowed": bool(gate_pass),
        "cross_family_locked_scale_success_claim_allowed": False,
        "paper_claim_allowed": False,
        "training_started": False,
        "far_aggregation_started": False,
        "sanitizer_started": False,
        "payload_diversity_started": False,
        "next_allowed_action": (
            "Record route decision for Llama 32-block dev diagnostic or expert review; "
            "do not make paper-facing claim from this 4-block diagnostic."
            if gate_pass
            else "Analyze failed gates before any rerun or scale-up."
        ),
    }

    write_json_new(output_dir / "review_summary.json", summary)
    write_csv(
        output_dir / "first_token_event_per_block.csv",
        per_block_rows,
        [
            "shard_index",
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
            "source_shard_dir",
        ],
    )
    write_csv(
        output_dir / "trace_binding_by_shard.csv",
        trace_rows,
        ["shard_index", "status", "checked_rows", "invalid_rows"],
    )
    write_csv(
        output_dir / "full_phrase_decode_report_only.csv",
        full_phrase_rows,
        sorted({key for row in full_phrase_rows for key in row.keys()}),
    )
    write_json_new(output_dir / "shard_summaries.json", {"shards": shard_summaries})
    review_lines = [
        "# R4 After-877895 Llama Small Diagnostic Review",
        "",
        f"Status: `{status}`",
        "",
        "This review is artifact-only over completed job `879102`. It does not make a paper-facing claim.",
        "",
        "## Gate Summary",
        "",
        f"- Complete shards: `{len(shards)}/{args.expected_shards}`",
        f"- Generated rows: `{len(generated_rows)}/{int(args.expected_shards) * int(args.expected_generated_rows_per_shard)}`",
        f"- Protected strict accepts: `{intish(protected.get('accepts'))}/{args.expected_shards}`",
        f"- Protected accepts ignoring quality: `{intish(protected.get('accepts_ignoring_quality'))}/{args.expected_shards}`",
        f"- Raw accepts: `{intish(raw.get('accepts'))}/{args.expected_shards}`",
        f"- Wrong-key accepts: `{intish(wrong_key.get('accepts'))}/{args.expected_shards}`",
        f"- Wrong-payload accepts: `{intish(wrong_payload.get('accepts'))}/{args.expected_shards}`",
        f"- Global duplicate extra rows: `{generation_summary['global_duplicate_response_hash_extra_rows']}`",
        f"- Within-block duplicate count: `{duplicate_within_block}`",
        f"- Technical forbidden public surface count: `{technical_forbidden}`",
        f"- Trace binding invalid rows: `{trace_invalid}/{trace_checked}`",
        "",
        "## Claim Control",
        "",
        "- This is a 4-block second-family diagnostic, not a locked-scale or paper-facing result.",
        "- Full-phrase decoder outputs are report-only and do not support a text-only phrase success claim.",
        "- FAR, sanitizer, payload diversity, and paper claims remain locked.",
    ]
    write_text_new(output_dir / "review.md", "\n".join(review_lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
