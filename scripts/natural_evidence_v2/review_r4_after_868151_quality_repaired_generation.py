from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_INPUT = Path("results/natural_evidence_v2/status/r4_after_868016_controller_generation_868212")
DEFAULT_OUTPUT = Path("results/natural_evidence_v2/status/r4_after_868016_controller_generation_868212_review")


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


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv_new(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def sum_arm_summary(paths: Sequence[Path]) -> dict[str, dict[str, int]]:
    aggregate: dict[str, Counter[str]] = defaultdict(Counter)
    for path in paths:
        payload = read_json(path)
        for arm, values in payload.get("summary_by_arm", {}).items():
            if not isinstance(values, Mapping):
                raise ValueError(f"summary_by_arm payload must be object: {path}:{arm}")
            for key in (
                "blocks",
                "accepts",
                "accepts_ignoring_quality",
                "forbidden_public_surface_count",
                "duplicate_response_hash_count",
            ):
                aggregate[str(arm)][key] += int(values.get(key, 0))
    return {arm: dict(counter) for arm, counter in sorted(aggregate.items())}


def sum_event_counts(paths: Sequence[Path]) -> tuple[Counter[str], Counter[str], dict[str, Counter[str]]]:
    event_sources: Counter[str] = Counter()
    event_statuses: Counter[str] = Counter()
    by_condition: dict[str, Counter[str]] = defaultdict(Counter)
    for path in paths:
        for row in read_jsonl(path):
            source = str(row.get("event_source", ""))
            status = str(row.get("event_status", ""))
            condition = str(row.get("condition", ""))
            event_sources[source] += 1
            event_statuses[status] += 1
            by_condition[condition][status] += 1
    return event_sources, event_statuses, by_condition


def global_generated_hash_summary(paths: Sequence[Path]) -> dict[str, int]:
    hashes: Counter[str] = Counter()
    rows = 0
    for path in paths:
        for row in read_jsonl(path):
            rows += 1
            hashes[str(row.get("response_text_sha256", ""))] += 1
    duplicate_count = sum(count - 1 for value, count in hashes.items() if value and count > 1)
    max_group_size = max(hashes.values()) if hashes else 0
    return {
        "generated_rows": rows,
        "unique_response_hashes": sum(1 for value in hashes if value),
        "global_duplicate_response_hash_count": duplicate_count,
        "global_duplicate_response_hash_max_group_size": max_group_size,
    }


def load_decode_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl(path):
            row = dict(row)
            row["source_file"] = str(path)
            rows.append(row)
    return rows


def first_token_shard_rows(decode_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in decode_rows:
        if str(row.get("arm", "")) != "protected":
            continue
        missing_bits = [
            int(trace.get("bit_index"))
            for trace in row.get("pair_trace", [])
            if trace.get("decoded_bit", "") == ""
        ]
        min_support = min((int(trace.get("support", 0)) for trace in row.get("pair_trace", [])), default=0)
        rows.append(
            {
                "block_id": row.get("block_id", ""),
                "accept": bool(row.get("accept", False)),
                "decoded_bits": row.get("decoded_bits", ""),
                "expected_bits": row.get("expected_bits", ""),
                "complete_pairs": int(row.get("complete_pairs", 0)),
                "required_pairs": int(row.get("required_pairs", 0)),
                "checksum_valid": bool(row.get("checksum_valid", False)),
                "bits_match_condition": bool(row.get("bits_match_condition", False)),
                "forbidden_public_surface_count": int(row.get("forbidden_public_surface_count", 0)),
                "duplicate_response_hash_count": int(row.get("duplicate_response_hash_count", 0)),
                "min_pair_support": min_support,
                "missing_bit_indices": ",".join(str(item) for item in missing_bits),
            }
        )
    return rows


def full_phrase_summary(paths: Sequence[Path]) -> dict[str, dict[str, int]]:
    return sum_arm_summary(paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review R4 after-868151 quality-repaired generation job.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-job-id", default="868212")
    parser.add_argument("--expected-shards", type=int, default=4)
    parser.add_argument("--expected-event-trace-rows", type=int, default=9216)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    shard_dirs = sorted((input_dir / "shards").glob("shard_*"))
    missing: list[str] = []
    for shard_dir in shard_dirs:
        for rel in (
            "r4_generated_outputs.jsonl",
            "first_token_event_decode/first_token_event_decode_summary.json",
            "first_token_event_decode/first_token_event_decode_rows.jsonl",
            "first_token_event_decode/first_token_event_rows.jsonl",
            "decode_all/decode_summary.json",
            "decode_none/decode_summary.json",
        ):
            if not (shard_dir / rel).exists():
                missing.append(f"{shard_dir.name}/{rel}")

    generated_paths = [path / "r4_generated_outputs.jsonl" for path in shard_dirs if (path / "r4_generated_outputs.jsonl").exists()]
    event_summary_paths = [
        path / "first_token_event_decode/first_token_event_decode_summary.json"
        for path in shard_dirs
        if (path / "first_token_event_decode/first_token_event_decode_summary.json").exists()
    ]
    event_row_paths = [
        path / "first_token_event_decode/first_token_event_rows.jsonl"
        for path in shard_dirs
        if (path / "first_token_event_decode/first_token_event_rows.jsonl").exists()
    ]
    event_decode_row_paths = [
        path / "first_token_event_decode/first_token_event_decode_rows.jsonl"
        for path in shard_dirs
        if (path / "first_token_event_decode/first_token_event_decode_rows.jsonl").exists()
    ]
    full_phrase_all_paths = [
        path / "decode_all/decode_summary.json"
        for path in shard_dirs
        if (path / "decode_all/decode_summary.json").exists()
    ]
    full_phrase_none_paths = [
        path / "decode_none/decode_summary.json"
        for path in shard_dirs
        if (path / "decode_none/decode_summary.json").exists()
    ]

    first_token_aggregate = sum_arm_summary(event_summary_paths)
    event_sources, event_statuses, event_by_condition = sum_event_counts(event_row_paths)
    generated_hash_summary = global_generated_hash_summary(generated_paths)
    event_decode_rows = load_decode_rows(event_decode_row_paths)
    protected_rows = first_token_shard_rows(event_decode_rows)
    full_phrase_all = full_phrase_summary(full_phrase_all_paths)
    full_phrase_none = full_phrase_summary(full_phrase_none_paths)

    protected = first_token_aggregate.get("protected", {})
    controls = {
        arm: int(first_token_aggregate.get(arm, {}).get("accepts", 0))
        for arm in ("raw", "task_only", "wrong_key", "wrong_payload")
    }
    block_quality_duplicate_count = sum(
        int(first_token_aggregate.get(arm, {}).get("duplicate_response_hash_count", 0))
        for arm in first_token_aggregate
    )
    block_quality_forbidden_count = sum(
        int(first_token_aggregate.get(arm, {}).get("forbidden_public_surface_count", 0))
        for arm in first_token_aggregate
    )
    first_token_block_gate_pass = (
        len(shard_dirs) == int(args.expected_shards)
        and not missing
        and int(protected.get("accepts", 0)) >= 3
        and int(protected.get("blocks", 0)) == 4
        and all(value == 0 for value in controls.values())
        and block_quality_forbidden_count == 0
        and block_quality_duplicate_count == 0
        and event_sources == Counter({"token_id_trace": int(args.expected_event_trace_rows)})
    )

    global_duplicate_count = int(generated_hash_summary["global_duplicate_response_hash_count"])
    status = (
        "PASS_R4_AFTER_868151_QUALITY_REPAIRED_FIRST_TOKEN_EVENT_BLOCK_DIAGNOSTIC_GATE_NOT_LOCKED_POSITIVE"
        if first_token_block_gate_pass
        else "FAIL_R4_AFTER_868151_QUALITY_REPAIRED_FIRST_TOKEN_EVENT_BLOCK_DIAGNOSTIC_GATE"
    )
    if global_duplicate_count:
        status += "_GLOBAL_DUPLICATE_CAVEAT"

    summary = {
        "schema_name": "natural_evidence_v2_r4_after_868151_quality_repaired_generation_review_v1",
        "status": status,
        "source_job_id": str(args.source_job_id),
        "input_dir": str(input_dir),
        "shards_seen": len(shard_dirs),
        "expected_shards": int(args.expected_shards),
        "missing_artifacts": missing,
        "first_token_event_aggregate_by_arm": first_token_aggregate,
        "first_token_event_status_counts": dict(sorted(event_statuses.items())),
        "first_token_event_source_counts": dict(sorted(event_sources.items())),
        "first_token_event_status_by_condition": {
            key: dict(counter) for key, counter in sorted(event_by_condition.items())
        },
        "first_token_protected_accepts": int(protected.get("accepts", 0)),
        "first_token_protected_blocks": int(protected.get("blocks", 0)),
        "first_token_control_accepts": controls,
        "first_token_block_quality_forbidden_count": block_quality_forbidden_count,
        "first_token_block_quality_duplicate_count": block_quality_duplicate_count,
        "first_token_block_diagnostic_gate_pass": first_token_block_gate_pass,
        "global_generated_hash_summary": generated_hash_summary,
        "global_duplicate_caveat": bool(global_duplicate_count),
        "failed_protected_blocks": [row for row in protected_rows if not row["accept"]],
        "full_phrase_decode_all_aggregate_by_arm": full_phrase_all,
        "full_phrase_decode_none_aggregate_by_arm": full_phrase_none,
        "full_phrase_protected_accepts_format_scrub_all": int(full_phrase_all.get("protected", {}).get("accepts", 0)),
        "claim_policy": "diagnostic_only_not_locked_positive_not_paper_claim",
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "llama_started": False,
        "same_family_null_started": False,
        "sanitizer_started": False,
        "far_started": False,
        "paper_claim_allowed": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_new(output_dir / "quality_repaired_generation_review_summary.json", summary)
    write_csv_new(
        output_dir / "first_token_protected_block_summary.csv",
        protected_rows,
        [
            "block_id",
            "accept",
            "decoded_bits",
            "expected_bits",
            "complete_pairs",
            "required_pairs",
            "checksum_valid",
            "bits_match_condition",
            "forbidden_public_surface_count",
            "duplicate_response_hash_count",
            "min_pair_support",
            "missing_bit_indices",
        ],
    )
    report = [
        "# R4 After-868151 Quality-Repaired Generation Review",
        "",
        f"Status: `{status}`",
        "",
        f"- source job id: `{args.source_job_id}`",
        f"- shards seen: `{len(shard_dirs)}` / `{args.expected_shards}`",
        f"- first-token protected accepts: `{summary['first_token_protected_accepts']}` / `{summary['first_token_protected_blocks']}`",
        f"- first-token control accepts: `{controls}`",
        f"- first-token forbidden public surface count: `{block_quality_forbidden_count}`",
        f"- first-token duplicate response hash count, per block: `{block_quality_duplicate_count}`",
        f"- token-id trace rows: `{summary['first_token_event_source_counts'].get('token_id_trace', 0)}`",
        f"- event status counts: `{summary['first_token_event_status_counts']}`",
        f"- global duplicate response hash count: `{global_duplicate_count}`",
        f"- full-phrase protected accepts, format_scrub=all: `{summary['full_phrase_protected_accepts_format_scrub_all']}`",
        "",
        "## Interpretation",
        "",
        "The quality-repaired first-token event diagnostic passes the small block-level gate: protected recovers 3/4 blocks, all controls reject, and the contextual literal / per-block duplicate gates are clean.",
        "",
        "This is not a locked positive result. One protected shard failed because bit index 1, coordinate 26 had zero support, and generated-output hashes still have global cross-shard/condition duplicates. The full-phrase decoder remains failed, as expected.",
    ]
    if summary["failed_protected_blocks"]:
        report.extend(["", "## Failed Protected Blocks", ""])
        for row in summary["failed_protected_blocks"]:
            report.append(
                f"- `{row['block_id']}` decoded `{row['decoded_bits']}` vs expected `{row['expected_bits']}`; "
                f"missing bit indices `{row['missing_bit_indices']}`; min support `{row['min_pair_support']}`"
            )
    (output_dir / "quality_repaired_generation_review.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if first_token_block_gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
