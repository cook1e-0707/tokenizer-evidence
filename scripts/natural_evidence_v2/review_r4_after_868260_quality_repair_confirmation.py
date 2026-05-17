from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


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


def write_csv_new(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review R4 after-868260 quality-repair confirmation job.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-job-id", default="868299")
    parser.add_argument("--expected-shards", type=int, default=4)
    parser.add_argument("--expected-rows-per-shard", type=int, default=3072)
    parser.add_argument("--protected-strict-min", type=int, default=None)
    parser.add_argument("--protected-ignoring-quality-min", type=int, default=None)
    parser.add_argument(
        "--pass-status",
        default="PASS_R4_AFTER_868260_QUALITY_REPAIR_CONFIRMATION_FIRST_TOKEN_EVENT_GATE",
    )
    parser.add_argument(
        "--fail-status",
        default="FAIL_R4_AFTER_868260_QUALITY_REPAIR_CONFIRMATION_FIRST_TOKEN_EVENT_GATE",
    )
    parser.add_argument(
        "--next-pass-action",
        default="record a reviewed 32-block first-token event dev diagnostic route",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    shard_dirs = sorted((input_dir / "shards").glob("shard_*"))
    missing: list[str] = []
    per_block_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    full_phrase_rows: list[dict[str, Any]] = []
    generated_hashes: Counter[str] = Counter()
    generated_rows = 0
    attempt_rows = 0
    event_statuses: Counter[str] = Counter()

    for shard_dir in shard_dirs:
        shard_id = shard_dir.name
        generated_path = shard_dir / "r4_generated_outputs.jsonl"
        attempts_path = shard_dir / "r4_generation_attempts.jsonl"
        trace_path = shard_dir / "trace_binding_validation.json"
        event_summary_path = shard_dir / "first_token_event_decode" / "first_token_event_decode_summary.json"
        event_block_path = shard_dir / "first_token_event_decode" / "first_token_event_per_block.csv"
        full_phrase_summary_path = shard_dir / "decode_all" / "decode_summary.json"
        for path in (generated_path, attempts_path, trace_path, event_summary_path, event_block_path, full_phrase_summary_path):
            if not path.exists():
                missing.append(f"{shard_id}:{path.relative_to(shard_dir)}")
        if generated_path.exists():
            generated = read_jsonl(generated_path)
            generated_rows += len(generated)
            generated_hashes.update(str(row.get("response_text_sha256", "")) for row in generated)
        if attempts_path.exists():
            attempt_rows += len(read_jsonl(attempts_path))
        if trace_path.exists():
            trace = read_json(trace_path)
            trace_rows.append(
                {
                    "shard_id": shard_id,
                    "checked_rows": int(trace.get("checked_rows", 0)),
                    "invalid_rows": int(trace.get("invalid_rows", 0)),
                    "status": str(trace.get("status", "")),
                }
            )
        if event_summary_path.exists():
            event_summary = read_json(event_summary_path)
            event_statuses.update({str(key): int(value) for key, value in event_summary.get("event_statuses", {}).items()})
        if event_block_path.exists():
            with event_block_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    row["shard_id"] = shard_id
                    per_block_rows.append(dict(row))
        if full_phrase_summary_path.exists():
            full_summary = read_json(full_phrase_summary_path)
            for arm, payload in full_summary.get("summary_by_arm", {}).items():
                if isinstance(payload, Mapping):
                    full_phrase_rows.append(
                        {
                            "shard_id": shard_id,
                            "arm": str(arm),
                            "blocks": int(payload.get("blocks", 0)),
                            "accepts": int(payload.get("accepts", 0)),
                            "forbidden_public_surface_count": int(payload.get("forbidden_public_surface_count", 0)),
                        }
                    )

    duplicate_response_hash_count = sum(count - 1 for value, count in generated_hashes.items() if value and count > 1)
    protected = [row for row in per_block_rows if row.get("arm") == "protected"]
    controls = [row for row in per_block_rows if row.get("arm") in {"raw", "task_only", "wrong_key", "wrong_payload"}]
    protected_accepts = sum(str(row.get("accept", "")).lower() == "true" for row in protected)
    protected_accepts_ignoring_quality = sum(
        str(row.get("accept_ignoring_quality", "")).lower() == "true" for row in protected
    )
    control_accepts: Counter[str] = Counter()
    control_accepts_ignoring_quality: Counter[str] = Counter()
    for row in controls:
        arm = str(row.get("arm", ""))
        control_accepts[arm] += int(str(row.get("accept", "")).lower() == "true")
        control_accepts_ignoring_quality[arm] += int(str(row.get("accept_ignoring_quality", "")).lower() == "true")
    protected_forbidden = sum(int(row.get("forbidden_public_surface_count", 0)) for row in protected)
    protected_duplicate = sum(int(row.get("duplicate_response_hash_count", 0)) for row in protected)
    trace_checked_rows = sum(int(row["checked_rows"]) for row in trace_rows)
    trace_invalid_rows = sum(int(row["invalid_rows"]) for row in trace_rows)
    full_phrase_protected_accepts = sum(
        int(row["accepts"]) for row in full_phrase_rows if str(row.get("arm", "")) == "protected"
    )
    full_phrase_forbidden = sum(int(row["forbidden_public_surface_count"]) for row in full_phrase_rows)

    protected_strict_min = int(args.protected_strict_min) if args.protected_strict_min is not None else int(args.expected_shards)
    protected_ignoring_min = (
        int(args.protected_ignoring_quality_min)
        if args.protected_ignoring_quality_min is not None
        else int(args.expected_shards)
    )
    expected_generated_rows = int(args.expected_shards) * int(args.expected_rows_per_shard)
    gate_pass = (
        len(shard_dirs) == int(args.expected_shards)
        and not missing
        and generated_rows == expected_generated_rows
        and attempt_rows >= expected_generated_rows
        and protected_accepts >= protected_strict_min
        and protected_accepts_ignoring_quality >= protected_ignoring_min
        and all(control_accepts.get(arm, 0) == 0 for arm in ("raw", "task_only", "wrong_key", "wrong_payload"))
        and all(control_accepts_ignoring_quality.get(arm, 0) == 0 for arm in ("raw", "task_only", "wrong_key", "wrong_payload"))
        and protected_forbidden == 0
        and protected_duplicate == 0
        and duplicate_response_hash_count == 0
        and trace_checked_rows == generated_rows
        and trace_invalid_rows == 0
    )
    status = str(args.pass_status) if gate_pass else str(args.fail_status)
    summary = {
        "schema_name": "natural_evidence_v2_r4_after_868260_quality_repair_confirmation_review_v1",
        "status": status,
        "source_job_id": str(args.source_job_id),
        "input_dir": str(input_dir),
        "shards_seen": len(shard_dirs),
        "expected_shards": int(args.expected_shards),
        "missing_artifacts": missing,
        "expected_generated_rows": expected_generated_rows,
        "expected_rows_per_shard": int(args.expected_rows_per_shard),
        "generated_rows": generated_rows,
        "attempt_rows": attempt_rows,
        "event_statuses": dict(sorted(event_statuses.items())),
        "protected_strict_accepts_min": protected_strict_min,
        "protected_strict_accepts": protected_accepts,
        "protected_accepts_ignoring_quality_min": protected_ignoring_min,
        "protected_accepts_ignoring_quality": protected_accepts_ignoring_quality,
        "protected_blocks": len(protected),
        "control_accepts": dict(sorted(control_accepts.items())),
        "control_accepts_ignoring_quality": dict(sorted(control_accepts_ignoring_quality.items())),
        "protected_forbidden_public_surface_count": protected_forbidden,
        "protected_duplicate_response_hash_count": protected_duplicate,
        "global_duplicate_response_hash_count": duplicate_response_hash_count,
        "trace_binding_checked_rows": trace_checked_rows,
        "trace_binding_invalid_rows": trace_invalid_rows,
        "trace_binding_validity": 1.0 if trace_checked_rows and trace_invalid_rows == 0 else 0.0,
        "full_phrase_protected_accepts_format_scrub_all": full_phrase_protected_accepts,
        "full_phrase_forbidden_public_surface_count_format_scrub_all": full_phrase_forbidden,
        "full_phrase_decoder_policy": "report_only_not_success_claim",
        "generation_diagnostic_gate_pass": gate_pass,
        "paper_claim_allowed": False,
        "training_started": False,
        "llama_started": False,
        "sanitizer_started": False,
        "far_started": False,
        "payload_diversity_claim_allowed": False,
        "next_allowed_action": (
            str(args.next_pass_action)
            if gate_pass
            else "artifact-only failure attribution before any rerun"
        ),
    }
    write_json_new(output_dir / "quality_repair_confirmation_review_summary.json", summary)
    write_csv_new(
        output_dir / "first_token_event_per_block.csv",
        per_block_rows,
        [
            "shard_id",
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
    write_csv_new(
        output_dir / "trace_binding_by_shard.csv",
        trace_rows,
        ["shard_id", "checked_rows", "invalid_rows", "status"],
    )
    write_csv_new(
        output_dir / "full_phrase_decode_summary.csv",
        full_phrase_rows,
        ["shard_id", "arm", "blocks", "accepts", "forbidden_public_surface_count"],
    )
    report = [
        "# R4 After-868260 Quality-Repair Confirmation Review",
        "",
        f"Status: `{status}`",
        "",
        f"- source job id: `{args.source_job_id}`",
        f"- shards seen: `{len(shard_dirs)}` / `{args.expected_shards}`",
        f"- generated rows: `{generated_rows}`",
        f"- generation attempts: `{attempt_rows}`",
        f"- protected strict accepts: `{protected_accepts}` / `{len(protected)}`",
        f"- protected accepts ignoring quality: `{protected_accepts_ignoring_quality}` / `{len(protected)}`",
        f"- control accepts: `{dict(sorted(control_accepts.items()))}`",
        f"- global duplicate response hash count: `{duplicate_response_hash_count}`",
        f"- protected forbidden public surface count: `{protected_forbidden}`",
        f"- trace binding invalid rows: `{trace_invalid_rows}` / `{trace_checked_rows}`",
        f"- full-phrase protected accepts, format_scrub=all: `{full_phrase_protected_accepts}`",
        "",
        "Interpretation:",
        "",
        (
            "This is a first-token event/controller trace diagnostic. It does not "
            "establish a text-only phrase decoder result and does not unlock paper-facing claims."
        ),
    ]
    (output_dir / "quality_repair_confirmation_review.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
