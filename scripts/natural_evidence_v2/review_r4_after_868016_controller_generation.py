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


def merge_summary(summary: Mapping[str, Any], scrub_mode: str, shard_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_arm = summary.get("summary_by_arm", {})
    if not isinstance(by_arm, Mapping):
        raise ValueError("decode summary missing summary_by_arm")
    for arm, payload in by_arm.items():
        if not isinstance(payload, Mapping):
            raise ValueError(f"decode summary arm payload must be object: {arm}")
        rows.append(
            {
                "shard_id": shard_id,
                "scrub_mode": scrub_mode,
                "arm": str(arm),
                "blocks": int(payload.get("blocks", 0)),
                "accepts": int(payload.get("accepts", 0)),
                "forbidden_public_surface_count": int(payload.get("forbidden_public_surface_count", 0)),
            }
        )
    return rows


def summarize_rows(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    counters: dict[str, Counter[str]] = {}
    for row in rows:
        key = f"{row['scrub_mode']}::{row['arm']}"
        counters.setdefault(key, Counter())
        counters[key]["blocks"] += int(row["blocks"])
        counters[key]["accepts"] += int(row["accepts"])
        counters[key]["forbidden_public_surface_count"] += int(row["forbidden_public_surface_count"])
    return {key: dict(counter) for key, counter in sorted(counters.items())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review R4 after-868016 controller generation/decode shards.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=4)
    parser.add_argument("--source-job-id", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    shard_dirs = sorted((input_dir / "shards").glob("shard_*"))
    shard_rows: list[dict[str, Any]] = []
    generated_hashes: Counter[str] = Counter()
    generated_rows = 0
    missing: list[str] = []
    for shard_dir in shard_dirs:
        shard_id = shard_dir.name
        generated_path = shard_dir / "r4_generated_outputs.jsonl"
        if not generated_path.exists():
            missing.append(f"{shard_id}:r4_generated_outputs.jsonl")
            continue
        generated = read_jsonl(generated_path)
        generated_rows += len(generated)
        generated_hashes.update(str(row.get("response_text_sha256", "")) for row in generated)
        for scrub_mode in ("all", "none"):
            summary_path = shard_dir / f"decode_{scrub_mode}" / "decode_summary.json"
            if not summary_path.exists():
                missing.append(f"{shard_id}:decode_{scrub_mode}/decode_summary.json")
                continue
            shard_rows.extend(merge_summary(read_json(summary_path), scrub_mode, shard_id))
    duplicate_response_hash_count = sum(count - 1 for value, count in generated_hashes.items() if value and count > 1)
    aggregate = summarize_rows(shard_rows)
    all_protected = aggregate.get("all::protected", {})
    control_keys = ["all::raw", "all::task_only", "all::wrong_key", "all::wrong_payload"]
    control_accepts = {key: int(aggregate.get(key, {}).get("accepts", 0)) for key in control_keys}
    forbidden_all = sum(
        int(row["forbidden_public_surface_count"]) for row in shard_rows if str(row["scrub_mode"]) == "all"
    )
    gate_pass = (
        len(shard_dirs) == int(args.expected_shards)
        and not missing
        and int(all_protected.get("accepts", 0)) >= 3
        and all(value == 0 for value in control_accepts.values())
        and forbidden_all == 0
        and duplicate_response_hash_count == 0
    )
    status = (
        "PASS_R4_AFTER_868016_CONTROLLER_GENERATION_DIAGNOSTIC_GATE"
        if gate_pass
        else "FAIL_R4_AFTER_868016_CONTROLLER_GENERATION_DIAGNOSTIC_GATE"
    )
    summary = {
        "status": status,
        "schema_name": "r4_after_868016_controller_generation_review_summary_v1",
        "source_job_id": str(args.source_job_id),
        "input_dir": str(input_dir),
        "shards_seen": len(shard_dirs),
        "expected_shards": int(args.expected_shards),
        "missing_artifacts": missing,
        "generated_rows": generated_rows,
        "duplicate_response_hash_count": duplicate_response_hash_count,
        "aggregate_by_scrub_and_arm": aggregate,
        "protected_accepts_format_scrub_all": int(all_protected.get("accepts", 0)),
        "protected_blocks_format_scrub_all": int(all_protected.get("blocks", 0)),
        "control_accepts_format_scrub_all": control_accepts,
        "forbidden_public_surface_count_format_scrub_all": forbidden_all,
        "generation_diagnostic_gate_pass": gate_pass,
        "training_started": False,
        "llama_started": False,
        "far_started": False,
        "sanitizer_started": False,
        "paper_claim_allowed": False,
    }
    write_json_new(output_dir / "controller_generation_review_summary.json", summary)
    write_csv_new(
        output_dir / "controller_generation_shard_summary.csv",
        shard_rows,
        ["shard_id", "scrub_mode", "arm", "blocks", "accepts", "forbidden_public_surface_count"],
    )
    report = [
        "# R4 After-868016 Controller Generation Review",
        "",
        f"Status: `{status}`",
        "",
        f"- source job id: `{args.source_job_id}`",
        f"- shards seen: `{len(shard_dirs)}` / `{args.expected_shards}`",
        f"- generated rows: `{generated_rows}`",
        f"- protected accepts, format_scrub=all: `{summary['protected_accepts_format_scrub_all']}` / `{summary['protected_blocks_format_scrub_all']}`",
        f"- control accepts, format_scrub=all: `{control_accepts}`",
        f"- forbidden public surfaces, format_scrub=all: `{forbidden_all}`",
        f"- duplicate response hashes: `{duplicate_response_hash_count}`",
        "",
        "This is a diagnostic generation review only. It does not unlock paper-facing claims.",
    ]
    (output_dir / "controller_generation_review.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
