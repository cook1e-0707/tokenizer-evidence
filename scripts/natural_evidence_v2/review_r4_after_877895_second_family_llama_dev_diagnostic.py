#!/usr/bin/env python3
"""Review the R4 after-877895 second-family Llama 32-block dev diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_cover_natural_common import write_json_new, write_text_new  # noqa: E402


ARMS = ("protected", "raw", "task_only", "wrong_key", "wrong_payload")
CONTROL_ARMS = ("raw", "task_only", "wrong_key", "wrong_payload")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job-id", default="879248")
    parser.add_argument("--expected-shards", type=int, default=32)
    parser.add_argument("--expected-generated-rows-per-shard", type=int, default=2048)
    parser.add_argument("--protected-strict-accepts-min", type=int, default=28)
    parser.add_argument("--protected-ignoring-quality-accepts-min", type=int, default=30)
    parser.add_argument("--control-accepts-max", type=int, default=0)
    parser.add_argument("--global-duplicate-extra-max", type=int, default=0)
    parser.add_argument("--technical-forbidden-max", type=int, default=0)
    parser.add_argument("--trace-invalid-max", type=int, default=0)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            yield payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
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


def summarize_generated(path: Path) -> tuple[int, Counter[str], Counter[str], Counter[str], Counter[str], int, int]:
    rows = 0
    by_condition: Counter[str] = Counter()
    response_hashes: Counter[str] = Counter()
    generation_ids: Counter[str] = Counter()
    wrong_replay: Counter[str] = Counter()
    for row in iter_jsonl(path):
        rows += 1
        by_condition[str(row.get("generation_condition", row.get("arm", "")))] += 1
        response_hash = str(row.get("output_text_sha256") or row.get("response_text_sha256") or "")
        generation_id = str(row.get("generation_id", ""))
        if response_hash:
            response_hashes[response_hash] += 1
        if generation_id:
            generation_ids[generation_id] += 1
        wrong_replay["wrong_key"] += int(boolish(row.get("wrong_key_replay_accept")))
        wrong_replay["wrong_payload"] += int(boolish(row.get("wrong_payload_replay_accept")))
    duplicate_extra = sum(count - 1 for count in response_hashes.values() if count > 1)
    duplicate_generation_id_extra = sum(count - 1 for count in generation_ids.values() if count > 1)
    return rows, by_condition, response_hashes, generation_ids, wrong_replay, duplicate_extra, duplicate_generation_id_extra


def main() -> int:
    args = parse_args()
    run_root = resolve(args.run_root)
    output_dir = resolve(args.output_dir)
    shards = discover_shards(run_root)

    per_block_rows: list[dict[str, Any]] = []
    full_phrase_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []
    errors: list[str] = []

    generated_rows = 0
    rows_by_condition: Counter[str] = Counter()
    response_hashes: Counter[str] = Counter()
    generation_ids: Counter[str] = Counter()
    wrong_replay_accepts: Counter[str] = Counter()
    event_sources: Counter[str] = Counter()
    event_statuses: Counter[str] = Counter()

    if len(shards) != args.expected_shards:
        errors.append(f"expected {args.expected_shards} shards, found {sorted(shards)}")

    for shard_index, shard_dir in sorted(shards.items()):
        generated_path = shard_dir / "r4_generated_outputs.jsonl"
        first_token_path = shard_dir / "first_token_event_decode/first_token_event_per_block.csv"
        first_event_path = shard_dir / "first_token_event_decode/first_token_event_rows.jsonl"
        trace_path = shard_dir / "trace_binding_validation.json"

        if not generated_path.exists():
            errors.append(f"missing generated outputs: {generated_path}")
            continue
        (
            shard_generated_rows,
            shard_by_condition,
            shard_response_hashes,
            shard_generation_ids,
            shard_wrong_replay,
            _duplicate_extra,
            _duplicate_generation_id_extra,
        ) = summarize_generated(generated_path)
        generated_rows += shard_generated_rows
        rows_by_condition.update(shard_by_condition)
        response_hashes.update(shard_response_hashes)
        generation_ids.update(shard_generation_ids)
        wrong_replay_accepts.update(shard_wrong_replay)
        if shard_generated_rows != args.expected_generated_rows_per_shard:
            errors.append(f"shard_{shard_index:02d} generated rows mismatch: {shard_generated_rows}")

        shard_first_token = read_csv(first_token_path) if first_token_path.exists() else []
        if not shard_first_token:
            errors.append(f"missing first-token per-block rows for shard_{shard_index:02d}")
        for row in shard_first_token:
            row["shard_index"] = shard_index
            row["source_shard_dir"] = str(shard_dir)
            per_block_rows.append(dict(row))

        if first_event_path.exists():
            for row in iter_jsonl(first_event_path):
                event_sources[str(row.get("event_source", ""))] += 1
                event_statuses[str(row.get("event_status", ""))] += 1
        else:
            errors.append(f"missing first-token event rows for shard_{shard_index:02d}")

        for mode in ("decode_all", "decode_none"):
            path = shard_dir / mode / "per_block_decode.csv"
            if not path.exists():
                errors.append(f"missing full-phrase report-only rows for shard_{shard_index:02d} {mode}")
                continue
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
                "generated_rows": shard_generated_rows,
                "rows_by_condition": dict(sorted(shard_by_condition.items())),
                "first_token_summary_by_arm": summarize_first_token(shard_first_token),
                "trace_invalid_rows": trace_rows[-1]["invalid_rows"] if trace_rows else "",
            }
        )

    first_token_summary = summarize_first_token(per_block_rows)
    trace_checked = sum(row["checked_rows"] for row in trace_rows)
    trace_invalid = sum(row["invalid_rows"] for row in trace_rows)
    global_duplicate_extra = sum(count - 1 for count in response_hashes.values() if count > 1)
    global_duplicate_groups = sum(1 for count in response_hashes.values() if count > 1)
    max_duplicate_group = max((count for count in response_hashes.values() if count > 1), default=1)
    duplicate_generation_id_extra = sum(count - 1 for count in generation_ids.values() if count > 1)
    duplicate_generation_id_groups = sum(1 for count in generation_ids.values() if count > 1)
    max_generation_id_group = max((count for count in generation_ids.values() if count > 1), default=1)

    full_phrase_summary: dict[str, Counter[str]] = {}
    for row in full_phrase_rows:
        key = f"{row.get('format_mode', '')}:{row.get('arm', '')}"
        full_phrase_summary.setdefault(key, Counter())
        full_phrase_summary[key]["blocks"] += 1
        full_phrase_summary[key]["accepts"] += int(boolish(row.get("accept")))
        full_phrase_summary[key]["forbidden_public_surface_count"] += intish(row.get("forbidden_public_surface_count"))

    protected = first_token_summary.get("protected", {})
    controls = {arm: first_token_summary.get(arm, {}) for arm in CONTROL_ARMS}
    control_accepts = {arm: intish(summary.get("accepts")) for arm, summary in controls.items()}
    technical_forbidden = sum(
        intish(first_token_summary.get(arm, {}).get("forbidden_public_surface_count")) for arm in ARMS
    )
    duplicate_within_block = sum(
        intish(first_token_summary.get(arm, {}).get("duplicate_response_hash_count")) for arm in ARMS
    )
    expected_generated_rows = args.expected_shards * args.expected_generated_rows_per_shard
    gate_pass = (
        not errors
        and generated_rows == expected_generated_rows
        and intish(protected.get("accepts")) >= args.protected_strict_accepts_min
        and intish(protected.get("accepts_ignoring_quality")) >= args.protected_ignoring_quality_accepts_min
        and all(value <= args.control_accepts_max for value in control_accepts.values())
        and global_duplicate_extra <= args.global_duplicate_extra_max
        and duplicate_within_block == 0
        and technical_forbidden <= args.technical_forbidden_max
        and trace_invalid <= args.trace_invalid_max
        and wrong_replay_accepts["wrong_key"] == 0
        and wrong_replay_accepts["wrong_payload"] == 0
    )
    status = (
        f"PASS_R4_AFTER_877895_SECOND_FAMILY_LLAMA_DEV_DIAGNOSTIC_{args.job_id}_REVIEWED"
        if gate_pass
        else f"FAIL_R4_AFTER_877895_SECOND_FAMILY_LLAMA_DEV_DIAGNOSTIC_{args.job_id}_REVIEWED_NO_ADOPT"
    )
    generation_summary = {
        "generated_rows": generated_rows,
        "rows_by_condition": dict(sorted(rows_by_condition.items())),
        "unique_response_hashes": len(response_hashes),
        "global_duplicate_response_hash_extra_rows": global_duplicate_extra,
        "global_duplicate_response_hash_group_count": global_duplicate_groups,
        "max_response_hash_group_size": max_duplicate_group,
        "duplicate_generation_id_extra_rows": duplicate_generation_id_extra,
        "duplicate_generation_id_group_count": duplicate_generation_id_groups,
        "max_generation_id_group_size": max_generation_id_group,
        "wrong_key_replay_accept_rows": wrong_replay_accepts["wrong_key"],
        "wrong_payload_replay_accept_rows": wrong_replay_accepts["wrong_payload"],
    }
    summary = {
        "schema_name": "r4_after_877895_second_family_llama_dev_diagnostic_review_v1",
        "status": status,
        "errors": errors,
        "job_id": str(args.job_id),
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "complete_shards": len(shards),
        "expected_shards": args.expected_shards,
        "generated_rows": generated_rows,
        "expected_generated_rows": expected_generated_rows,
        "generation_summary": generation_summary,
        "first_token_event_summary_by_arm": first_token_summary,
        "control_accepts": control_accepts,
        "event_sources": dict(sorted(event_sources.items())),
        "event_statuses": dict(sorted(event_statuses.items())),
        "trace_binding": {
            "checked_rows": trace_checked,
            "invalid_rows": trace_invalid,
        },
        "technical_forbidden_public_surface_count": technical_forbidden,
        "duplicate_within_block_count": duplicate_within_block,
        "full_phrase_decoder_policy": "report_only_not_success_claim",
        "full_phrase_decoder_summary_by_mode_arm": {
            key: dict(counter) for key, counter in sorted(full_phrase_summary.items())
        },
        "same_family_raw_null_rejection_already_supported_by_877895": True,
        "second_family_dev_diagnostic_gate_pass": bool(gate_pass),
        "second_family_dev_diagnostic_internal_claim_allowed": bool(gate_pass),
        "cross_family_locked_scale_success_claim_allowed": False,
        "paper_claim_allowed": False,
        "training_started": False,
        "far_aggregation_started": False,
        "sanitizer_started": False,
        "payload_diversity_started": False,
        "text_only_phrase_success_claim_allowed": False,
        "next_allowed_action": (
            "Record reviewed Llama locked-scale route planning or expert review; do not make "
            "paper-facing or locked-scale transfer claims from this 32-block dev diagnostic."
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
    write_csv(output_dir / "trace_binding_by_shard.csv", trace_rows, ["shard_index", "status", "checked_rows", "invalid_rows"])
    write_csv(output_dir / "full_phrase_decode_report_only.csv", full_phrase_rows, sorted({k for r in full_phrase_rows for k in r}))
    write_json_new(output_dir / "shard_summaries.json", {"shards": shard_summaries})

    review_lines = [
        f"# R4 After-877895 Llama Dev Diagnostic {args.job_id} Review",
        "",
        f"Status: `{status}`",
        "",
        f"This review is artifact-only over completed job `{args.job_id}`. It does not make a paper-facing claim.",
        "",
        "## Gate Summary",
        "",
        f"- Complete shards: `{len(shards)}/{args.expected_shards}`",
        f"- Generated rows: `{generated_rows}/{expected_generated_rows}`",
        f"- Protected strict accepts: `{intish(protected.get('accepts'))}/{args.expected_shards}`",
        f"- Protected accepts ignoring quality: `{intish(protected.get('accepts_ignoring_quality'))}/{args.expected_shards}`",
        f"- Raw accepts: `{control_accepts.get('raw', 0)}/{args.expected_shards}`",
        f"- Task-only accepts: `{control_accepts.get('task_only', 0)}/{args.expected_shards}`",
        f"- Wrong-key accepts: `{control_accepts.get('wrong_key', 0)}/{args.expected_shards}`",
        f"- Wrong-payload accepts: `{control_accepts.get('wrong_payload', 0)}/{args.expected_shards}`",
        f"- Global duplicate extra rows: `{global_duplicate_extra}`",
        f"- Within-block duplicate count: `{duplicate_within_block}`",
        f"- Technical forbidden public surface count: `{technical_forbidden}`",
        f"- Trace binding invalid rows: `{trace_invalid}/{trace_checked}`",
        "",
        "## Report-Only Signals",
        "",
        "- Full-phrase decoder outputs are report-only and do not support a text-only phrase success claim.",
        f"- Full-phrase summary: `{json.dumps(summary['full_phrase_decoder_summary_by_mode_arm'], sort_keys=True)}`",
        "",
        "## Claim Control",
        "",
        "- This is a 32-block second-family dev diagnostic, not a locked-scale or paper-facing result.",
        "- It can support the internal statement that the first-token event route passes a Llama dev diagnostic if the gate passed.",
        "- Llama locked-scale transfer, FAR, sanitizer, payload diversity, and paper claims remain gated.",
    ]
    write_text_new(output_dir / "review.md", "\n".join(review_lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
