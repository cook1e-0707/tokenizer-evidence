from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]


ARMS = ("protected", "raw", "task_only", "wrong_key", "wrong_payload")
CONTROL_ARMS = ("raw", "task_only", "wrong_key", "wrong_payload")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate R4 after-869348 first-token event locked-scale generation shards. "
            "This is artifact-only: it reads generated/decode/trace artifacts and writes "
            "review tables. It does not generate, train, submit Slurm, run Llama, run FAR, "
            "or make paper-facing claims."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--shard-roots",
        type=Path,
        nargs="+",
        required=True,
        help="One or more directories containing shard_XX subdirectories. Later roots may fill missing shards.",
    )
    parser.add_argument("--expected-shards", type=int, default=96)
    parser.add_argument("--protected-strict-min", type=int, default=85)
    parser.add_argument("--protected-ignoring-quality-min", type=int, default=90)
    parser.add_argument("--control-accepts-max", type=int, default=0)
    parser.add_argument("--global-duplicate-extra-max", type=int, default=0)
    parser.add_argument("--within-block-duplicate-max", type=int, default=0)
    parser.add_argument("--forbidden-public-surface-max", type=int, default=0)
    parser.add_argument("--expected-generated-rows-per-shard", type=int, default=3072)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write an incomplete summary and exit 0 instead of failing when some shards are missing.",
    )
    parser.add_argument(
        "--allow-missing-generated-jsonl",
        action="store_true",
        help="Allow summary-only aggregation when generated JSONL files are unavailable locally.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            yield payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def int_value(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(value)


def shard_id_from_dir(path: Path) -> int:
    name = path.name
    if not name.startswith("shard_"):
        raise ValueError(f"unexpected shard directory name: {path}")
    return int(name.split("_", 1)[1])


def required_summary_files(shard_dir: Path) -> list[Path]:
    return [
        shard_dir / "first_token_event_decode/first_token_event_decode_summary.json",
        shard_dir / "first_token_event_decode/first_token_event_per_block.csv",
        shard_dir / "decode_all/decode_summary.json",
        shard_dir / "decode_all/per_block_decode.csv",
        shard_dir / "decode_none/decode_summary.json",
        shard_dir / "decode_none/per_block_decode.csv",
        shard_dir / "trace_binding_validation.json",
    ]


def shard_complete(shard_dir: Path, *, require_generated_jsonl: bool) -> bool:
    files = required_summary_files(shard_dir)
    if require_generated_jsonl:
        files.append(shard_dir / "r4_generated_outputs.jsonl")
    return all(path.is_file() and path.stat().st_size > 0 for path in files)


def discover_shards(
    roots: Sequence[Path],
    *,
    expected_shards: int,
    require_generated_jsonl: bool,
) -> tuple[dict[int, Path], dict[int, list[Path]], list[int], list[int]]:
    complete: dict[int, Path] = {}
    partial: dict[int, list[Path]] = {index: [] for index in range(expected_shards)}
    duplicate_complete: dict[int, list[Path]] = {}
    for root in roots:
        for shard_dir in sorted(root.glob("shard_*")):
            if not shard_dir.is_dir():
                continue
            shard_index = shard_id_from_dir(shard_dir)
            if shard_index < 0 or shard_index >= expected_shards:
                raise ValueError(f"unexpected shard index {shard_index}: {shard_dir}")
            if shard_complete(shard_dir, require_generated_jsonl=require_generated_jsonl):
                if shard_index in complete:
                    duplicate_complete.setdefault(shard_index, [complete[shard_index]]).append(shard_dir)
                else:
                    complete[shard_index] = shard_dir
            else:
                partial.setdefault(shard_index, []).append(shard_dir)
    if duplicate_complete:
        raise ValueError(
            "duplicate complete shard artifacts found; refusing aggregation: "
            + json.dumps({str(k): [str(v) for v in vals] for k, vals in duplicate_complete.items()}, sort_keys=True)
        )
    missing = [index for index in range(expected_shards) if index not in complete and not partial.get(index)]
    incomplete = [index for index in range(expected_shards) if index not in complete]
    return complete, partial, missing, incomplete


def summarize_first_token(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, dict[str, int]] = {
        arm: {
            "blocks": 0,
            "accepts": 0,
            "accepts_ignoring_quality": 0,
            "forbidden_public_surface_count": 0,
            "duplicate_response_hash_count": 0,
        }
        for arm in ARMS
    }
    for row in rows:
        arm = str(row["arm"])
        if arm not in by_arm:
            raise ValueError(f"unexpected arm in first-token decode row: {arm}")
        by_arm[arm]["blocks"] += 1
        by_arm[arm]["accepts"] += int(bool_value(row.get("accept")))
        by_arm[arm]["accepts_ignoring_quality"] += int(bool_value(row.get("accept_ignoring_quality")))
        by_arm[arm]["forbidden_public_surface_count"] += int_value(row.get("forbidden_public_surface_count"))
        by_arm[arm]["duplicate_response_hash_count"] += int_value(row.get("duplicate_response_hash_count"))
    return by_arm


def summarize_full_phrase(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    by_mode_arm: dict[str, dict[str, int]] = {}
    for row in rows:
        mode = str(row["format_scrub_mode"])
        arm = str(row["arm"])
        key = f"{mode}:{arm}"
        if key not in by_mode_arm:
            by_mode_arm[key] = {"blocks": 0, "accepts": 0, "forbidden_public_surface_count": 0}
        by_mode_arm[key]["blocks"] += 1
        by_mode_arm[key]["accepts"] += int(bool_value(row.get("accept")))
        by_mode_arm[key]["forbidden_public_surface_count"] += int_value(row.get("forbidden_public_surface_count"))
    return by_mode_arm


def summarize_generated_jsonl(shard_dirs: Mapping[int, Path]) -> dict[str, Any]:
    response_hash_counts: Counter[str] = Counter()
    generation_id_counts: Counter[str] = Counter()
    rows_by_condition: Counter[str] = Counter()
    row_count = 0
    files = []
    for shard_index, shard_dir in sorted(shard_dirs.items()):
        path = shard_dir / "r4_generated_outputs.jsonl"
        files.append(str(path))
        for row in read_jsonl(path):
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
        "generated_output_files": files,
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
    require_jsonl = not args.allow_missing_generated_jsonl
    complete, partial, missing, incomplete = discover_shards(
        roots,
        expected_shards=args.expected_shards,
        require_generated_jsonl=require_jsonl,
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
    trace_checked_rows = sum(int(row["checked_rows"]) for row in trace_rows)
    trace_invalid_rows = sum(int(row["invalid_rows"]) for row in trace_rows)
    trace_status_pass = bool(trace_rows) and all(row["status"] == "PASS_R4_FIRST_TOKEN_EVENT_TRACE_BINDING" for row in trace_rows)

    if require_jsonl:
        generation_summary = summarize_generated_jsonl(complete)
    else:
        generation_summary = {
            "generated_rows": None,
            "global_duplicate_response_hash_extra_rows": None,
            "global_duplicate_response_hash_group_count": None,
            "max_response_hash_group_size": None,
            "duplicate_generation_id_extra_rows": None,
            "duplicate_generation_id_group_count": None,
            "max_generation_id_group_size": None,
            "note": "generated JSONL duplicate scan skipped by --allow-missing-generated-jsonl",
        }

    expected_generated_rows = int(args.expected_shards) * int(args.expected_generated_rows_per_shard)
    protected = first_token_summary["protected"]
    control_strict_accepts = {arm: first_token_summary[arm]["accepts"] for arm in CONTROL_ARMS}
    control_ignoring_quality_accepts = {
        arm: first_token_summary[arm]["accepts_ignoring_quality"] for arm in CONTROL_ARMS
    }
    forbidden_public_surface_count = sum(arm["forbidden_public_surface_count"] for arm in first_token_summary.values())
    within_block_duplicate_response_hash_count = sum(
        arm["duplicate_response_hash_count"] for arm in first_token_summary.values()
    )
    generated_rows_ok = (
        generation_summary["generated_rows"] == expected_generated_rows
        if generation_summary["generated_rows"] is not None
        else False
    )
    global_duplicate_extra = generation_summary["global_duplicate_response_hash_extra_rows"]
    global_duplicate_ok = (
        global_duplicate_extra <= int(args.global_duplicate_extra_max)
        if global_duplicate_extra is not None
        else False
    )
    all_complete = len(complete) == int(args.expected_shards)
    gate_pass = (
        all_complete
        and protected["blocks"] == int(args.expected_shards)
        and protected["accepts"] >= int(args.protected_strict_min)
        and protected["accepts_ignoring_quality"] >= int(args.protected_ignoring_quality_min)
        and all(value <= int(args.control_accepts_max) for value in control_strict_accepts.values())
        and all(value <= int(args.control_accepts_max) for value in control_ignoring_quality_accepts.values())
        and within_block_duplicate_response_hash_count <= int(args.within_block_duplicate_max)
        and forbidden_public_surface_count <= int(args.forbidden_public_surface_max)
        and trace_status_pass
        and trace_invalid_rows == 0
        and trace_checked_rows == expected_generated_rows
        and generated_rows_ok
        and global_duplicate_ok
    )
    status = (
        "PASS_R4_AFTER_869348_LOCKED_SCALE_GENERATION_GATE"
        if gate_pass
        else (
            "INCOMPLETE_R4_AFTER_869348_LOCKED_SCALE_GENERATION_NO_GATE"
            if not all_complete
            else "FAIL_R4_AFTER_869348_LOCKED_SCALE_GENERATION_GATE"
        )
    )

    summary = {
        "schema_name": "natural_evidence_v2_r4_after_869348_locked_scale_generation_aggregate_v1",
        "status": status,
        "scale_gate_pass": bool(gate_pass),
        "all_shards_complete": all_complete,
        "complete_shards": sorted(complete),
        "complete_shard_count": len(complete),
        "expected_shards": int(args.expected_shards),
        "missing_shards": missing,
        "incomplete_shards": incomplete,
        "shard_roots": [str(path) for path in roots],
        "first_token_event_summary_by_arm": first_token_summary,
        "full_phrase_decoder_report_only_summary": full_phrase_summary,
        "trace_binding": {
            "checked_rows": trace_checked_rows,
            "expected_checked_rows": expected_generated_rows,
            "invalid_rows": trace_invalid_rows,
            "all_shard_trace_status_pass": trace_status_pass,
        },
        "generation_duplicate_summary": generation_summary,
        "gate_targets": {
            "protected_strict_accepts_min": int(args.protected_strict_min),
            "protected_accepts_ignoring_quality_min": int(args.protected_ignoring_quality_min),
            "control_accepts_max_per_condition": int(args.control_accepts_max),
            "within_block_duplicate_response_hash_count_max": int(args.within_block_duplicate_max),
            "global_duplicate_response_hash_extra_rows_max": int(args.global_duplicate_extra_max),
            "technical_forbidden_public_surface_count_max": int(args.forbidden_public_surface_max),
            "trace_binding_validity_required": 1.0,
            "full_phrase_decoder_policy": "report_only_not_success_claim",
        },
        "claim_control": {
            "paper_claim_allowed": False,
            "training_allowed": False,
            "llama_allowed": False,
            "same_family_null_allowed": False,
            "sanitizer_allowed": False,
            "far_aggregation_allowed": False,
            "payload_diversity_tested": False,
            "text_only_phrase_decoder_success_claim": False,
        },
    }

    write_csv(
        output_dir / "locked_scale_shard_matrix.csv",
        shard_matrix,
        ["shard_index", "shard_id", "status", "selected_shard_dir", "partial_dirs"],
    )
    write_csv(
        output_dir / "locked_scale_first_token_blocks.csv",
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
        output_dir / "locked_scale_full_phrase_blocks.csv",
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
        output_dir / "locked_scale_trace_binding_by_shard.csv",
        trace_rows,
        ["shard_index", "shard_id", "status", "checked_rows", "invalid_rows", "source_shard_dir"],
    )
    write_json(output_dir / "locked_scale_generation_duplicate_summary.json", generation_summary)
    write_json(output_dir / "locked_scale_summary.json", summary)
    print(json.dumps({"status": status, "output_dir": str(output_dir), "complete_shards": len(complete)}, sort_keys=True))

    if not all_complete and not args.allow_incomplete:
        return 2
    if all_complete and not gate_pass:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
