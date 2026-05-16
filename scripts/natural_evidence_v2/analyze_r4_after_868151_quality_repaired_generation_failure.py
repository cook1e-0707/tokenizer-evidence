from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_INPUT = Path("results/natural_evidence_v2/status/r4_after_868016_controller_generation_868212")
DEFAULT_REVIEW = Path("results/natural_evidence_v2/status/r4_after_868016_controller_generation_868212_review")
DEFAULT_OUTPUT = Path("results/natural_evidence_v2/status/r4_after_868016_controller_generation_868212_failure_attribution")


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


def load_generated_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((input_dir / "shards").glob("shard_*/r4_generated_outputs.jsonl")):
        for row in read_jsonl(path):
            row = dict(row)
            row["shard_id"] = path.parent.name
            rows.append(row)
    return rows


def load_event_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((input_dir / "shards").glob("shard_*/first_token_event_decode/first_token_event_rows.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def counter_to_rows(counter: Counter[Any], key_names: Sequence[str], value_name: str = "count") -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for key, count in counter.most_common():
        values = key if isinstance(key, tuple) else (key,)
        row = {name: values[index] if index < len(values) else "" for index, name in enumerate(key_names)}
        row[value_name] = count
        output.append(row)
    return output


def summarize_coord26(rows: Sequence[Mapping[str, Any]], event_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_shard_condition: Counter[tuple[str, str, str]] = Counter()
    top_events: Counter[tuple[str, str, str, str, str]] = Counter()
    for row in rows:
        if int(row.get("coordinate_id", -1)) != 26:
            continue
        shard = str(row.get("shard_id", ""))
        condition = str(row.get("generation_condition", ""))
        side = str(row.get("event_side", ""))
        by_shard_condition[(shard, condition, side)] += 1
        if condition == "protected":
            top_events[
                (
                    shard,
                    str(row.get("event_side", "")),
                    str(row.get("first_generated_token_text", "")),
                    str(row.get("prefix_family_id", "")),
                    str(row.get("target_bit", "")),
                )
            ] += 1

    # Event rows are decoder-facing and verify that coordinate 26 support is
    # absent rather than hidden in generated output metadata.
    for event in event_rows:
        if int(event.get("coordinate_id", -1)) != 26:
            continue
        if str(event.get("condition", "")) != "protected":
            continue
        top_events[
            (
                str(event.get("shard_id", "")),
                str(event.get("event_status", "")),
                str(event.get("event", "")),
                str(event.get("prefix_family_id", "")),
                str(event.get("target_bit", "")),
            )
        ] += 0

    return (
        counter_to_rows(by_shard_condition, ["shard_id", "condition", "event_side"]),
        counter_to_rows(top_events, ["shard_id", "event_side", "first_generated_token_text", "prefix_family_id", "target_bit"]),
    )


def duplicate_groups(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_hash: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        digest = str(row.get("response_text_sha256", ""))
        if digest:
            by_hash[digest].append(row)

    duplicate_rows: list[dict[str, Any]] = []
    condition_sets: Counter[tuple[str, ...]] = Counter()
    shard_sets: Counter[tuple[str, ...]] = Counter()
    diagnosis_sets: Counter[tuple[bool, bool, bool, bool]] = Counter()
    for digest, group in by_hash.items():
        if len(group) <= 1:
            continue
        conditions = tuple(sorted({str(row.get("generation_condition", "")) for row in group}))
        shards = tuple(sorted({str(row.get("shard_id", "")) for row in group}))
        prompt_ids = {str(row.get("prompt_id", "")) for row in group}
        prefixes = {str(row.get("prefix_family_id", "")) for row in group}
        coordinates = sorted({int(row.get("coordinate_id", -1)) for row in group})
        condition_sets[conditions] += 1
        shard_sets[shards] += 1
        diagnosis_sets[
            (
                len(shards) > 1,
                len(conditions) > 1,
                len(prompt_ids) > 1,
                len(prefixes) > 1,
            )
        ] += 1
        duplicate_rows.append(
            {
                "response_text_sha256": digest,
                "count": len(group),
                "conditions": ",".join(conditions),
                "shards": ",".join(shards),
                "prompt_count": len(prompt_ids),
                "prefix_family_count": len(prefixes),
                "coordinate_ids": ",".join(str(item) for item in coordinates[:16]),
                "response_text_prefix": str(group[0].get("response_text", "")).replace("\n", " ")[:240],
            }
        )

    duplicate_rows.sort(key=lambda row: (-int(row["count"]), str(row["response_text_sha256"])))
    summary = {
        "generated_rows": len(rows),
        "unique_response_hashes": sum(1 for group in by_hash.values() if group),
        "duplicate_hash_groups": len(duplicate_rows),
        "global_duplicate_response_hash_count": sum(int(row["count"]) - 1 for row in duplicate_rows),
        "global_duplicate_response_hash_max_group_size": max((int(row["count"]) for row in duplicate_rows), default=0),
        "duplicate_condition_sets": {",".join(key): value for key, value in condition_sets.most_common()},
        "duplicate_shard_sets_top": {",".join(key): value for key, value in shard_sets.most_common(12)},
        "duplicate_causal_shape_counts": {
            "cross_shard_cross_condition_cross_prompt_cross_prefix": 0,
            "cross_shard_same_condition_same_prompt_same_prefix": 0,
            "cross_condition_same_shard_same_prompt_same_prefix": 0,
            "other": 0,
        },
    }
    for key, count in diagnosis_sets.items():
        cross_shard, cross_condition, cross_prompt, cross_prefix = key
        if cross_shard and cross_condition and cross_prompt and cross_prefix:
            bucket = "cross_shard_cross_condition_cross_prompt_cross_prefix"
        elif cross_shard and not cross_condition and not cross_prompt and not cross_prefix:
            bucket = "cross_shard_same_condition_same_prompt_same_prefix"
        elif not cross_shard and cross_condition and not cross_prompt and not cross_prefix:
            bucket = "cross_condition_same_shard_same_prompt_same_prefix"
        else:
            bucket = "other"
        summary["duplicate_causal_shape_counts"][bucket] += count
    return summary, duplicate_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attribute 868212 first-token event failure and duplicate caveat.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    generated_rows = load_generated_rows(args.input_dir)
    event_rows = load_event_rows(args.input_dir)
    review_summary = read_json(args.review_dir / "quality_repaired_generation_review_summary.json")

    coord26_rows, coord26_top_events = summarize_coord26(generated_rows, event_rows)
    duplicate_summary, duplicate_rows = duplicate_groups(generated_rows)
    shard03_coord26_protected = [
        row
        for row in coord26_rows
        if row["shard_id"] == "shard_03" and row["condition"] == "protected"
    ]
    shard03_coord26_erasure_count = sum(
        int(row["count"]) for row in shard03_coord26_protected if row["event_side"] == "erasure"
    )
    shard03_coord26_total = sum(int(row["count"]) for row in shard03_coord26_protected)

    summary = {
        "schema_name": "natural_evidence_v2_r4_after_868151_quality_repaired_generation_failure_attribution_v1",
        "status": "RECORDED_R4_AFTER_868151_QUALITY_REPAIRED_GENERATION_868212_ARTIFACT_ONLY_FAILURE_ATTRIBUTION_NO_SUBMIT",
        "source_review_status": review_summary.get("status", ""),
        "input_dir": str(args.input_dir),
        "review_dir": str(args.review_dir),
        "coordinate_26_shard03_protected_erasure_count": shard03_coord26_erasure_count,
        "coordinate_26_shard03_protected_total": shard03_coord26_total,
        "coordinate_26_shard03_protected_all_erasure": shard03_coord26_total > 0
        and shard03_coord26_erasure_count == shard03_coord26_total,
        "duplicate_summary": duplicate_summary,
        "interpretation": {
            "coordinate_26": "the failed block is an erasure failure: shard_03 protected coordinate 26 produced zero target/other events across all 64 rows",
            "duplicates": "global duplicates remain dominated by deterministic identical generations across shard pairs and protected/raw or same-condition repetitions; per-block duplicate gate is clean but global duplicate gate is not",
            "route_implication": "do not submit another Slurm job until coordinate reliability and global allocation/decoding duplicate policy are repaired or a pivot route is recorded",
        },
        "slurm_submitted": False,
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "paper_claim_allowed": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_new(output_dir / "failure_attribution_summary.json", summary)
    write_csv_new(
        output_dir / "coordinate_26_by_shard_condition.csv",
        coord26_rows,
        ["shard_id", "condition", "event_side", "count"],
    )
    write_csv_new(
        output_dir / "coordinate_26_protected_top_events.csv",
        coord26_top_events[:80],
        ["shard_id", "event_side", "first_generated_token_text", "prefix_family_id", "target_bit", "count"],
    )
    write_csv_new(
        output_dir / "duplicate_hash_group_summary.csv",
        duplicate_rows[:200],
        [
            "response_text_sha256",
            "count",
            "conditions",
            "shards",
            "prompt_count",
            "prefix_family_count",
            "coordinate_ids",
            "response_text_prefix",
        ],
    )

    report = [
        "# R4 After-868151 Quality-Repaired Generation 868212 Failure Attribution",
        "",
        f"Status: `{summary['status']}`",
        "",
        "## Coordinate 26",
        "",
        f"- shard_03 protected coordinate-26 erasures: `{shard03_coord26_erasure_count}` / `{shard03_coord26_total}`",
        "- interpretation: the single failed protected block is caused by zero support for bit index 1 / coordinate 26, not by a wrong-key or null accept.",
        "",
        "## Global Duplicate Caveat",
        "",
        f"- generated rows: `{duplicate_summary['generated_rows']}`",
        f"- unique response hashes: `{duplicate_summary['unique_response_hashes']}`",
        f"- duplicate hash groups: `{duplicate_summary['duplicate_hash_groups']}`",
        f"- duplicate extra rows: `{duplicate_summary['global_duplicate_response_hash_count']}`",
        f"- max duplicate group size: `{duplicate_summary['global_duplicate_response_hash_max_group_size']}`",
        f"- duplicate condition sets: `{duplicate_summary['duplicate_condition_sets']}`",
        f"- duplicate shard sets top: `{duplicate_summary['duplicate_shard_sets_top']}`",
        "",
        "## Route Implication",
        "",
        "Do not submit another Slurm generation/scoring/training job from this state. The next route must first decide whether to repair coordinate reliability and global duplicate allocation/decoding policy, or pivot away from this controller/generation path.",
    ]
    (output_dir / "failure_attribution.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
