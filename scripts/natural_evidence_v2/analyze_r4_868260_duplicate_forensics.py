from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SOURCE = (
    ROOT
    / "results/natural_evidence_v2/status/"
    "r4_after_868212_repaired_first_token_event_generation_868260"
)
DEFAULT_REVIEW = (
    ROOT
    / "results/natural_evidence_v2/status/"
    "r4_after_868212_repaired_first_token_event_generation_868260_review"
)
DEFAULT_FAILURE = (
    ROOT
    / "results/natural_evidence_v2/status/"
    "r4_after_868212_repaired_first_token_event_generation_868260_failure_analysis/"
    "failure_analysis_summary.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/natural_evidence_v2/status/r4_868260_duplicate_forensics_20260517"
)


def read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object in {path}:{line_no}")
            payload.setdefault("_source_file", str(path))
            rows.append(payload)
    return rows


def scrub_text(text: str) -> str:
    text = re.sub(r"(?im)^\s*(?:[-*•]|\d+[.)]|step\s+\d+\s*:)\s*", " ", text)
    text = re.sub(r"(?im)^\s*[a-z][\w -]{0,40}:\s*", " ", text)
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compact(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return "|".join(str(item) for item in sorted(value))
    return str(value)


def sample(values: Iterable[Any], limit: int = 8) -> str:
    seen: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.append(text)
        if len(seen) >= limit:
            break
    return "|".join(seen)


def classify_domain(prompt_text: str) -> str:
    text = prompt_text.lower()
    rules = [
        ("home_maintenance_plumbing", ("plumbing", "sink", "leak", "water", "repair", "electrical", "home")),
        ("gardening", ("garden", "watering", "soil", "plants")),
        ("volunteer_coordination", ("volunteer", "community", "team working")),
        ("education_parent_update", ("teacher", "parent", "student", "classroom")),
        ("small_business_cafe", ("cafe", "customer", "closing checklist", "staff")),
        ("library_volunteer", ("library", "books", "returned")),
        ("planning_guidance", ("plan", "schedule", "organizing", "prepare")),
        ("troubleshooting_guidance", ("troubleshoot", "issue", "problem", "mistake")),
    ]
    for label, needles in rules:
        if any(needle in text for needle in needles):
            return label
    return "other"


def load_block_status(review_dir: Path) -> dict[str, dict[str, str]]:
    path = review_dir / "first_token_protected_block_summary.csv"
    statuses: dict[str, dict[str, str]] = {}
    if not path.exists():
        return statuses
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            statuses[str(row.get("block_id", ""))] = dict(row)
    return statuses


def enrich_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        response_text = str(item.get("response_text", ""))
        item["response_hash"] = str(item.get("response_text_sha256") or sha256_text(response_text))
        item["format_scrub_hash"] = sha256_text(scrub_text(response_text))
        item["arm"] = str(item.get("generation_condition", item.get("arm", "")))
        item["shard_id"] = str(item.get("replicate_group_id", item.get("shard_id", "")))
        item["block_id"] = f"{item['shard_id']}_block_00" if item["shard_id"] else ""
        item["prompt_prefix_pair"] = f"{item.get('prompt_id', '')}::{item.get('prefix_family_id', '')}"
        item["task_domain"] = classify_domain(str(item.get("prompt_text", "")))
        item["controller_event"] = "::".join(
            [
                str(item.get("first_generated_token_id", "")),
                str(item.get("event_side", "")),
                str(item.get("event_bucket_side", "")),
                str(item.get("controller_bonus_nats", "")),
                str(item.get("controller_penalty_nats", "")),
                str(item.get("controller_max_target_mass", "")),
            ]
        )
        enriched.append(item)
    return enriched


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, ""))].append(row)
    return groups


def summarize_duplicate_groups(groups: Mapping[str, list[dict[str, Any]]]) -> dict[str, int]:
    duplicate_groups = [members for members in groups.values() if len(members) > 1]
    return {
        "duplicate_groups": len(duplicate_groups),
        "duplicate_extra_rows": sum(len(members) - 1 for members in duplicate_groups),
        "max_group_size": max((len(members) for members in duplicate_groups), default=0),
        "unique_groups": len(groups),
    }


def write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (value.rstrip() if isinstance(value, str) else value) for key, value in row.items()})


def duplicate_group_rows(groups: Mapping[str, list[dict[str, Any]]], key_name: str, block_status: Mapping[str, Mapping[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group_key, members in groups.items():
        if len(members) <= 1:
            continue
        block_ids = sorted({str(row.get("block_id", "")) for row in members if row.get("block_id")})
        accepted_related = any(str(block_status.get(block_id, {}).get("accept", "")).lower() == "true" for block_id in block_ids)
        quality_fail_related = any(str(block_status.get(block_id, {}).get("accept", "")).lower() == "false" for block_id in block_ids)
        first = members[0]
        out.append(
            {
                key_name: group_key,
                "count": len(members),
                "extra_rows": len(members) - 1,
                "arms": sample(row.get("arm") for row in members),
                "shards": sample(row.get("shard_id") for row in members),
                "blocks": sample(block_ids),
                "prompt_ids": sample(row.get("prompt_id") for row in members),
                "prefix_families": sample(row.get("prefix_family_id") for row in members),
                "task_domains": sample(row.get("task_domain") for row in members),
                "controller_events": sample(row.get("controller_event") for row in members),
                "first_token_ids": sample(row.get("first_generated_token_id") for row in members),
                "first_token_texts": sample(row.get("first_generated_token_text") for row in members),
                "accepted_block_related": accepted_related,
                "quality_fail_block_related": quality_fail_related,
                "response_excerpt": str(first.get("response_text", ""))[:240].replace("\n", " "),
            }
        )
    return sorted(out, key=lambda row: (-int(row["count"]), str(row[key_name])))


def aggregate_by_keys(rows: list[dict[str, Any]], keys: list[str], *, duplicate_key: str = "response_hash") -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(str(row.get(key, "")) for key in keys)].append(row)
    out: list[dict[str, Any]] = []
    for bucket_key, members in buckets.items():
        groups = group_by(members, duplicate_key)
        dup_summary = summarize_duplicate_groups(groups)
        record = {key: bucket_key[idx] for idx, key in enumerate(keys)}
        record.update(
            {
                "rows": len(members),
                "unique_response_hashes": dup_summary["unique_groups"],
                "duplicate_groups": dup_summary["duplicate_groups"],
                "duplicate_extra_rows": dup_summary["duplicate_extra_rows"],
                "max_group_size": dup_summary["max_group_size"],
                "prompt_ids": sample(row.get("prompt_id") for row in members),
                "prefix_families": sample(row.get("prefix_family_id") for row in members),
                "first_token_ids": sample(row.get("first_generated_token_id") for row in members),
            }
        )
        out.append(record)
    return sorted(out, key=lambda row: (-int(row["duplicate_extra_rows"]), compact(row)))


def analyze(source_dir: Path, review_dir: Path, failure_summary_path: Path, output_dir: Path) -> dict[str, Any]:
    row_paths = sorted(source_dir.glob("shards/shard_*/r4_generated_outputs.jsonl"))
    if not row_paths:
        raise FileNotFoundError(f"no generated output shards found under {source_dir}")
    rows = enrich_rows([row for path in row_paths for row in read_jsonl(path)])
    block_status = load_block_status(review_dir)
    failure = read_json(failure_summary_path) if failure_summary_path.exists() else {}

    exact_groups = group_by(rows, "response_hash")
    scrub_groups = group_by(rows, "format_scrub_hash")
    exact_summary = summarize_duplicate_groups(exact_groups)
    scrub_summary = summarize_duplicate_groups(scrub_groups)

    duplicate_groups = duplicate_group_rows(exact_groups, "response_hash", block_status)
    scrub_duplicate_groups = duplicate_group_rows(scrub_groups, "format_scrub_hash", block_status)

    write_csv(
        output_dir / "duplicate_groups_by_response_hash.csv",
        duplicate_groups,
        [
            "response_hash",
            "count",
            "extra_rows",
            "arms",
            "shards",
            "blocks",
            "prompt_ids",
            "prefix_families",
            "task_domains",
            "controller_events",
            "first_token_ids",
            "first_token_texts",
            "accepted_block_related",
            "quality_fail_block_related",
            "response_excerpt",
        ],
    )
    write_csv(
        output_dir / "duplicate_after_format_scrub.csv",
        scrub_duplicate_groups,
        [
            "format_scrub_hash",
            "count",
            "extra_rows",
            "arms",
            "shards",
            "blocks",
            "prompt_ids",
            "prefix_families",
            "task_domains",
            "controller_events",
            "first_token_ids",
            "first_token_texts",
            "accepted_block_related",
            "quality_fail_block_related",
            "response_excerpt",
        ],
    )
    write_csv(
        output_dir / "duplicate_by_arm_condition.csv",
        aggregate_by_keys(rows, ["arm"]),
        ["arm", "rows", "unique_response_hashes", "duplicate_groups", "duplicate_extra_rows", "max_group_size", "prompt_ids", "prefix_families", "first_token_ids"],
    )
    write_csv(
        output_dir / "duplicate_by_shard_block.csv",
        aggregate_by_keys(rows, ["shard_id", "block_id", "arm"]),
        [
            "shard_id",
            "block_id",
            "arm",
            "rows",
            "unique_response_hashes",
            "duplicate_groups",
            "duplicate_extra_rows",
            "max_group_size",
            "prompt_ids",
            "prefix_families",
            "first_token_ids",
        ],
    )
    write_csv(
        output_dir / "duplicate_by_prompt_prefix_pair.csv",
        aggregate_by_keys(rows, ["prompt_id", "prefix_family_id", "arm"]),
        [
            "prompt_id",
            "prefix_family_id",
            "arm",
            "rows",
            "unique_response_hashes",
            "duplicate_groups",
            "duplicate_extra_rows",
            "max_group_size",
            "prompt_ids",
            "prefix_families",
            "first_token_ids",
        ],
    )
    write_csv(
        output_dir / "duplicate_by_controller_event.csv",
        aggregate_by_keys(rows, ["controller_event", "arm"]),
        [
            "controller_event",
            "arm",
            "rows",
            "unique_response_hashes",
            "duplicate_groups",
            "duplicate_extra_rows",
            "max_group_size",
            "prompt_ids",
            "prefix_families",
            "first_token_ids",
        ],
    )
    write_csv(
        output_dir / "duplicate_by_task_domain.csv",
        aggregate_by_keys(rows, ["task_domain", "arm"]),
        [
            "task_domain",
            "arm",
            "rows",
            "unique_response_hashes",
            "duplicate_groups",
            "duplicate_extra_rows",
            "max_group_size",
            "prompt_ids",
            "prefix_families",
            "first_token_ids",
        ],
    )
    write_csv(
        output_dir / "duplicate_within_block.csv",
        [row for row in aggregate_by_keys(rows, ["block_id", "arm"]) if int(row["duplicate_extra_rows"]) > 0],
        ["block_id", "arm", "rows", "unique_response_hashes", "duplicate_groups", "duplicate_extra_rows", "max_group_size", "prompt_ids", "prefix_families", "first_token_ids"],
    )

    cross_arm_groups = sum(1 for members in exact_groups.values() if len({row.get("arm") for row in members}) > 1)
    cross_shard_groups = sum(1 for members in exact_groups.values() if len({row.get("shard_id") for row in members}) > 1)
    cross_prompt_groups = sum(1 for members in exact_groups.values() if len({row.get("prompt_id") for row in members}) > 1)
    within_block_extra = Counter()
    for row in aggregate_by_keys(rows, ["block_id", "arm"]):
        within_block_extra[f"{row['block_id']}::{row['arm']}"] = int(row["duplicate_extra_rows"])

    summary = {
        "schema_name": "natural_evidence_v2_r4_868260_duplicate_forensics_v1",
        "status": "RECORDED_R4_868260_DUPLICATE_FORENSICS_ARTIFACT_ONLY_NO_SUBMIT",
        "source_job_id": "868260",
        "source_dir": str(source_dir),
        "generated_rows": len(rows),
        "unique_response_hashes": exact_summary["unique_groups"],
        "global_duplicate_extra_rows": exact_summary["duplicate_extra_rows"],
        "global_duplicate_hash_groups": exact_summary["duplicate_groups"],
        "global_duplicate_hash_max_group_size": exact_summary["max_group_size"],
        "format_scrub_unique_hashes": scrub_summary["unique_groups"],
        "format_scrub_duplicate_extra_rows": scrub_summary["duplicate_extra_rows"],
        "format_scrub_duplicate_hash_groups": scrub_summary["duplicate_groups"],
        "format_scrub_worsens_duplicates": scrub_summary["duplicate_extra_rows"] > exact_summary["duplicate_extra_rows"],
        "cross_arm_duplicate_groups": cross_arm_groups,
        "cross_shard_duplicate_groups": cross_shard_groups,
        "cross_prompt_duplicate_groups": cross_prompt_groups,
        "within_block_duplicate_extra_rows": dict(sorted(within_block_extra.items())),
        "duplicate_condition_sets_from_failure_analysis": failure.get("duplicate_summary", {}).get("top_duplicate_condition_sets", {}),
        "affected_quality_fail_blocks": failure.get("protected_quality_fail_blocks", []),
        "interpretation": {
            "primary": "Duplicates are a strict-quality blocker; 868260 remains failed and is not reclassified.",
            "decoding_or_sampling": "Rows were produced under deterministic_greedy_first_step_controller, so exact duplicates are plausibly driven by deterministic decoding plus repeated/near-repeated prompt-prefix cylinders.",
            "prompt_driven_check": "cross_prompt_duplicate_groups and duplicate_by_prompt_prefix_pair.csv separate prompt/prefix reuse from global deterministic collapse.",
            "controller_driven_check": "duplicate_by_controller_event.csv groups duplicates by first token and controller settings; all rows share the same controller gain policy in this run.",
            "format_scrub_check": "duplicate_after_format_scrub.csv shows whether trivial formatting normalization increases duplicate pressure.",
        },
        "slurm_allowed": False,
        "generation_started": False,
        "training_started": False,
        "paper_claim_allowed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "duplicate_forensics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "# R4 868260 Duplicate Forensics",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This is an artifact-only analysis. It does not submit Slurm, reclassify `868260`, or unlock a paper-facing positive claim.",
        "",
        "## Core Counts",
        "",
        f"- generated rows: `{summary['generated_rows']}`",
        f"- unique exact response hashes: `{summary['unique_response_hashes']}`",
        f"- global duplicate extra rows: `{summary['global_duplicate_extra_rows']}`",
        f"- exact duplicate hash groups: `{summary['global_duplicate_hash_groups']}`",
        f"- max exact duplicate group size: `{summary['global_duplicate_hash_max_group_size']}`",
        f"- format-scrub duplicate extra rows: `{summary['format_scrub_duplicate_extra_rows']}`",
        f"- format-scrub worsens duplicates: `{summary['format_scrub_worsens_duplicates']}`",
        f"- cross-arm duplicate groups: `{summary['cross_arm_duplicate_groups']}`",
        f"- cross-shard duplicate groups: `{summary['cross_shard_duplicate_groups']}`",
        f"- cross-prompt duplicate groups: `{summary['cross_prompt_duplicate_groups']}`",
        "",
        "## Interpretation",
        "",
        "- `868260` is best read as signal-present but strict-quality failed.",
        "- The exact duplicate rate is too high for a paper-facing natural-output claim.",
        "- Deterministic greedy decoding and repeated natural prompt-prefix cylinders are both plausible duplicate sources; the CSV slices isolate them without rerunning generation.",
        "- Future reruns must use a precommitted duplicate-safe policy that is blind to decode success and applied identically to all arms.",
    ]
    (output_dir / "duplicate_forensics_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze duplicate forensics for R4 job 868260.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--failure-summary", type=Path, default=DEFAULT_FAILURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = analyze(args.source_dir, args.review_dir, args.failure_summary, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
