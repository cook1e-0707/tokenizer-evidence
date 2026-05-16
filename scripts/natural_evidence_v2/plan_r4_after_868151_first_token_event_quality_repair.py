from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_cover_natural_common import (  # noqa: E402
    read_jsonl,
    sha256_file,
    write_csv_new,
    write_json_new,
    write_jsonl_new,
    write_text_new,
)


DEFAULT_ROWS = (
    ROOT
    / "results/natural_evidence_v2/status/r4_after_868016_reliability_coordinate_pivot_rows_20260516/reliability_surface_mass_rows.jsonl"
)
DEFAULT_QUALITY_AUDIT = (
    ROOT
    / "results/natural_evidence_v2/status/r4_after_868151_first_token_event_quality_audit_20260516/quality_audit_summary.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/natural_evidence_v2/status/r4_after_868151_first_token_event_quality_repair_plan_20260516"
)

COORDINATION_RE = re.compile(r"\bcoordinat(?:e|es|ed|ing|ion|or|ors)\b", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan the artifact-only R4 after-868151 first-token event quality repair. "
            "This writes literal-domain policy and duplicate-safe row allocation artifacts; "
            "it does not run generation, model scoring, training, or Slurm."
        )
    )
    parser.add_argument("--score-rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--quality-audit", type=Path, default=DEFAULT_QUALITY_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--rows-per-coordinate-per-shard", type=int, default=64)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def row_key(row: Mapping[str, Any]) -> str:
    key = row.get("row_key")
    if key:
        return str(key)
    return "|".join(
        [
            str(row["prompt_id"]),
            str(int(row["coordinate_id"])),
            str(row.get("prefix_family_id", "")),
            str(row.get("target_surface_id", row.get("target_surface", ""))),
        ]
    )


def duplicate_pair_key(row: Mapping[str, Any]) -> str:
    return f"{int(row['prompt_index'])}|{row.get('prefix_family_id', '')}"


def is_coordination_domain_prompt(text: str) -> bool:
    return bool(COORDINATION_RE.search(text))


def count_coordination_domain_prompts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prompt_to_text: dict[int, str] = {}
    for row in rows:
        prompt_to_text[int(row["prompt_index"])] = str(row.get("prompt_text", ""))
    coordination_prompt_indices = sorted(
        prompt_index
        for prompt_index, text in prompt_to_text.items()
        if is_coordination_domain_prompt(text)
    )
    return {
        "total_prompt_count": len(prompt_to_text),
        "coordination_domain_prompt_count": len(coordination_prompt_indices),
        "coordination_domain_prompt_indices": coordination_prompt_indices,
    }


def literal_policy() -> dict[str, Any]:
    return {
        "schema_name": "natural_evidence_v2_r4_after_868151_contextual_literal_policy_v1",
        "status": "CONTEXTUAL_LITERAL_POLICY_PLANNED_ARTIFACT_ONLY",
        "hard_forbid_literals": [
            "bucket",
            "fingerprint",
            "watermark",
            "payload",
            "secret key",
            "decoder",
            "hidden signal",
        ],
        "contextual_literals": {
            "coordinate": {
                "ordinary_domain_allowed": True,
                "allowed_domain_cues": [
                    "coordinate",
                    "coordinates",
                    "coordinated",
                    "coordinating",
                    "coordination",
                    "coordinator",
                    "coordinators",
                ],
                "technical_cues": [
                    "bit",
                    "bucket",
                    "checksum",
                    "codeword",
                    "coordinate id",
                    "decoder",
                    "hidden",
                    "payload",
                    "secret",
                    "slot",
                    "token id",
                    "watermark",
                ],
                "gate_policy": (
                    "ordinary coordination-domain use may be classified as non-technical, "
                    "but technical coordinate use remains a forbidden public surface"
                ),
            }
        },
        "final_generation_gate": {
            "technical_public_literal_count_max": 0,
            "ordinary_domain_literal_count_reported_separately": True,
        },
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "slurm_submitted": False,
        "paper_claim_allowed": False,
    }


def build_duplicate_safe_allocation(
    rows: Sequence[Mapping[str, Any]],
    *,
    shards: int,
    rows_per_coordinate_per_shard: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if shards <= 0:
        raise ValueError("shards must be positive")
    if rows_per_coordinate_per_shard <= 0:
        raise ValueError("rows_per_coordinate_per_shard must be positive")
    coordinates = sorted({int(row["coordinate_id"]) for row in rows})
    expected_per_coordinate = shards * rows_per_coordinate_per_shard
    by_coordinate = Counter(int(row["coordinate_id"]) for row in rows)
    bad_coordinates = {
        str(coord): int(count)
        for coord, count in sorted(by_coordinate.items())
        if int(count) != expected_per_coordinate
    }
    if bad_coordinates:
        raise ValueError(f"coordinate row count mismatch: {bad_coordinates}")

    shard_pairs: list[set[str]] = [set() for _ in range(shards)]
    shard_coord_counts: list[Counter[int]] = [Counter() for _ in range(shards)]
    assigned_rows: list[dict[str, Any]] = []
    seen_row_keys: set[str] = set()

    pair_counts = Counter(duplicate_pair_key(row) for row in rows)
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            -pair_counts[duplicate_pair_key(row)],
            int(row["prompt_index"]),
            str(row.get("prefix_family_id", "")),
            int(row["coordinate_id"]),
            row_key(row),
        ),
    )

    for row in ordered_rows:
        key = row_key(row)
        if key in seen_row_keys:
            raise ValueError(f"duplicate row_key: {key}")
        seen_row_keys.add(key)
        coord = int(row["coordinate_id"])
        pair_key = duplicate_pair_key(row)
        candidates = [
            shard
            for shard in range(shards)
            if shard_coord_counts[shard][coord] < rows_per_coordinate_per_shard
            and pair_key not in shard_pairs[shard]
        ]
        if not candidates:
            raise ValueError(f"unable to allocate row without shard duplicate pair: {key}")
        shard = min(
            candidates,
            key=lambda item: (
                shard_coord_counts[item][coord],
                sum(shard_coord_counts[item].values()),
                item,
            ),
        )
        shard_pairs[shard].add(pair_key)
        shard_coord_counts[shard][coord] += 1
        assigned = dict(row)
        assigned["assigned_shard_index"] = shard
        assigned["replicate_group_id"] = f"quality_repair_shard_{shard:02d}"
        assigned["duplicate_pair_key"] = pair_key
        assigned["quality_repair_allocation_policy"] = "per_shard_unique_prompt_index_prefix_family"
        assigned_rows.append(assigned)

    expected_rows_per_shard = len(coordinates) * rows_per_coordinate_per_shard
    shard_summaries: list[dict[str, Any]] = []
    for shard in range(shards):
        coord_counts = {str(coord): int(shard_coord_counts[shard][coord]) for coord in coordinates}
        duplicate_pair_count = len(shard_pairs[shard])
        rows_in_shard = sum(shard_coord_counts[shard].values())
        if rows_in_shard != expected_rows_per_shard:
            raise ValueError(f"shard {shard} row count mismatch: {rows_in_shard}")
        if duplicate_pair_count != rows_in_shard:
            raise ValueError(f"shard {shard} has duplicate prompt/prefix pairs")
        if any(count != rows_per_coordinate_per_shard for count in coord_counts.values()):
            raise ValueError(f"shard {shard} coordinate count mismatch: {coord_counts}")
        shard_summaries.append(
            {
                "shard_index": shard,
                "replicate_group_id": f"quality_repair_shard_{shard:02d}",
                "row_count": rows_in_shard,
                "unique_prompt_prefix_pairs": duplicate_pair_count,
                "duplicate_prompt_prefix_pair_count": rows_in_shard - duplicate_pair_count,
                "coordinate_counts": coord_counts,
            }
        )

    manifest = {
        "schema_name": "natural_evidence_v2_r4_after_868151_duplicate_safe_allocation_manifest_v1",
        "status": "PASS_DUPLICATE_SAFE_ROW_ALLOCATION_ARTIFACT_ONLY",
        "allocation_policy": "per_shard_unique_prompt_index_prefix_family",
        "duplicate_pair_key": ["prompt_index", "prefix_family_id"],
        "shards": shards,
        "coordinates": coordinates,
        "rows_per_coordinate_per_shard": rows_per_coordinate_per_shard,
        "rows_per_shard": expected_rows_per_shard,
        "total_rows": len(assigned_rows),
        "shard_summaries": shard_summaries,
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "slurm_submitted": False,
        "paper_claim_allowed": False,
    }
    return assigned_rows, manifest


def write_report(
    *,
    output_dir: Path,
    summary: Mapping[str, Any],
    coordination_analysis: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    text = f"""# R4 After-868151 First-Token Event Quality Repair Plan

Date: 2026-05-16

## Status

`{summary["status"]}`

This is an artifact-only planning record. No generation, scoring, training,
Slurm submission, Llama, FAR, sanitizer, payload-diversity route, or paper claim
is unlocked by this artifact.

## Literal Policy

The previous quality audit showed `coordinate` literal hits. The prompt bank
contains {coordination_analysis["coordination_domain_prompt_count"]} / {coordination_analysis["total_prompt_count"]}
coordination-domain prompts, so simply deleting that domain would leave too few
rows to preserve the existing 4-shard x 768-row diagnostic scope. The planned
repair is a contextual matcher:

- hard-forbid technical literals such as `bucket`, `fingerprint`, `watermark`,
  `payload`, `secret key`, `decoder`, and `hidden signal`;
- treat `coordinate` as technical only when hidden-channel technical cues are
  present;
- still require zero technical public literal hits in any future generation
  route.

## Duplicate-Output Mitigation

The repaired allocation manifest assigns all {manifest["total_rows"]} rows to
{manifest["shards"]} shards while enforcing one unique `(prompt_index,
prefix_family_id)` pair per shard. Each shard has {manifest["rows_per_shard"]}
rows and {manifest["rows_per_coordinate_per_shard"]} rows per selected
coordinate.

This does not prove future generation will have zero duplicate response hashes,
but it removes the deterministic duplicate source caused by evaluating multiple
coordinates with the same prompt/prefix inside a shard.

## Next Allowed Action

Patch the generation wrapper to consume the allocation manifest and patch the
decoder/quality gate to use the reviewed contextual literal policy. Then run
local and remote plan-only validation again. Do not submit Slurm from this
artifact alone.
"""
    write_text_new(output_dir / "quality_repair_plan_report.md", text)


def write_domain_prompt_csv(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    prompt_rows: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        prompt_rows[int(row["prompt_index"])] = row
    output_rows = []
    for prompt_index, row in sorted(prompt_rows.items()):
        prompt_text = str(row.get("prompt_text", ""))
        output_rows.append(
            {
                "prompt_index": prompt_index,
                "prompt_id": str(row.get("prompt_id", "")),
                "coordination_domain": is_coordination_domain_prompt(prompt_text),
                "prompt_text": prompt_text,
            }
        )
    write_csv_new(
        output_dir / "coordination_domain_prompt_analysis.csv",
        output_rows,
        ["prompt_index", "prompt_id", "coordination_domain", "prompt_text"],
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.score_rows if args.score_rows.is_absolute() else ROOT / args.score_rows
    quality_audit_path = args.quality_audit if args.quality_audit.is_absolute() else ROOT / args.quality_audit
    rows = read_jsonl(rows_path)
    quality_audit = read_json(quality_audit_path)
    coordination_analysis = count_coordination_domain_prompts(rows)
    assigned_rows, manifest = build_duplicate_safe_allocation(
        rows,
        shards=int(args.shards),
        rows_per_coordinate_per_shard=int(args.rows_per_coordinate_per_shard),
    )
    policy = literal_policy()
    enough_rows_after_domain_exclusion = (
        (coordination_analysis["total_prompt_count"] - coordination_analysis["coordination_domain_prompt_count"])
        * len(manifest["coordinates"])
        >= manifest["total_rows"]
    )
    summary = {
        "schema_name": "natural_evidence_v2_r4_after_868151_first_token_event_quality_repair_plan_v1",
        "status": "PASS_R4_AFTER_868151_FIRST_TOKEN_EVENT_QUALITY_REPAIR_PLAN_ARTIFACT_ONLY",
        "source_score_rows": str(rows_path),
        "source_score_rows_sha256": sha256_file(rows_path),
        "source_quality_audit": str(quality_audit_path),
        "source_quality_audit_status": quality_audit.get("status", ""),
        "source_quality_audit_sha256": sha256_file(quality_audit_path),
        "source_coordinate_literal_hits": int(quality_audit.get("coordinate_literal_hits", 0)),
        "source_duplicate_response_hash_count": int(quality_audit.get("duplicate_response_hash_count", 0)),
        "coordination_domain_prompt_count": coordination_analysis["coordination_domain_prompt_count"],
        "coordination_domain_exclusion_preserves_scope": bool(enough_rows_after_domain_exclusion),
        "literal_policy": "contextual_coordinate_domain_policy",
        "literal_policy_path": str(output_dir / "contextual_literal_policy.json"),
        "allocation_manifest_path": str(output_dir / "row_allocation_manifest.json"),
        "allocation_rows_path": str(output_dir / "row_allocation_rows.jsonl"),
        "allocation_status": manifest["status"],
        "shards": manifest["shards"],
        "rows_per_shard": manifest["rows_per_shard"],
        "duplicate_prompt_prefix_pair_count_per_shard_max": max(
            int(item["duplicate_prompt_prefix_pair_count"]) for item in manifest["shard_summaries"]
        ),
        "future_generation_wrapper_must_consume_allocation_manifest": True,
        "future_decoder_must_use_contextual_literal_policy": True,
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "slurm_submitted": False,
        "paper_claim_allowed": False,
    }
    write_json_new(output_dir / "contextual_literal_policy.json", policy)
    write_json_new(output_dir / "row_allocation_manifest.json", manifest)
    write_jsonl_new(output_dir / "row_allocation_rows.jsonl", assigned_rows)
    write_domain_prompt_csv(output_dir, rows)
    write_json_new(output_dir / "quality_repair_plan_summary.json", summary)
    write_report(
        output_dir=output_dir,
        summary=summary,
        coordination_analysis=coordination_analysis,
        manifest=manifest,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
