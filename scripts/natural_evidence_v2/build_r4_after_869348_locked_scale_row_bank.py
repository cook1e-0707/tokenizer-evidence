from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.build_r4_after_868348_global_unique_row_bank import (  # noqa: E402
    DEFAULT_CODEBOOK,
    DEFAULT_SURFACE_BANK,
    build_rows,
)
from scripts.natural_evidence_v2.r4_cover_natural_common import write_json_new, write_text_new  # noqa: E402


DEFAULT_PROMPTS = ROOT / "results/natural_evidence_v2/prompts/r4_cover_natural_prompt_bank_20260512_dev2048/locked_prompts.jsonl"
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/natural_evidence_v2/status/r4_after_869348_global_unique_locked_scale_row_bank_plan_20260518"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the artifact-only held-out locked-scale row bank after the "
            "869348 32-block dev diagnostic pass. This does not tokenize, "
            "score, generate, train, enable an allowlist, or submit Slurm."
        )
    )
    parser.add_argument("--surface-bank", type=Path, default=DEFAULT_SURFACE_BANK)
    parser.add_argument("--codebook", type=Path, default=DEFAULT_CODEBOOK)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-shards", type=int, default=96)
    parser.add_argument("--prompts-per-shard", type=int, default=64)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def rewrite_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        shard_index = int(row["assigned_shard_index"])
        prompt_id = str(row["prompt_id"])
        coordinate = int(row["coordinate_id"])
        prefix_family_id = str(row["prefix_family_id"])
        surface_id = str(row["target_surface_id"])
        old_key = str(row.get("row_key", ""))
        row.pop("source_failure_job_id", None)
        row.pop("source_failure_root_cause", None)
        row.update(
            {
                "schema_name": "natural_evidence_v2_r4_after_869348_global_unique_locked_scale_row_bank_row_v1",
                "artifact_role": "r4_after_869348_global_unique_locked_scale_row_bank_not_tokenized_not_scored",
                "source_dev_pass_job_id": "869348",
                "source_dev_pass_status": "PASS_R4_AFTER_868348_GLOBAL_UNIQUE_DEV_DIAGNOSTIC_GATE",
                "locked_scale_candidate": True,
                "payload_diversity_tested": False,
                "same_contract_only": True,
                "allocation_policy": "locked_scale_global_unique_prompt_prefix_pairs_16_prefix_rotated_by_prompt_and_coordinate",
                "replicate_group_id": f"first_token_event_locked_scale_shard_{shard_index:02d}",
                "row_key": (
                    f"{prompt_id}|{coordinate}|{prefix_family_id}|{surface_id}|"
                    f"locked{shard_index:02d}_{old_key.rsplit('_', 1)[-1]}"
                ),
            }
        )


def rewrite_manifest(manifest: dict[str, Any]) -> None:
    manifest.pop("source_failure_job_id", None)
    manifest.pop("source_failure_interpretation", None)
    manifest.update(
        {
            "schema_name": "natural_evidence_v2_r4_after_869348_global_unique_locked_scale_row_bank_manifest_v1",
            "status": "PASS_R4_AFTER_869348_GLOBAL_UNIQUE_LOCKED_SCALE_ROW_BANK_BUILT_ARTIFACT_ONLY_NO_SUBMIT",
            "source_dev_pass_job_id": "869348",
            "source_dev_pass_status": "PASS_R4_AFTER_868348_GLOBAL_UNIQUE_DEV_DIAGNOSTIC_GATE",
            "source_dev_pass_review": (
                "results/natural_evidence_v2/status/"
                "r4_after_868348_global_unique_dev_diagnostic_869348_review/"
                "quality_repair_confirmation_review_summary.json"
            ),
            "locked_scale_candidate": True,
            "payload_diversity_tested": False,
            "same_contract_only": True,
            "reclassifies_868348": False,
            "reclassifies_869348": False,
            "next_allowed_action": (
                "Run locked-scale row-bank route validation and static/Qwen tokenizer boundary preflight. "
                "Do not submit generation until the tokenizer preflight passes and a reviewed locked-scale "
                "H200 route is recorded."
            ),
        }
    )


def write_report(path: Path, manifest: Mapping[str, Any]) -> None:
    text = f"""# R4 After-869348 Global-Unique Locked-Scale Row Bank Plan

Date: 2026-05-18

## Status

`{manifest['status']}`

This artifact-only plan builds the held-out locked-scale row bank after the
`869348` Qwen first-token-event 32-block dev diagnostic passed. It uses the
locked split from the reviewed R4 cover-natural prompt bank, keeps the same
`a55e` contract and first-token event controller surface/codebook, and allocates
96 globally unique prompt/prefix shards.

It does not tokenize, score, generate outputs, train, enable an allowlist entry,
submit Slurm, or create a paper-facing claim.

## Key Counts

- selected prompts: `{manifest['selected_prompt_count']}`
- selected coordinates: `{manifest['selected_coordinate_count']}`
- total row cylinders: `{manifest['row_count']}`
- rows per shard: `{manifest['rows_per_shard']}`
- unique content prompt/prefix pairs: `{manifest['unique_content_prompt_prefix_pairs']}`
- duplicate content prompt/prefix extra rows: `{manifest['duplicate_content_prompt_prefix_pair_extra_rows']}`
- max prefix template fraction: `{manifest['max_prefix_template_fraction']:.4f}`

## Next Allowed Action

Artifact-only route validation and tokenizer boundary preflight for this locked
row bank. No generation or locked-scale Slurm submission is allowed until those
checks pass and a reviewed route is recorded.
"""
    write_text_new(path, text)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {output_dir}")
    surface_bank = read_json(args.surface_bank)
    codebook = read_json(args.codebook)
    prompts = read_jsonl(args.prompts)
    rows, coordinate_rows, manifest, prefix_inventory = build_rows(
        surface_bank=surface_bank,
        codebook=codebook,
        prompts=prompts,
        target_shards=int(args.target_shards),
        prompts_per_shard=int(args.prompts_per_shard),
        surface_bank_path=args.surface_bank,
        codebook_path=args.codebook,
        prompts_path=args.prompts,
    )
    rewrite_rows(rows)
    rewrite_manifest(manifest)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(output_dir / "row_allocation_rows.jsonl", rows)
    write_csv(
        output_dir / "coordinate_bucket_compatibility.csv",
        coordinate_rows,
        [
            "coordinate_id",
            "expected_codeword_bit",
            "target_entry_count",
            "opposite_entry_count",
            "current_two_way_scorer_compatible",
        ],
    )
    write_csv(
        output_dir / "prefix_template_inventory.csv",
        prefix_inventory,
        ["prefix_family_id", "assistant_prefix_before_surface", "row_count", "row_fraction"],
    )
    write_json_new(output_dir / "row_allocation_manifest.json", manifest)
    write_report(output_dir / "row_allocation_report.md", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
