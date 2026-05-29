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
from scripts.natural_evidence_v2.r4_cover_natural_common import sha256_file, write_json_new, write_text_new  # noqa: E402


DEFAULT_PROMPTS = (
    ROOT / "results/natural_evidence_v2/prompts/r4_after_870987_prefar_organic_null_prompts_20260521/locked_prompts.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/natural_evidence_v2/status/r4_after_870987_prefar_organic_null_row_bank_v2_plan_20260521"
)
LOCKED_SCALE_SUMMARY = (
    "results/natural_evidence_v2/status/"
    "r4_after_869348_locked_scale_generation_870210_plus_870987_aggregate_20260519/"
    "locked_scale_summary.json"
)
STANDARD_CONTROL_SUMMARY = (
    "results/natural_evidence_v2/status/"
    "r4_after_870987_prefar_standard_control_generation_871250_contextual_window_v2_aggregate_20260521/"
    "prefar_standard_control_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build artifact-only R4 pre-FAR organic-null row bank.")
    parser.add_argument("--surface-bank", type=Path, default=DEFAULT_SURFACE_BANK)
    parser.add_argument("--codebook", type=Path, default=DEFAULT_CODEBOOK)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-shards", type=int, default=256)
    parser.add_argument("--prompts-per-shard", type=int, default=64)
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
        row.update(
            {
                "schema_name": "natural_evidence_v2_r4_after_870987_prefar_organic_null_row_bank_row_v1",
                "artifact_role": "r4_after_870987_prefar_organic_null_row_bank_not_tokenized_not_scored",
                "source_locked_scale_jobs": ["870210", "870987"],
                "source_locked_scale_status": "PASS_R4_AFTER_869348_LOCKED_SCALE_GENERATION_GATE",
                "source_locked_scale_summary": LOCKED_SCALE_SUMMARY,
                "source_standard_control_summary": STANDARD_CONTROL_SUMMARY,
                "prefar_null_candidate": True,
                "standard_control_null_expansion": False,
                "organic_null": True,
                "payload_diversity_tested": False,
                "same_contract_only": True,
                "contract_id": "a55e",
                "allocation_policy": "prefar_organic_null_global_unique_prompt_prefix_pairs",
                "replicate_group_id": f"first_token_event_prefar_organic_null_shard_{shard_index:03d}",
                "row_key": (
                    f"{prompt_id}|{coordinate}|{prefix_family_id}|{surface_id}|"
                    f"organic{shard_index:03d}_{old_key.rsplit('_', 1)[-1]}"
                ),
                "generation_conditions": ["raw"],
                "generation_started": False,
                "model_scoring_started": False,
                "training_started": False,
                "slurm_submitted": False,
                "paper_claim_allowed": False,
            }
        )


def rewrite_manifest(manifest: dict[str, Any], *, prompts_path: Path) -> None:
    manifest.update(
        {
            "schema_name": "natural_evidence_v2_r4_after_870987_prefar_organic_null_row_bank_manifest_v1",
            "status": "PASS_R4_AFTER_870987_PREFAR_ORGANIC_NULL_ROW_BANK_BUILT_ARTIFACT_ONLY_NO_SUBMIT",
            "source_locked_scale_jobs": ["870210", "870987"],
            "source_locked_scale_status": "PASS_R4_AFTER_869348_LOCKED_SCALE_GENERATION_GATE",
            "source_locked_scale_summary": LOCKED_SCALE_SUMMARY,
            "source_standard_control_summary": STANDARD_CONTROL_SUMMARY,
            "prefar_null_candidate": True,
            "standard_control_null_expansion": False,
            "organic_null": True,
            "payload_diversity_tested": False,
            "same_contract_only": True,
            "contract_id": "a55e",
            "target_organic_null_blocks": 256,
            "prompts_path": str(prompts_path.relative_to(ROOT)) if prompts_path.is_relative_to(ROOT) else str(prompts_path),
            "prompts_sha256": sha256_file(prompts_path),
            "generation_conditions": ["raw"],
            "generation_started": False,
            "model_scoring_started": False,
            "training_started": False,
            "slurm_submitted": False,
            "paper_claim_allowed": False,
            "next_allowed_action": (
                "Validate organic-null row bank, run Qwen tokenizer boundary preflight, "
                "then prepare raw-only generation/decode wrapper before any Slurm submission."
            ),
        }
    )


def write_report(path: Path, manifest: Mapping[str, Any]) -> None:
    text = f"""# R4 After-870987 Pre-FAR Organic-Null Row Bank Plan

Date: 2026-05-21

Status: `{manifest['status']}`

This artifact-only plan builds the organic-null row bank for the R4 Qwen
same-contract first-token event pre-FAR null package. It does not tokenize,
score, generate, train, enable an allowlist entry, submit Slurm, aggregate FAR,
or create a paper-facing claim.

```text
source locked-scale jobs: 870210 + 870987
source standard-control package: 871250 contextual-window aggregate
target organic-null shards: {manifest['target_shards']}
rows per shard: {manifest['rows_per_shard']}
row cylinders: {manifest['row_count']}
selected prompts: {manifest['selected_prompt_count']}
generation conditions: raw only
```

Next allowed action: validate this row bank, run Qwen tokenizer boundary
preflight, then prepare raw-only generation/decode wrapper before any Slurm
submission.
"""
    write_text_new(path, text)


def main() -> int:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {output_dir}")
    surface_bank_path = resolve(args.surface_bank)
    codebook_path = resolve(args.codebook)
    prompts_path = resolve(args.prompts)
    rows, coordinate_rows, manifest, prefix_inventory = build_rows(
        surface_bank=read_json(surface_bank_path),
        codebook=read_json(codebook_path),
        prompts=read_jsonl(prompts_path),
        target_shards=int(args.target_shards),
        prompts_per_shard=int(args.prompts_per_shard),
        surface_bank_path=surface_bank_path,
        codebook_path=codebook_path,
        prompts_path=prompts_path,
    )
    rewrite_rows(rows)
    rewrite_manifest(manifest, prompts_path=prompts_path)
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
