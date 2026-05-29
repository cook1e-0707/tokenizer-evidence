from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_cover_natural_common import (  # noqa: E402
    read_json,
    read_jsonl,
    sha256_file,
    technical_literal_hits,
    write_json_new,
    write_text_new,
)
from scripts.natural_evidence_v2.score_r4_surface_teacher_forced_mass import r4_row_surface_contract  # noqa: E402


DEFAULT_ROW_BANK_DIR = (
    ROOT / "results/natural_evidence_v2/status/r4_after_870987_prefar_standard_control_row_bank_plan_20260519"
)
DEFAULT_LOCKED_SCALE_ROWS = (
    ROOT / "results/natural_evidence_v2/status/r4_after_869348_global_unique_locked_scale_row_bank_plan_20260518/row_allocation_rows.jsonl"
)
DEFAULT_LOCKED_SCALE_SUMMARY = (
    ROOT
    / "results/natural_evidence_v2/status/r4_after_869348_locked_scale_generation_870210_plus_870987_aggregate_20260519/locked_scale_summary.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/natural_evidence_v2/status/r4_after_870987_prefar_standard_control_row_bank_validation_20260519"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the artifact-only R4 after-870987 pre-FAR standard-control "
            "row bank. This does not tokenize, score, generate, train, enable an "
            "allowlist entry, or submit Slurm."
        )
    )
    parser.add_argument("--row-bank-dir", type=Path, default=DEFAULT_ROW_BANK_DIR)
    parser.add_argument("--locked-scale-rows", type=Path, default=DEFAULT_LOCKED_SCALE_ROWS)
    parser.add_argument("--locked-scale-summary", type=Path, default=DEFAULT_LOCKED_SCALE_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def int_field(data: Mapping[str, Any], field: str, default: int = -1) -> int:
    try:
        return int(data.get(field, default))
    except (TypeError, ValueError):
        return default


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def validate_locked_scale_summary(summary: Mapping[str, Any], errors: list[str]) -> None:
    if summary.get("status") != "PASS_R4_AFTER_869348_LOCKED_SCALE_GENERATION_GATE":
        fail(errors, "locked-scale summary must be PASS")
    if summary.get("scale_gate_pass") is not True:
        fail(errors, "locked-scale scale_gate_pass must be true")
    by_arm = summary.get("first_token_event_summary_by_arm", {})
    if not isinstance(by_arm, Mapping):
        fail(errors, "locked-scale first_token_event_summary_by_arm missing")
        return
    for arm in ("raw", "task_only", "wrong_key", "wrong_payload"):
        arm_summary = by_arm.get(arm, {})
        if not isinstance(arm_summary, Mapping):
            fail(errors, f"locked-scale {arm} summary missing")
            continue
        if int_field(arm_summary, "blocks") != 96:
            fail(errors, f"locked-scale {arm} blocks must be 96")
        if int_field(arm_summary, "accepts") != 0:
            fail(errors, f"locked-scale {arm} accepts must be 0")


def validate_rows(
    rows: list[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    previous_rows: list[Mapping[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    if len(rows) != 163840:
        fail(errors, f"row count must be 163840, found {len(rows)}")
    previous_prompt_ids = {str(row.get("prompt_id", "")) for row in previous_rows if row.get("prompt_id")}
    prompt_ids = Counter(str(row.get("prompt_id", "")) for row in rows)
    overlap = sorted(set(prompt_ids) & previous_prompt_ids)
    if overlap:
        fail(errors, f"row bank overlaps locked-scale prompts: {overlap[:5]}")

    row_keys = Counter(str(row.get("row_key", "")) for row in rows)
    duplicate_row_keys = sum(count - 1 for count in row_keys.values() if count > 1)
    if duplicate_row_keys:
        fail(errors, f"duplicate row_key extra rows: {duplicate_row_keys}")
    content_pairs = Counter(str(row.get("content_duplicate_pair_key", "")) for row in rows)
    duplicate_content_pairs = sum(count - 1 for count in content_pairs.values() if count > 1)
    if duplicate_content_pairs:
        fail(errors, f"duplicate content prompt/prefix extra rows: {duplicate_content_pairs}")
    prompt_pairs = Counter(str(row.get("duplicate_pair_key", "")) for row in rows)
    duplicate_prompt_pairs = sum(count - 1 for count in prompt_pairs.values() if count > 1)
    if duplicate_prompt_pairs:
        fail(errors, f"duplicate prompt/prefix extra rows: {duplicate_prompt_pairs}")

    shards: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    coordinates = Counter()
    prefixes = Counter()
    for index, row in enumerate(rows):
        try:
            shard = int(row.get("assigned_shard_index", -1))
            coordinate = int(row["coordinate_id"])
        except Exception as exc:  # noqa: BLE001
            fail(errors, f"row {index} missing shard/coordinate: {type(exc).__name__}:{exc}")
            continue
        shards[shard].append(row)
        coordinates[coordinate] += 1
        prefixes[str(row.get("prefix_family_id", ""))] += 1
        if row.get("schema_name") != "natural_evidence_v2_r4_after_870987_prefar_standard_control_row_bank_row_v1":
            fail(errors, f"row {index} schema_name mismatch")
        if row.get("contract_id") != "a55e":
            fail(errors, f"row {index} contract_id must be a55e")
        if row.get("prefar_null_candidate") is not True or row.get("standard_control_null_expansion") is not True:
            fail(errors, f"row {index} pre-FAR standard-control flags invalid")
        if row.get("organic_null") is not False:
            fail(errors, f"row {index} organic_null must be false")
        if row.get("payload_diversity_tested") is not False or row.get("same_contract_only") is not True:
            fail(errors, f"row {index} payload/contract scope flags invalid")
        if row.get("generation_started") or row.get("model_scoring_started") or row.get("training_started"):
            fail(errors, f"row {index} has non-artifact-only execution flag")
        if row.get("slurm_submitted") or row.get("paper_claim_allowed"):
            fail(errors, f"row {index} has forbidden submission/claim flag")
        prompt_text = str(row.get("prompt_text", ""))
        hits = technical_literal_hits(prompt_text)
        if hits:
            fail(errors, f"row {index} prompt technical literal hits: {hits}")
        lowered = prompt_text.lower()
        if "step " in lowered or "exactly 16" in lowered or "slot" in lowered:
            fail(errors, f"row {index} prompt contains structural literal")
        try:
            r4_row_surface_contract(row)
        except Exception as exc:  # noqa: BLE001
            fail(errors, f"row {index} surface contract failed: {type(exc).__name__}:{exc}")

    if sorted(shards) != list(range(160)):
        fail(errors, f"expected shards 0..159, found {sorted(shards)[:5]}..{sorted(shards)[-5:] if shards else []}")
    for shard, shard_rows in sorted(shards.items()):
        if len(shard_rows) != 1024:
            fail(errors, f"shard {shard} row count must be 1024, found {len(shard_rows)}")
        shard_coord_counts = Counter(int(row["coordinate_id"]) for row in shard_rows)
        if len(shard_coord_counts) != 16 or set(shard_coord_counts.values()) != {64}:
            fail(errors, f"shard {shard} must have 64 rows for each of 16 coordinates")
    if len(coordinates) != 16 or set(coordinates.values()) != {10240}:
        fail(errors, f"expected 16 coordinates with 10240 rows each, found {dict(sorted(coordinates.items()))}")
    if len(prefixes) != 16 or set(prefixes.values()) != {10240}:
        fail(errors, f"expected 16 prefixes with 10240 rows each, found {dict(sorted(prefixes.items()))}")
    if len(prompt_ids) != 10240 or set(prompt_ids.values()) != {16}:
        fail(errors, f"expected 10240 prompts used exactly 16 times, found {len(prompt_ids)} prompts")

    if manifest.get("status") != "PASS_R4_AFTER_870987_PREFAR_STANDARD_CONTROL_ROW_BANK_BUILT_ARTIFACT_ONLY_NO_SUBMIT":
        fail(errors, f"manifest status mismatch: {manifest.get('status')}")
    for field, expected in (
        ("row_count", 163840),
        ("target_shards", 160),
        ("rows_per_shard", 1024),
        ("selected_prompt_count", 10240),
        ("selected_coordinate_count", 16),
        ("unique_content_prompt_prefix_pairs", 163840),
        ("duplicate_content_prompt_prefix_pair_extra_rows", 0),
        ("duplicate_prompt_prefix_pair_extra_rows", 0),
        ("previous_locked_scale_prompt_overlap_count", 0),
        ("existing_standard_control_blocks_per_arm", 96),
        ("target_standard_control_blocks_per_arm", 256),
        ("additional_standard_control_blocks_per_arm", 160),
    ):
        if int_field(manifest, field) != expected:
            fail(errors, f"manifest {field} must be {expected}")
    for field in ("prefar_null_candidate", "standard_control_null_expansion", "same_contract_only"):
        if manifest.get(field) is not True:
            fail(errors, f"manifest {field} must be true")
    for field in ("organic_null", "payload_diversity_tested", "generation_started", "model_scoring_started", "training_started", "slurm_submitted", "paper_claim_allowed"):
        if manifest.get(field) is not False:
            fail(errors, f"manifest {field} must be false")

    return {
        "row_count": len(rows),
        "shard_count": len(shards),
        "rows_per_shard": sorted({len(items) for items in shards.values()}),
        "selected_coordinate_count": len(coordinates),
        "rows_per_coordinate": sorted(set(coordinates.values())),
        "prefix_template_count": len(prefixes),
        "rows_per_prefix_template": sorted(set(prefixes.values())),
        "prompt_count": len(prompt_ids),
        "rows_per_prompt": sorted(set(prompt_ids.values())),
        "unique_content_prompt_prefix_pairs": len(content_pairs),
        "duplicate_content_prompt_prefix_pair_extra_rows": duplicate_content_pairs,
        "unique_prompt_prefix_pairs": len(prompt_pairs),
        "duplicate_prompt_prefix_pair_extra_rows": duplicate_prompt_pairs,
        "duplicate_row_key_extra_rows": duplicate_row_keys,
        "previous_locked_scale_prompt_overlap_count": len(overlap),
    }


def write_report(output_dir: Path, summary: Mapping[str, Any]) -> None:
    metrics = summary["metrics"]
    text = f"""# R4 After-870987 Pre-FAR Standard-Control Row Bank Validation

Date: 2026-05-19

Status: `{summary['status']}`

This validation is artifact-only. It does not tokenize, score, generate, train,
enable an allowlist entry, submit Slurm, aggregate FAR, or create a paper-facing
claim.

```text
rows: {metrics['row_count']}
shards: {metrics['shard_count']}
prompts: {metrics['prompt_count']}
unique content prompt/prefix pairs: {metrics['unique_content_prompt_prefix_pairs']}
duplicate content prompt/prefix extra rows: {metrics['duplicate_content_prompt_prefix_pair_extra_rows']}
previous locked-scale prompt overlap: {metrics['previous_locked_scale_prompt_overlap_count']}
```

Next allowed action: actual Qwen tokenizer/controller preflight route planning
for the standard-control null row bank, plus organic-null wrapper design.
"""
    write_text_new(output_dir / "row_bank_validation_report.md", text)


def main() -> int:
    args = parse_args()
    row_bank_dir = resolve(args.row_bank_dir)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = row_bank_dir / "row_allocation_rows.jsonl"
    manifest_path = row_bank_dir / "row_allocation_manifest.json"
    rows = read_jsonl(rows_path)
    manifest = read_json(manifest_path)
    previous_rows = read_jsonl(resolve(args.locked_scale_rows))
    locked_summary = read_json(resolve(args.locked_scale_summary))
    errors: list[str] = []
    validate_locked_scale_summary(locked_summary, errors)
    metrics = validate_rows(rows, manifest, previous_rows=previous_rows, errors=errors)
    status = (
        "PASS_R4_AFTER_870987_PREFAR_STANDARD_CONTROL_ROW_BANK_VALIDATION_NO_SUBMIT"
        if not errors
        else "FAIL_R4_AFTER_870987_PREFAR_STANDARD_CONTROL_ROW_BANK_VALIDATION_NO_SUBMIT"
    )
    summary = {
        "schema_name": "natural_evidence_v2_r4_after_870987_prefar_standard_control_row_bank_validation_v1",
        "status": status,
        "errors": errors,
        "row_bank_dir": str(row_bank_dir.relative_to(ROOT)) if row_bank_dir.is_relative_to(ROOT) else str(row_bank_dir),
        "row_bank_rows_sha256": sha256_file(rows_path),
        "row_bank_manifest_sha256": sha256_file(manifest_path),
        "locked_scale_summary": str(resolve(args.locked_scale_summary).relative_to(ROOT)),
        "metrics": metrics,
        "slurm_allowed": False,
        "generation_started": False,
        "training_started": False,
        "paper_claim_allowed": False,
        "next_allowed_action": (
            "Run actual Qwen tokenizer/controller preflight route planning for the standard-control "
            "null row bank, and design the organic-null wrapper before any Slurm submission."
        ),
    }
    write_json_new(output_dir / "row_bank_validation_summary.json", summary)
    write_report(output_dir, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
