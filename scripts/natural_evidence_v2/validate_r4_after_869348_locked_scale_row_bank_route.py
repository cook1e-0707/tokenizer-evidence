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
from scripts.natural_evidence_v2.score_r4_surface_teacher_forced_mass import (  # noqa: E402
    r4_row_surface_contract,
)


DEFAULT_ROW_BANK_DIR = (
    ROOT / "results/natural_evidence_v2/status/r4_after_869348_global_unique_locked_scale_row_bank_plan_20260518"
)
DEFAULT_DEV_REVIEW = (
    ROOT
    / "results/natural_evidence_v2/status/r4_after_868348_global_unique_dev_diagnostic_869348_review/"
    "quality_repair_confirmation_review_summary.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/natural_evidence_v2/status/r4_after_869348_locked_scale_row_bank_route_validation_20260518"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the artifact-only held-out locked-scale row-bank route after 869348. "
            "This does not tokenize, score, generate, train, enable an allowlist, or submit Slurm."
        )
    )
    parser.add_argument("--row-bank-dir", type=Path, default=DEFAULT_ROW_BANK_DIR)
    parser.add_argument("--dev-review", type=Path, default=DEFAULT_DEV_REVIEW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def int_field(data: Mapping[str, Any], field: str, default: int = -1) -> int:
    try:
        return int(data.get(field, default))
    except (TypeError, ValueError):
        return default


def validate_dev_review(dev_review: Mapping[str, Any], errors: list[str]) -> None:
    if dev_review.get("status") != "PASS_R4_AFTER_868348_GLOBAL_UNIQUE_DEV_DIAGNOSTIC_GATE":
        fail(errors, "869348 dev review status must pass")
    for field, expected in (
        ("source_job_id", "869348"),
        ("protected_strict_accepts", 32),
        ("protected_accepts_ignoring_quality", 32),
        ("global_duplicate_response_hash_count", 0),
        ("protected_duplicate_response_hash_count", 0),
        ("protected_forbidden_public_surface_count", 0),
        ("trace_binding_invalid_rows", 0),
    ):
        actual = dev_review.get(field)
        if isinstance(expected, int):
            actual = int_field(dev_review, field)
        if actual != expected:
            fail(errors, f"869348 dev review {field} must be {expected}, found {actual}")
    controls = dev_review.get("control_accepts", {})
    if not isinstance(controls, Mapping) or any(int(controls.get(arm, -1)) != 0 for arm in ("raw", "task_only", "wrong_key", "wrong_payload")):
        fail(errors, "869348 dev review controls must be zero")


def validate_rows(rows: list[Mapping[str, Any]], manifest: Mapping[str, Any], errors: list[str]) -> dict[str, Any]:
    if len(rows) != 98304:
        fail(errors, f"row count must be 98304, found {len(rows)}")

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
    splits = Counter()
    prompt_ids = Counter()
    coord_prefix = Counter()
    for index, row in enumerate(rows):
        try:
            shard = int(row.get("assigned_shard_index", -1))
            coordinate = int(row["coordinate_id"])
        except Exception as exc:  # noqa: BLE001 - report row context in validation output.
            fail(errors, f"row {index} missing shard/coordinate: {type(exc).__name__}:{exc}")
            continue
        shards[shard].append(row)
        coordinates[coordinate] += 1
        prefix = str(row.get("prefix_family_id", ""))
        prefixes[prefix] += 1
        splits[str(row.get("split", ""))] += 1
        prompt_ids[str(row.get("prompt_id", ""))] += 1
        coord_prefix[(coordinate, prefix)] += 1

        if row.get("schema_name") != "natural_evidence_v2_r4_after_869348_global_unique_locked_scale_row_bank_row_v1":
            fail(errors, f"row {index} schema_name mismatch")
        if row.get("contract_id") != "a55e":
            fail(errors, f"row {index} contract_id is not a55e")
        if row.get("source_dev_pass_job_id") != "869348":
            fail(errors, f"row {index} source_dev_pass_job_id mismatch")
        if row.get("split") != "locked":
            fail(errors, f"row {index} split must be locked")
        if row.get("payload_diversity_tested") is not False or row.get("same_contract_only") is not True:
            fail(errors, f"row {index} payload/contract scope flags invalid")
        if row.get("generation_started") or row.get("model_scoring_started") or row.get("training_started"):
            fail(errors, f"row {index} has non-artifact-only execution flag")
        if row.get("slurm_submitted") or row.get("paper_claim_allowed"):
            fail(errors, f"row {index} has forbidden route/claim flag")
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

    if sorted(shards) != list(range(96)):
        fail(errors, f"expected shards 0..95, found {sorted(shards)}")
    for shard, shard_rows in sorted(shards.items()):
        if len(shard_rows) != 1024:
            fail(errors, f"shard {shard} row count must be 1024, found {len(shard_rows)}")
        shard_coord_counts = Counter(int(row["coordinate_id"]) for row in shard_rows)
        if set(shard_coord_counts.values()) != {64} or len(shard_coord_counts) != 16:
            fail(errors, f"shard {shard} must have 64 rows for each of 16 coordinates")
        shard_content_pairs = Counter(str(row.get("content_duplicate_pair_key", "")) for row in shard_rows)
        shard_duplicate_pairs = sum(count - 1 for count in shard_content_pairs.values() if count > 1)
        if shard_duplicate_pairs:
            fail(errors, f"shard {shard} duplicate content prompt/prefix pairs: {shard_duplicate_pairs}")

    if len(coordinates) != 16 or set(coordinates.values()) != {6144}:
        fail(errors, f"expected 16 coordinates with 6144 rows each, found {dict(sorted(coordinates.items()))}")
    if len(prefixes) != 16 or set(prefixes.values()) != {6144}:
        fail(errors, f"expected 16 prefixes with 6144 rows each, found {dict(sorted(prefixes.items()))}")
    if splits != Counter({"locked": len(rows)}):
        fail(errors, f"all rows must use locked split, found {dict(splits)}")
    if len(prompt_ids) != 6144 or set(prompt_ids.values()) != {16}:
        fail(errors, f"expected 6144 locked prompts used exactly 16 times, found {len(prompt_ids)} prompts")

    if manifest.get("status") != "PASS_R4_AFTER_869348_GLOBAL_UNIQUE_LOCKED_SCALE_ROW_BANK_BUILT_ARTIFACT_ONLY_NO_SUBMIT":
        fail(errors, f"manifest status mismatch: {manifest.get('status')}")
    for field, expected in (
        ("row_count", 98304),
        ("target_shards", 96),
        ("rows_per_shard", 1024),
        ("selected_prompt_count", 6144),
        ("selected_coordinate_count", 16),
        ("unique_content_prompt_prefix_pairs", 98304),
        ("duplicate_content_prompt_prefix_pair_extra_rows", 0),
        ("duplicate_prompt_prefix_pair_extra_rows", 0),
    ):
        if int_field(manifest, field) != expected:
            fail(errors, f"manifest {field} must be {expected}")
    if manifest.get("locked_scale_candidate") is not True:
        fail(errors, "manifest locked_scale_candidate must be true")
    if manifest.get("payload_diversity_tested") is not False or manifest.get("same_contract_only") is not True:
        fail(errors, "manifest payload/contract scope flags invalid")
    if manifest.get("generation_started") or manifest.get("model_scoring_started") or manifest.get("training_started"):
        fail(errors, "manifest has non-artifact-only execution flag")
    if manifest.get("slurm_submitted") or manifest.get("paper_claim_allowed"):
        fail(errors, "manifest has forbidden route/claim flag")

    return {
        "row_count": len(rows),
        "shard_count": len(shards),
        "rows_per_shard": sorted({len(items) for items in shards.values()}),
        "selected_coordinate_count": len(coordinates),
        "rows_per_coordinate": sorted(set(coordinates.values())),
        "prefix_template_count": len(prefixes),
        "rows_per_prefix_template": sorted(set(prefixes.values())),
        "locked_prompt_count": len(prompt_ids),
        "rows_per_locked_prompt": sorted(set(prompt_ids.values())),
        "unique_content_prompt_prefix_pairs": len(content_pairs),
        "duplicate_content_prompt_prefix_pair_extra_rows": duplicate_content_pairs,
        "unique_prompt_prefix_pairs": len(prompt_pairs),
        "duplicate_prompt_prefix_pair_extra_rows": duplicate_prompt_pairs,
        "duplicate_row_key_extra_rows": duplicate_row_keys,
        "max_coordinate_prefix_pair_count": max(coord_prefix.values()) if coord_prefix else 0,
    }


def write_report(output_dir: Path, *, summary: Mapping[str, Any]) -> None:
    text = f"""# R4 After-869348 Locked-Scale Row Bank Route Validation

Date: 2026-05-18

Status: `{summary['status']}`

This validation is artifact-only. It does not tokenize, score, generate, train,
enable an allowlist entry, or submit Slurm.

```text
row bank: {summary['row_bank_dir']}
rows: {summary['metrics']['row_count']}
shards: {summary['metrics']['shard_count']}
locked prompts: {summary['metrics']['locked_prompt_count']}
unique content prompt/prefix pairs: {summary['metrics']['unique_content_prompt_prefix_pairs']}
duplicate content prompt/prefix extra rows: {summary['metrics']['duplicate_content_prompt_prefix_pair_extra_rows']}
coordinates: {summary['metrics']['selected_coordinate_count']}
```

Next allowed action: static and actual Qwen tokenizer/controller preflight for
this locked row bank. No generation or locked-scale submission is allowed until
those checks pass and a reviewed H200 route is recorded.
"""
    write_text_new(output_dir / "route_validation_report.md", text)


def main() -> int:
    args = parse_args()
    row_bank_dir = args.row_bank_dir if args.row_bank_dir.is_absolute() else ROOT / args.row_bank_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = row_bank_dir / "row_allocation_rows.jsonl"
    manifest_path = row_bank_dir / "row_allocation_manifest.json"
    rows = read_jsonl(rows_path)
    manifest = read_json(manifest_path)
    dev_review = read_json(args.dev_review if args.dev_review.is_absolute() else ROOT / args.dev_review)
    errors: list[str] = []
    validate_dev_review(dev_review, errors)
    metrics = validate_rows(rows, manifest, errors)
    status = (
        "PASS_R4_AFTER_869348_LOCKED_SCALE_ROW_BANK_ROUTE_VALIDATION_NO_SUBMIT"
        if not errors
        else "FAIL_R4_AFTER_869348_LOCKED_SCALE_ROW_BANK_ROUTE_VALIDATION_NO_SUBMIT"
    )
    summary = {
        "schema_name": "natural_evidence_v2_r4_after_869348_locked_scale_row_bank_route_validation_v1",
        "status": status,
        "errors": errors,
        "source_dev_pass_job_id": "869348",
        "row_bank_dir": str(row_bank_dir.relative_to(ROOT)),
        "row_allocation_rows": str(rows_path.relative_to(ROOT)),
        "row_allocation_rows_sha256": sha256_file(rows_path),
        "row_allocation_manifest": str(manifest_path.relative_to(ROOT)),
        "row_allocation_manifest_sha256": sha256_file(manifest_path),
        "source_dev_review": str((args.dev_review if args.dev_review.is_absolute() else ROOT / args.dev_review).relative_to(ROOT)),
        "source_dev_review_sha256": sha256_file(args.dev_review if args.dev_review.is_absolute() else ROOT / args.dev_review),
        "metrics": metrics,
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "slurm_submitted": False,
        "paper_claim_allowed": False,
        "next_allowed_action": (
            "Run static and actual Qwen tokenizer/controller preflight for this locked row bank; no generation "
            "or Slurm locked-scale submission until those pass and a reviewed H200 route is recorded."
        ),
    }
    write_json_new(output_dir / "route_validation_summary.json", summary)
    write_report(output_dir, summary=summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
