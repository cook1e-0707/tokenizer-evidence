from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_cover_natural_common import sha256_file, write_json_new, write_text_new  # noqa: E402


DEFAULT_INPUT_DIR = ROOT / "results/natural_evidence_v2/status/r4_after_870987_prefar_organic_null_row_bank_v2_plan_20260521"
DEFAULT_OUTPUT_DIR = ROOT / "results/natural_evidence_v2/status/r4_after_870987_prefar_organic_null_row_bank_v2_validation_20260521"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate R4 pre-FAR organic-null row bank artifacts.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-shards", type=int, default=256)
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


def add_error(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def main() -> int:
    args = parse_args()
    input_dir = resolve(args.input_dir)
    output_dir = resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {output_dir}")
    rows_path = input_dir / "row_allocation_rows.jsonl"
    manifest_path = input_dir / "row_allocation_manifest.json"
    rows = read_jsonl(rows_path)
    manifest = read_json(manifest_path)

    manifest_rows_per_shard = int(manifest.get("rows_per_shard", 0))
    manifest_prompts_per_shard = int(manifest.get("prompts_per_shard", 0))
    expected_rows = int(args.expected_shards) * manifest_rows_per_shard
    errors: list[str] = []
    shard_counts = Counter(int(row.get("assigned_shard_index", -1)) for row in rows)
    row_keys = Counter(str(row.get("row_key", "")) for row in rows)
    generation_ids = Counter(str(row.get("generation_id", "")) for row in rows if row.get("generation_id"))
    prompt_prefix = Counter(
        f"{row.get('prompt_id')}::{row.get('prefix_family_id')}::{row.get('assistant_prefix_before_surface')}"
        for row in rows
    )

    add_error(errors, len(rows) == expected_rows, f"row count {len(rows)} != expected {expected_rows}")
    add_error(errors, manifest.get("status") == "PASS_R4_AFTER_870987_PREFAR_ORGANIC_NULL_ROW_BANK_BUILT_ARTIFACT_ONLY_NO_SUBMIT", "manifest status is not organic-null row-bank pass")
    add_error(errors, bool(manifest.get("organic_null")) is True, "manifest organic_null must be true")
    add_error(errors, manifest.get("generation_conditions") == ["raw"], "manifest generation_conditions must be raw only")
    add_error(errors, int(manifest.get("target_organic_null_blocks", -1)) == int(args.expected_shards), "target organic blocks mismatch")
    add_error(errors, all(row.get("organic_null") is True for row in rows), "all rows must be organic_null=true")
    add_error(errors, all(row.get("standard_control_null_expansion") is False for row in rows), "organic rows must not be standard-control rows")
    add_error(errors, all(row.get("generation_conditions") == ["raw"] for row in rows), "all rows must be raw-only generation")
    add_error(errors, all(row.get("generation_started") is False for row in rows), "generation_started must remain false")
    add_error(errors, all(row.get("slurm_submitted") is False for row in rows), "slurm_submitted must remain false")
    add_error(errors, all(row.get("paper_claim_allowed") is False for row in rows), "paper_claim_allowed must remain false")
    add_error(errors, set(shard_counts) == set(range(int(args.expected_shards))), "assigned shard indexes must be complete")
    add_error(errors, manifest_rows_per_shard > 0, "manifest rows_per_shard must be positive")
    add_error(errors, manifest_prompts_per_shard == int(args.prompts_per_shard), "manifest prompts_per_shard mismatch")
    add_error(errors, all(count == manifest_rows_per_shard for count in shard_counts.values()), "each shard must have manifest rows_per_shard rows")
    add_error(errors, not any(count > 1 for count in row_keys.values()), "row_key values must be unique")
    add_error(errors, not any(count > 1 for count in prompt_prefix.values()), "prompt/prefix pairs must be unique")
    add_error(errors, not generation_ids, "artifact-only row bank must not contain generation ids")

    summary = {
        "schema_name": "natural_evidence_v2_r4_after_870987_prefar_organic_null_row_bank_validation_v1",
        "status": "PASS_R4_AFTER_870987_PREFAR_ORGANIC_NULL_ROW_BANK_VALIDATION_NO_SUBMIT" if not errors else "FAIL_R4_AFTER_870987_PREFAR_ORGANIC_NULL_ROW_BANK_VALIDATION_NO_SUBMIT",
        "errors": errors,
        "input_dir": str(input_dir.relative_to(ROOT)) if input_dir.is_relative_to(ROOT) else str(input_dir),
        "row_allocation_rows": str(rows_path.relative_to(ROOT)) if rows_path.is_relative_to(ROOT) else str(rows_path),
        "row_allocation_rows_sha256": sha256_file(rows_path),
        "row_allocation_manifest": str(manifest_path.relative_to(ROOT)) if manifest_path.is_relative_to(ROOT) else str(manifest_path),
        "row_allocation_manifest_sha256": sha256_file(manifest_path),
        "rows": len(rows),
        "expected_rows": expected_rows,
        "expected_shards": int(args.expected_shards),
        "rows_per_shard": manifest_rows_per_shard,
        "prompts_per_shard": manifest_prompts_per_shard,
        "generation_conditions": ["raw"],
        "generation_started": False,
        "slurm_submitted": False,
        "training_started": False,
        "paper_claim_allowed": False,
        "next_allowed_action": "Run actual Qwen tokenizer boundary preflight for organic-null rows; no generation or Slurm submission yet.",
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_new(output_dir / "validation_summary.json", summary)
    write_text_new(
        output_dir / "validation_report.md",
        "# R4 Pre-FAR Organic-Null Row Bank Validation\n\n"
        f"Status: `{summary['status']}`\n\n"
        f"Rows: {len(rows)} / {expected_rows}\n\n"
        f"Errors: {len(errors)}\n\n"
        "This validation is artifact-only and does not start generation, training, scoring, or Slurm.\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
