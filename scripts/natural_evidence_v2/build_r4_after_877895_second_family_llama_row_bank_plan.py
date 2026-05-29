#!/usr/bin/env python3
"""Build the artifact-only Llama candidate row-bank plan after 877895.

This copies tokenizer-neutral R4 first-token event rows into a second-family
candidate package and marks the actual Llama tokenizer preflight as pending.
It does not run a tokenizer, load a model, submit Slurm, generate, or train.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROWS = (
    ROOT
    / "results/natural_evidence_v2/status/"
    / "r4_after_877840_same_family_raw_null_v8_row_bank_plan_20260526/"
    / "row_allocation_rows.jsonl"
)
DEFAULT_SOURCE_MANIFEST = DEFAULT_SOURCE_ROWS.with_name("row_allocation_manifest.json")
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results/natural_evidence_v2/status/"
    / "r4_after_877895_second_family_llama_row_bank_plan_20260526"
)
EXPECTED_ROWS = 65536
EXPECTED_PROMPTS = 4096
EXPECTED_SHARDS = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-rows", type=Path, default=DEFAULT_SOURCE_ROWS)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    source_rows = args.source_rows if args.source_rows.is_absolute() else ROOT / args.source_rows
    source_manifest = args.source_manifest if args.source_manifest.is_absolute() else ROOT / args.source_manifest
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output dir: {output_dir}")
    if not source_rows.exists():
        raise FileNotFoundError(source_rows)
    if not source_manifest.exists():
        raise FileNotFoundError(source_manifest)

    manifest = read_json(source_manifest)
    rows = iter_rows(source_rows)
    prompt_ids = {str(row.get("prompt_id", "")) for row in rows}
    replicate_groups = {str(row.get("replicate_group_id", "")) for row in rows}
    token_id_fields = sorted(
        {
            key
            for row in rows[:256]
            for key in row
            if "token_id" in key.lower() or "first_token_ids" in key.lower()
        }
    )
    errors: list[str] = []
    if len(rows) != EXPECTED_ROWS:
        errors.append(f"expected {EXPECTED_ROWS} rows, got {len(rows)}")
    if len(prompt_ids) != EXPECTED_PROMPTS:
        errors.append(f"expected {EXPECTED_PROMPTS} prompt ids, got {len(prompt_ids)}")
    if len(replicate_groups) != EXPECTED_SHARDS:
        errors.append(f"expected {EXPECTED_SHARDS} replicate groups, got {len(replicate_groups)}")
    if token_id_fields:
        errors.append(f"source rows contain tokenizer-specific token id fields: {token_id_fields}")
    if manifest.get("status") != "PASS_R4_AFTER_877840_SAME_FAMILY_RAW_NULL_V8_ROW_BANK_BUILT_ARTIFACT_ONLY_NO_SUBMIT":
        errors.append("source manifest is not the reviewed v8 row-bank build")

    output_dir.mkdir(parents=True)
    output_rows = output_dir / "row_allocation_rows.jsonl"
    with output_rows.open("w", encoding="utf-8") as handle:
        for row in rows:
            copied = dict(row)
            copied["second_family_route_id"] = "r4_after_877895_second_family_llama"
            copied["second_family_model_id"] = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            copied["second_family_tokenizer_preflight_started"] = False
            copied["second_family_tokenizer_preflight_status"] = "PENDING"
            copied["qwen_same_family_source_row"] = True
            copied.pop("qwen_tokenizer_validation_started", None)
            handle.write(json.dumps(copied, sort_keys=True) + "\n")

    summary = {
        "schema_name": "r4_after_877895_second_family_llama_row_bank_plan_v1",
        "status": (
            "PASS_R4_AFTER_877895_SECOND_FAMILY_LLAMA_ROW_BANK_PLAN_ARTIFACT_ONLY_TOKENIZER_PENDING"
            if not errors
            else "FAIL_R4_AFTER_877895_SECOND_FAMILY_LLAMA_ROW_BANK_PLAN"
        ),
        "errors": errors,
        "artifact_only": True,
        "source_rows": str(source_rows.relative_to(ROOT)),
        "source_rows_sha256": sha256_file(source_rows),
        "source_manifest": str(source_manifest.relative_to(ROOT)),
        "source_manifest_sha256": sha256_file(source_manifest),
        "output_rows": str(output_rows.relative_to(ROOT)),
        "output_rows_sha256": sha256_file(output_rows),
        "selected_row_count": len(rows),
        "selected_prompt_count": len(prompt_ids),
        "target_shards": len(replicate_groups),
        "tokenizer_specific_token_id_fields_detected": token_id_fields,
        "candidate_model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "candidate_model_config": "configs/model/llama3_1_8b_instruct.yaml",
        "tokenizer_preflight_started": False,
        "model_scoring_started": False,
        "generation_started": False,
        "training_started": False,
        "slurm_submitted": False,
        "allowlist_enabled": False,
        "next_allowed_action": (
            "Prepare artifact-only Llama tokenizer preflight route validation for this row bank; "
            "do not submit Slurm until route config/wrapper/hash/allowlist preflights pass."
        ),
    }
    (output_dir / "row_bank_plan_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = f"""# R4 After-877895 Second-Family Llama Row-Bank Plan

Status: `{summary["status"]}`

This is artifact-only. It does not run the Llama tokenizer, load model weights,
submit Slurm, generate, or train.

## Facts

- Source rows: `{summary["source_rows"]}`
- Output rows: `{summary["output_rows"]}`
- Rows: `{summary["selected_row_count"]}`
- Prompts: `{summary["selected_prompt_count"]}`
- Shards: `{summary["target_shards"]}`
- Tokenizer-specific token-id fields detected in source sample:
  `{summary["tokenizer_specific_token_id_fields_detected"]}`

The candidate row bank is tokenizer-neutral and marked Llama-tokenizer-pending.
The next route must run a tokenizer-only boundary preflight before any Llama
model scoring or generation.
"""
    (output_dir / "row_bank_plan_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": summary["status"], "output_dir": str(output_dir)}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
