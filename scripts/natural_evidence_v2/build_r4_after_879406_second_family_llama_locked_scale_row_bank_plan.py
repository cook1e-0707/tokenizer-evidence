#!/usr/bin/env python3
"""Build the artifact-only Llama locked-scale row-bank plan after 879406.

This copies the reviewed tokenizer-neutral 96-shard Qwen locked-scale row bank
into a second-family Llama locked-scale candidate package. It does not tokenize,
load a model, generate, train, enable an allowlist, or submit Slurm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROWS = (
    ROOT
    / "results/natural_evidence_v2/status/"
    / "r4_after_869348_global_unique_locked_scale_row_bank_plan_20260518/"
    / "row_allocation_rows.jsonl"
)
DEFAULT_SOURCE_MANIFEST = DEFAULT_SOURCE_ROWS.with_name("row_allocation_manifest.json")
DEFAULT_LLAMA_DEV_REVIEW = (
    ROOT
    / "results/natural_evidence_v2/status/"
    / "r4_after_879391_second_family_llama_dev_diagnostic_policy_v4_879406_review_r1/"
    / "review_summary.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results/natural_evidence_v2/status/"
    / "r4_after_879406_second_family_llama_locked_scale_row_bank_plan_20260527"
)
EXPECTED_ROWS = 98_304
EXPECTED_PROMPTS = 6_144
EXPECTED_SHARDS = 96
EXPECTED_ROWS_PER_SHARD = 1_024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-rows", type=Path, default=DEFAULT_SOURCE_ROWS)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--llama-dev-review", type=Path, default=DEFAULT_LLAMA_DEV_REVIEW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    source_rows = args.source_rows if args.source_rows.is_absolute() else ROOT / args.source_rows
    source_manifest = args.source_manifest if args.source_manifest.is_absolute() else ROOT / args.source_manifest
    llama_dev_review_path = args.llama_dev_review if args.llama_dev_review.is_absolute() else ROOT / args.llama_dev_review
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output dir: {output_dir}")
    if not source_rows.exists():
        raise FileNotFoundError(source_rows)
    if not source_manifest.exists():
        raise FileNotFoundError(source_manifest)
    if not llama_dev_review_path.exists():
        raise FileNotFoundError(llama_dev_review_path)

    manifest = read_json(source_manifest)
    llama_dev_review = read_json(llama_dev_review_path)
    rows = iter_jsonl(source_rows)
    prompt_ids = {str(row.get("prompt_id", "")) for row in rows}
    shard_counts: dict[int, int] = {}
    token_id_fields = sorted(
        {
            key
            for row in rows[:256]
            for key in row
            if "token_id" in key.lower() or "first_token_ids" in key.lower()
        }
    )
    for row in rows:
        shard_index = int(row.get("assigned_shard_index", -1))
        shard_counts[shard_index] = shard_counts.get(shard_index, 0) + 1

    errors: list[str] = []
    if len(rows) != EXPECTED_ROWS:
        errors.append(f"expected {EXPECTED_ROWS} rows, got {len(rows)}")
    if len(prompt_ids) != EXPECTED_PROMPTS:
        errors.append(f"expected {EXPECTED_PROMPTS} prompt ids, got {len(prompt_ids)}")
    if len(shard_counts) != EXPECTED_SHARDS:
        errors.append(f"expected {EXPECTED_SHARDS} shards, got {len(shard_counts)}")
    if set(shard_counts) != set(range(EXPECTED_SHARDS)):
        errors.append("assigned_shard_index must cover 0..95 exactly")
    if any(count != EXPECTED_ROWS_PER_SHARD for count in shard_counts.values()):
        errors.append("each shard must contain exactly 1024 rows")
    if token_id_fields:
        errors.append(f"source rows contain tokenizer-specific token id fields: {token_id_fields}")
    if manifest.get("status") != "PASS_R4_AFTER_869348_GLOBAL_UNIQUE_LOCKED_SCALE_ROW_BANK_BUILT_ARTIFACT_ONLY_NO_SUBMIT":
        errors.append("source manifest is not the reviewed 96-shard locked-scale row bank")
    if llama_dev_review.get("status") != "PASS_R4_AFTER_877895_SECOND_FAMILY_LLAMA_DEV_DIAGNOSTIC_879406_REVIEWED":
        errors.append("Llama dev diagnostic 879406 must be reviewed pass before locked-scale row-bank planning")
    if llama_dev_review.get("second_family_dev_diagnostic_gate_pass") is not True:
        errors.append("Llama dev diagnostic gate pass must be true")

    output_dir.mkdir(parents=True)
    output_rows = output_dir / "row_allocation_rows.jsonl"
    with output_rows.open("w", encoding="utf-8") as handle:
        for row in rows:
            copied = dict(row)
            copied.update(
                {
                    "schema_name": "natural_evidence_v2_r4_after_879406_second_family_llama_locked_scale_row_v1",
                    "artifact_role": "r4_after_879406_second_family_llama_locked_scale_row_bank_not_tokenized_not_scored",
                    "second_family_route_id": "r4_after_879406_second_family_llama_locked_scale",
                    "second_family_model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                    "second_family_tokenizer_preflight_started": False,
                    "second_family_tokenizer_preflight_status": "PENDING",
                    "source_llama_dev_pass_job_id": "879406",
                    "source_llama_dev_pass_status": "PASS_R4_AFTER_877895_SECOND_FAMILY_LLAMA_DEV_DIAGNOSTIC_879406_REVIEWED",
                    "source_qwen_locked_scale_row_bank": rel(source_rows),
                    "llama_locked_scale_candidate": True,
                    "payload_diversity_tested": False,
                    "same_contract_only": True,
                    "paper_claim_allowed": False,
                    "locked_scale_cross_family_claim_allowed": False,
                }
            )
            handle.write(json.dumps(copied, sort_keys=True) + "\n")

    summary = {
        "schema_name": "r4_after_879406_second_family_llama_locked_scale_row_bank_plan_v1",
        "status": (
            "PASS_R4_AFTER_879406_SECOND_FAMILY_LLAMA_LOCKED_SCALE_ROW_BANK_PLAN_ARTIFACT_ONLY_TOKENIZER_PENDING"
            if not errors
            else "FAIL_R4_AFTER_879406_SECOND_FAMILY_LLAMA_LOCKED_SCALE_ROW_BANK_PLAN"
        ),
        "errors": errors,
        "artifact_only": True,
        "source_rows": rel(source_rows),
        "source_rows_sha256": sha256_file(source_rows),
        "source_manifest": rel(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_llama_dev_review": rel(llama_dev_review_path),
        "source_llama_dev_review_sha256": sha256_file(llama_dev_review_path),
        "output_rows": rel(output_rows),
        "output_rows_sha256": sha256_file(output_rows),
        "selected_row_count": len(rows),
        "selected_prompt_count": len(prompt_ids),
        "target_shards": len(shard_counts),
        "rows_per_shard": EXPECTED_ROWS_PER_SHARD,
        "tokenizer_specific_token_id_fields_detected": token_id_fields,
        "candidate_model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "tokenizer_preflight_started": False,
        "model_scoring_started": False,
        "generation_started": False,
        "training_started": False,
        "slurm_submitted": False,
        "allowlist_enabled": False,
        "paper_claim_allowed": False,
        "locked_scale_cross_family_claim_allowed": False,
        "next_allowed_action": (
            "Prepare artifact-only Llama locked-scale tokenizer preflight route validation. "
            "Do not submit generation until tokenizer preflight passes and a reviewed locked-scale H200 route is recorded."
        ),
    }
    write_json(output_dir / "row_bank_plan_summary.json", summary)
    report = f"""# R4 After-879406 Second-Family Llama Locked-Scale Row-Bank Plan

Status: `{summary["status"]}`

This is artifact-only. It copies the reviewed tokenizer-neutral 96-shard Qwen
locked-scale row bank into a Llama locked-scale candidate package after the
`879406` 32-block Llama dev diagnostic passed.

It does not run a tokenizer, load model weights, submit Slurm, generate, train,
or create a paper-facing or locked-scale transfer claim.

## Counts

- rows: `{summary["selected_row_count"]}`
- prompts: `{summary["selected_prompt_count"]}`
- shards: `{summary["target_shards"]}`
- rows per shard: `{summary["rows_per_shard"]}`
- tokenizer-specific token-id fields detected: `{summary["tokenizer_specific_token_id_fields_detected"]}`

## Next Allowed Action

Artifact-only Llama tokenizer-boundary preflight route validation for this
96-shard row bank. No Llama locked-scale generation may be submitted until that
preflight and a reviewed H200 route pass.
"""
    (output_dir / "row_bank_plan_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": summary["status"], "output_dir": rel(output_dir)}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
