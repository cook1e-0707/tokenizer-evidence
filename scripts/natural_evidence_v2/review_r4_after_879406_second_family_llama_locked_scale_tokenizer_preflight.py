#!/usr/bin/env python3
"""Review the R4 after-879406 Llama locked-scale tokenizer preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_cover_natural_common import read_json, sha256_file, write_json_new, write_text_new  # noqa: E402


EXPECTED_JOB_ID = "879455"
EXPECTED_ROWS = 98304
EXPECTED_TOKENIZER = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PASS_STATUS = "PASS_R4_AFTER_879406_SECOND_FAMILY_LLAMA_LOCKED_SCALE_TOKENIZER_PREFLIGHT_879455_REVIEWED"
FAIL_STATUS = "FAIL_R4_AFTER_879406_SECOND_FAMILY_LLAMA_LOCKED_SCALE_TOKENIZER_PREFLIGHT_879455_NO_GENERATION"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--sacct", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job-id", default=EXPECTED_JOB_ID)
    return parser.parse_args()


def find_one(root: Path, pattern: str, errors: list[str]) -> Path | None:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        errors.append(f"expected one {pattern} under {root}, found {len(matches)}")
        return None
    return matches[0]


def bool_false(summary: Mapping[str, Any], key: str, errors: list[str]) -> None:
    if summary.get(key) is not False:
        errors.append(f"{key} must be false")


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root if args.raw_root.is_absolute() else ROOT / args.raw_root
    sacct_path = args.sacct if args.sacct.is_absolute() else ROOT / args.sacct
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    errors: list[str] = []

    summary_path = find_one(raw_root, "**/llama3_1_8b_instruct/r4_prefix_native_tokenizer_boundary_preflight_summary.json", errors)
    route_summary_path = find_one(raw_root, "**/route_validation_llama3_1_8b_instruct/route_validation_summary.json", errors)

    tokenizer_summary: dict[str, Any] = {}
    route_summary: dict[str, Any] = {}
    if summary_path is not None:
        tokenizer_summary = read_json(summary_path)
    if route_summary_path is not None:
        route_summary = read_json(route_summary_path)

    if tokenizer_summary.get("status") != "PASS_QWEN_TOKENIZER_BOUNDARY_PREFLIGHT":
        errors.append("tokenizer preflight summary must pass")
    if int(tokenizer_summary.get("checked_row_count", -1)) != EXPECTED_ROWS:
        errors.append(f"checked_row_count must be {EXPECTED_ROWS}")
    if int(tokenizer_summary.get("score_row_count", -1)) != EXPECTED_ROWS:
        errors.append(f"score_row_count must be {EXPECTED_ROWS}")
    for key in (
        "failed_row_count",
        "empty_target_id_row_count",
        "empty_other_id_row_count",
        "target_other_overlap_row_count",
    ):
        if int(tokenizer_summary.get(key, -1)) != 0:
            errors.append(f"{key} must be 0")
    if tokenizer_summary.get("tokenizer_name") != EXPECTED_TOKENIZER:
        errors.append("tokenizer_name mismatch")
    if tokenizer_summary.get("tokenizer_preflight_started") is not True:
        errors.append("tokenizer_preflight_started must be true")
    for key in (
        "model_forward_pass_started",
        "generation_started",
        "training_started",
        "same_family_null_started",
        "sanitizer_benchmark_started",
        "far_aggregation_started",
        "paper_claim_allowed",
        "scoring_job_submitted",
        "scoring_authorized",
    ):
        bool_false(tokenizer_summary, key, errors)

    if route_summary.get("status") != "PASS_R4_AFTER_879406_SECOND_FAMILY_LLAMA_LOCKED_SCALE_TOKENIZER_ROUTE_VALIDATION_NO_SUBMIT":
        errors.append("route validation summary must pass the locked-scale tokenizer route")

    sacct_text = sacct_path.read_text(encoding="utf-8") if sacct_path.exists() else ""
    if args.job_id not in sacct_text:
        errors.append("sacct record missing job id")
    if "COMPLETED" not in sacct_text or "0:0" not in sacct_text:
        errors.append("sacct record must show COMPLETED 0:0")

    status = PASS_STATUS if not errors else FAIL_STATUS
    summary = {
        "schema_name": "r4_after_879406_second_family_llama_locked_scale_tokenizer_preflight_review_v1",
        "status": status,
        "errors": errors,
        "job_id": args.job_id,
        "tokenizer_name": tokenizer_summary.get("tokenizer_name"),
        "checked_row_count": tokenizer_summary.get("checked_row_count"),
        "failed_row_count": tokenizer_summary.get("failed_row_count"),
        "empty_target_id_row_count": tokenizer_summary.get("empty_target_id_row_count"),
        "empty_other_id_row_count": tokenizer_summary.get("empty_other_id_row_count"),
        "target_other_overlap_row_count": tokenizer_summary.get("target_other_overlap_row_count"),
        "tokenizer_summary": str(summary_path.relative_to(ROOT)) if summary_path else None,
        "tokenizer_summary_sha256": sha256_file(summary_path) if summary_path else None,
        "route_validation_summary": str(route_summary_path.relative_to(ROOT)) if route_summary_path else None,
        "route_validation_summary_sha256": sha256_file(route_summary_path) if route_summary_path else None,
        "sacct": str(sacct_path.relative_to(ROOT)) if sacct_path.exists() else str(sacct_path),
        "sacct_sha256": sha256_file(sacct_path) if sacct_path.exists() else None,
        "tokenizer_preflight_passed": not errors,
        "locked_scale_generation_allowed_by_this_review": False,
        "paper_claim_allowed": False,
        "model_forward_started": False,
        "generation_started": False,
        "training_started": False,
        "next_allowed_action": (
            "Prepare a separate reviewed Llama locked-scale generation route; no generation "
            "submission is authorized by this tokenizer review alone."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_new(output_dir / "review_summary.json", summary)
    report = f"""# R4 After 879406 Llama Locked-Scale Tokenizer Preflight Review

Status: `{status}`

Job `{args.job_id}` completed as tokenizer-only preflight.

Key facts:

```text
checked rows: {summary["checked_row_count"]}
failed rows: {summary["failed_row_count"]}
empty target ids: {summary["empty_target_id_row_count"]}
empty other ids: {summary["empty_other_id_row_count"]}
target/other overlaps: {summary["target_other_overlap_row_count"]}
tokenizer: {summary["tokenizer_name"]}
```

This review does not authorize paper-facing claims. It only confirms that the
Llama tokenizer can score the locked-scale first-token event row bank without
boundary failures.

Next allowed action: {summary["next_allowed_action"]}
"""
    write_text_new(output_dir / "review_report.md", report)
    print(json.dumps({"status": status, "output_dir": str(output_dir)}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
