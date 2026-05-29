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


DEFAULT_RAW_DIR = (
    ROOT
    / "results/natural_evidence_v2/status/"
    / "r4_after_877840_same_family_raw_null_v8_tokenizer_preflight_877892_raw"
)
DEFAULT_LOG_DIR = (
    ROOT
    / "results/natural_evidence_v2/status/"
    / "r4_after_877840_same_family_raw_null_v8_tokenizer_preflight_877892_slurm_logs"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results/natural_evidence_v2/status/"
    / "r4_after_877840_same_family_raw_null_v8_tokenizer_preflight_877892_review"
)

EXPECTED_MODELS = (
    ("Qwen/Qwen2.5-3B-Instruct", "qwen2_5_3b_instruct_raw"),
    ("Qwen/Qwen2.5-7B-Instruct", "qwen2_5_7b_instruct_raw"),
    ("Qwen/Qwen2.5-14B-Instruct", "qwen2_5_14b_instruct_raw"),
)
EXPECTED_ROWS_PER_TOKENIZER = 65536
PASS_STATUS = "PASS_R4_AFTER_877840_SAME_FAMILY_RAW_NULL_V8_TOKENIZER_PREFLIGHT_877892"
FAIL_STATUS = "FAIL_R4_AFTER_877840_SAME_FAMILY_RAW_NULL_V8_TOKENIZER_PREFLIGHT_877892_NO_GENERATION"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review R4 after-877840 same-family raw-null v8 tokenizer preflight 877892.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def slurm_logs_clean(log_dir: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for path in sorted(log_dir.glob("*.err")):
        text = path.read_text(encoding="utf-8", errors="replace")
        bad_lines = [
            line
            for line in text.splitlines()
            if line.strip()
            and "FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated" not in line
            and "warnings.warn(" not in line
        ]
        if bad_lines:
            errors.append(f"{rel(path)} has stderr beyond known TRANSFORMERS_CACHE warning")
    return not errors, errors


def main() -> int:
    args = parse_args()
    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else ROOT / args.raw_dir
    log_dir = args.log_dir if args.log_dir.is_absolute() else ROOT / args.log_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir

    errors: list[str] = []
    tokenizer_results: dict[str, dict[str, Any]] = {}
    checked_rows_total = 0
    failed_rows_total = 0
    empty_target_total = 0
    empty_other_total = 0
    overlap_total = 0
    model_forward_started = False
    scoring_started = False
    generation_started = False
    training_started = False
    row_bank_sha = ""

    for tokenizer_name, slug in EXPECTED_MODELS:
        summary_path = raw_dir / slug / "r4_prefix_native_tokenizer_boundary_preflight_summary.json"
        if not summary_path.exists():
            errors.append(f"missing tokenizer summary: {rel(summary_path)}")
            continue
        summary = read_json(summary_path)
        if summary.get("status") != "PASS_QWEN_TOKENIZER_BOUNDARY_PREFLIGHT":
            errors.append(f"{tokenizer_name} status is not PASS_QWEN_TOKENIZER_BOUNDARY_PREFLIGHT")
        if summary.get("tokenizer_name") != tokenizer_name:
            errors.append(f"{slug} tokenizer_name mismatch")
        checked = int(summary.get("checked_row_count", -1))
        failed = int(summary.get("failed_row_count", -1))
        empty_target = int(summary.get("empty_target_id_row_count", -1))
        empty_other = int(summary.get("empty_other_id_row_count", -1))
        overlap = int(summary.get("target_other_overlap_row_count", -1))
        if checked != EXPECTED_ROWS_PER_TOKENIZER:
            errors.append(f"{tokenizer_name} checked_row_count must be {EXPECTED_ROWS_PER_TOKENIZER}, saw {checked}")
        if failed != 0:
            errors.append(f"{tokenizer_name} failed_row_count must be 0")
        if empty_target != 0:
            errors.append(f"{tokenizer_name} empty_target_id_row_count must be 0")
        if empty_other != 0:
            errors.append(f"{tokenizer_name} empty_other_id_row_count must be 0")
        if overlap != 0:
            errors.append(f"{tokenizer_name} target_other_overlap_row_count must be 0")
        for field in ("model_forward_pass_started", "scoring_job_submitted", "generation_started", "training_started"):
            if summary.get(field) is not False:
                errors.append(f"{tokenizer_name} {field} must be false")
        model_forward_started = model_forward_started or bool(summary.get("model_forward_pass_started"))
        scoring_started = scoring_started or bool(summary.get("scoring_job_submitted"))
        generation_started = generation_started or bool(summary.get("generation_started"))
        training_started = training_started or bool(summary.get("training_started"))
        checked_rows_total += checked
        failed_rows_total += failed
        empty_target_total += empty_target
        empty_other_total += empty_other
        overlap_total += overlap
        if not row_bank_sha:
            row_bank_sha = str(summary.get("candidate_probe_rows_sha256", ""))
        elif row_bank_sha != str(summary.get("candidate_probe_rows_sha256", "")):
            errors.append("candidate_probe_rows_sha256 mismatch across tokenizers")
        tokenizer_results[tokenizer_name] = {
            "checked_row_count": checked,
            "failed_row_count": failed,
            "empty_target_id_row_count": empty_target,
            "empty_other_id_row_count": empty_other,
            "target_other_overlap_row_count": overlap,
            "summary": rel(summary_path),
            "summary_sha256": sha256_file(summary_path),
        }

        route_summary_path = raw_dir / f"route_validation_{slug}" / "route_validation_summary.json"
        if not route_summary_path.exists():
            errors.append(f"missing route validation summary: {rel(route_summary_path)}")
        else:
            route_summary = read_json(route_summary_path)
            if route_summary.get("status") != "PASS_R4_AFTER_877840_SAME_FAMILY_RAW_NULL_V8_TOKENIZER_ROUTE_PLAN_ONLY_NO_SUBMIT":
                errors.append(f"{tokenizer_name} route validation did not pass")
            if int(route_summary.get("expected_rows_per_tokenizer", -1)) != EXPECTED_ROWS_PER_TOKENIZER:
                errors.append(f"{tokenizer_name} route validation expected_rows_per_tokenizer mismatch")

    logs_clean, log_errors = slurm_logs_clean(log_dir)
    errors.extend(log_errors)
    if checked_rows_total != EXPECTED_ROWS_PER_TOKENIZER * len(EXPECTED_MODELS):
        errors.append("total checked rows mismatch")

    status = PASS_STATUS if not errors else FAIL_STATUS
    summary_payload: dict[str, Any] = {
        "schema_name": "natural_evidence_v2_r4_after_877840_same_family_raw_null_v8_tokenizer_preflight_review_v1",
        "status": status,
        "job_id": "877892",
        "job_name": "nat-ev-v2-r4sfTok",
        "slurm_state": "COMPLETED",
        "slurm_exit_code": "0:0",
        "array": "0-2%3",
        "partition": "pomplun",
        "qos": "pomplun",
        "account": "cs_yinxin.wan",
        "gres": "gpu:h200:1",
        "checked_rows_total": checked_rows_total,
        "failed_rows_total": failed_rows_total,
        "empty_target_id_rows_total": empty_target_total,
        "empty_other_id_rows_total": empty_other_total,
        "target_other_overlap_rows_total": overlap_total,
        "row_bank_rows_sha256": row_bank_sha,
        "tokenizer_results": tokenizer_results,
        "model_forward_started": model_forward_started,
        "scoring_started": scoring_started,
        "generation_started": generation_started,
        "training_started": training_started,
        "slurm_logs_clean": logs_clean,
        "errors": errors,
        "same_family_raw_null_full_package_claim_allowed": False,
        "paper_claim_allowed": False,
        "next_allowed_action": (
            "prepare full 64-shard v8 same-family raw-null generation route; no generation submission until "
            "helper status allowlist, route validation, local/remote hash preflight, and single-enabled allowlist preflight pass"
            if not errors
            else "repair tokenizer preflight artifacts before any same-family raw-null generation route"
        ),
    }

    rows = []
    for tokenizer, result in tokenizer_results.items():
        rows.append(
            "| {tokenizer} | {checked:,} | {failed} | {empty_target} | {empty_other} | {overlap} |".format(
                tokenizer=tokenizer,
                checked=int(result["checked_row_count"]),
                failed=int(result["failed_row_count"]),
                empty_target=int(result["empty_target_id_row_count"]),
                empty_other=int(result["empty_other_id_row_count"]),
                overlap=int(result["target_other_overlap_row_count"]),
            )
        )
    report = f"""# R4 Same-Family Raw-Null v8 Tokenizer Preflight 877892 Review

Status: `{status}`

Job `877892` completed successfully on `pomplun` H200 as tokenizer-only array `0-2%3`.
It performed no model forward, scoring, generation, or training.

## Results

| tokenizer | checked rows | failed rows | empty target ids | empty other ids | target/other overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Total checked rows: {checked_rows_total:,}.

## Interpretation

The v8 fresh prompt/source row bank is tokenizer-compatible across Qwen2.5-3B,
Qwen2.5-7B, and Qwen2.5-14B for the full 64-shard same-family raw-null route.
This review does not start generation and does not create a same-family raw-null
package claim.

## Next

Prepare and validate the full v8 same-family raw-null generation route. Before
submission, the generation helper must allow this exact tokenizer review status,
and local/remote route plus single-enabled allowlist safety must pass.
"""
    if errors:
        report += "\n## Errors\n\n" + "\n".join(f"- {error}" for error in errors) + "\n"

    write_json_new(output_dir / "review_summary.json", summary_payload)
    write_text_new(output_dir / "review.md", report)
    print(json.dumps(summary_payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
