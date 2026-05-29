from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.classify_r4_forbidden_surface_context_v2 import classify_text  # noqa: E402
from scripts.natural_evidence_v2.r4_cover_natural_common import sha256_file, write_json_new, write_text_new  # noqa: E402


DEFAULT_INPUT_DIR = (
    ROOT
    / "results/natural_evidence_v2/status/"
    / "r4_after_870987_same_family_raw_null_v2_row_bank_plan_20260523"
)
DEFAULT_FAILURE_PROMPTS = (
    ROOT
    / "results/natural_evidence_v2/status/"
    / "r4_after_870987_same_family_raw_null_generation_875168_failure_review_20260523/"
    / "forbidden_collision_by_prompt.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results/natural_evidence_v2/status/"
    / "r4_after_870987_same_family_raw_null_v2_row_bank_validation_20260523"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate same-family raw-null v2 row bank artifacts.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--failure-prompts", type=Path, default=DEFAULT_FAILURE_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-shards", type=int, default=64)
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


def read_failure_prompt_ids(path: Path) -> set[str]:
    prompt_ids: set[str] = set()
    if not path.exists():
        return prompt_ids
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            prompt_id = str(row.get("prompt_id", "")).strip()
            if prompt_id:
                prompt_ids.add(prompt_id)
    return prompt_ids


def extract_domain(prompt_text: str) -> str:
    match = re.search(r"working on (.*?), with emphasis on ", prompt_text)
    return match.group(1).strip() if match else ""


def row_static_text(row: Mapping[str, Any]) -> str:
    parts = [
        str(row.get("prompt_text", "")),
        str(row.get("assistant_prefix_before_surface", "")),
        str(row.get("target_response_text", "")),
        str(row.get("target_surface", "")),
    ]
    for key in ("bucket_0_surfaces", "bucket_1_surfaces"):
        value = row.get(key, [])
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
    return "\n".join(parts)


def add_error(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def main() -> int:
    args = parse_args()
    input_dir = resolve(args.input_dir)
    output_dir = resolve(args.output_dir)
    failure_prompts = resolve(args.failure_prompts)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {output_dir}")
    rows_path = input_dir / "row_allocation_rows.jsonl"
    manifest_path = input_dir / "row_allocation_manifest.json"
    rows = read_jsonl(rows_path)
    manifest = read_json(manifest_path)
    collision_prompt_ids = read_failure_prompt_ids(failure_prompts)

    errors: list[str] = []
    expected_rows = int(args.expected_shards) * int(args.prompts_per_shard) * 16
    shard_counts = Counter(int(row.get("assigned_shard_index", -1)) for row in rows)
    prompt_counts = Counter(str(row.get("prompt_id", "")) for row in rows)
    prompt_shards: dict[str, set[int]] = defaultdict(set)
    row_keys = Counter(str(row.get("row_key", "")) for row in rows)
    prompt_prefix = Counter(
        f"{row.get('prompt_id')}::{row.get('prefix_family_id')}::{row.get('assistant_prefix_before_surface')}"
        for row in rows
    )
    selected_domains = Counter()
    technical_static_rows = 0
    ambiguous_static_rows = 0

    for row in rows:
        prompt_id = str(row.get("prompt_id", ""))
        prompt_shards[prompt_id].add(int(row.get("assigned_shard_index", -1)))
        selected_domains[extract_domain(str(row.get("prompt_text", "")))] += 1
        classification = classify_text(row_static_text(row))
        technical_static_rows += classification.technical_forbidden_public_surface_count
        ambiguous_static_rows += classification.ambiguous_forbidden_surface_count

    denied_domains = set(str(item) for item in manifest.get("denied_domains", []))
    source_collision_reused = sorted(set(prompt_counts).intersection(collision_prompt_ids))
    denied_domain_rows = [
        str(row.get("prompt_id", ""))
        for row in rows
        if extract_domain(str(row.get("prompt_text", ""))) in denied_domains
    ]

    add_error(errors, manifest.get("status") == "PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V2_ROW_BANK_BUILT_ARTIFACT_ONLY_NO_SUBMIT", "manifest status is not v2 row-bank pass")
    add_error(errors, len(rows) == expected_rows, f"row count {len(rows)} != expected {expected_rows}")
    add_error(errors, set(shard_counts) == set(range(int(args.expected_shards))), "assigned shard indexes must be complete 0..expected-1")
    add_error(errors, all(count == int(args.prompts_per_shard) * 16 for count in shard_counts.values()), "each shard must have prompts_per_shard*16 rows")
    add_error(errors, len(prompt_counts) == int(args.expected_shards) * int(args.prompts_per_shard), "selected prompt count mismatch")
    add_error(errors, all(count == 16 for count in prompt_counts.values()), "each prompt must have exactly 16 rows")
    add_error(errors, all(len(shards) == 1 for shards in prompt_shards.values()), "each prompt must stay inside one shard")
    add_error(errors, not any(count > 1 for count in row_keys.values()), "row_key values must be unique")
    add_error(errors, not any(count > 1 for count in prompt_prefix.values()), "prompt/prefix pairs must be unique")
    add_error(errors, not source_collision_reused, f"875168 collision prompt ids reused: {source_collision_reused[:10]}")
    add_error(errors, not denied_domain_rows, f"denied prompt domains reused in {len(denied_domain_rows)} rows")
    add_error(errors, technical_static_rows == 0, f"static technical forbidden hits: {technical_static_rows}")
    add_error(errors, ambiguous_static_rows == 0, f"static ambiguous forbidden hits: {ambiguous_static_rows}")
    add_error(errors, all(row.get("generation_conditions") == ["raw"] for row in rows), "all rows must be raw-only")
    add_error(errors, all(row.get("generation_started") is False for row in rows), "generation_started must be false")
    add_error(errors, all(row.get("model_scoring_started") is False for row in rows), "model_scoring_started must be false")
    add_error(errors, all(row.get("training_started") is False for row in rows), "training_started must be false")
    add_error(errors, all(row.get("slurm_submitted") is False for row in rows), "slurm_submitted must be false")
    add_error(errors, all(row.get("paper_claim_allowed") is False for row in rows), "paper_claim_allowed must be false")

    status = (
        "PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V2_ROW_BANK_VALIDATION_NO_SUBMIT"
        if not errors
        else "FAIL_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V2_ROW_BANK_VALIDATION_NO_SUBMIT"
    )
    summary = {
        "schema_name": "natural_evidence_v2_r4_after_870987_same_family_raw_null_v2_row_bank_validation_v1",
        "status": status,
        "errors": errors,
        "input_dir": str(input_dir.relative_to(ROOT)) if input_dir.is_relative_to(ROOT) else str(input_dir),
        "row_allocation_rows": str(rows_path.relative_to(ROOT)) if rows_path.is_relative_to(ROOT) else str(rows_path),
        "row_allocation_rows_sha256": sha256_file(rows_path),
        "row_allocation_manifest": str(manifest_path.relative_to(ROOT)) if manifest_path.is_relative_to(ROOT) else str(manifest_path),
        "row_allocation_manifest_sha256": sha256_file(manifest_path),
        "failure_prompts": str(failure_prompts.relative_to(ROOT)) if failure_prompts.is_relative_to(ROOT) else str(failure_prompts),
        "rows": len(rows),
        "expected_rows": expected_rows,
        "selected_prompts": len(prompt_counts),
        "expected_prompts": int(args.expected_shards) * int(args.prompts_per_shard),
        "expected_shards": int(args.expected_shards),
        "prompts_per_shard": int(args.prompts_per_shard),
        "rows_per_shard": int(args.prompts_per_shard) * 16,
        "source_collision_prompt_reuse_count": len(source_collision_reused),
        "denied_domain_row_count": len(denied_domain_rows),
        "static_technical_forbidden_hits": technical_static_rows,
        "static_ambiguous_forbidden_hits": ambiguous_static_rows,
        "selected_domain_row_counts": dict(sorted(selected_domains.items())),
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "slurm_submitted": False,
        "paper_claim_allowed": False,
        "next_allowed_action": "If this validation passes, prepare tokenizer preflight / single H200 route review for same-family raw-null v2; no Slurm submission yet.",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_new(output_dir / "validation_summary.json", summary)
    write_text_new(
        output_dir / "validation_report.md",
        "# R4 Same-Family Raw-Null V2 Row Bank Validation\n\n"
        f"Status: `{status}`\n\n"
        f"Rows: {len(rows)} / {expected_rows}\n\n"
        f"Selected prompts: {len(prompt_counts)}\n\n"
        f"Errors: {len(errors)}\n\n"
        "This validation is artifact-only and does not tokenize, score, generate, train, enable allowlist, or submit Slurm.\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
