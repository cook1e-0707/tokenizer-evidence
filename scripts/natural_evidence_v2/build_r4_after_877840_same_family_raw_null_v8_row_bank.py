from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.build_r4_after_870987_same_family_raw_null_v3_row_bank import (  # noqa: E402
    DEFAULT_SOURCE_ROWS,
    candidate_is_clean,
    extract_domain,
    group_prompt_rows,
    read_jsonl,
    resolve,
    rewrite_selected_rows,
    validate_rows,
    write_csv,
)
from scripts.natural_evidence_v2.r4_cover_natural_common import (  # noqa: E402
    sha256_file,
    write_json_new,
    write_jsonl_new,
    write_text_new,
)


DEFAULT_FEASIBILITY = (
    ROOT
    / "results/natural_evidence_v2/status/"
    / "r4_after_877840_full_same_family_raw_null_feasibility_20260526/"
    / "feasibility_summary.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results/natural_evidence_v2/status/"
    / "r4_after_877840_same_family_raw_null_v8_row_bank_plan_20260526"
)
DEFAULT_VALIDATION_DIR = (
    ROOT
    / "results/natural_evidence_v2/status/"
    / "r4_after_877840_same_family_raw_null_v8_row_bank_validation_20260526"
)


VENUES = (
    "community art class",
    "after school reading circle",
    "neighborhood newsletter team",
    "parent teacher welcome group",
    "bike skills workshop",
    "language practice meetup",
    "youth robotics club",
    "music lesson coordinator group",
    "apartment welcome committee",
    "food pantry sign-in desk",
    "book discussion group",
    "local walking club",
    "shared kitchen planning team",
    "school science night crew",
    "museum volunteer orientation",
    "health clinic front desk",
    "small office onboarding group",
    "community mediation circle",
    "student mentoring program",
    "local newsletter editing team",
    "craft lesson signup group",
    "town hall reception team",
    "college study group",
    "neighborhood mural planning team",
    "sports equipment checkout desk",
    "senior phone tree group",
    "child care pickup team",
    "career fair welcome desk",
    "market survey table",
    "tenant welcome packet team",
    "music recital planning group",
    "community grant review team",
)

ACTIVITIES = (
    "schedule coordination",
    "handoff planning",
    "feedback review",
    "welcome message drafting",
    "attendance follow-up",
    "role assignment",
    "question collection",
    "progress check-in",
)

EMPHASES = (
    "keeping the tone calm",
    "making responsibilities easy to understand",
    "reducing confusion for newcomers",
    "helping people respond on time",
    "keeping communication concise",
    "making the next meeting smoother",
    "supporting a mixed-experience group",
    "making decisions easy to revisit",
)

FOCUSES = (
    "clear handoffs",
    "common mistakes",
    "simple follow-up",
    "shared expectations",
    "useful reminders",
    "practical wording",
    "small coordination details",
    "fair division of work",
)

AUDIENCES = (
    "a volunteer group",
    "a new team",
    "a small committee",
    "a coordinator",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an artifact-only v8 full same-family raw-null row bank after "
            "the 877840 capacity-limited pass and full-source feasibility blocker. "
            "This creates a fresh natural prompt/source pool and validates it; it "
            "does not tokenize, generate, enable allowlists, submit Slurm, or unlock claims."
        )
    )
    parser.add_argument("--template-source-rows", type=Path, default=DEFAULT_SOURCE_ROWS)
    parser.add_argument("--feasibility-summary", type=Path, default=DEFAULT_FEASIBILITY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--target-shards", type=int, default=64)
    parser.add_argument("--prompts-per-shard", type=int, default=64)
    parser.add_argument("--min-domains", type=int, default=32)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_text_for(index: int) -> tuple[str, str]:
    venue = VENUES[index % len(VENUES)]
    activity = ACTIVITIES[(index // len(VENUES)) % len(ACTIVITIES)]
    emphasis = EMPHASES[(index // (len(VENUES) * len(ACTIVITIES))) % len(EMPHASES)]
    focus = FOCUSES[(index // (len(VENUES) * len(ACTIVITIES) * len(EMPHASES))) % len(FOCUSES)]
    audience = AUDIENCES[
        (index // (len(VENUES) * len(ACTIVITIES) * len(EMPHASES) * len(FOCUSES))) % len(AUDIENCES)
    ]
    domain = f"{venue} {activity}"
    prompt = (
        f"Give practical advice for {audience} working on {domain}, with emphasis on {emphasis}. "
        f"Focus on {focus}. Write a useful, ordinary answer in short paragraphs or natural bullets. "
        "Do not use numbered steps, fixed line labels, special terminology, or headings."
    )
    return domain, prompt


def patch_rows_to_v8(prompt_groups: list[list[dict[str, Any]]], *, prompts_per_shard: int) -> list[dict[str, Any]]:
    rewritten = rewrite_selected_rows(prompt_groups, prompts_per_shard=int(prompts_per_shard))
    patched: list[dict[str, Any]] = []
    for row in rewritten:
        new_row = dict(row)
        row_key = str(new_row.get("row_key", ""))
        if row_key.startswith("sfrawv3|"):
            row_key = row_key.replace("sfrawv3|", "sfrawv8|", 1)
        else:
            row_key = f"sfrawv8|{row_key}"
        new_row.update(
            {
                "schema_name": "natural_evidence_v2_r4_after_877840_same_family_raw_null_v8_row_bank_row_v1",
                "artifact_role": "r4_after_877840_same_family_raw_null_v8_row_bank_not_tokenized_not_scored",
                "same_family_raw_null_v3": False,
                "same_family_raw_null_v8": True,
                "source_capacity_limited_pass_job_id": "877840",
                "source_capacity_limited_pass_status": "PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_GENERATION_GATE",
                "source_feasibility_blocker": (
                    "BLOCK_R4_AFTER_877840_FULL_SAME_FAMILY_RAW_NULL_CURRENT_SOURCE_INSUFFICIENT_NO_SUBMIT"
                ),
                "allocation_policy": "same_family_raw_null_v8_fresh_prompt_source_expansion_after_877840",
                "row_key": row_key,
                "generation_conditions": ["raw"],
                "generation_started": False,
                "model_scoring_started": False,
                "training_started": False,
                "slurm_submitted": False,
                "paper_claim_allowed": False,
                "same_family_raw_null_pass_claim_allowed": False,
            }
        )
        patched.append(new_row)
    return patched


def build_prompt_groups(template_groups: list[list[dict[str, Any]]], target_prompts: int) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    prompt_groups: list[list[dict[str, Any]]] = []
    prompt_manifest_rows: list[dict[str, Any]] = []
    for prompt_index in range(target_prompts):
        template = template_groups[prompt_index % len(template_groups)]
        domain, prompt_text = prompt_text_for(prompt_index)
        prompt_hash = sha256_text(prompt_text)
        prompt_id = f"r4_sfraw_v8_{prompt_hash[:24]}"
        shard_index = prompt_index // 64
        slot_index = prompt_index % 64
        rows: list[dict[str, Any]] = []
        for template_row in sorted(
            template,
            key=lambda item: (int(item.get("coordinate_id", -1)), str(item.get("prefix_family_id", ""))),
        ):
            row = dict(template_row)
            coordinate = int(row["coordinate_id"])
            prefix_family_id = str(row["prefix_family_id"])
            surface_id = str(row["target_surface_id"])
            assistant_prefix = str(row.get("assistant_prefix_before_surface", ""))
            row.update(
                {
                    "prompt_id": prompt_id,
                    "prompt_index": prompt_index,
                    "prompt_text": prompt_text,
                    "prompt_text_sha256": prompt_hash,
                    "source_prompt_id": str(template_row.get("prompt_id", "")),
                    "source_prompt_domain": domain,
                    "source_template_row_key": str(template_row.get("row_key", "")),
                    "assigned_shard_index": shard_index,
                    "prompt_slot_index": slot_index,
                    "replicate_group_id": f"same_family_raw_null_v8_shard_{shard_index:03d}",
                    "duplicate_pair_key": f"{prompt_id}::{prefix_family_id}",
                    "content_duplicate_pair_key": f"{prompt_hash}::{prefix_family_id}::{assistant_prefix}",
                    "row_key": (
                        f"sfrawv3|shard{shard_index:03d}|slot{slot_index:02d}|"
                        f"{prompt_id}|c{coordinate:02d}|{prefix_family_id}|{surface_id}"
                    ),
                }
            )
            rows.append(row)
        prompt_groups.append(rows)
        prompt_manifest_rows.append(
            {
                "prompt_index": prompt_index,
                "prompt_id": prompt_id,
                "prompt_text_sha256": prompt_hash,
                "source_prompt_domain": domain,
                "assigned_shard_index": shard_index,
                "prompt_slot_index": slot_index,
                "prompt_text": prompt_text,
            }
        )
    return prompt_groups, prompt_manifest_rows


def write_report(path: Path, summary: dict[str, Any]) -> None:
    text = f"""# R4 After-877840 Same-Family Raw-Null V8 Row Bank

Status: `{summary['status']}`

This artifact-only package responds to the full-source feasibility blocker after
the capacity-limited `877840` pass. It builds a fresh rule-generated natural
prompt/source pool for a full 64-shard same-family raw-null package while
preserving the reviewed first-token event surface/coordinate contract. It does
not tokenize, score, generate, enable the allowlist, submit Slurm, or unlock
claims.

```text
template source rows: {summary['template_source_rows']}
template source rows sha256: {summary['template_source_rows_sha256']}
feasibility source: {summary['feasibility_summary']}
target shards: {summary['target_shards']}
prompts per shard: {summary['prompts_per_shard']}
selected prompts: {summary['selected_prompt_count']}
selected rows: {summary['selected_row_count']}
selected domains: {summary['selected_domain_count']}
static technical hits: {summary['validation']['static_technical_forbidden_hits']}
static ambiguous hits: {summary['validation']['static_ambiguous_forbidden_hits']}
```

## Next Allowed Action

If reviewed, prepare tokenizer-only route validation for this v8 row bank. Do
not submit generation until tokenizer preflight and single-entry allowlist
preflight pass.
"""
    write_text_new(path, text)


def main() -> int:
    args = parse_args()
    template_source_rows = resolve(args.template_source_rows)
    feasibility_summary = resolve(args.feasibility_summary)
    output_dir = resolve(args.output_dir)
    validation_dir = resolve(args.validation_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {output_dir}")
    if validation_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing validation dir: {validation_dir}")

    feasibility = read_json(feasibility_summary)
    if feasibility.get("status") != "BLOCK_R4_AFTER_877840_FULL_SAME_FAMILY_RAW_NULL_CURRENT_SOURCE_INSUFFICIENT_NO_SUBMIT":
        raise ValueError("v8 row-bank expansion requires the reviewed current-source insufficiency blocker")

    grouped = group_prompt_rows(read_jsonl(template_source_rows))
    clean_templates: list[list[dict[str, Any]]] = []
    for rows in grouped.values():
        clean, _reason = candidate_is_clean(rows)
        if clean and len(rows) == 16:
            clean_templates.append(rows)
    if not clean_templates:
        raise RuntimeError("no clean template prompt groups available")

    target_prompts = int(args.target_shards) * int(args.prompts_per_shard)
    prompt_groups, prompt_manifest_rows = build_prompt_groups(clean_templates, target_prompts)
    selected_rows = patch_rows_to_v8(prompt_groups, prompts_per_shard=int(args.prompts_per_shard))
    validation = validate_rows(
        selected_rows,
        expected_shards=int(args.target_shards),
        prompts_per_shard=int(args.prompts_per_shard),
        denied_prompt_ids=set(),
        denied_domains=set(),
    )
    validation["schema_name"] = "natural_evidence_v2_r4_after_877840_same_family_raw_null_v8_row_bank_validation_v1"
    validation["status"] = (
        "PASS_R4_AFTER_877840_SAME_FAMILY_RAW_NULL_V8_ROW_BANK_VALIDATION_NO_SUBMIT"
        if not validation["errors"]
        else "FAIL_R4_AFTER_877840_SAME_FAMILY_RAW_NULL_V8_ROW_BANK_VALIDATION_NO_SUBMIT"
    )
    validation["next_allowed_action"] = (
        "If reviewed, prepare tokenizer-only route validation for same-family raw-null v8; no generation submission yet."
    )
    selected_domain_count = len({row["source_prompt_domain"] for row in prompt_manifest_rows})
    if selected_domain_count < int(args.min_domains):
        validation["errors"].append(
            f"selected domain count {selected_domain_count} is below min_domains {int(args.min_domains)}"
        )
        validation["status"] = "FAIL_R4_AFTER_877840_SAME_FAMILY_RAW_NULL_V8_ROW_BANK_VALIDATION_NO_SUBMIT"

    manifest = {
        "schema_name": "natural_evidence_v2_r4_after_877840_same_family_raw_null_v8_row_bank_manifest_v1",
        "status": (
            "PASS_R4_AFTER_877840_SAME_FAMILY_RAW_NULL_V8_ROW_BANK_BUILT_ARTIFACT_ONLY_NO_SUBMIT"
            if validation["status"].startswith("PASS_")
            else "FAIL_R4_AFTER_877840_SAME_FAMILY_RAW_NULL_V8_ROW_BANK_BUILD_NO_SUBMIT"
        ),
        "template_source_rows": str(template_source_rows.relative_to(ROOT))
        if template_source_rows.is_relative_to(ROOT)
        else str(template_source_rows),
        "template_source_rows_sha256": sha256_file(template_source_rows),
        "feasibility_summary": str(feasibility_summary.relative_to(ROOT))
        if feasibility_summary.is_relative_to(ROOT)
        else str(feasibility_summary),
        "feasibility_summary_sha256": sha256_file(feasibility_summary),
        "source_capacity_limited_pass_job_id": "877840",
        "source_capacity_limited_pass_status": "PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_GENERATION_GATE",
        "target_shards": int(args.target_shards),
        "prompts_per_shard": int(args.prompts_per_shard),
        "selected_prompt_count": target_prompts,
        "selected_row_count": len(selected_rows),
        "selected_domain_count": selected_domain_count,
        "clean_template_prompt_count": len(clean_templates),
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "slurm_submitted": False,
        "tokenizer_preflight_started": False,
        "paper_claim_allowed": False,
        "same_family_raw_null_full_package_claim_allowed": False,
        "next_allowed_action": validation["next_allowed_action"],
        "validation": validation,
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl_new(output_dir / "row_allocation_rows.jsonl", selected_rows)
    write_jsonl_new(output_dir / "fresh_prompt_source.jsonl", prompt_manifest_rows)
    manifest["row_allocation_rows_sha256"] = sha256_file(output_dir / "row_allocation_rows.jsonl")
    manifest["fresh_prompt_source_sha256"] = sha256_file(output_dir / "fresh_prompt_source.jsonl")
    write_json_new(output_dir / "row_allocation_manifest.json", manifest)
    write_report(output_dir / "row_bank_report.md", manifest)
    domain_counts = Counter(row["source_prompt_domain"] for row in prompt_manifest_rows)
    write_csv(
        output_dir / "selected_prompt_domains.csv",
        [
            {"source_prompt_domain": domain, "prompt_count": count}
            for domain, count in sorted(domain_counts.items())
        ],
        ["source_prompt_domain", "prompt_count"],
    )

    validation_dir.mkdir(parents=True, exist_ok=False)
    validation.update(
        {
            "input_dir": str(output_dir.relative_to(ROOT)) if output_dir.is_relative_to(ROOT) else str(output_dir),
            "row_allocation_rows": str((output_dir / "row_allocation_rows.jsonl").relative_to(ROOT)),
            "row_allocation_rows_sha256": manifest["row_allocation_rows_sha256"],
            "fresh_prompt_source": str((output_dir / "fresh_prompt_source.jsonl").relative_to(ROOT)),
            "fresh_prompt_source_sha256": manifest["fresh_prompt_source_sha256"],
            "row_allocation_manifest": str((output_dir / "row_allocation_manifest.json").relative_to(ROOT)),
            "row_allocation_manifest_sha256": sha256_file(output_dir / "row_allocation_manifest.json"),
            "source_capacity_limited_pass_job_id": "877840",
            "selected_domain_count": selected_domain_count,
            "min_domains": int(args.min_domains),
        }
    )
    write_json_new(validation_dir / "validation_summary.json", validation)
    write_text_new(
        validation_dir / "validation_report.md",
        "# R4 After-877840 Same-Family Raw-Null V8 Row Bank Validation\n\n"
        f"Status: `{validation['status']}`\n\n"
        f"Rows: {validation['rows']} / {validation['expected_rows']}\n\n"
        f"Selected prompts: {validation['selected_prompts']}\n\n"
        f"Selected domains: {selected_domain_count}\n\n"
        f"Errors: {len(validation['errors'])}\n\n"
        "This validation is artifact-only and does not tokenize, score, generate, train, enable allowlist, or submit Slurm.\n",
    )
    print(json.dumps({"manifest": manifest, "validation": validation}, indent=2, sort_keys=True))
    return 0 if validation["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
