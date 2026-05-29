from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.classify_r4_forbidden_surface_context_v2 import classify_text  # noqa: E402
from scripts.natural_evidence_v2.r4_cover_natural_common import (  # noqa: E402
    sha256_file,
    write_csv_new,
    write_json_new,
    write_jsonl_new,
    write_text_new,
)


DEFAULT_SOURCE_ROWS = (
    ROOT
    / "results/natural_evidence_v2/status/"
    / "r4_after_870987_prefar_organic_null_row_bank_v2_plan_20260521/row_allocation_rows.jsonl"
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
    / "r4_after_870987_same_family_raw_null_v2_row_bank_plan_20260523"
)
DEFAULT_DENY_DOMAINS = (
    "local history archive sorting",
    "library display refresh",
    "hardware store aisle cleanup",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build artifact-only R4 same-family raw-null v2 row bank after 875168 forbidden-literal collision."
    )
    parser.add_argument("--source-rows", type=Path, default=DEFAULT_SOURCE_ROWS)
    parser.add_argument("--failure-prompts", type=Path, default=DEFAULT_FAILURE_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-shards", type=int, default=64)
    parser.add_argument("--prompts-per-shard", type=int, default=64)
    parser.add_argument("--deny-domain", action="append", default=[])
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


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


def read_failure_csv(path: Path) -> tuple[set[str], set[str], Counter[str]]:
    prompt_ids: set[str] = set()
    domains: set[str] = set()
    term_counts: Counter[str] = Counter()
    if not path.exists():
        return prompt_ids, domains, term_counts
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            prompt_id = str(row.get("prompt_id", "")).strip()
            if prompt_id:
                prompt_ids.add(prompt_id)
            domain = extract_domain(str(row.get("prompt_text", "")))
            if domain:
                domains.add(domain)
            for item in str(row.get("terms", "")).split(";"):
                if not item:
                    continue
                term = item.split(":", 1)[0].strip()
                if term:
                    term_counts[term] += 1
    return prompt_ids, domains, term_counts


def extract_domain(prompt_text: str) -> str:
    match = re.search(r"working on (.*?), with emphasis on ", prompt_text)
    return match.group(1).strip() if match else ""


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    text = f"""# R4 Same-Family Raw-Null V2 Row Bank Plan

Status: `{summary['status']}`

This artifact-only package repairs the `875168` prompt-domain / hard-forbidden
literal collision. It does not reinterpret `875168`, relax the hard forbidden
policy, tokenize, score, generate, enable the allowlist, submit Slurm, or unlock
claims.

```text
source rows: {summary['source_rows']}
source rows sha256: {summary['source_rows_sha256']}
failure prompt artifact: {summary['failure_prompts']}
target shards: {summary['target_shards']}
prompts per shard: {summary['prompts_per_shard']}
selected prompts: {summary['selected_prompt_count']}
selected rows: {summary['selected_row_count']}
denied prompt ids from 875168: {summary['denied_prompt_id_count']}
denied domains: {', '.join(summary['denied_domains'])}
```

The next action is static validation of this v2 row bank and lexical preflight.
No Slurm rerun is allowed until that validation passes and a reviewed route
decision is recorded.
"""
    write_text_new(path, text)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    write_jsonl_new(path, rows)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    write_csv_new(path, rows, fieldnames)


def group_prompt_rows(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("prompt_id", ""))].append(row)
    return grouped


def row_static_text(row: Mapping[str, Any]) -> str:
    fields: list[str] = [
        str(row.get("prompt_text", "")),
        str(row.get("assistant_prefix_before_surface", "")),
        str(row.get("target_response_text", "")),
        str(row.get("target_surface", "")),
    ]
    for key in ("bucket_0_surfaces", "bucket_1_surfaces"):
        value = row.get(key, [])
        if isinstance(value, list):
            fields.extend(str(item) for item in value)
    return "\n".join(fields)


def candidate_is_clean(rows: list[dict[str, Any]]) -> tuple[bool, str]:
    if len(rows) != 16:
        return False, f"prompt row count is {len(rows)}, expected 16"
    for row in rows:
        classification = classify_text(row_static_text(row))
        if classification.technical_forbidden_public_surface_count:
            return False, f"technical static literal: {classification.technical_hits}"
        if classification.ambiguous_forbidden_surface_count:
            return False, f"ambiguous static literal: {classification.ambiguous_hits}"
    return True, ""


def rewrite_selected_rows(prompt_rows: list[list[dict[str, Any]]], *, prompts_per_shard: int) -> list[dict[str, Any]]:
    rewritten: list[dict[str, Any]] = []
    for prompt_index, rows in enumerate(prompt_rows):
        shard_index = prompt_index // prompts_per_shard
        slot_index = prompt_index % prompts_per_shard
        for row in sorted(rows, key=lambda item: (int(item.get("coordinate_id", -1)), str(item.get("prefix_family_id", "")))):
            new_row = dict(row)
            prompt_id = str(new_row["prompt_id"])
            coordinate = int(new_row["coordinate_id"])
            prefix_family_id = str(new_row["prefix_family_id"])
            surface_id = str(new_row["target_surface_id"])
            new_row.update(
                {
                    "schema_name": "natural_evidence_v2_r4_after_870987_same_family_raw_null_v2_row_bank_row_v1",
                    "artifact_role": "r4_after_870987_same_family_raw_null_v2_row_bank_not_tokenized_not_scored",
                    "same_family_raw_null_v2": True,
                    "source_failure_job_id": "875168",
                    "source_failure_status": "FAIL_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_GENERATION_GATE",
                    "source_failure_root_cause": "prompt_domain_hard_forbidden_literal_collision",
                    "source_prompt_id": prompt_id,
                    "source_assigned_shard_index": int(row.get("assigned_shard_index", -1)),
                    "source_prompt_domain": extract_domain(str(row.get("prompt_text", ""))),
                    "allocation_policy": "same_family_raw_null_v2_forbidden_literal_collision_repair",
                    "assigned_shard_index": shard_index,
                    "prompt_slot_index": slot_index,
                    "replicate_group_id": f"same_family_raw_null_v2_shard_{shard_index:03d}",
                    "row_key": (
                        f"sfrawv2|shard{shard_index:03d}|slot{slot_index:02d}|"
                        f"{prompt_id}|c{coordinate:02d}|{prefix_family_id}|{surface_id}"
                    ),
                    "generation_conditions": ["raw"],
                    "generation_started": False,
                    "model_scoring_started": False,
                    "training_started": False,
                    "slurm_submitted": False,
                    "paper_claim_allowed": False,
                    "same_family_raw_null_pass_claim_allowed": False,
                }
            )
            rewritten.append(new_row)
    return rewritten


def main() -> int:
    args = parse_args()
    source_rows = resolve(args.source_rows)
    failure_prompts = resolve(args.failure_prompts)
    output_dir = resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {output_dir}")

    collision_prompt_ids, collision_domains, collision_terms = read_failure_csv(failure_prompts)
    denied_domains = sorted(set(DEFAULT_DENY_DOMAINS).union(collision_domains).union(args.deny_domain))
    grouped = group_prompt_rows(read_jsonl(source_rows))
    accepted_by_domain: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    rejection_counts: Counter[str] = Counter()

    for prompt_id, rows in grouped.items():
        prompt_text = str(rows[0].get("prompt_text", "")) if rows else ""
        domain = extract_domain(prompt_text)
        if prompt_id in collision_prompt_ids:
            rejection_counts["875168_collision_prompt_id"] += 1
            continue
        if domain in denied_domains:
            rejection_counts[f"denied_domain::{domain}"] += 1
            continue
        clean, reason = candidate_is_clean(rows)
        if not clean:
            rejection_counts[f"static_lexical::{reason}"] += 1
            continue
        accepted_by_domain[domain].append(rows)

    for rows_for_domain in accepted_by_domain.values():
        rows_for_domain.sort(key=lambda item: str(item[0].get("prompt_id", "")))

    selected_prompt_groups: list[list[dict[str, Any]]] = []
    target_prompt_count = int(args.target_shards) * int(args.prompts_per_shard)
    domains = sorted(accepted_by_domain)
    domain_offsets = {domain: 0 for domain in domains}
    while len(selected_prompt_groups) < target_prompt_count:
        progressed = False
        for domain in domains:
            offset = domain_offsets[domain]
            candidates = accepted_by_domain[domain]
            if offset >= len(candidates):
                continue
            selected_prompt_groups.append(candidates[offset])
            domain_offsets[domain] += 1
            progressed = True
            if len(selected_prompt_groups) >= target_prompt_count:
                break
        if not progressed:
            break

    status = (
        "PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V2_ROW_BANK_BUILT_ARTIFACT_ONLY_NO_SUBMIT"
        if len(selected_prompt_groups) == target_prompt_count
        else "FAIL_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V2_ROW_BANK_INSUFFICIENT_PROMPTS_NO_SUBMIT"
    )
    selected_rows = rewrite_selected_rows(selected_prompt_groups, prompts_per_shard=int(args.prompts_per_shard))
    output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = output_dir / "row_allocation_rows.jsonl"
    write_jsonl(rows_path, selected_rows)

    selected_inventory = [
        {
            "selected_prompt_order": index,
            "assigned_shard_index": index // int(args.prompts_per_shard),
            "prompt_slot_index": index % int(args.prompts_per_shard),
            "prompt_id": rows[0].get("prompt_id", ""),
            "prompt_domain": extract_domain(str(rows[0].get("prompt_text", ""))),
            "prompt_text_sha256": rows[0].get("prompt_text_sha256", ""),
            "prompt_text": rows[0].get("prompt_text", ""),
        }
        for index, rows in enumerate(selected_prompt_groups)
    ]
    write_csv(
        output_dir / "selected_prompt_inventory.csv",
        selected_inventory,
        [
            "selected_prompt_order",
            "assigned_shard_index",
            "prompt_slot_index",
            "prompt_id",
            "prompt_domain",
            "prompt_text_sha256",
            "prompt_text",
        ],
    )
    domain_inventory = [
        {
            "prompt_domain": domain,
            "available_clean_prompts": len(accepted_by_domain[domain]),
            "selected_prompts": domain_offsets.get(domain, 0),
        }
        for domain in domains
    ]
    write_csv(output_dir / "domain_selection_inventory.csv", domain_inventory, ["prompt_domain", "available_clean_prompts", "selected_prompts"])
    rejection_inventory = [{"reason": reason, "prompt_count": count} for reason, count in sorted(rejection_counts.items())]
    write_csv(output_dir / "rejection_inventory.csv", rejection_inventory, ["reason", "prompt_count"])

    summary = {
        "schema_name": "natural_evidence_v2_r4_after_870987_same_family_raw_null_v2_row_bank_manifest_v1",
        "status": status,
        "source_rows": str(source_rows.relative_to(ROOT)) if source_rows.is_relative_to(ROOT) else str(source_rows),
        "source_rows_sha256": sha256_file(source_rows),
        "failure_prompts": str(failure_prompts.relative_to(ROOT)) if failure_prompts.is_relative_to(ROOT) else str(failure_prompts),
        "failure_prompts_sha256": sha256_file(failure_prompts) if failure_prompts.exists() else "",
        "source_failure_job_id": "875168",
        "source_failure_root_cause": "prompt_domain_hard_forbidden_literal_collision",
        "selected_row_count": len(selected_rows),
        "selected_prompt_count": len(selected_prompt_groups),
        "target_prompt_count": target_prompt_count,
        "target_shards": int(args.target_shards),
        "prompts_per_shard": int(args.prompts_per_shard),
        "rows_per_prompt": 16,
        "rows_per_shard": 16 * int(args.prompts_per_shard),
        "generation_conditions": ["raw"],
        "denied_domains": denied_domains,
        "denied_prompt_id_count": len(collision_prompt_ids),
        "collision_term_counts": dict(sorted(collision_terms.items())),
        "accepted_domain_count": len(domains),
        "domain_selection_inventory": "domain_selection_inventory.csv",
        "rejection_inventory": "rejection_inventory.csv",
        "row_allocation_rows": "row_allocation_rows.jsonl",
        "row_allocation_rows_sha256": sha256_file(rows_path),
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "slurm_submitted": False,
        "allowlist_enabled": False,
        "paper_claim_allowed": False,
        "same_family_raw_null_pass_claim_allowed": False,
        "next_allowed_action": "Validate same-family raw-null v2 row bank and run lexical/tokenizer preflights before any Slurm rerun.",
    }
    write_json_new(output_dir / "row_allocation_manifest.json", summary)
    write_report(output_dir / "row_allocation_report.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
