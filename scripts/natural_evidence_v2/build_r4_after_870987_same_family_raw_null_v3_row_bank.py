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
DEFAULT_875168_FAILURE_PROMPTS = (
    ROOT
    / "results/natural_evidence_v2/status/"
    / "r4_after_870987_same_family_raw_null_generation_875168_failure_review_20260523/"
    / "forbidden_collision_by_prompt.csv"
)
DEFAULT_875471_FAILURE_PROMPTS = (
    ROOT
    / "results/natural_evidence_v2/status/"
    / "r4_after_870987_same_family_raw_null_generation_875471_failure_review_20260523/"
    / "forbidden_by_prompt.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results/natural_evidence_v2/status/"
    / "r4_after_870987_same_family_raw_null_v3_row_bank_plan_20260523"
)
DEFAULT_VALIDATION_DIR = (
    ROOT
    / "results/natural_evidence_v2/status/"
    / "r4_after_870987_same_family_raw_null_v3_row_bank_validation_20260523"
)
DEFAULT_DENY_DOMAINS = (
    "local history archive sorting",
    "library display refresh",
    "hardware store aisle cleanup",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate artifact-only R4 same-family raw-null v3 row bank after 875471 residual forbidden failure."
    )
    parser.add_argument("--source-rows", type=Path, default=DEFAULT_SOURCE_ROWS)
    parser.add_argument("--failure-prompts-875168", type=Path, default=DEFAULT_875168_FAILURE_PROMPTS)
    parser.add_argument("--failure-prompts-875471", type=Path, default=DEFAULT_875471_FAILURE_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
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


def extract_domain(prompt_text: str) -> str:
    match = re.search(r"working on (.*?), with emphasis on ", prompt_text)
    return match.group(1).strip() if match else ""


def read_failure_csv(path: Path) -> tuple[set[str], set[str]]:
    prompt_ids: set[str] = set()
    domains: set[str] = set()
    if not path.exists():
        return prompt_ids, domains
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            prompt_id = str(row.get("prompt_id", "")).strip()
            if prompt_id:
                prompt_ids.add(prompt_id)
            domain = str(row.get("source_prompt_domain", "")).strip() or extract_domain(str(row.get("prompt_text", "")))
            if domain:
                domains.add(domain)
    return prompt_ids, domains


def row_static_text(row: Mapping[str, Any]) -> str:
    fields = [
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


def group_prompt_rows(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("prompt_id", ""))].append(row)
    return grouped


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


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    text = f"""# R4 Same-Family Raw-Null V3 Row Bank Plan

Status: `{summary['status']}`

This artifact-only package repairs the residual forbidden-surface failure from
`875471`. It does not reinterpret `875471`, relax the hard forbidden policy,
tokenize, score, generate, enable the allowlist, submit Slurm, or unlock
claims.

```text
source rows: {summary['source_rows']}
source rows sha256: {summary['source_rows_sha256']}
875168 failure prompt artifact: {summary['failure_prompts_875168']}
875471 failure prompt artifact: {summary['failure_prompts_875471']}
target shards: {summary['target_shards']}
prompts per shard: {summary['prompts_per_shard']}
selected prompts: {summary['selected_prompt_count']}
selected rows: {summary['selected_row_count']}
denied prompt ids: {summary['denied_prompt_id_count']}
denied domains: {', '.join(summary['denied_domains'])}
```

The next action is tokenizer preflight route planning only if this row bank and
its validation are reviewed. No Slurm rerun is allowed from this artifact alone.
"""
    write_text_new(path, text)


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
                    "schema_name": "natural_evidence_v2_r4_after_870987_same_family_raw_null_v3_row_bank_row_v1",
                    "artifact_role": "r4_after_870987_same_family_raw_null_v3_row_bank_not_tokenized_not_scored",
                    "same_family_raw_null_v3": True,
                    "source_failure_job_ids": ["875168", "875471"],
                    "source_failure_status": "FAIL_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_GENERATION_GATE",
                    "source_failure_root_cause": "residual_forbidden_literal_collision",
                    "source_prompt_id": prompt_id,
                    "source_assigned_shard_index": int(row.get("assigned_shard_index", -1)),
                    "source_prompt_domain": extract_domain(str(row.get("prompt_text", ""))),
                    "allocation_policy": "same_family_raw_null_v3_residual_forbidden_literal_repair",
                    "assigned_shard_index": shard_index,
                    "prompt_slot_index": slot_index,
                    "replicate_group_id": f"same_family_raw_null_v3_shard_{shard_index:03d}",
                    "row_key": (
                        f"sfrawv3|shard{shard_index:03d}|slot{slot_index:02d}|"
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


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    write_csv_new(path, rows, fieldnames)


def validate_rows(rows: list[dict[str, Any]], *, expected_shards: int, prompts_per_shard: int, denied_prompt_ids: set[str], denied_domains: set[str]) -> dict[str, Any]:
    errors: list[str] = []
    expected_rows = expected_shards * prompts_per_shard * 16
    shard_counts = Counter(int(row.get("assigned_shard_index", -1)) for row in rows)
    prompt_counts = Counter(str(row.get("prompt_id", "")) for row in rows)
    row_keys = Counter(str(row.get("row_key", "")) for row in rows)
    prompt_shards: dict[str, set[int]] = defaultdict(set)
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

    source_collision_reused = sorted(set(prompt_counts).intersection(denied_prompt_ids))
    denied_domain_rows = [
        str(row.get("prompt_id", ""))
        for row in rows
        if extract_domain(str(row.get("prompt_text", ""))) in denied_domains
    ]

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(len(rows) == expected_rows, f"row count {len(rows)} != expected {expected_rows}")
    require(set(shard_counts) == set(range(expected_shards)), "assigned shard indexes must be complete")
    require(all(count == prompts_per_shard * 16 for count in shard_counts.values()), "each shard must have prompts_per_shard*16 rows")
    require(len(prompt_counts) == expected_shards * prompts_per_shard, "selected prompt count mismatch")
    require(all(count == 16 for count in prompt_counts.values()), "each prompt must have exactly 16 rows")
    require(all(len(shards) == 1 for shards in prompt_shards.values()), "each prompt must stay inside one shard")
    require(not any(count > 1 for count in row_keys.values()), "row_key values must be unique")
    require(not any(count > 1 for count in prompt_prefix.values()), "prompt/prefix pairs must be unique")
    require(not source_collision_reused, f"denied prompt ids reused: {source_collision_reused[:10]}")
    require(not denied_domain_rows, f"denied prompt domains reused in {len(denied_domain_rows)} rows")
    require(technical_static_rows == 0, f"static technical forbidden hits: {technical_static_rows}")
    require(ambiguous_static_rows == 0, f"static ambiguous forbidden hits: {ambiguous_static_rows}")
    require(all(row.get("generation_conditions") == ["raw"] for row in rows), "all rows must be raw-only")
    require(all(row.get("generation_started") is False for row in rows), "generation_started must be false")
    require(all(row.get("model_scoring_started") is False for row in rows), "model_scoring_started must be false")
    require(all(row.get("training_started") is False for row in rows), "training_started must be false")
    require(all(row.get("slurm_submitted") is False for row in rows), "slurm_submitted must be false")
    require(all(row.get("paper_claim_allowed") is False for row in rows), "paper_claim_allowed must be false")
    status = (
        "PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V3_ROW_BANK_VALIDATION_NO_SUBMIT"
        if not errors
        else "FAIL_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V3_ROW_BANK_VALIDATION_NO_SUBMIT"
    )
    return {
        "schema_name": "natural_evidence_v2_r4_after_870987_same_family_raw_null_v3_row_bank_validation_v1",
        "status": status,
        "errors": errors,
        "rows": len(rows),
        "expected_rows": expected_rows,
        "selected_prompts": len(prompt_counts),
        "expected_prompts": expected_shards * prompts_per_shard,
        "expected_shards": expected_shards,
        "prompts_per_shard": prompts_per_shard,
        "rows_per_shard": prompts_per_shard * 16,
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
        "next_allowed_action": "If reviewed, prepare actual Qwen tokenizer preflight route for same-family raw-null v3; no Slurm submission yet.",
    }


def main() -> int:
    args = parse_args()
    source_rows = resolve(args.source_rows)
    failure_875168 = resolve(args.failure_prompts_875168)
    failure_875471 = resolve(args.failure_prompts_875471)
    output_dir = resolve(args.output_dir)
    validation_dir = resolve(args.validation_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {output_dir}")
    if validation_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing validation dir: {validation_dir}")

    prompt_ids_875168, domains_875168 = read_failure_csv(failure_875168)
    prompt_ids_875471, domains_875471 = read_failure_csv(failure_875471)
    denied_prompt_ids = set(prompt_ids_875168).union(prompt_ids_875471)
    denied_domains = sorted(set(DEFAULT_DENY_DOMAINS).union(domains_875168).union(domains_875471).union(args.deny_domain))

    grouped = group_prompt_rows(read_jsonl(source_rows))
    accepted_by_domain: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    rejection_counts: Counter[str] = Counter()
    for prompt_id, rows in grouped.items():
        prompt_text = str(rows[0].get("prompt_text", "")) if rows else ""
        domain = extract_domain(prompt_text)
        if prompt_id in denied_prompt_ids:
            rejection_counts["denied_prompt_id"] += 1
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

    if len(selected_prompt_groups) < target_prompt_count:
        raise RuntimeError(f"only selected {len(selected_prompt_groups)} prompts, need {target_prompt_count}")

    selected_rows = rewrite_selected_rows(selected_prompt_groups, prompts_per_shard=int(args.prompts_per_shard))
    validation = validate_rows(
        selected_rows,
        expected_shards=int(args.target_shards),
        prompts_per_shard=int(args.prompts_per_shard),
        denied_prompt_ids=denied_prompt_ids,
        denied_domains=set(denied_domains),
    )

    manifest = {
        "schema_name": "natural_evidence_v2_r4_after_870987_same_family_raw_null_v3_row_bank_manifest_v1",
        "status": "PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V3_ROW_BANK_BUILT_ARTIFACT_ONLY_NO_SUBMIT",
        "source_rows": str(source_rows.relative_to(ROOT)) if source_rows.is_relative_to(ROOT) else str(source_rows),
        "source_rows_sha256": sha256_file(source_rows),
        "failure_prompts_875168": str(failure_875168.relative_to(ROOT)) if failure_875168.is_relative_to(ROOT) else str(failure_875168),
        "failure_prompts_875471": str(failure_875471.relative_to(ROOT)) if failure_875471.is_relative_to(ROOT) else str(failure_875471),
        "target_shards": int(args.target_shards),
        "prompts_per_shard": int(args.prompts_per_shard),
        "selected_prompt_count": len(selected_prompt_groups),
        "selected_row_count": len(selected_rows),
        "denied_prompt_id_count": len(denied_prompt_ids),
        "denied_domains": denied_domains,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "validation_status": validation["status"],
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "slurm_submitted": False,
        "paper_claim_allowed": False,
        "same_family_raw_null_pass_claim_allowed": False,
        "next_allowed_action": "review this v3 row bank and validation; then prepare tokenizer preflight route if accepted",
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl_new(output_dir / "row_allocation_rows.jsonl", selected_rows)
    manifest["row_allocation_rows_sha256"] = sha256_file(output_dir / "row_allocation_rows.jsonl")
    write_json_new(output_dir / "row_allocation_manifest.json", manifest)
    write_report(output_dir / "row_bank_report.md", manifest)
    write_csv(
        output_dir / "selected_prompt_domains.csv",
        [{"source_prompt_domain": domain, "prompt_count": count // 16} for domain, count in sorted(validation["selected_domain_row_counts"].items())],
        ["source_prompt_domain", "prompt_count"],
    )

    validation_dir.mkdir(parents=True, exist_ok=False)
    validation.update(
        {
            "input_dir": str(output_dir.relative_to(ROOT)) if output_dir.is_relative_to(ROOT) else str(output_dir),
            "row_allocation_rows": str((output_dir / "row_allocation_rows.jsonl").relative_to(ROOT)),
            "row_allocation_rows_sha256": manifest["row_allocation_rows_sha256"],
            "row_allocation_manifest": str((output_dir / "row_allocation_manifest.json").relative_to(ROOT)),
            "row_allocation_manifest_sha256": sha256_file(output_dir / "row_allocation_manifest.json"),
            "denied_domains": denied_domains,
            "denied_prompt_id_count": len(denied_prompt_ids),
        }
    )
    write_json_new(validation_dir / "validation_summary.json", validation)
    write_text_new(
        validation_dir / "validation_report.md",
        "# R4 Same-Family Raw-Null V3 Row Bank Validation\n\n"
        f"Status: `{validation['status']}`\n\n"
        f"Rows: {validation['rows']} / {validation['expected_rows']}\n\n"
        f"Selected prompts: {validation['selected_prompts']}\n\n"
        f"Errors: {len(validation['errors'])}\n\n"
        "This validation is artifact-only and does not tokenize, score, generate, train, enable allowlist, or submit Slurm.\n",
    )
    print(json.dumps({"manifest": manifest, "validation": validation}, indent=2, sort_keys=True))
    return 0 if validation["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
