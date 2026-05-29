from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.build_r4_after_870987_same_family_raw_null_v3_row_bank import (
    DEFAULT_875168_FAILURE_PROMPTS,
    DEFAULT_875471_FAILURE_PROMPTS,
    DEFAULT_DENY_DOMAINS,
    DEFAULT_SOURCE_ROWS,
    candidate_is_clean,
    extract_domain,
    group_prompt_rows,
    read_failure_csv,
    read_jsonl,
    resolve,
    rewrite_selected_rows,
    validate_rows,
    write_csv,
)
from scripts.natural_evidence_v2.r4_cover_natural_common import (
    sha256_file,
    write_json_new,
    write_jsonl_new,
    write_text_new,
)


DEFAULT_875777_FAILURE_PROMPTS = (
    ROOT
    / "results/natural_evidence_v2/status/"
    / "r4_after_870987_same_family_raw_null_generation_875777_failure_review_20260524/"
    / "forbidden_by_prompt.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results/natural_evidence_v2/status/"
    / "r4_after_870987_same_family_raw_null_v4_row_bank_plan_20260524"
)
DEFAULT_VALIDATION_DIR = (
    ROOT
    / "results/natural_evidence_v2/status/"
    / "r4_after_870987_same_family_raw_null_v4_row_bank_validation_20260524"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and validate artifact-only R4 same-family raw-null v4 row bank after "
            "875777 residual forbidden failure."
        )
    )
    parser.add_argument("--source-rows", type=Path, default=DEFAULT_SOURCE_ROWS)
    parser.add_argument("--failure-prompts-875168", type=Path, default=DEFAULT_875168_FAILURE_PROMPTS)
    parser.add_argument("--failure-prompts-875471", type=Path, default=DEFAULT_875471_FAILURE_PROMPTS)
    parser.add_argument("--failure-prompts-875777", type=Path, default=DEFAULT_875777_FAILURE_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--target-shards", type=int, default=64)
    parser.add_argument("--prompts-per-shard", type=int, default=64)
    parser.add_argument("--deny-domain", action="append", default=[])
    return parser.parse_args()


def read_failure_875777(path: Path) -> tuple[set[str], set[str]]:
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


def patch_rows_to_v4(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    patched: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        row_key = str(new_row.get("row_key", "")).replace("sfrawv3|", "sfrawv4|", 1)
        new_row.update(
            {
                "schema_name": "natural_evidence_v2_r4_after_870987_same_family_raw_null_v4_row_bank_row_v1",
                "artifact_role": "r4_after_870987_same_family_raw_null_v4_row_bank_not_tokenized_not_scored",
                "same_family_raw_null_v3": False,
                "same_family_raw_null_v4": True,
                "source_failure_job_ids": ["875168", "875471", "875777"],
                "allocation_policy": "same_family_raw_null_v4_residual_forbidden_literal_repair",
                "row_key": row_key,
            }
        )
        patched.append(new_row)
    return patched


def write_report(path: Path, summary: dict[str, Any]) -> None:
    text = f"""# R4 Same-Family Raw-Null V4 Row Bank Plan

Status: `{summary['status']}`

This artifact-only package repairs the residual forbidden-surface failure from
`875777`. It does not reinterpret `875777`, relax the hard forbidden policy,
tokenize, score, generate, enable the allowlist, submit Slurm, or unlock claims.

```text
source rows: {summary['source_rows']}
source rows sha256: {summary['source_rows_sha256']}
875168 failure prompt artifact: {summary['failure_prompts_875168']}
875471 failure prompt artifact: {summary['failure_prompts_875471']}
875777 failure prompt artifact: {summary['failure_prompts_875777']}
target shards: {summary['target_shards']}
prompts per shard: {summary['prompts_per_shard']}
selected prompts: {summary['selected_prompt_count']}
selected rows: {summary['selected_row_count']}
denied prompt ids: {summary['denied_prompt_id_count']}
denied domains: {', '.join(summary['denied_domains'])}
```

The next action is tokenizer preflight route planning only if this row bank and
its validation are reviewed. No generation rerun is allowed from this artifact
alone.
"""
    write_text_new(path, text)


def main() -> int:
    args = parse_args()
    source_rows = resolve(args.source_rows)
    failure_875168 = resolve(args.failure_prompts_875168)
    failure_875471 = resolve(args.failure_prompts_875471)
    failure_875777 = resolve(args.failure_prompts_875777)
    output_dir = resolve(args.output_dir)
    validation_dir = resolve(args.validation_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {output_dir}")
    if validation_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing validation dir: {validation_dir}")

    prompt_ids_875168, domains_875168 = read_failure_csv(failure_875168)
    prompt_ids_875471, domains_875471 = read_failure_csv(failure_875471)
    prompt_ids_875777, domains_875777 = read_failure_875777(failure_875777)
    denied_prompt_ids = set(prompt_ids_875168).union(prompt_ids_875471).union(prompt_ids_875777)
    denied_domains = sorted(
        set(DEFAULT_DENY_DOMAINS)
        .union(domains_875168)
        .union(domains_875471)
        .union(domains_875777)
        .union(args.deny_domain)
    )

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

    selected_rows = patch_rows_to_v4(
        rewrite_selected_rows(selected_prompt_groups, prompts_per_shard=int(args.prompts_per_shard))
    )
    validation = validate_rows(
        selected_rows,
        expected_shards=int(args.target_shards),
        prompts_per_shard=int(args.prompts_per_shard),
        denied_prompt_ids=denied_prompt_ids,
        denied_domains=set(denied_domains),
    )
    validation["schema_name"] = "natural_evidence_v2_r4_after_870987_same_family_raw_null_v4_row_bank_validation_v1"
    validation["status"] = validation["status"].replace("_V3_", "_V4_")
    validation[
        "next_allowed_action"
    ] = "If reviewed, prepare actual Qwen tokenizer preflight route for same-family raw-null v4; no generation submission yet."

    manifest = {
        "schema_name": "natural_evidence_v2_r4_after_870987_same_family_raw_null_v4_row_bank_manifest_v1",
        "status": "PASS_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_V4_ROW_BANK_BUILT_ARTIFACT_ONLY_NO_SUBMIT",
        "source_rows": str(source_rows.relative_to(ROOT)) if source_rows.is_relative_to(ROOT) else str(source_rows),
        "source_rows_sha256": sha256_file(source_rows),
        "failure_prompts_875168": str(failure_875168.relative_to(ROOT))
        if failure_875168.is_relative_to(ROOT)
        else str(failure_875168),
        "failure_prompts_875471": str(failure_875471.relative_to(ROOT))
        if failure_875471.is_relative_to(ROOT)
        else str(failure_875471),
        "failure_prompts_875777": str(failure_875777.relative_to(ROOT))
        if failure_875777.is_relative_to(ROOT)
        else str(failure_875777),
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
        "next_allowed_action": "review this v4 row bank and validation; then prepare tokenizer preflight route if accepted",
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl_new(output_dir / "row_allocation_rows.jsonl", selected_rows)
    manifest["row_allocation_rows_sha256"] = sha256_file(output_dir / "row_allocation_rows.jsonl")
    write_json_new(output_dir / "row_allocation_manifest.json", manifest)
    write_report(output_dir / "row_bank_report.md", manifest)
    write_csv(
        output_dir / "selected_prompt_domains.csv",
        [
            {"source_prompt_domain": domain, "prompt_count": count // 16}
            for domain, count in sorted(validation["selected_domain_row_counts"].items())
        ],
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
        "# R4 Same-Family Raw-Null V4 Row Bank Validation\n\n"
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
