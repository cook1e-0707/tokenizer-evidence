from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.build_r4_after_870987_same_family_raw_null_v3_row_bank import (  # noqa: E402
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
)
from scripts.natural_evidence_v2.build_r4_after_870987_same_family_raw_null_v4_row_bank import (  # noqa: E402
    DEFAULT_875777_FAILURE_PROMPTS,
    read_failure_875777,
)
from scripts.natural_evidence_v2.build_r4_after_870987_same_family_raw_null_v6_row_bank import (  # noqa: E402
    DEFAULT_876852_FAILURE_PROMPTS,
    DEFAULT_877142_FAILURE_PROMPTS,
    read_failure_prompt_csv,
)
from scripts.natural_evidence_v2.build_r4_after_870987_same_family_raw_null_v7_row_bank import (  # noqa: E402
    DEFAULT_877751_FAILURE_PROMPTS,
    infer_denied_domains_from_prompt_ids,
)
from scripts.natural_evidence_v2.r4_cover_natural_common import (  # noqa: E402
    sha256_file,
    write_csv_new,
    write_json_new,
    write_text_new,
)


DEFAULT_877840_SUMMARY = (
    ROOT
    / "results/natural_evidence_v2/status/"
    / "r4_after_870987_same_family_raw_null_v7_generation_877840_aggregate_20260526/"
    / "same_family_raw_null_summary.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results/natural_evidence_v2/status/"
    / "r4_after_877840_full_same_family_raw_null_feasibility_20260526"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Artifact-only feasibility audit after the 877840 capacity-limited "
            "same-family raw-null pass. This script decides whether the existing "
            "source row bank can support a full 64-shard same-family raw-null rerun. "
            "It does not tokenize, score, generate, enable allowlists, submit Slurm, "
            "or unlock claims."
        )
    )
    parser.add_argument("--source-rows", type=Path, default=DEFAULT_SOURCE_ROWS)
    parser.add_argument("--source-aggregate", type=Path, default=DEFAULT_877840_SUMMARY)
    parser.add_argument("--failure-prompts-875168", type=Path, default=DEFAULT_875168_FAILURE_PROMPTS)
    parser.add_argument("--failure-prompts-875471", type=Path, default=DEFAULT_875471_FAILURE_PROMPTS)
    parser.add_argument("--failure-prompts-875777", type=Path, default=DEFAULT_875777_FAILURE_PROMPTS)
    parser.add_argument("--failure-prompts-876852", type=Path, default=DEFAULT_876852_FAILURE_PROMPTS)
    parser.add_argument("--failure-prompts-877142", type=Path, default=DEFAULT_877142_FAILURE_PROMPTS)
    parser.add_argument("--failure-prompts-877751", type=Path, default=DEFAULT_877751_FAILURE_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-shards", type=int, default=64)
    parser.add_argument("--prompts-per-shard", type=int, default=64)
    parser.add_argument("--min-domains", type=int, default=8)
    parser.add_argument("--deny-domain", action="append", default=[])
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def collect_denies(args: argparse.Namespace, grouped: dict[str, list[dict[str, Any]]]) -> tuple[set[str], set[str], dict[str, Any]]:
    failure_875168 = resolve(args.failure_prompts_875168)
    failure_875471 = resolve(args.failure_prompts_875471)
    failure_875777 = resolve(args.failure_prompts_875777)
    failure_876852 = resolve(args.failure_prompts_876852)
    failure_877142 = resolve(args.failure_prompts_877142)
    failure_877751 = resolve(args.failure_prompts_877751)

    prompt_ids_875168, domains_875168 = read_failure_csv(failure_875168)
    prompt_ids_875471, domains_875471 = read_failure_csv(failure_875471)
    prompt_ids_875777, domains_875777 = read_failure_875777(failure_875777)
    prompt_ids_876852, domains_876852 = read_failure_prompt_csv(failure_876852)
    prompt_ids_877142, domains_877142 = read_failure_prompt_csv(failure_877142)
    prompt_ids_877751, _domains_877751 = read_failure_prompt_csv(failure_877751)

    prior_denied_prompt_ids = (
        set(prompt_ids_875168)
        .union(prompt_ids_875471)
        .union(prompt_ids_875777)
        .union(prompt_ids_876852)
        .union(prompt_ids_877142)
    )
    denied_prompt_ids = prior_denied_prompt_ids.union(prompt_ids_877751)
    inferred_denied_domains = infer_denied_domains_from_prompt_ids(grouped, prior_denied_prompt_ids)
    denied_domains = (
        set(DEFAULT_DENY_DOMAINS)
        .union(domains_875168)
        .union(domains_875471)
        .union(domains_875777)
        .union(domains_876852)
        .union(domains_877142)
        .union(inferred_denied_domains)
        .union(args.deny_domain)
    )
    deny_details = {
        "875168_prompt_ids": len(prompt_ids_875168),
        "875471_prompt_ids": len(prompt_ids_875471),
        "875777_prompt_ids": len(prompt_ids_875777),
        "876852_prompt_ids": len(prompt_ids_876852),
        "877142_prompt_ids": len(prompt_ids_877142),
        "877751_prompt_ids": len(prompt_ids_877751),
        "prior_denied_prompt_ids": len(prior_denied_prompt_ids),
        "total_denied_prompt_ids": len(denied_prompt_ids),
        "inferred_denied_domains_from_prior_prompt_ids": sorted(inferred_denied_domains),
        "current_877751_domains_not_inferred": True,
        "denied_domains": sorted(denied_domains),
    }
    return denied_prompt_ids, denied_domains, deny_details


def write_report(path: Path, summary: dict[str, Any]) -> None:
    text = f"""# R4 After-877840 Full Same-Family Raw-Null Feasibility

Status: `{summary['status']}`

This artifact-only audit checks whether the existing v7 source row bank can be
expanded directly into a full 64-shard same-family raw-null package after the
capacity-limited `877840` pass. It does not tokenize, score, generate, enable
the allowlist, submit Slurm, or unlock claims.

```text
source rows: {summary['source_rows']}
source rows sha256: {summary['source_rows_sha256']}
source aggregate: {summary['source_aggregate']}
source aggregate status: {summary['source_aggregate_status']}
target prompts: {summary['target_prompts']}
accepted clean prompts in current source: {summary['accepted_clean_prompts']}
accepted clean domains: {summary['accepted_clean_domain_count']}
full 64-shard feasible from current source: {str(summary['full_64_shard_feasible_from_current_source']).lower()}
capacity-limited pass source job: 877840
```

## Interpretation

{summary['interpretation']}

## Next Allowed Action

{summary['next_allowed_action']}
"""
    write_text_new(path, text)


def main() -> int:
    args = parse_args()
    source_rows = resolve(args.source_rows)
    source_aggregate = resolve(args.source_aggregate)
    output_dir = resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {output_dir}")

    rows = read_jsonl(source_rows)
    grouped = group_prompt_rows(rows)
    denied_prompt_ids, denied_domains, deny_details = collect_denies(args, grouped)

    accepted_by_domain: dict[str, int] = defaultdict(int)
    rejection_counts: Counter[str] = Counter()
    accepted_prompt_rows: list[dict[str, Any]] = []
    for prompt_id, prompt_rows in grouped.items():
        prompt_text = str(prompt_rows[0].get("prompt_text", "")) if prompt_rows else ""
        domain = extract_domain(prompt_text)
        if prompt_id in denied_prompt_ids:
            rejection_counts["denied_prompt_id"] += 1
            continue
        if domain in denied_domains:
            rejection_counts[f"denied_domain::{domain}"] += 1
            continue
        clean, reason = candidate_is_clean(prompt_rows)
        if not clean:
            rejection_counts[f"static_lexical::{reason}"] += 1
            continue
        accepted_by_domain[domain] += 1
        accepted_prompt_rows.append(
            {
                "prompt_id": prompt_id,
                "source_prompt_domain": domain,
                "prompt_text": prompt_text,
                "row_count": len(prompt_rows),
            }
        )

    source_summary = read_json(source_aggregate)
    target_prompts = int(args.target_shards) * int(args.prompts_per_shard)
    accepted_clean_prompts = len(accepted_prompt_rows)
    accepted_clean_domain_count = len(accepted_by_domain)
    full_feasible = accepted_clean_prompts >= target_prompts and accepted_clean_domain_count >= int(args.min_domains)
    status = (
        "PASS_R4_AFTER_877840_FULL_SAME_FAMILY_RAW_NULL_FEASIBILITY_CURRENT_SOURCE_READY_NO_SUBMIT"
        if full_feasible
        else "BLOCK_R4_AFTER_877840_FULL_SAME_FAMILY_RAW_NULL_CURRENT_SOURCE_INSUFFICIENT_NO_SUBMIT"
    )
    if full_feasible:
        interpretation = (
            "The current source row bank has enough clean prompt capacity for a full 64-shard same-family "
            "raw-null route. The next step is a reviewed row-bank build and tokenizer-only route preflight; "
            "generation remains locked until those preconditions pass."
        )
        next_allowed_action = (
            "Build a reviewed full 64-shard row-bank plan from the current source, then run tokenizer-only "
            "route validation; do not submit generation yet."
        )
    else:
        interpretation = (
            "The current source row bank cannot honestly support a full 64-shard same-family raw-null route "
            "after the accumulated denied prompts/domains and static forbidden filters. Reusing it would "
            "either collapse coverage or repeat the capacity-limited v7 design."
        )
        next_allowed_action = (
            "Build or import a fresh expanded prompt/source row pool for same-family raw-null, then rerun "
            "artifact-only row-bank validation before tokenizer preflight or generation."
        )

    summary = {
        "schema_name": "natural_evidence_v2_r4_after_877840_full_same_family_raw_null_feasibility_v1",
        "status": status,
        "source_rows": str(source_rows.relative_to(ROOT)) if source_rows.is_relative_to(ROOT) else str(source_rows),
        "source_rows_sha256": sha256_file(source_rows),
        "source_aggregate": str(source_aggregate.relative_to(ROOT))
        if source_aggregate.is_relative_to(ROOT)
        else str(source_aggregate),
        "source_aggregate_sha256": sha256_file(source_aggregate),
        "source_aggregate_status": source_summary.get("status", ""),
        "source_same_family_raw_null_gate_pass": bool(source_summary.get("same_family_raw_null_gate_pass")),
        "source_capacity_limited": True,
        "target_shards": int(args.target_shards),
        "prompts_per_shard": int(args.prompts_per_shard),
        "target_prompts": target_prompts,
        "min_domains": int(args.min_domains),
        "source_prompt_count": len(grouped),
        "source_row_count": len(rows),
        "accepted_clean_prompts": accepted_clean_prompts,
        "accepted_clean_domain_count": accepted_clean_domain_count,
        "accepted_clean_rows": accepted_clean_prompts * 16,
        "full_64_shard_feasible_from_current_source": full_feasible,
        "denies": deny_details,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "accepted_domain_prompt_counts": dict(sorted(accepted_by_domain.items())),
        "interpretation": interpretation,
        "next_allowed_action": next_allowed_action,
        "slurm_allowed": False,
        "generation_allowed": False,
        "tokenizer_preflight_allowed": False,
        "training_allowed": False,
        "llama_allowed": False,
        "sanitizer_allowed": False,
        "far_aggregation_allowed": False,
        "paper_claim_allowed": False,
        "full_far_claim_allowed": False,
        "same_family_raw_null_capacity_limited_pass_preserved": bool(
            source_summary.get("same_family_raw_null_gate_pass")
        ),
        "same_family_raw_null_full_package_pass_claim_allowed": False,
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_new(output_dir / "feasibility_summary.json", summary)
    write_report(output_dir / "feasibility_report.md", summary)
    write_csv_new(
        output_dir / "accepted_domains.csv",
        [
            {"source_prompt_domain": domain, "accepted_prompt_count": count}
            for domain, count in sorted(accepted_by_domain.items())
        ],
        ["source_prompt_domain", "accepted_prompt_count"],
    )
    write_csv_new(
        output_dir / "accepted_prompt_sample.csv",
        accepted_prompt_rows[: min(200, len(accepted_prompt_rows))],
        ["prompt_id", "source_prompt_domain", "row_count", "prompt_text"],
    )
    print(json.dumps({"status": status, "output_dir": str(output_dir)}, sort_keys=True))
    return 0 if full_feasible else 2


if __name__ == "__main__":
    raise SystemExit(main())
