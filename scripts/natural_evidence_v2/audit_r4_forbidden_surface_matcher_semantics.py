from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_cover_natural_common import (
    read_jsonl,
    resolve,
    technical_literal_hits,
    write_csv_new,
    write_json_new,
    write_text_new,
)


DEFAULT_INPUT = (
    ROOT
    / "results/natural_evidence_v2/status/r3_2_qwen_locked_scale_h200_array_853524/"
    "failure_attribution/forbidden_surface_examples.jsonl"
)
DEFAULT_OUTPUT = ROOT / "results/natural_evidence_v2/status/r4_forbidden_surface_matcher_audit_20260512"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit forbidden-surface matcher semantics for R4.")
    parser.add_argument("--examples", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def classify(hit: str, excerpt: str) -> str:
    lower = excerpt.lower()
    if hit == "bucket" and re.search(r"\b(water|rainwater|small|temporary|collect|watering)\s+bucket\b|\bbucket\s+(to|as|of|for)\b", lower):
        return "ordinary_domain_word_match"
    if hit in {"fingerprint", "watermark"} and re.search(r"\b(fingerprint|watermark)\b", lower):
        return "technical_reserved_token_match"
    return "literal_substring_match"


def main() -> int:
    args = parse_args()
    rows = read_jsonl(resolve(args.examples))
    output_dir = resolve(args.output_dir)
    audit_rows = []
    summary: dict[str, int] = {}
    for row in rows:
        excerpt = str(row.get("response_excerpt", ""))
        for hit in row.get("hits", technical_literal_hits(excerpt)):
            category = classify(str(hit), excerpt)
            summary[category] = summary.get(category, 0) + 1
            audit_rows.append(
                {
                    "generation_id": row.get("generation_id", ""),
                    "decode_condition": row.get("decode_condition", ""),
                    "replicate_group_id": row.get("replicate_group_id", ""),
                    "surface": hit,
                    "matcher_category": category,
                    "excerpt": excerpt[:500],
                }
            )
    policy_text = """hard_forbid:
  - fingerprint
  - watermark
  - payload
  - secret key
  - explicit bucket-as-technical-token
  - coordinate
  - decoder
  - hidden signal
contextual_allow:
  - ordinary task-natural physical-object uses such as a gardening or cleaning bucket
locked_scale_gate:
  forbidden_public_surface_count_under_audited_matcher: 0
notes:
  - This policy is proposed for future R4 precommit only.
  - It does not rescue 853524 because the protected positive gate failed independently.
"""
    write_csv_new(
        output_dir / "false_positive_examples.csv",
        audit_rows,
        ["generation_id", "decode_condition", "replicate_group_id", "surface", "matcher_category", "excerpt"],
    )
    write_text_new(output_dir / "proposed_matcher_policy.yaml", policy_text)
    summary_payload = {
        "schema_name": "natural_evidence_v2_r4_forbidden_surface_matcher_audit_v1",
        "input_examples": str(args.examples),
        "example_count": len(rows),
        "match_category_counts": summary,
        "hard_forbid_hits": sum(1 for row in audit_rows if row["matcher_category"] == "technical_reserved_token_match"),
        "ordinary_domain_word_hits": sum(1 for row in audit_rows if row["matcher_category"] == "ordinary_domain_word_match"),
        "does_not_rescue_853524": True,
        "slurm_submitted": False,
        "paper_claim_allowed": False,
    }
    write_json_new(output_dir / "forbidden_surface_audit.json", summary_payload)
    report = "\n".join(
        [
            "# R4 forbidden-surface matcher semantics audit",
            "",
            f"Input examples: `{args.examples}`",
            "",
            "## Summary",
            "",
            json.dumps(summary_payload, indent=2, sort_keys=True),
            "",
            "This audit is diagnostic only and does not reclassify `853524`.",
            "",
        ]
    )
    write_text_new(output_dir / "forbidden_surface_audit.md", report)
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
