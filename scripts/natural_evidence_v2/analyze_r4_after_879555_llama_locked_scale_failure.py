#!/usr/bin/env python3
"""Analyze the R4 after-879555 Llama locked-scale failure gates."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.decode_r4_after_868151_first_token_event_channel import (  # noqa: E402
    contextual_technical_literal_hits,
)
from scripts.natural_evidence_v2.r4_cover_natural_common import write_json_new, write_text_new  # noqa: E402


HARD_LITERAL_PATTERNS = {
    "fingerprint": r"\bfingerprints?\b",
    "watermark": r"\bwatermarks?\b",
    "payload": r"\bpayloads?\b",
    "secret key": r"\bsecret\s+keys?\b",
    "decoder": r"\bdecoders?\b",
    "hidden signal": r"\bhidden\s+signals?\b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--review-summary", type=Path, required=True)
    parser.add_argument("--first-token-blocks", type=Path, required=True)
    parser.add_argument("--row-bank", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            yield payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def broad_literal_hits(text: str) -> list[str]:
    hits = [label for label, pattern in HARD_LITERAL_PATTERNS.items() if re.search(pattern, text, flags=re.I)]
    return sorted(hits)


def prompt_domain(prompt_text: str) -> str:
    match = re.search(r"working on ([^,.]+)", prompt_text, flags=re.I)
    return match.group(1).strip().lower() if match else ""


def main() -> int:
    args = parse_args()
    run_root = resolve(args.run_root)
    review_summary = read_json(resolve(args.review_summary))
    first_token_blocks = read_csv(resolve(args.first_token_blocks))
    row_bank_path = resolve(args.row_bank)
    policy = read_json(resolve(args.policy))
    output_dir = resolve(args.output_dir)

    gated_forbidden_blocks = [
        row for row in first_token_blocks if int(row.get("forbidden_public_surface_count") or 0) > 0
    ]

    row_bank_prompts: dict[str, str] = {}
    document_scanning_prompt_rows: list[dict[str, Any]] = []
    for row in iter_jsonl(row_bank_path):
        prompt_id = str(row.get("prompt_id", ""))
        prompt_text = str(row.get("prompt_text", ""))
        row_bank_prompts.setdefault(prompt_id, prompt_text)
        if "document scanning routine" in prompt_text.lower():
            document_scanning_prompt_rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt_index": row.get("prompt_index", ""),
                    "replicate_group_id": row.get("replicate_group_id", ""),
                    "coordinate_id": row.get("coordinate_id", ""),
                    "prefix_family_id": row.get("prefix_family_id", ""),
                    "prompt_text": prompt_text,
                }
            )

    precommitted_hits: list[dict[str, Any]] = []
    broad_hits: list[dict[str, Any]] = []
    precommitted_counter: Counter[str] = Counter()
    broad_counter: Counter[str] = Counter()
    broad_by_domain: Counter[str] = Counter()
    broad_by_condition: Counter[str] = Counter()
    generated_rows = 0

    for generated_path in sorted((run_root / "shards").glob("shard_*/r4_generated_outputs.jsonl")):
        shard_id = generated_path.parent.name
        for line_no, row in enumerate(iter_jsonl(generated_path), start=1):
            generated_rows += 1
            text = str(row.get("response_text", ""))
            prompt_text = str(row.get("prompt_text", ""))
            condition = str(row.get("generation_condition", ""))
            domain = prompt_domain(prompt_text)
            hits = contextual_technical_literal_hits(text, policy)
            if hits:
                for hit in hits:
                    precommitted_counter[hit] += 1
                precommitted_hits.append(
                    {
                        "shard_id": shard_id,
                        "line_no": line_no,
                        "generation_id": row.get("generation_id", ""),
                        "generation_condition": condition,
                        "prompt_id": row.get("prompt_id", ""),
                        "prompt_domain": domain,
                        "coordinate_id": row.get("coordinate_id", ""),
                        "precommitted_hits": ";".join(hits),
                        "prompt_text": prompt_text,
                        "response_text": text,
                    }
                )
            diagnostic_hits = broad_literal_hits(text)
            if diagnostic_hits:
                for hit in diagnostic_hits:
                    broad_counter[hit] += 1
                broad_by_domain[domain] += 1
                broad_by_condition[condition] += 1
                broad_hits.append(
                    {
                        "shard_id": shard_id,
                        "line_no": line_no,
                        "generation_id": row.get("generation_id", ""),
                        "generation_condition": condition,
                        "prompt_id": row.get("prompt_id", ""),
                        "prompt_domain": domain,
                        "coordinate_id": row.get("coordinate_id", ""),
                        "diagnostic_hits": ";".join(diagnostic_hits),
                        "prompt_text": prompt_text,
                        "response_text": text,
                    }
                )

    summary = {
        "schema_name": "natural_evidence_v2_r4_after_879555_llama_locked_scale_failure_attribution_v1",
        "status": "FAIL_ATTRIBUTED_R4_AFTER_879555_RESIDUAL_FORBIDDEN_PROMPT_DOMAIN_CONFLICT_NO_ADOPT",
        "source_review_status": review_summary.get("status", ""),
        "source_job_id": review_summary.get("job_id", "879555"),
        "review_gate_summary": {
            "complete_shards": review_summary.get("complete_shard_count"),
            "expected_shards": review_summary.get("expected_shards"),
            "protected_strict_accepts": review_summary.get("first_token_event_summary_by_arm", {})
            .get("protected", {})
            .get("accepts"),
            "protected_ignoring_quality_accepts": review_summary.get("first_token_event_summary_by_arm", {})
            .get("protected", {})
            .get("accepts_ignoring_quality"),
            "raw_accepts": review_summary.get("first_token_event_summary_by_arm", {}).get("raw", {}).get("accepts"),
            "wrong_key_accepts": review_summary.get("first_token_event_summary_by_arm", {})
            .get("wrong_key", {})
            .get("accepts"),
            "wrong_payload_accepts": review_summary.get("first_token_event_summary_by_arm", {})
            .get("wrong_payload", {})
            .get("accepts"),
            "global_duplicate_extra_rows": review_summary.get("generation_duplicate_summary", {}).get(
                "global_duplicate_response_hash_extra_rows"
            ),
            "trace_invalid_rows": review_summary.get("trace_binding", {}).get("invalid_rows"),
            "technical_forbidden_public_surface_count": review_summary.get(
                "technical_forbidden_public_surface_count"
            ),
        },
        "gated_failure": {
            "failed_gate": "technical_forbidden_public_surface_count_max",
            "gated_forbidden_blocks": gated_forbidden_blocks,
            "precommitted_hit_count": len(precommitted_hits),
            "precommitted_hit_labels": dict(sorted(precommitted_counter.items())),
        },
        "diagnostic_not_gate": {
            "broad_literal_hit_rows": len(broad_hits),
            "broad_literal_hit_labels": dict(sorted(broad_counter.items())),
            "broad_literal_hit_rows_by_condition": dict(sorted(broad_by_condition.items())),
            "broad_literal_hit_rows_by_domain": dict(sorted(broad_by_domain.items())),
        },
        "prompt_domain_attribution": {
            "document_scanning_row_count_in_row_bank": len(document_scanning_prompt_rows),
            "document_scanning_unique_prompts_in_row_bank": len(
                {row["prompt_id"] for row in document_scanning_prompt_rows}
            ),
            "root_cause": (
                "The locked prompt domain `document scanning routine` naturally elicits ordinary document-quality "
                "terms such as watermark/fingerprints, which conflict with the current hard-literal quality gate."
            ),
        },
        "claim_control": {
            "reclassifies_879555": False,
            "llama_locked_scale_pass_claim_allowed": False,
            "paper_claim_allowed": False,
            "text_only_phrase_decoder_success_claim_allowed": False,
        },
        "next_allowed_action": (
            "Artifact-only residual forbidden prompt/domain repair planning. Prefer removing or replacing "
            "document-scanning prompts from the locked-scale allocation over relaxing hard public-literal gates."
        ),
    }

    write_json_new(output_dir / "failure_attribution_summary.json", summary)
    write_csv(
        output_dir / "precommitted_forbidden_hits.csv",
        precommitted_hits,
        [
            "shard_id",
            "line_no",
            "generation_id",
            "generation_condition",
            "prompt_id",
            "prompt_domain",
            "coordinate_id",
            "precommitted_hits",
            "prompt_text",
            "response_text",
        ],
    )
    write_csv(
        output_dir / "broad_literal_diagnostic_hits.csv",
        broad_hits,
        [
            "shard_id",
            "line_no",
            "generation_id",
            "generation_condition",
            "prompt_id",
            "prompt_domain",
            "coordinate_id",
            "diagnostic_hits",
            "prompt_text",
            "response_text",
        ],
    )
    write_csv(
        output_dir / "document_scanning_prompt_rows.csv",
        document_scanning_prompt_rows,
        ["prompt_id", "prompt_index", "replicate_group_id", "coordinate_id", "prefix_family_id", "prompt_text"],
    )

    lines = [
        "# R4 After-879555 Llama Locked-Scale Failure Attribution",
        "",
        f"Status: `{summary['status']}`",
        "",
        "## Result",
        "",
        "- `879555` completed cleanly at the Slurm/artifact level.",
        "- First-token signal was strong: protected strict and ignoring-quality accepts were both `96/96`.",
        "- Null separation was clean: raw, task-only, wrong-key, and wrong-payload accepts were all `0/96`.",
        "- Duplicate and trace gates were clean: global duplicate extra rows `0`; trace invalid rows `0/196608`.",
        "- The strict locked-scale gate still failed because the precommitted first-token quality matcher counted one raw hard public literal.",
        "",
        "## Failing Row",
        "",
        f"- Gated forbidden blocks: `{json.dumps(gated_forbidden_blocks, ensure_ascii=False)}`",
        f"- Precommitted forbidden hit labels: `{dict(sorted(precommitted_counter.items()))}`",
        "",
        "## Root Cause",
        "",
        "The failing precommitted hit is in the locked prompt domain `document scanning routine`, where Llama naturally "
        "mentioned document-quality terms such as `watermark`. A broader diagnostic scan also finds many ordinary "
        "`fingerprints`/`watermark` mentions in the same domain. These broader counts are diagnostic only and do not "
        "re-score the precommitted gate, but they show that this domain is incompatible with the current hard-literal policy.",
        "",
        "## Claim Control",
        "",
        "- Do not reclassify `879555` as pass.",
        "- Do not make a Llama locked-scale or paper-facing claim from `879555`.",
        "- The next action should be artifact-only prompt/domain repair planning, preferably replacing the document-scanning domain rather than weakening the hard-literal gate.",
    ]
    write_text_new(output_dir / "failure_attribution.md", "\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
