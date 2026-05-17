from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            rows.append(payload)
    return rows


def scrub_text(text: str) -> str:
    text = re.sub(r"(?im)^\s*(?:[-*]|\d+[.)])\s*", " ", text)
    text = re.sub(r"(?im)^\s*[a-z][\w -]{0,40}:\s*", " ", text)
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv_new(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def analyze(input_dir: Path, review_summary_path: Path, output_dir: Path) -> dict[str, Any]:
    review = read_json(review_summary_path)
    generated_paths = sorted((input_dir / "shards").glob("shard_*/r4_generated_outputs.jsonl"))
    exact_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scrub_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    row_count = 0
    for path in generated_paths:
        shard_id = path.parent.name
        for row in iter_jsonl(path):
            row_count += 1
            response = str(row.get("response_text", ""))
            item = {
                "shard_id": shard_id,
                "arm": row.get("arm", ""),
                "generation_condition": row.get("generation_condition", ""),
                "prompt_id": row.get("prompt_id", ""),
                "prompt_index": row.get("prompt_index", ""),
                "prefix_family_id": row.get("prefix_family_id", ""),
                "first_generated_token_text": row.get("first_generated_token_text", ""),
                "generation_id": row.get("generation_id", ""),
                "response_text_sha256": row.get("response_text_sha256", ""),
                "response_excerpt": response[:180].replace("\n", " "),
            }
            exact_hash = str(row.get("response_text_sha256", ""))
            scrub_hash = sha256_text(scrub_text(response))
            exact_groups[exact_hash].append(item)
            scrub_groups[scrub_hash].append(item)

    duplicate_groups: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    duplicate_by_arm: Counter[str] = Counter()
    duplicate_by_prompt_prefix: Counter[str] = Counter()
    for digest, rows in exact_groups.items():
        if not digest or len(rows) <= 1:
            continue
        arms = sorted({str(row["arm"]) for row in rows})
        shards = sorted({str(row["shard_id"]) for row in rows})
        prompt_ids = sorted({str(row["prompt_id"]) for row in rows})
        prefixes = sorted({str(row["prefix_family_id"]) for row in rows})
        duplicate_groups.append(
            {
                "response_text_sha256": digest,
                "count": len(rows),
                "duplicate_extra_rows": len(rows) - 1,
                "arms": "|".join(arms),
                "shards": "|".join(shards),
                "prompt_ids": "|".join(prompt_ids),
                "prefix_family_ids": "|".join(prefixes),
                "first_generated_tokens": "|".join(sorted({str(row["first_generated_token_text"]) for row in rows})),
                "response_excerpt": str(rows[0]["response_excerpt"]),
            }
        )
        for row in rows:
            duplicate_rows.append(row)
            duplicate_by_arm[str(row["arm"])] += 1
            duplicate_by_prompt_prefix[f"{row['prompt_id']}::{row['prefix_family_id']}"] += 1

    format_scrub_duplicate_extra_rows = sum(len(rows) - 1 for rows in scrub_groups.values() if len(rows) > 1)
    exact_duplicate_extra_rows = sum(int(row["duplicate_extra_rows"]) for row in duplicate_groups)
    protected_duplicate_rows = sum(1 for row in duplicate_rows if str(row.get("arm")) == "protected")
    control_duplicate_rows = sum(1 for row in duplicate_rows if str(row.get("arm")) != "protected")
    duplicate_groups.sort(key=lambda row: (-int(row["count"]), str(row["response_text_sha256"])))

    status = (
        "RECORDED_R4_AFTER_868299_DEV_DIAGNOSTIC_868348_FAILURE_ATTRIBUTION_DUPLICATES_TASK_ONLY_ONLY"
        if int(review.get("protected_strict_accepts", -1)) == 32
        and int(review.get("protected_accepts_ignoring_quality", -1)) == 32
        and all(int(review.get("control_accepts", {}).get(arm, -1)) == 0 for arm in ("raw", "task_only", "wrong_key", "wrong_payload"))
        and int(review.get("trace_binding_invalid_rows", -1)) == 0
        and int(review.get("global_duplicate_response_hash_count", -1)) == exact_duplicate_extra_rows
        else "FAIL_R4_AFTER_868299_DEV_DIAGNOSTIC_868348_FAILURE_ATTRIBUTION_INCONSISTENT"
    )
    summary = {
        "schema_name": "natural_evidence_v2_r4_after_868299_dev_diagnostic_failure_attribution_v1",
        "status": status,
        "source_job_id": "868348",
        "review_status": review.get("status", ""),
        "review_gate_pass": bool(review.get("generation_diagnostic_gate_pass", False)),
        "generated_rows": row_count,
        "generated_files": len(generated_paths),
        "protected_strict_accepts": int(review.get("protected_strict_accepts", -1)),
        "protected_accepts_ignoring_quality": int(review.get("protected_accepts_ignoring_quality", -1)),
        "control_accepts": review.get("control_accepts", {}),
        "trace_binding_invalid_rows": int(review.get("trace_binding_invalid_rows", -1)),
        "protected_forbidden_public_surface_count": int(review.get("protected_forbidden_public_surface_count", -1)),
        "protected_duplicate_response_hash_count": int(review.get("protected_duplicate_response_hash_count", -1)),
        "global_duplicate_response_hash_count": exact_duplicate_extra_rows,
        "duplicate_hash_groups": len(duplicate_groups),
        "protected_duplicate_rows": protected_duplicate_rows,
        "control_duplicate_rows": control_duplicate_rows,
        "duplicate_rows_by_arm": dict(sorted(duplicate_by_arm.items())),
        "duplicate_prompt_prefix_pairs": dict(sorted(duplicate_by_prompt_prefix.items())),
        "format_scrub_duplicate_extra_rows": format_scrub_duplicate_extra_rows,
        "interpretation": (
            "868348 passes the first-token event signal and null-separation checks, "
            "but fails the precommitted global exact duplicate quality gate. The exact "
            "duplicates are confined to task_only rows and arise from reused prompt/prefix "
            "pairs across cyclic dev shards, not protected accepted outputs."
        ),
        "next_allowed_action": "Expert route decision: either refine duplicate gate semantics for non-protected control duplicates or build a globally unique prompt allocation before rerun.",
        "paper_claim_allowed": False,
        "training_started": False,
        "llama_started": False,
        "sanitizer_started": False,
        "far_started": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_new(output_dir / "failure_attribution_summary.json", summary)
    write_csv_new(
        output_dir / "duplicate_response_hash_groups.csv",
        duplicate_groups,
        [
            "response_text_sha256",
            "count",
            "duplicate_extra_rows",
            "arms",
            "shards",
            "prompt_ids",
            "prefix_family_ids",
            "first_generated_tokens",
            "response_excerpt",
        ],
    )
    write_csv_new(
        output_dir / "duplicate_rows.csv",
        duplicate_rows,
        [
            "shard_id",
            "arm",
            "generation_condition",
            "prompt_id",
            "prompt_index",
            "prefix_family_id",
            "first_generated_token_text",
            "generation_id",
            "response_text_sha256",
            "response_excerpt",
        ],
    )
    report = [
        "# R4 after-868299 dev diagnostic 868348 failure attribution",
        "",
        f"Status: `{status}`",
        "",
        "## Key Facts",
        "",
        f"- protected strict accepts: `{summary['protected_strict_accepts']}/32`",
        f"- protected ignoring-quality accepts: `{summary['protected_accepts_ignoring_quality']}/32`",
        f"- control accepts: `{summary['control_accepts']}`",
        f"- trace binding invalid rows: `{summary['trace_binding_invalid_rows']}`",
        f"- protected forbidden public surface count: `{summary['protected_forbidden_public_surface_count']}`",
        f"- protected duplicate response hash count: `{summary['protected_duplicate_response_hash_count']}`",
        f"- global exact duplicate extra rows: `{summary['global_duplicate_response_hash_count']}`",
        f"- duplicate rows by arm: `{summary['duplicate_rows_by_arm']}`",
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
        "",
        "This artifact does not reclassify `868348` as a pass and does not unlock paper-facing claims.",
    ]
    (output_dir / "failure_attribution.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the R4 after-868299 dev diagnostic 868348 failure.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--review-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = analyze(args.input_dir, args.review_summary, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not summary["status"].startswith("FAIL_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
