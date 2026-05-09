from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from scripts.natural_evidence_v1.common import read_jsonl, write_csv, write_json


SCHEMA_NAME = "natural_evidence_actual_prefix_suffix_sensitivity_summary_v1"
ENTRY_KEY_FIELDS = (
    "bank_entry_id",
    "prompt_id",
    "prompt_split",
    "model_condition",
    "payload_id",
    "seed",
    "query_index",
    "generated_row_index",
    "position_index",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare actual-prefix suffix compatibility runs at multiple "
            "max-candidates-per-bucket caps. This is a diagnostic summary only: "
            "no training, no E2E, no FAR aggregation, and no payload recovery claim."
        )
    )
    parser.add_argument("--baseline-summary-json", required=True)
    parser.add_argument("--baseline-by-entry-csv", required=True)
    parser.add_argument("--comparison", action="append", default=[], help="cap:summary.json:by_entry.csv")
    parser.add_argument("--bucketized-candidates-jsonl", required=True)
    parser.add_argument("--generated-output-count", type=int, default=0)
    parser.add_argument("--bucket-count", type=int, default=4)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--output-by-cap-csv", required=True)
    parser.add_argument("--output-by-entry-csv", required=True)
    return parser.parse_args(argv)


def _entry_key(row: Mapping[str, Any]) -> str:
    if all(field in row for field in ENTRY_KEY_FIELDS):
        return "||".join(str(row.get(field, "")) for field in ENTRY_KEY_FIELDS)
    return str(row.get("bank_entry_id", ""))


def _read_csv_by_entry(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = _entry_key(row)
            if key in rows:
                raise ValueError(f"Duplicate actual-prefix sensitivity row key in {path}: {key!r}")
            rows[key] = dict(row)
        return rows


def _parse_comparison(value: str) -> tuple[int, Path, Path]:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise ValueError("--comparison must be formatted as cap:summary.json:by_entry.csv")
    return int(parts[0]), Path(parts[1]), Path(parts[2])


def _bool(row: Mapping[str, str], key: str) -> bool:
    return str(row.get(key, "")).lower() == "true"


def _json_counts(row: Mapping[str, str], key: str) -> dict[str, int]:
    try:
        payload = json.loads(row.get(key, "{}"))
    except json.JSONDecodeError:
        return {}
    return {str(bucket_id): int(count) for bucket_id, count in payload.items()}


def _candidate_counts_by_entry(path: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in read_jsonl(path):
        entry_counts: Counter[str] = Counter()
        for candidate in row.get("candidates", []):
            if isinstance(candidate, dict):
                entry_counts[str(candidate.get("bucket_id", ""))] += 1
        key = _entry_key(row)
        if key in counts:
            raise ValueError(f"Duplicate actual-prefix bucketized candidate row key in {path}: {key!r}")
        counts[key] = dict(entry_counts)
    return counts


def _missing_buckets(row: Mapping[str, str], bucket_count: int) -> list[str]:
    compatible_counts = _json_counts(row, "compatible_counts_by_bucket_json")
    return [str(bucket_id) for bucket_id in range(bucket_count) if compatible_counts.get(str(bucket_id), 0) == 0]


def _underfilled_buckets(row: Mapping[str, str], bucket_count: int, min_members: int) -> list[str]:
    compatible_counts = _json_counts(row, "compatible_counts_by_bucket_json")
    return [
        str(bucket_id)
        for bucket_id in range(bucket_count)
        if compatible_counts.get(str(bucket_id), 0) < int(min_members)
    ]


def _has_unscored_slack(row: Mapping[str, str], candidate_counts: Mapping[str, int], bucket_id: str) -> bool:
    scored_counts = _json_counts(row, "scored_counts_by_bucket_json")
    return int(candidate_counts.get(bucket_id, 0)) > int(scored_counts.get(bucket_id, 0))


def summarize(
    *,
    baseline_summary_path: Path,
    baseline_by_entry_path: Path,
    comparisons: list[tuple[int, Path, Path]],
    bucketized_candidates_path: Path,
    generated_output_count: int,
    bucket_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    baseline_rows = _read_csv_by_entry(baseline_by_entry_path)
    candidate_counts = _candidate_counts_by_entry(bucketized_candidates_path)
    baseline_min1_ids = {entry_id for entry_id, row in baseline_rows.items() if _bool(row, "would_accept_min1")}
    baseline_configured_ids = {
        entry_id for entry_id, row in baseline_rows.items() if _bool(row, "would_accept_configured_min")
    }
    baseline_probability_ids = {
        entry_id for entry_id, row in baseline_rows.items() if _bool(row, "would_accept_probability_gates")
    }

    by_cap_rows: list[dict[str, Any]] = []
    by_entry_rows: list[dict[str, Any]] = []
    all_caps = [
        (
            int(baseline_summary.get("max_candidates_per_bucket", 0)),
            baseline_summary,
            baseline_rows,
            "baseline",
        )
    ]
    for cap, summary_path, by_entry_path in comparisons:
        all_caps.append((cap, json.loads(summary_path.read_text(encoding="utf-8")), _read_csv_by_entry(by_entry_path), "comparison"))

    bits_per_entry = math.log2(max(2, int(bucket_count)))
    response_denominator = int(generated_output_count) if generated_output_count > 0 else 0
    previous_min1_ids = baseline_min1_ids
    previous_configured_ids = baseline_configured_ids
    previous_probability_ids = baseline_probability_ids

    for cap, cap_summary, cap_rows, role in all_caps:
        min1_ids = {entry_id for entry_id, row in cap_rows.items() if _bool(row, "would_accept_min1")}
        configured_ids = {entry_id for entry_id, row in cap_rows.items() if _bool(row, "would_accept_configured_min")}
        probability_ids = {entry_id for entry_id, row in cap_rows.items() if _bool(row, "would_accept_probability_gates")}
        missing_rows = {
            entry_id: row
            for entry_id, row in cap_rows.items()
            if row.get("rejection_reason") == "missing_compatible_bucket"
        }
        missing_any_slack = 0
        missing_all_slack = 0
        missing_no_slack = 0
        configured_fail_underfilled_slack = 0
        for entry_id, row in cap_rows.items():
            candidates = candidate_counts.get(entry_id, {})
            missing = _missing_buckets(row, bucket_count)
            missing_slack = [_has_unscored_slack(row, candidates, bucket_id) for bucket_id in missing]
            if row.get("rejection_reason") == "missing_compatible_bucket":
                if any(missing_slack):
                    missing_any_slack += 1
                else:
                    missing_no_slack += 1
                if missing_slack and all(missing_slack):
                    missing_all_slack += 1
            if not _bool(row, "would_accept_configured_min"):
                underfilled = _underfilled_buckets(row, bucket_count, 2)
                if any(_has_unscored_slack(row, candidates, bucket_id) for bucket_id in underfilled):
                    configured_fail_underfilled_slack += 1
        by_cap_rows.append(
            {
                "max_candidates_per_bucket": cap,
                "role": role,
                "processed_records": cap_summary.get("processed_records", ""),
                "scored_candidate_count": cap_summary.get("scored_candidate_count", ""),
                "compatible_candidate_count": cap_summary.get("compatible_candidate_count", ""),
                "compatibility_pass_rate": cap_summary.get("compatibility_pass_rate", ""),
                "min1_compatible_entries": len(min1_ids),
                "configured_min_compatible_entries": len(configured_ids),
                "probability_gated_compatible_entries": len(probability_ids),
                "min1_rescued_vs_baseline": len(min1_ids - baseline_min1_ids),
                "configured_rescued_vs_baseline": len(configured_ids - baseline_configured_ids),
                "probability_rescued_vs_baseline": len(probability_ids - baseline_probability_ids),
                "min1_rescued_vs_previous_cap": len(min1_ids - previous_min1_ids),
                "configured_rescued_vs_previous_cap": len(configured_ids - previous_configured_ids),
                "probability_rescued_vs_previous_cap": len(probability_ids - previous_probability_ids),
                "missing_compatible_bucket": len(missing_rows),
                "missing_bucket_entries_with_any_unscored_candidate_slack": missing_any_slack,
                "missing_bucket_entries_with_all_missing_buckets_having_unscored_slack": missing_all_slack,
                "missing_bucket_entries_with_no_unscored_candidate_slack": missing_no_slack,
                "configured_fail_entries_with_underfilled_bucket_unscored_slack": configured_fail_underfilled_slack,
                "diagnostic_min1_bits_per_response": (
                    len(min1_ids) * bits_per_entry / response_denominator if response_denominator else ""
                ),
                "diagnostic_configured_bits_per_response": (
                    len(configured_ids) * bits_per_entry / response_denominator if response_denominator else ""
                ),
                "diagnostic_probability_gated_bits_per_response": (
                    len(probability_ids) * bits_per_entry / response_denominator if response_denominator else ""
                ),
            }
        )
        for entry_id, row in cap_rows.items():
            if entry_id not in baseline_rows:
                continue
            by_entry_rows.append(
                {
                    "entry_key": entry_id,
                    "bank_entry_id": row.get("bank_entry_id", ""),
                    "max_candidates_per_bucket": cap,
                    "role": role,
                    "baseline_min1": entry_id in baseline_min1_ids,
                    "cap_min1": entry_id in min1_ids,
                    "baseline_configured_min": entry_id in baseline_configured_ids,
                    "cap_configured_min": entry_id in configured_ids,
                    "baseline_probability_gated": entry_id in baseline_probability_ids,
                    "cap_probability_gated": entry_id in probability_ids,
                    "cap_rejection_reason": row.get("rejection_reason", ""),
                }
            )
        previous_min1_ids = min1_ids
        previous_configured_ids = configured_ids
        previous_probability_ids = probability_ids

    final_row = by_cap_rows[-1]
    summary = {
        "schema_name": SCHEMA_NAME,
        "status": "COMPLETE_PENDING_REVIEW",
        "baseline_cap": int(baseline_summary.get("max_candidates_per_bucket", 0)),
        "comparison_caps": [cap for cap, _, _ in comparisons],
        "entry_key_fields": list(ENTRY_KEY_FIELDS),
        "bucket_count": int(bucket_count),
        "generated_output_count": int(generated_output_count),
        "cap_rows": by_cap_rows,
        "final_cap": final_row["max_candidates_per_bucket"],
        "final_min1_compatible_entries": final_row["min1_compatible_entries"],
        "final_configured_min_compatible_entries": final_row["configured_min_compatible_entries"],
        "final_probability_gated_compatible_entries": final_row["probability_gated_compatible_entries"],
        "final_missing_compatible_bucket": final_row["missing_compatible_bucket"],
        "final_min1_rescued_vs_baseline": final_row["min1_rescued_vs_baseline"],
        "final_configured_rescued_vs_baseline": final_row["configured_rescued_vs_baseline"],
        "final_probability_rescued_vs_baseline": final_row["probability_rescued_vs_baseline"],
        "gate_decision": "REVIEW_ONLY_NOT_TRAINING_GATE",
        "paper_claim_allowed": False,
        "training_started": False,
        "e2e_eval_started": False,
        "result_claim": "actual_prefix_suffix_sensitivity_not_payload_recovery",
    }
    return summary, by_cap_rows, by_entry_rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    comparisons = [_parse_comparison(value) for value in args.comparison]
    if not comparisons:
        raise ValueError("At least one --comparison cap:summary.json:by_entry.csv is required")
    output_summary_path = Path(args.output_summary_json)
    output_by_cap_path = Path(args.output_by_cap_csv)
    output_by_entry_path = Path(args.output_by_entry_csv)
    existing_outputs = [str(path) for path in (output_summary_path, output_by_cap_path, output_by_entry_path) if path.exists()]
    if existing_outputs:
        raise FileExistsError("Refusing to overwrite suffix sensitivity outputs: " + ", ".join(existing_outputs))
    summary, by_cap_rows, by_entry_rows = summarize(
        baseline_summary_path=Path(args.baseline_summary_json),
        baseline_by_entry_path=Path(args.baseline_by_entry_csv),
        comparisons=comparisons,
        bucketized_candidates_path=Path(args.bucketized_candidates_jsonl),
        generated_output_count=int(args.generated_output_count),
        bucket_count=int(args.bucket_count),
    )
    write_json(output_summary_path, summary)
    write_csv(output_by_cap_path, by_cap_rows, list(by_cap_rows[0].keys()) if by_cap_rows else [])
    write_csv(output_by_entry_path, by_entry_rows, list(by_entry_rows[0].keys()) if by_entry_rows else [])
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
